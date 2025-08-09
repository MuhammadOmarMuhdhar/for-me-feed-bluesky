"""
Main ETL orchestrator that coordinates all ranking modules.
"""
import os
import sys
import argparse
import json
import logging
from datetime import datetime

# Add parent directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
env_file_path = os.path.join(project_root, '.env')
if os.path.exists(env_file_path):
    with open(env_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

from client.bigQuery import Client as BigQueryClient
from client.redis import Client as RedisClient
from experiments.abTesting import get_ab_test_manager

from ETL.ranking.config import LoggingConfig
from ETL.ranking.userData import get_active_users_with_keywords, get_user_embeddings, get_user_reading_level, process_user_keywords
from ETL.ranking.postCollector import collect_comprehensive_posts
from ETL.ranking.rankingEngine import calculate_rankings_with_feed_boosting
from ETL.ranking.cacheManager import cache_user_rankings, update_default_feed_if_needed

# Configure logging
LoggingConfig.configure_logging()
logger = logging.getLogger(__name__)


def main():
    """Main ETL process"""
    parser = argparse.ArgumentParser(description='Feed Ranking ETL')
    parser.add_argument('--test-mode', default='false', help='Run in test mode with limited users')
    args = parser.parse_args()
    
    test_mode = args.test_mode.lower() == 'true'
    batch_id = f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting Feed Ranking ETL (test_mode={test_mode}, batch_id={batch_id})")
    
    try:
        # Initialize clients
        credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
        bq_client = BigQueryClient(credentials_json, os.environ['BIGQUERY_PROJECT_ID'])
        redis_client = RedisClient()
        
        
        # Get users with their stored keywords (Redis + BigQuery hybrid)
        users = get_active_users_with_keywords(redis_client, bq_client, test_mode)
        
        if not users:
            logger.warning("No users found to process, but will still update default feed")
            success_count, error_count = 0, 0
        else:
            logger.info(f"Processing {len(users)} users")
            success_count = 0
            error_count = 0
        
        for user in users:
            user_handle = user.get('handle', '')
            user_id = user.get('user_id', user_handle)
            
            try:
                logger.info(f"Processing user: {user_handle}")
                
                # Step 1: Get user embeddings, keywords, and reading level
                user_embeddings = get_user_embeddings(user_id, bq_client)
                user_keywords = user.get('keywords')
                user_reading_level = get_user_reading_level(user_id, bq_client)
                
                # Process keywords using enhanced format detection
                user_terms, user_sentiment_context, keywords_format = process_user_keywords(user_keywords)
                
                # Users can now get feeds based on network alone, even without keywords/embeddings
                if not user_terms and user_embeddings is None:
                    logger.info(f"No keywords or embeddings for {user_handle}, using network-only approach")
                
                # Log keyword format for monitoring
                logger.info(f"User {user_handle} using {keywords_format} keywords format with {len(user_terms)} terms")
                
                # A/B Testing: Determine if user should get enhanced ranking
                ab_test_manager = get_ab_test_manager()
                use_enhanced, experiment_group = ab_test_manager.should_use_enhanced_ranking(user_id)
                
                # Override user_sentiment_context based on A/B test group
                if not use_enhanced:
                    logger.info(f"User {user_handle} in A/B test group '{experiment_group}' - using basic ranking")
                    user_sentiment_context = None  # Force basic ranking
                else:
                    logger.info(f"User {user_handle} in A/B test group '{experiment_group}' - using enhanced ranking")
                
                # Cache experiment group assignment in Redis for consistency
                redis_client.set_user_experiment_group(user_id, 'enhanced_keywords_experiment', experiment_group)
                
                # Step 2: Comprehensive post collection (feeds + 1st/2nd degree network)
                posts_to_rank = collect_comprehensive_posts(
                    user_embeddings=user_embeddings,
                    user_keywords=user_terms,
                    user_did=user_id,
                    bq_client=bq_client,
                    redis_client=redis_client
                )
                
                if not posts_to_rank:
                    logger.warning(f"No posts collected for {user_handle}, skipping")
                    continue
                
                # Step 3: Calculate rankings with enhanced BM25 and feed boosting
                if user_terms:
                    ranked_posts = calculate_rankings_with_feed_boosting(
                        user_terms, 
                        posts_to_rank,
                        user_sentiment_context=user_sentiment_context,
                        user_reading_level=user_reading_level
                    )
                else:
                    # Fallback for users with embeddings but no keywords
                    logger.info(f"No keywords for {user_handle}, using basic scoring")
                    for post in posts_to_rank:
                        post['bm25_score'] = 1.0  # Base score
                        source = post.get('source', 'unknown')
                        if source == 'feed':
                            feed_similarity = post.get('feed_similarity', 0)
                            feed_boost = 1.0 + (feed_similarity * 2.0)
                            post['final_score'] = 1.0 * feed_boost
                        else:
                            post['final_score'] = 1.0
                    ranked_posts = sorted(posts_to_rank, key=lambda x: x.get('final_score', 0), reverse=True)
                
                if not ranked_posts:
                    logger.warning(f"No rankings calculated for {user_handle}, skipping")
                    continue
                
                # Step 4: Posts already ranked by priority (network > feed > search)
                logger.info(f"Applied priority-based ranking for user {user_handle}")
                
                # Step 5: Cache results
                cache_success = cache_user_rankings(redis_client, user_id, ranked_posts)
                
                if cache_success:
                    success_count += 1
                    embeddings_status = "with embeddings" if user_embeddings is not None else "no embeddings"
                    keywords_count = len(user_terms) if user_terms else 0
                    
                    # Log experiment metrics for A/B testing analysis
                    experiment_metrics = {
                        'experiment_group': experiment_group,
                        'keywords_format': keywords_format,
                        'user_reading_level': user_reading_level,
                        'posts_ranked': len(ranked_posts),
                        'keywords_count': keywords_count,
                        'has_embeddings': user_embeddings is not None,
                        'has_sentiment_context': user_sentiment_context is not None,
                        'ranking_method': 'enhanced' if user_sentiment_context else 'basic'
                    }
                    
                    # Add sentiment analysis stats if available
                    if user_sentiment_context and ranked_posts:
                        sentiment_boosted = len([p for p in ranked_posts[:50] if p.get('sentiment_multiplier', 1.0) > 1.05])
                        sentiment_penalized = len([p for p in ranked_posts[:50] if p.get('sentiment_multiplier', 1.0) < 0.95])
                        experiment_metrics.update({
                            'sentiment_boosted_posts': sentiment_boosted,
                            'sentiment_penalized_posts': sentiment_penalized
                        })
                    
                    ab_test_manager.log_ranking_experiment(user_id, experiment_metrics)
                    redis_client.log_experiment_metrics(user_id, experiment_metrics)
                    
                    logger.info(f"Successfully processed {user_handle} ({embeddings_status}, {keywords_count} keywords, {experiment_group} group)")
                else:
                    error_count += 1
                    logger.error(f"Failed to cache results for {user_handle}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {user_handle}: {e}")
                continue
        
        # Update default feed for new users (once daily)
        try:
            update_default_feed_if_needed(redis_client)
        except Exception as e:
            logger.error(f"Failed to update default feed: {e}")
        
        # Final summary
        logger.info(f"ETL Complete! Success: {success_count}, Errors: {error_count}")
        
        # Show cache stats
        stats = redis_client.get_stats()
        cached_users = redis_client.get_cached_users()
        logger.info(f"Redis Stats: {stats}")
        logger.info(f"Total cached feeds: {len(cached_users)}")
        logger.info(f"ETL Performance: Network-focused collection (feeds + network) with embedding-based feed filtering")
        
    except Exception as e:
        logger.error(f"ETL failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()