import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.bluesky.newPosts import Client as BlueskyClient
from client.bigQuery import Client as BigQueryClient
from client.redis import Client as RedisClient
from ranking.bm25Similarity import compute_bm25_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_users_with_keywords_from_bigquery(bq_client: BigQueryClient, test_mode: bool = False) -> List[Dict]:
    """Get active users with their stored keywords from BigQuery"""
    try:
        # Only limit in test mode for development
        limit_clause = "LIMIT 5" if test_mode else ""
        
        query = f"""
        SELECT 
            user_id,
            handle,
            display_name,
            last_seen_at,
            keywords
        FROM `{bq_client.project_id}.data.users`
        WHERE is_active = TRUE
        ORDER BY last_seen_at DESC
        {limit_clause}
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved {len(users)} active users with keywords from BigQuery")
        return users
        
    except Exception as e:
        logger.warning(f"Could not get users from BigQuery: {e}")

def get_user_keywords_as_terms(user_keywords: List) -> List[str]:
    """Convert stored keywords to term list for BM25"""
    try:
        terms = []
        
        # Handle different keyword formats from BigQuery
        if isinstance(user_keywords, list):
            for kw in user_keywords:
                if isinstance(kw, dict) and 'keyword' in kw:
                    # BigQuery STRUCT format: {'keyword': 'startup', 'score': 0.8}
                    keyword = kw['keyword']
                    score = kw.get('score', 1.0)
                    # Add keyword multiple times based on score (weight)
                    weight = max(1, int(score * 5))  # Convert score to repeat count
                    terms.extend([keyword] * weight)
                elif isinstance(kw, str):
                    # Simple string format
                    terms.append(kw)
        
        logger.info(f"Converted {len(set(terms))} unique keywords to {len(terms)} weighted terms")
        return terms
        
    except Exception as e:
        logger.error(f"Failed to process user keywords: {e}")
        return []

def collect_posts_to_rank(user_keywords: List[str], following_list: List = None) -> List[Dict]:
    """Collect posts to rank using current system"""
    try:
        posts_client = BlueskyClient()
        posts_client.login()
        
        # Use keyword-based post collection (no user_data needed)
        # Create mock user_data structure for compatibility
        mock_user_data = {
            'posts': [{'text': ' '.join(user_keywords[:10])}],  # Use keywords as mock content
            'reposts': [],
            'replies': [],
            'likes': []
        }
        
        new_posts = posts_client.get_posts_hybrid(
            user_data=mock_user_data,
            following_list=following_list or [],
            target_count=1000, 
            time_hours=0.15,
            following_ratio=0.6,
            keyword_ratio=0.4,
            keyword_extraction_method="advanced",
            include_reposts=True,
            repost_weight=0.0
        )
        
        logger.info(f"Collected {len(new_posts)} posts to rank")
        return new_posts
        
    except Exception as e:
        logger.error(f"Failed to collect posts: {e}")
        return []

def calculate_rankings(user_terms: List[str], posts: List[Dict]) -> List[Dict]:
    """Calculate TF-IDF/BM25 rankings using pre-stored keywords"""
    try:
        if not user_terms:
            logger.warning("No user terms provided for ranking")
            return []
        
        # Calculate BM25 similarity using stored keywords
        ranked_posts = compute_bm25_similarity(user_terms, posts)
        
        # Sort by BM25 score
        ranked_posts = sorted(ranked_posts, key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        logger.info(f"Calculated rankings for {len(ranked_posts)} posts using {len(set(user_terms))} unique keywords")
        return ranked_posts
        
    except Exception as e:
        logger.error(f"Failed to calculate rankings: {e}")
        return []

def cache_user_rankings(redis_client: RedisClient, user_id: str, ranked_posts: List[Dict]) -> bool:
    """Cache minimal feed data for a user with re-ranking support"""
    try:
        # Get existing cache to preserve unconsumed posts
        existing_feed = redis_client.get_user_feed(user_id) or []
        existing_posts = {post['post_uri']: post['score'] for post in existing_feed} if existing_feed else {}
        
        # Create cache with only essential data for Bluesky feed API
        new_posts = []
        for post in ranked_posts:
            post_uri = post.get('uri', '')
            if not post_uri:
                continue
                
            new_post = {
                'post_uri': post_uri,
                'score': float(post.get('bm25_score', 0))  # Ensure Python float
            }
            new_posts.append(new_post)
        
        # Merge new posts with existing unconsumed posts
        merged_posts = {}
        
        # Add new posts (they get priority with fresh scores)
        for post in new_posts:
            merged_posts[post['post_uri']] = post
        
        # Add existing posts that aren't in new batch (preserve unconsumed)
        for uri, score in existing_posts.items():
            if uri not in merged_posts:
                merged_posts[uri] = {'post_uri': uri, 'score': score}
        
        # Sort by score and keep top 500
        final_feed = sorted(merged_posts.values(), key=lambda x: x['score'], reverse=True)[:500]
        
        success = redis_client.set_user_feed(user_id, final_feed, ttl=900)  # 15 minutes TTL
        
        if success:
            new_count = len(new_posts)
            total_count = len(final_feed)
            preserved_count = total_count - len([p for p in new_posts if p['post_uri'] in existing_posts])
            logger.info(f"Cached {total_count} posts for user {user_id} ({new_count} new, {preserved_count} preserved)")
        else:
            logger.error(f"Failed to cache posts for user {user_id}")
            
        return success
        
    except Exception as e:
        logger.error(f"Failed to cache rankings for {user_id}: {e}")
        return False

# Keyword extraction removed - keywords are now pre-stored in BigQuery
# This will be handled by the separate user discovery process

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
        
        # Get users with their stored keywords
        users = get_users_with_keywords_from_bigquery(bq_client, test_mode)
        
        if not users:
            logger.error("No users found to process")
            return
        
        logger.info(f"Processing {len(users)} users")
        
        success_count = 0
        error_count = 0
        
        for user in users:
            user_handle = user.get('handle', '')
            user_id = user.get('user_id', user_handle)
            
            try:
                logger.info(f"Processing user: {user_handle}")
                
                # Step 1: Get stored keywords and convert to terms
                user_keywords = user.get('keywords', [])
                user_terms = get_user_keywords_as_terms(user_keywords)
                
                if not user_terms:
                    logger.warning(f"No keywords found for {user_handle}, skipping")
                    continue
                
                # Step 2: Collect posts to rank (using keywords for targeting)
                posts_to_rank = collect_posts_to_rank(user_terms)
                
                if not posts_to_rank:
                    logger.warning(f"No posts to rank for {user_handle}, skipping")
                    continue
                
                # Step 3: Calculate rankings using stored keywords
                ranked_posts = calculate_rankings(user_terms, posts_to_rank)
                
                if not ranked_posts:
                    logger.warning(f"No rankings calculated for {user_handle}, skipping")
                    continue
                
                # Step 4: Cache results
                cache_success = cache_user_rankings(redis_client, user_id, ranked_posts)
                
                if cache_success:
                    success_count += 1
                    logger.info(f"Successfully processed {user_handle} with {len(user_terms)} keywords")
                else:
                    error_count += 1
                    logger.error(f"Failed to cache results for {user_handle}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {user_handle}: {e}")
                continue
        
        # Final summary
        logger.info(f"ETL Complete! Success: {success_count}, Errors: {error_count}")
        
        # Show cache stats
        stats = redis_client.get_stats()
        cached_users = redis_client.get_cached_users()
        logger.info(f"Redis Stats: {stats}")
        logger.info(f"Total cached feeds: {len(cached_users)}")
        logger.info(f"ETL Performance: No user data collection, used pre-stored keywords only")
        
    except Exception as e:
        logger.error(f"ETL failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()