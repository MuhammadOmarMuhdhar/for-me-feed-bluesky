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
            keywords
        FROM `{bq_client.project_id}.data.users`
        WHERE last_request_at >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 30 DAY)
        ORDER BY last_request_at DESC
        {limit_clause}
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved {len(users)} active users with keywords from BigQuery")
        return users
        
    except Exception as e:
        logger.warning(f"Could not get users from BigQuery: {e}")
        return []

def get_user_keywords_as_terms(user_keywords) -> List[str]:
    """Convert stored keywords to term list for BM25"""
    try:
        import json
        terms = []
        
        # Handle JSON string from BigQuery
        if isinstance(user_keywords, str):
            keywords_list = json.loads(user_keywords)
        elif isinstance(user_keywords, list):
            keywords_list = user_keywords
        else:
            logger.warning(f"Unexpected keywords format: {type(user_keywords)}")
            return []
        
        # Process keywords list
        for kw in keywords_list:
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

def collect_posts_to_rank(user_keywords: List[str], user_did: str = None) -> List[Dict]:
    """Collect posts to rank using hybrid system with following network"""
    try:
        posts_client = BlueskyClient()
        posts_client.login()
        
        # Get user's following list for network effects
        following_list = []
        if user_did:
            try:
                from client.bluesky.userData import Client as UserDataClient
                user_client = UserDataClient()
                user_client.login()
                # Get following (first 100 should be enough for network effects)
                following_data = user_client.get_user_follows(user_did, limit=100)
                following_list = [f['did'] for f in following_data] if following_data else []
                logger.info(f"Retrieved {len(following_list)} following accounts for user {user_did}")
                
                # Debug: Log first few following accounts
                if following_list:
                    logger.info(f"Sample following DIDs: {following_list[:3]}...")
                else:
                    logger.warning(f"No following accounts found for user {user_did}")
                    
            except Exception as e:
                logger.error(f"Failed to get following list for {user_did}: {e}")
                import traceback
                logger.error(f"Following list error traceback: {traceback.format_exc()}")
                following_list = []
        
        # Create mock user_data structure for compatibility
        mock_user_data = {
            'posts': [{'text': ' '.join(user_keywords[:10])}],  # Use keywords as mock content
            'reposts': [],
            'replies': [],
            'likes': []
        }
        
        new_posts = posts_client.get_posts_hybrid(
            user_data=mock_user_data,
            following_list=following_list,  # Now populated with real following!
            target_count=1000, 
            time_hours=6,  # 6 hours for better content variety
            following_ratio=0.6,  
            keyword_ratio=0.4,
            keyword_extraction_method="advanced",
            include_reposts=True,
            repost_weight=0.5
        )
        
        # Debug: Analyze post sources
        if new_posts:
            following_posts = 0
            keyword_posts = 0
            
            for post in new_posts:
                author_did = post.get('author', {}).get('did', '')
                if author_did in following_list:
                    following_posts += 1
                else:
                    keyword_posts += 1
            
            logger.info(f"Post source breakdown: {following_posts} from network, {keyword_posts} from keywords")
            logger.info(f"Network effectiveness: {following_posts/len(new_posts)*100:.1f}% from following")
        
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
        
        # Merge new posts with existing unconsumed posts (deduplication via URI)
        merged_posts = {}
        duplicate_count = 0
        
        # Add new posts (they get priority with fresh scores)
        for post in new_posts:
            uri = post['post_uri']
            if uri in existing_posts:
                duplicate_count += 1
            merged_posts[uri] = post  # Always use new score for duplicates
        
        # Add existing posts that aren't in new batch (preserve unconsumed)
        for uri, score in existing_posts.items():
            if uri not in merged_posts:
                merged_posts[uri] = {'post_uri': uri, 'score': score}
        
        # Sort by score and keep top 500
        final_feed = sorted(merged_posts.values(), key=lambda x: x['score'], reverse=True)[:500]
        
        logger.info(f"Deduplication: {duplicate_count} duplicate posts found and updated with fresh scores")
        
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

def update_default_feed_if_needed(redis_client: RedisClient):
    """Update default feed if it's expired or missing (once daily)"""
    try:
        # Check if default feed exists and is fresh
        existing_default = redis_client.get_default_feed()
        if existing_default:
            logger.info("Default feed already cached and fresh, skipping update")
            return
        
        logger.info("Default feed expired or missing, generating new default feed...")
        
        # Generate default feed with popular posts from last 24 hours
        bluesky_client = BlueskyClient()
        bluesky_client.login()
        
        # Get popular posts using multiple search queries
        popular_posts = bluesky_client.get_top_posts_multiple_queries(
            queries=["the", "a", "I", "you", "this", "today", "new", "just", "really", "good"],
            target_count=100,
            time_hours=24  # Last 24 hours
        )
        
        if not popular_posts:
            logger.warning("No popular posts found for default feed")
            return
        
        # Convert to feed format (same as personalized feeds)
        formatted_posts = []
        for post in popular_posts:
            formatted_post = {
                "post_uri": post['uri'],
                "combined_score": post['engagement_score'],
                "like_count": post['like_count'],
                "repost_count": post['repost_count'],
                "reply_count": post['reply_count'],
                "created_at": post['created_at'],
                "author_handle": post['author']['handle'],
                "text": post['text'][:200],  # Truncate for cache efficiency
                "source": "default_daily"
            }
            formatted_posts.append(formatted_post)
        
        # Cache for 24 hours
        success = redis_client.set_default_feed(formatted_posts, ttl=86400)
        
        if success:
            logger.info(f"Successfully updated default feed with {len(formatted_posts)} posts")
        else:
            logger.error("Failed to cache default feed")
            
    except Exception as e:
        logger.error(f"Error updating default feed: {e}")
        raise

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
                
                # Step 1: Get stored keywords and convert to terms
                user_keywords = user.get('keywords')  # Don't default to []
                user_terms = get_user_keywords_as_terms(user_keywords)
                
                if not user_terms:
                    logger.warning(f"No keywords found for {user_handle}, skipping")
                    continue
                
                # Step 2: Collect posts to rank (using keywords + following network)
                posts_to_rank = collect_posts_to_rank(user_terms, user_id)
                
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
        logger.info(f"ETL Performance: No user data collection, used pre-stored keywords only")
        
    except Exception as e:
        logger.error(f"ETL failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()