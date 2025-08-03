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


def get_active_users_with_keywords(redis_client: RedisClient, bq_client: BigQueryClient, test_mode: bool = False) -> List[Dict]:
    """Get active users with their stored keywords from Redis + BigQuery"""
    try:
        # Get active users from Redis (fast, real-time activity)
        active_user_ids = redis_client.get_active_users(days=30)
        
        if not active_user_ids:
            logger.warning("No active users found in Redis, falling back to BigQuery")
            return get_users_with_keywords_from_bigquery_fallback(bq_client, test_mode)
        
        # Limit in test mode
        if test_mode:
            active_user_ids = active_user_ids[:5]
        
        logger.info(f"Found {len(active_user_ids)} active users from Redis")
        
        # Get keywords for active users from BigQuery
        if not active_user_ids:
            return []
        
        # Create parameterized query for active users
        user_ids_str = "', '".join(active_user_ids)
        query = f"""
        SELECT 
            user_id,
            handle,
            keywords
        FROM `{bq_client.project_id}.data.users`
        WHERE user_id IN ('{user_ids_str}')
        AND keywords IS NOT NULL
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved keywords for {len(users)} active users from BigQuery")
        return users
        
    except Exception as e:
        logger.error(f"Error getting active users: {e}")
        logger.warning("Falling back to BigQuery-only approach")
        return get_users_with_keywords_from_bigquery_fallback(bq_client, test_mode)

def get_users_with_keywords_from_bigquery_fallback(bq_client: BigQueryClient, test_mode: bool = False) -> List[Dict]:
    """Fallback: Get active users with their stored keywords from BigQuery only"""
    try:
        # Only limit in test mode for development
        limit_clause = "LIMIT 5" if test_mode else ""
        
        query = f"""
        SELECT 
            user_id,
            handle,
            keywords
        FROM `{bq_client.project_id}.data.users`
        WHERE last_request_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY last_request_at DESC
        {limit_clause}
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved {len(users)} active users with keywords from BigQuery (fallback)")
        return users
        
    except Exception as e:
        logger.warning(f"Could not get users from BigQuery: {e}")
        return []

def get_user_keywords_as_terms(user_keywords) -> List[str]:
    """Convert stored keywords to term list for BM25"""
    try:
        import json
        terms = []
        
        # Handle different formats of keywords from BigQuery JSON
        if isinstance(user_keywords, list):
            keywords_list = user_keywords
        elif isinstance(user_keywords, str):
            # JSON string that needs parsing
            try:
                keywords_list = json.loads(user_keywords)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in keywords: {user_keywords}")
                return []
        elif user_keywords is None:
            return []
        else:
            logger.warning(f"Unexpected keywords format: {type(user_keywords)}")
            return []
        
        # Keywords are stored as simple string arrays
        for kw in keywords_list:
            if isinstance(kw, str):
                # Simple string format - add multiple times for weighting
                terms.extend([kw] * 3)  # Default weight of 3
            else:
                logger.warning(f"Unexpected keyword type: {type(kw)}")
        
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
                # Get following 
                following_data = user_client.get_all_user_follows(user_did)
                following_list = following_data if following_data else []  # Keep full objects
                logger.info(f"Retrieved {len(following_list)} following accounts for user {user_did}")
                
                # Debug: Log first few following accounts
                if following_list:
                    sample_dids = [f.get('did', 'unknown') for f in following_list[:3]]
                    logger.info(f"Sample following DIDs: {sample_dids}...")
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
            following_list=following_list,  
            target_count=10000, 
            time_hours=3,  
            following_ratio=0.6,  
            keyword_ratio=0.4,
            keyword_extraction_method="advanced",
            include_reposts=True,
            repost_weight=0.5
        )
        
        # Debug: Check what we got back
        logger.info(f"get_posts_hybrid returned {len(new_posts) if new_posts else 0} items")
        if new_posts:
            logger.info(f"First post type: {type(new_posts[0])}")
            if len(new_posts) > 0 and isinstance(new_posts[0], dict):
                logger.info(f"First post keys: {list(new_posts[0].keys())}")
            elif len(new_posts) > 0:
                logger.info(f"First post content: {new_posts[0][:100] if isinstance(new_posts[0], str) else new_posts[0]}")
        
        # Debug: Analyze post sources
        if new_posts:
            following_posts = 0
            keyword_posts = 0
            
            # Create set of following DIDs for fast lookup
            following_dids = {f.get('did', '') for f in following_list if isinstance(f, dict)}
            
            for i, post in enumerate(new_posts):
                try:
                    # Ensure post is a dictionary
                    if not isinstance(post, dict):
                        logger.warning(f"Post {i} is not a dictionary: {type(post)} - {post}")
                        continue
                        
                    author_did = post.get('author', {}).get('did', '')
                    if author_did in following_dids:
                        following_posts += 1
                    else:
                        keyword_posts += 1
                except Exception as e:
                    logger.error(f"Error processing post {i}: {e}")
                    continue
            
            logger.info(f"Post source breakdown: {following_posts} from network, {keyword_posts} from keywords")
            if len(new_posts) > 0:
                logger.info(f"Network effectiveness: {following_posts/len(new_posts)*100:.1f}% from following")
        
        logger.info(f"Collected {len(new_posts)} posts to rank")
        
        return new_posts
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to collect posts: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def load_blocked_users() -> set:
    """Load blocked users from moderation file"""
    try:
        moderation_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'moderation', 'moderation.txt')
        
        if not os.path.exists(moderation_file):
            logger.warning(f"Moderation file not found: {moderation_file}")
            return set()
        
        with open(moderation_file, 'r', encoding='utf-8') as f:
            blocked_users = {line.strip() for line in f if line.strip()}
        
        logger.info(f"Loaded {len(blocked_users):,} blocked users for moderation")
        return blocked_users
        
    except Exception as e:
        logger.error(f"Failed to load moderation file: {e}")
        return set()

def filter_blocked_posts(posts: List[Dict], blocked_users: set) -> List[Dict]:
    """Filter out posts from blocked users"""
    if not blocked_users:
        return posts
    
    original_count = len(posts)
    filtered_posts = []
    
    for post in posts:
        author_handle = post.get('author', {}).get('handle', '')
        if author_handle not in blocked_users:
            filtered_posts.append(post)
    
    blocked_count = original_count - len(filtered_posts)
    if blocked_count > 0:
        filter_rate = blocked_count / original_count * 100
        logger.info(f"Moderation: blocked {blocked_count}/{original_count} posts ({filter_rate:.1f}%)")
    
    return filtered_posts

def calculate_rankings(user_terms: List[str], posts: List[Dict]) -> List[Dict]:
    """Calculate TF-IDF/BM25 rankings using pre-stored keywords"""
    try:
        if not user_terms:
            logger.warning("No user terms provided for ranking")
            return []
        
        # Load blocked users and filter posts before ranking
        blocked_users = load_blocked_users()
        filtered_posts = filter_blocked_posts(posts, blocked_users)
        
        if len(filtered_posts) < len(posts):
            logger.info(f"Moderation filtering: {len(posts)} -> {len(filtered_posts)} posts")
        
        # Calculate BM25 similarity using stored keywords on filtered posts
        ranked_posts = compute_bm25_similarity(user_terms, filtered_posts)
        
        # Sort by BM25 score
        ranked_posts = sorted(ranked_posts, key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        logger.info(f"Calculated rankings for {len(ranked_posts)} posts using {len(set(user_terms))} unique keywords")
        return ranked_posts
        
    except Exception as e:
        logger.error(f"Failed to calculate rankings: {e}")
        return []

def boost_network_posts(ranked_posts: List[Dict], user_did: str) -> List[Dict]:
    """
    Boost posts from user's network to ensure visibility throughout the feed
    
    Args:
        ranked_posts: List of posts with BM25 scores
        user_did: User's DID to get their following list
        
    Returns:
        Re-ranked posts with network posts boosted
    """
    try:
        # Get user's following list for boost identification
        following_dids = set()
        try:
            from client.bluesky.userData import Client as UserDataClient
            user_client = UserDataClient()
            user_client.login()
            following_data = user_client.get_all_user_follows(user_did)
            following_dids = {f.get('did', '') for f in following_data if isinstance(f, dict)}
            logger.info(f"Retrieved {len(following_dids)} following DIDs for network boost")
        except Exception as e:
            logger.warning(f"Could not get following list for network boost: {e}")
            return ranked_posts
        
        if not following_dids:
            return ranked_posts
        
        network_boosted = 0
        boost_factor = 2.5  # Strong boost to ensure network visibility
        
        for post in ranked_posts:
            author_did = post.get('author', {}).get('did', '')
            if author_did in following_dids:
                # Apply strong boost to network posts
                original_score = post.get('bm25_score', 0)
                post['bm25_score'] = original_score * boost_factor
                post['network_boosted'] = True
                network_boosted += 1
            else:
                post['network_boosted'] = False
        
        # Re-sort after boosting
        ranked_posts.sort(key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        logger.info(f"Network boost applied: {network_boosted} posts boosted by {boost_factor}x")
        
        # Log distribution in top positions
        network_in_top_50 = sum(1 for post in ranked_posts[:50] if post.get('network_boosted', False))
        logger.info(f"Network representation in top 50: {network_in_top_50} posts ({network_in_top_50/50*100:.1f}%)")
        
        return ranked_posts
        
    except Exception as e:
        logger.error(f"Failed to boost network posts: {e}")
        return ranked_posts

def cache_user_rankings(redis_client: RedisClient, user_id: str, ranked_posts: List[Dict]) -> bool:
    """Cache minimal feed data for a user with re-ranking support"""
    try:
        # Get existing cache to preserve unconsumed posts
        existing_feed = redis_client.get_user_feed(user_id) or []
        existing_posts = {post.get('post_uri', post.get('uri', '')): post for post in existing_feed} if existing_feed else {}
        
        # Create cache with only essential data for Bluesky feed API
        new_posts = []
        for post in ranked_posts:
            post_uri = post.get('uri', '')
            if not post_uri:
                continue
                
            # Store only essential fields for feed server (70% memory reduction)
            new_post = {
                'post_uri': post_uri,
                'uri': post_uri,  # Required for consumption tracking
                'score': float(post.get('bm25_score', 0)),  # Required for sorting
                'post_type': post.get('post_type', 'original'),  # Required for repost detection
                'followed_user': post.get('followed_user', None)  # Required for repost attribution
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
        for uri, post_data in existing_posts.items():
            if uri not in merged_posts:
                # Preserve existing post but extract only essential fields
                if isinstance(post_data, dict) and ('score' in post_data or 'combined_score' in post_data):
                    # Extract only essential fields from existing data
                    score = post_data.get('score', post_data.get('combined_score', 0))
                    merged_posts[uri] = {
                        'post_uri': uri,
                        'uri': uri,
                        'score': float(score),
                        'post_type': post_data.get('post_type', 'original'),
                        'followed_user': post_data.get('followed_user', None)
                    }
                else:
                    # Fallback for old cache format (just score) - minimal fields only
                    score = post_data if isinstance(post_data, (int, float)) else 0
                    merged_posts[uri] = {
                        'post_uri': uri,
                        'uri': uri,
                        'score': float(score),
                        'post_type': 'original',
                        'followed_user': None
                    }
        
        final_feed = sorted(merged_posts.values(), key=lambda x: x['score'], reverse=True)[:1500]
        
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
        
        # Convert to minimal feed format (consistent with personalized feeds)
        formatted_posts = []
        for post in popular_posts:
            formatted_post = {
                "post_uri": post['uri'],
                "uri": post['uri'],  # Required for consumption tracking
                "score": post['engagement_score'],  # Required for sorting
                "post_type": "original",  # Default for trending posts
                "followed_user": None  # Not applicable for trending
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
                
                # Step 3.5: Boost network posts to ensure visibility
                ranked_posts = boost_network_posts(ranked_posts, user_id)
                logger.info(f"Applied network visibility boost for user {user_handle}")
                
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