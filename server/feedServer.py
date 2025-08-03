import os
import sys
import json
import logging
import base64
import threading
from datetime import datetime
from typing import Optional, Dict, List
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.redis import Client as RedisClient
from client.bluesky.newPosts import Client as BlueskyPostsClient
from client.bigQuery import Client as BigQueryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeedServer:
    def __init__(self):
        """Initialize feed server"""
        self.redis_client = RedisClient()
        self.bluesky_client = None
        self.bigquery_client = None
        self.app = FastAPI()
        self.setup_routes()
        
    def decode_user_did(self, auth_header: str) -> Optional[str]:
        """Extract user DID from JWT token"""
        try:
            if not auth_header or not auth_header.startswith('Bearer '):
                return None
                
            jwt_token = auth_header.replace('Bearer ', '')
            parts = jwt_token.split('.')
            if len(parts) != 3:
                return None
                
            # Decode payload
            payload = parts[1]
            payload += '=' * (4 - len(payload) % 4)
            
            decoded = base64.b64decode(payload)
            payload_data = json.loads(decoded)
            
            return payload_data.get('iss')
            
        except Exception as e:
            logger.warning(f"Failed to decode JWT: {e}")
            return None

    def get_bigquery_client(self):
        """Get or create BigQuery client"""
        if self.bigquery_client is None:
            try:
                import json
                credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
                project_id = os.environ['BIGQUERY_PROJECT_ID']
                self.bigquery_client = BigQueryClient(credentials_json, project_id)
                logger.info("BigQuery client initialized for request logging")
            except Exception as e:
                logger.error(f"Failed to initialize BigQuery client: {e}")
                return None
        return self.bigquery_client

    def log_new_user_to_bigquery(self, user_did: str, feed_uri: str):
        """Log new user to BigQuery for discovery (called only once per user)"""
        logger.info(f"Logging new user {user_did} to BigQuery")
        
        try:
            bq_client = self.get_bigquery_client()
            if not bq_client:
                logger.warning(f"BigQuery client not available, skipping new user logging")
                return

            current_time = datetime.utcnow()
            
            # INSERT with duplicate protection (handles any remaining race conditions)
            from google.cloud import bigquery
            
            # Use MERGE for absolute duplicate protection
            insert_query = f"""
            MERGE `{bq_client.project_id}.data.users` AS target
            USING (
                SELECT 
                    @user_id AS user_id,
                    @handle AS handle,
                    @keywords AS keywords,
                    @timestamp AS last_request_at,
                    @request_count AS request_count,
                    @timestamp AS created_at,
                    @timestamp AS updated_at
            ) AS source
            ON target.user_id = source.user_id
            WHEN NOT MATCHED THEN
                INSERT (user_id, handle, keywords, last_request_at, request_count, created_at, updated_at)
                VALUES (source.user_id, source.handle, source.keywords, source.last_request_at, source.request_count, source.created_at, source.updated_at)
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_did),
                    bigquery.ScalarQueryParameter("handle", "STRING", ''),
                    bigquery.ScalarQueryParameter("keywords", "JSON", []),
                    bigquery.ScalarQueryParameter("timestamp", "TIMESTAMP", current_time),
                    bigquery.ScalarQueryParameter("request_count", "INTEGER", 1)
                ]
            )
            
            logger.info(f"Executing BigQuery INSERT for new user...")
            query_job = bq_client.client.query(insert_query, job_config=job_config)
            result = query_job.result()
            logger.info(f"SUCCESS: Inserted new user record for {user_did}")
                
        except Exception as e:
            logger.error(f"FAILED to log new user to BigQuery: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Don't fail the request if logging fails

    def is_truly_new_user(self, user_did: str) -> bool:
        """Check if user is truly new by verifying both Redis and BigQuery"""
        try:
            # First check Redis (fast)
            if not self.redis_client.is_new_user(user_did):
                logger.debug(f"User {user_did} found in Redis activity - not new")
                return False
            
            # If Redis says new, double-check BigQuery to prevent duplicates
            bq_client = self.get_bigquery_client()
            if bq_client:
                from google.cloud import bigquery
                
                check_query = f"""
                SELECT COUNT(*) as count 
                FROM `{bq_client.project_id}.data.users` 
                WHERE user_id = @user_id
                """
                
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("user_id", "STRING", user_did)
                    ]
                )
                
                query_job = bq_client.client.query(check_query, job_config=job_config)
                result = query_job.result()
                count = list(result)[0]['count']
                
                if count > 0:
                    logger.info(f"User {user_did} exists in BigQuery but not Redis - syncing activity")
                    # Sync to Redis to prevent future BigQuery checks
                    self.redis_client.track_user_activity(user_did)
                    return False
                else:
                    logger.info(f"User {user_did} confirmed new in both Redis and BigQuery")
                    return True
            else:
                logger.warning("BigQuery client not available, assuming new user")
                return True
                
        except Exception as e:
            logger.error(f"Error checking if user {user_did} is new: {e}")
            return False  # Default to not new to prevent duplicates

    def handle_user_request(self, user_did: str, feed_uri: str):
        """Handle user request with Redis activity tracking and new-user BigQuery logging"""
        if not user_did:
            return
        
        # Check if user is truly new BEFORE tracking activity in Redis
        is_new = self.is_truly_new_user(user_did)
        
        # Always track activity in Redis (fast, no duplicates possible)
        self.redis_client.track_user_activity(user_did)
        
        # Only log truly new users to BigQuery (prevents duplicates)
        if is_new:
            logger.info(f"New user detected: {user_did}")
            self.log_new_user_to_bigquery(user_did, feed_uri)
        else:
            logger.debug(f"Existing user activity tracked: {user_did}")

    def get_bluesky_client(self):
        """Get or create authenticated Bluesky client"""
        if self.bluesky_client is None:
            try:
                self.bluesky_client = BlueskyPostsClient()
                self.bluesky_client.login()
                logger.info("Bluesky client authenticated for trending posts")
            except Exception as e:
                logger.error(f"Failed to authenticate Bluesky client: {e}")
                return None
        return self.bluesky_client

    def get_trending_posts(self) -> List[Dict]:
        """Get trending posts with Redis caching"""
        try:
            # Check Redis cache first
            cached_trending = self.redis_client.get_trending_posts()
            if cached_trending:
                logger.info(f"Retrieved {len(cached_trending)} trending posts from cache")
                return cached_trending

            # Cache miss - fetch from Bluesky
            logger.info("Trending posts cache miss, fetching from Bluesky...")
            
            client = self.get_bluesky_client()
            if not client:
                logger.error("Could not get Bluesky client for trending posts")
                return []

            # Fetch trending posts (last 6 hours, top 100 posts)
            trending_posts = client.get_top_posts_multiple_queries(
                queries=["the", "a", "I", "you", "this", "today", "new", "just", "now", "really"],
                target_count=100,
                time_hours=6
            )

            if not trending_posts:
                logger.warning("No trending posts retrieved from Bluesky")
                return []

            # Convert to minimal feed format (consistent with user feeds)
            formatted_posts = []
            for post in trending_posts:
                formatted_post = {
                    "post_uri": post['uri'],
                    "uri": post['uri'],  # Required for consumption tracking
                    "score": post['engagement_score'],  # Required for sorting
                    "post_type": "original",  # Default for trending posts
                    "followed_user": None  # Not applicable for trending
                }
                formatted_posts.append(formatted_post)

            # Cache for 30 minutes
            self.redis_client.cache_trending_posts(formatted_posts, ttl=1800)
            logger.info(f"Cached {len(formatted_posts)} trending posts for 30 minutes")
            
            return formatted_posts

        except Exception as e:
            logger.error(f"Error fetching trending posts: {e}")
            return []

    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        def root():
            return {"status": "healthy", "service": "feed-server"}

        @self.app.get("/.well-known/did.json")
        def get_did_document():
            hostname = os.getenv('FEEDGEN_HOSTNAME', 'localhost')
            return {
                "@context": ["https://www.w3.org/ns/did/v1"],
                "id": f"did:web:{hostname}",
                "service": [{
                    "id": "#bsky_fg",
                    "type": "BskyFeedGenerator", 
                    "serviceEndpoint": f"https://{hostname}"
                }]
            }

        @self.app.get("/xrpc/app.bsky.feed.getFeedSkeleton")
        def get_feed_skeleton(request: Request, feed: str):
            try:
                # Get user from auth header
                auth_header = request.headers.get('authorization', '')
                user_did = self.decode_user_did(auth_header)
                
                # Fast check: Look in Redis cache for user feed
                cached_posts = []
                if user_did:
                    cached_posts = self.redis_client.get_user_feed(user_did) or []
                
                if cached_posts:
                    # EXISTING USER: Serve personalized feed, track activity
                    if user_did:
                        self.handle_user_request(user_did, feed)
                    logger.info(f"Retrieved {len(cached_posts)} personalized posts for user {user_did or 'anonymous'}")
                    
                    # Get unconsumed posts (top 20) and consumed posts for flowing feed
                    unconsumed_posts = self.redis_client.filter_unconsumed_posts(user_did, cached_posts)
                    consumed_posts = self.redis_client.get_consumed_posts_for_feed(user_did, cached_posts)
                    
                    # Create flowing feed with 100-post limit while preserving structure
                    MAX_POSTS = 100
                    unconsumed_limit = min(20, len(unconsumed_posts))
                    remaining_slots = MAX_POSTS - unconsumed_limit

                    # Take up to 20 unconsumed posts, then fill remaining slots with consumed posts
                    flowing_feed = unconsumed_posts[:unconsumed_limit] + consumed_posts[:remaining_slots]
                    cached_posts = flowing_feed
                    
                    logger.info(f"Flowing feed: {len(unconsumed_posts[:unconsumed_limit])} unconsumed + {len(consumed_posts[:remaining_slots])} consumed = {len(flowing_feed)} total posts")
                    
                else:
                    # POTENTIALLY NEW USER: Serve default feed, track activity and log if new
                    if user_did:
                        self.handle_user_request(user_did, feed)
                    else:
                        logger.info("No user DID extracted from auth header")
                    
                    # Get default feed
                    default_posts = self.redis_client.get_default_feed()
                    if default_posts:
                        cached_posts = default_posts[:20]  # Apply window to default feed too
                        logger.info(f"Serving {len(cached_posts)} default posts to new user {user_did or 'anonymous'}")
                    else:
                        # Last resort - fetch trending posts
                        trending_posts = self.get_trending_posts()
                        if trending_posts:
                            cached_posts = trending_posts[:20]  # Apply window to trending feed too
                            logger.info(f"Serving {len(cached_posts)} trending posts to new user {user_did or 'anonymous'}")
                        else:
                            logger.warning("No default or trending posts available for fallback")
                
                # Build feed items from windowed posts
                feed_items = []
                for post in cached_posts:
                    if not post.get("post_uri"):
                        continue
                        
                    feed_item = {"post": post["post_uri"]}
                    
                    # Add repost information if this is a repost
                    if post.get('post_type') == 'repost' and post.get('followed_user'):
                        feed_item["reason"] = {
                            "type": "repost",
                            "by": post.get('followed_user')
                        }
                    
                    feed_items.append(feed_item)
                
                # Mark only new unconsumed posts as consumed (don't re-mark already consumed posts)
                if user_did and 'unconsumed_posts' in locals():
                    new_posts_to_mark = unconsumed_posts[:20]  # Only the 20 new posts served
                    if new_posts_to_mark:
                        new_uris = [post.get("uri") for post in new_posts_to_mark if post.get("uri")]
                        if new_uris:
                            self.redis_client.mark_posts_consumed(user_did, new_uris)
                            logger.debug(f"Marked {len(new_uris)} new posts as consumed for user {user_did}")
                elif user_did and cached_posts:
                    # Fallback for new users or when no personalized feed exists
                    served_uris = [post.get("uri") for post in cached_posts if post.get("uri")]
                    if served_uris:
                        self.redis_client.mark_posts_consumed(user_did, served_uris)
                        logger.debug(f"Marked {len(served_uris)} posts as consumed for user {user_did}")
                
                response = {"feed": feed_items}
                
                logger.info(f"Served {len(feed_items)} posts to user {user_did or 'anonymous'}")
                return response
                
            except Exception as e:
                logger.error(f"Error serving feed: {e}")
                return {"feed": []}

        @self.app.get("/xrpc/app.bsky.feed.describeFeedGenerator")  
        def describe_feed_generator():
            return {
                "encoding": "application/json",
                "body": {
                    "did": os.getenv('FEED_DID', 'did:plc:your-feed'),
                    "feeds": [{
                        "uri": os.getenv('FEED_URI', 'at://your-feed/app.bsky.feed.generator/personalized'),
                        "cid": os.getenv('FEED_CID', 'bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi')
                    }]
                }
            }

        @self.app.get("/health")
        def health_check():
            try:
                stats = self.redis_client.get_stats()
                return {
                    "status": "healthy",
                    "redis_memory": stats.get('used_memory_human', '0B'),
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {"status": "unhealthy", "error": str(e)}

        @self.app.get("/stats")
        def get_stats():
            try:
                redis_stats = self.redis_client.get_stats()
                cached_users = self.redis_client.get_cached_users()
                
                return {
                    "cached_users": len(cached_users),
                    "redis_memory": redis_stats.get('used_memory_human', '0B'),
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return {"error": str(e)}

# Global app instance
feed_server = FeedServer()
app = feed_server.app

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting feed server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)