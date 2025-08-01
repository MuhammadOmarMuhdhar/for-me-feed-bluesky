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

    def log_request_to_bigquery(self, user_did: str, feed_uri: str):
        """Log feed request directly to BigQuery for user discovery"""
        try:
            bq_client = self.get_bigquery_client()
            if not bq_client:
                logger.warning("BigQuery client not available, skipping request logging")
                return

            import pandas as pd
            
            # Check if user already exists
            existing_user_query = f"""
            SELECT user_id, request_count, last_request_at
            FROM `{bq_client.project_id}.data.users` 
            WHERE user_id = '{user_did}'
            LIMIT 1
            """
            
            existing_result = bq_client.query(existing_user_query)
            current_time = datetime.utcnow()
            
            if not existing_result.empty:
                # Update existing user
                current_count = existing_result.iloc[0]['request_count'] or 0
                new_count = current_count + 1
                
                update_query = f"""
                UPDATE `{bq_client.project_id}.data.users`
                SET last_request_at = '{current_time.isoformat()}',
                    request_count = {new_count},
                    updated_at = '{current_time.isoformat()}'
                WHERE user_id = '{user_did}'
                """
                
                bq_client.query(update_query)
                logger.info(f"Updated request count for existing user {user_did}: {new_count}")
                
            else:
                # Create new user record with minimal data
                user_data = [{
                    'user_id': user_did,
                    'handle': '',  # Will be populated by user discovery ETL
                    'keywords': [],
                    'last_request_at': current_time,
                    'request_count': 1,
                    'created_at': current_time,
                    'updated_at': current_time
                }]
                
                df = pd.DataFrame(user_data)
                bq_client.append(df, 'data', 'users', create_if_not_exists=True)
                logger.info(f"Created new user record for {user_did}")
                
        except Exception as e:
            logger.error(f"Failed to log request to BigQuery: {e}")
            # Don't fail the request if logging fails

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

            # Convert to feed format
            formatted_posts = []
            for post in trending_posts:
                formatted_post = {
                    "post_uri": post['uri'],
                    "combined_score": post['engagement_score'],
                    "like_count": post['like_count'],
                    "repost_count": post['repost_count'],
                    "reply_count": post['reply_count'],
                    "created_at": post['created_at'],
                    "author_handle": post['author']['handle'],
                    "text": post['text'][:200],  # Truncate for cache efficiency
                    "source": "trending"
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
        def get_feed_skeleton(request: Request, feed: str, limit: int = 50, cursor: str = None):
            try:
                # Validate limit
                limit = min(max(limit, 1), 100)
                
                # Get user from auth header
                auth_header = request.headers.get('authorization', '')
                user_did = self.decode_user_did(auth_header)
                
                # Fast check: Look in Redis cache for user feed
                cached_posts = []
                if user_did:
                    cached_posts = self.redis_client.get_user_feed(user_did) or []
                
                if cached_posts:
                    # EXISTING USER: Serve personalized feed, log async
                    if user_did:
                        threading.Thread(
                            target=self.log_request_to_bigquery, 
                            args=(user_did, feed),
                            daemon=True
                        ).start()
                    logger.info(f"Serving {len(cached_posts)} personalized posts to user {user_did or 'anonymous'}")
                else:
                    # POTENTIALLY NEW USER: Serve default feed, log async
                    if user_did:
                        threading.Thread(
                            target=self.log_request_to_bigquery, 
                            args=(user_did, feed),
                            daemon=True
                        ).start()
                    
                    # Get default feed
                    default_posts = self.redis_client.get_default_feed()
                    if default_posts:
                        cached_posts = default_posts
                        logger.info(f"Serving {len(default_posts)} default posts to new user {user_did or 'anonymous'}")
                    else:
                        # Last resort - fetch trending posts
                        trending_posts = self.get_trending_posts()
                        if trending_posts:
                            cached_posts = trending_posts
                            logger.info(f"Serving {len(trending_posts)} trending posts to new user {user_did or 'anonymous'}")
                        else:
                            logger.warning("No default or trending posts available for fallback")
                
                # Handle pagination with sequential scroll simulation
                if not cursor:  # First request (refresh)
                    # Get last scroll position and advance sequentially
                    last_scroll = self.redis_client.get_value(f"scroll_start:{user_did}") or "0"
                    try:
                        last_position = int(last_scroll)
                    except:
                        last_position = 0
                    
                    # Advance by 20 posts each refresh, stay within top third
                    max_scroll = min(100, len(cached_posts) // 3) if cached_posts else 0
                    simulated_scroll = (last_position + 20) % max_scroll if max_scroll > 0 else 0
                    start_idx = simulated_scroll
                    
                    # Store new position for this session
                    self.redis_client.set_value(f"scroll_start:{user_did}", str(simulated_scroll), ttl=60)  # 1min
                    logger.info(f"Simulated scroll to position {simulated_scroll} for user {user_did or 'anonymous'}")
                else:
                    # Normal pagination from stored start point
                    scroll_start_str = self.redis_client.get_value(f"scroll_start:{user_did}") or "0"
                    try:
                        scroll_start = int(scroll_start_str)
                        start_idx = scroll_start + int(cursor)
                    except:
                        start_idx = int(cursor) if cursor else 0
                
                # Prepare response
                end_idx = start_idx + limit
                posts_slice = cached_posts[start_idx:end_idx]
                
                feed_items = [
                    {"post": post["post_uri"]}
                    for post in posts_slice
                    if post.get("post_uri")
                ]
                
                # Set next cursor
                next_cursor = None
                if end_idx < len(cached_posts):
                    next_cursor = str(end_idx)
                
                response = {"feed": feed_items}
                if next_cursor:
                    response["cursor"] = next_cursor
                
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