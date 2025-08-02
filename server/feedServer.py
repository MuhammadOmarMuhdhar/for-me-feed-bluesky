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
        """Log feed request directly to BigQuery for user discovery using UPSERT"""
        logger.info(f"Starting log_request_to_bigquery for user {user_did}")
        
        try:
            logger.info(f"Getting BigQuery client...")
            bq_client = self.get_bigquery_client()
            if not bq_client:
                logger.warning(f"BigQuery client not available, skipping request logging")
                return

            current_time = datetime.utcnow()
            logger.info(f"Building INSERT query for {user_did} at {current_time.isoformat()}")
            
            # Simple INSERT - duplicates will be handled by ETL cleanup
            insert_query = f"""
            INSERT INTO `{bq_client.project_id}.data.users` 
            (user_id, handle, keywords, last_request_at, request_count, created_at, updated_at)
            VALUES (
                '{user_did}',
                '',
                JSON '[]',
                TIMESTAMP('{current_time.isoformat()}'),
                1,
                TIMESTAMP('{current_time.isoformat()}'),
                TIMESTAMP('{current_time.isoformat()}')
            )
            """
            
            logger.info(f"Executing BigQuery INSERT...")
            result = bq_client.query(insert_query)
            logger.info(f"BigQuery query result: {result}")
            logger.info(f"SUCCESS: Inserted user record for {user_did}")
                
        except Exception as e:
            logger.error(f"FAILED to log request to BigQuery: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
                    # EXISTING USER: Serve personalized feed, log request
                    if user_did:
                        logger.info(f"Logging request for existing user {user_did}")
                        self.log_request_to_bigquery(user_did, feed)
                    logger.info(f"Retrieved {len(cached_posts)} personalized posts for user {user_did or 'anonymous'}")
                    
                    # Filter out posts user has already consumed to prevent duplicates
                    unconsumed_posts = self.redis_client.filter_unconsumed_posts(user_did, cached_posts)
                    logger.info(f"Filtered to {len(unconsumed_posts)} unconsumed posts for user {user_did}")
                    
                    # Apply 20-post window to unconsumed posts
                    cached_posts = unconsumed_posts[:20]
                    
                else:
                    # POTENTIALLY NEW USER: Serve default feed, log request
                    if user_did:
                        logger.info(f"Logging request for new user {user_did}")
                        self.log_request_to_bigquery(user_did, feed)
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
                
                # Mark served posts as consumed to prevent future duplicates
                if user_did and cached_posts:
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