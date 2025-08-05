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
            
            # Create DataFrame with new user data (excluding JSON columns for now)
            import pandas as pd
            new_user_df = pd.DataFrame({
                'user_id': [user_did],
                'handle': [''],
                'last_request_at': [current_time],
                'request_count': [1],
                'created_at': [current_time],
                'updated_at': [current_time]
            })
            
            # Use BigQuery client's append method
            bq_client.append(new_user_df, 'data', 'users')
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

    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        def root():
            return {"status": "healthy", "service": "feed-server"}

        @self.app.get("/.well-known/did.json")
        def get_did_document():
            try:
                # Check Redis cache first for instant response
                cached_did = self.redis_client.client.get("did_document")
                if cached_did:
                    logger.debug("Serving DID document from cache")
                    return json.loads(cached_did)
                
                # Generate DID document if not cached
                hostname = os.getenv('FEEDGEN_HOSTNAME', 'localhost')
                did_doc = {
                    "@context": ["https://www.w3.org/ns/did/v1"],
                    "id": f"did:web:{hostname}",
                    "service": [{
                        "id": "#bsky_fg",
                        "type": "BskyFeedGenerator", 
                        "serviceEndpoint": f"https://{hostname}"
                    }]
                }
                
                # Cache for 24 hours (DID documents rarely change)
                self.redis_client.client.set("did_document", 
                                           json.dumps(did_doc), 
                                           ex=86400)
                logger.info("Generated and cached DID document")
                return did_doc
                
            except Exception as e:
                # Fallback to uncached generation if Redis fails
                logger.warning(f"DID caching failed, serving uncached: {e}")
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
        def get_feed_skeleton(request: Request, feed: str, cursor: Optional[str] = None, limit: int = 50):
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
                    
                    # Split posts into fresh vs old for cursor-based pagination
                    fresh_posts, old_posts = self.redis_client.split_posts_by_consumption(user_did, cached_posts)
                    
                    if cursor is None:
                        # First request: serve fresh posts only
                        if len(fresh_posts) > 0:
                            cached_posts = fresh_posts[:limit]
                            logger.info(f"Serving {len(cached_posts)} fresh posts (no cursor)")
                        else:
                            # No fresh posts - serve first page of old posts
                            cached_posts = old_posts[:limit] 
                            logger.info(f"No fresh posts - serving {len(cached_posts)} old posts (no cursor)")
                    else:
                        # Subsequent request: serve old posts with pagination
                        try:
                            offset = int(cursor)
                            cached_posts = old_posts[offset:offset + limit]
                            logger.info(f"Serving {len(cached_posts)} old posts from offset {offset}")
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid cursor format: {cursor}, falling back to first page")
                            cached_posts = old_posts[:limit]
                    
                else:
                    # POTENTIALLY NEW USER: Serve default feed, track activity and log if new
                    if user_did:
                        self.handle_user_request(user_did, feed)
                    else:
                        logger.info("No user DID extracted from auth header")
                    
                    # Get default feed with pagination
                    default_posts = self.redis_client.get_default_feed()
                    if default_posts:
                        offset = 0
                        if cursor:
                            try:
                                offset = int(cursor)
                            except (ValueError, TypeError):
                                offset = 0
                        cached_posts = default_posts[offset:offset + limit]
                        logger.info(f"Serving {len(cached_posts)} default posts to new user {user_did or 'anonymous'}")
                    else:
                        logger.warning("No default posts available for fallback")
                
                # Build feed items from windowed posts
                feed_items = []
                for post in cached_posts:
                    if not post.get("uri"):
                        continue
                        
                    feed_item = {"post": post["uri"]}
                    
                    # TODO: Add repost reasons later with proper AT-URIs
                    # Temporarily removed to fix schema validation errors
                    
                    feed_items.append(feed_item)
                
                # Mark whatever we served as consumed
                if user_did and cached_posts:
                    served_uris = [post.get("uri") for post in cached_posts if post.get("uri")]
                    if served_uris:
                        self.redis_client.mark_posts_consumed(user_did, served_uris)
                        logger.debug(f"Marked {len(served_uris)} posts as consumed for user {user_did}")
                
                # Generate cursor for next page
                next_cursor = None
                if user_did and cached_posts:
                    # Get fresh/old posts for cursor calculation
                    fresh_posts, old_posts = self.redis_client.split_posts_by_consumption(user_did, self.redis_client.get_user_feed(user_did) or [])
                    
                    if cursor is None:
                        # First request served fresh posts, next cursor starts old posts
                        if len(old_posts) > 0:
                            next_cursor = str(0)  # Start old posts from beginning
                    else:
                        # Subsequent request, check if more old posts available
                        current_offset = int(cursor) if cursor else 0
                        next_offset = current_offset + limit
                        if next_offset < len(old_posts):
                            next_cursor = str(next_offset)
                elif not user_did and cached_posts:
                    # Default feed pagination
                    default_posts = self.redis_client.get_default_feed() or []
                    current_offset = int(cursor) if cursor else 0
                    next_offset = current_offset + limit
                    if next_offset < len(default_posts):
                        next_cursor = str(next_offset)
                
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