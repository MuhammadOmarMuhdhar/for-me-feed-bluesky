#!/usr/bin/env python3
"""
Feed Server for Bluesky AT Protocol
Serves personalized rankings from Redis cache
"""

import os
import sys
import json
import logging
import base64
from datetime import datetime
from typing import Optional, Dict, List
from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.redis import Client as RedisClient

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

    def log_request(self, user_did: str, feed_uri: str):
        """Log feed request for user discovery ETL"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"{timestamp} - GET /xrpc/app.bsky.feed.getFeedSkeleton - Feed: {feed_uri} - User: {user_did}"
            
            log_file = os.getenv('FEED_LOG_FILE', '/tmp/feed-server.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(log_entry + '\n')
                
            logger.info(f"Logged request for user: {user_did}")
            
        except Exception as e:
            logger.warning(f"Failed to log request: {e}")

    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        def root():
            return {"status": "healthy", "service": "feed-server"}

        @self.app.get("/xrpc/app.bsky.feed.getFeedSkeleton")
        def get_feed_skeleton(request: Request, feed: str, limit: int = 50, cursor: str = None):
            try:
                # Validate limit
                limit = min(max(limit, 1), 100)
                
                # Get user from auth header
                auth_header = request.headers.get('authorization', '')
                user_did = self.decode_user_did(auth_header)
                
                # Log request for user discovery
                if user_did:
                    self.log_request(user_did, feed)
                
                # Get cached rankings
                cached_posts = []
                if user_did:
                    cached_posts = self.redis_client.get_user_feed(user_did) or []
                
                # Handle pagination
                start_idx = 0
                if cursor:
                    try:
                        start_idx = int(cursor)
                    except:
                        start_idx = 0
                
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