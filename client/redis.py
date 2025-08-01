import redis
import json
import logging
from typing import Dict, List, Optional
import os
from datetime import datetime

class Client:
    def __init__(self, redis_url: str = None):
        """
        Initialize Redis client for feed rankings cache
        
        Args:
            redis_url: Redis connection URL (from environment)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not redis_url:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def set_user_feed(self, user_id: str, ranked_posts: List[Dict], ttl: int = 900) -> bool:
        """
        Store ranked posts for a user
        
        Args:
            user_id: User identifier 
            ranked_posts: List of ranked post dictionaries
            ttl: Time to live in seconds (default: 15 minutes)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"feed:{user_id}"
            
            # Store as JSON with metadata
            feed_data = {
                'posts': ranked_posts,
                'updated_at': datetime.utcnow().isoformat(),
                'count': len(ranked_posts)
            }
            
            # Use set + expire instead of setex to avoid Redis Cloud issues
            json_data = json.dumps(feed_data, default=str)
            result = self.client.set(key, json_data)
            if result:
                self.client.expire(key, ttl)
            
            self.logger.info(f"Stored feed for user {user_id}: {len(ranked_posts)} posts")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to store feed for user {user_id}: {e}")
            return False
    
    def get_user_feed(self, user_id: str) -> Optional[List[Dict]]:
        """
        Retrieve ranked posts for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of ranked posts or None if not found/expired
        """
        try:
            key = f"feed:{user_id}"
            data = self.client.get(key)
            
            if not data:
                self.logger.info(f"No cached feed found for user {user_id}")
                return None
            
            feed_data = json.loads(data)
            posts = feed_data.get('posts', [])
            
            self.logger.info(f"Retrieved feed for user {user_id}: {len(posts)} posts")
            return posts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve feed for user {user_id}: {e}")
            return None
    
    def delete_user_feed(self, user_id: str) -> bool:
        """Delete cached feed for a user"""
        try:
            key = f"feed:{user_id}"
            result = self.client.delete(key)
            self.logger.info(f"Deleted feed cache for user {user_id}")
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to delete feed for user {user_id}: {e}")
            return False
    
    def get_cached_users(self) -> List[str]:
        """Get list of users with cached feeds"""
        try:
            keys = self.client.keys("feed:*")
            user_ids = [key.replace("feed:", "") for key in keys]
            self.logger.info(f"Found {len(user_ids)} cached feeds")
            return user_ids
        except Exception as e:
            self.logger.error(f"Failed to get cached users: {e}")
            return []
    
    def clear_all_feeds(self) -> bool:
        """Clear all cached feeds (for maintenance)"""
        try:
            keys = self.client.keys("feed:*")
            if keys:
                result = self.client.delete(*keys)
                self.logger.info(f"Cleared {result} cached feeds")
                return True
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear feeds: {e}")
            return False
    
    def cache_trending_posts(self, trending_posts: List[Dict], ttl: int = 1800) -> bool:
        """
        Cache trending posts for new users
        
        Args:
            trending_posts: List of trending post dictionaries
            ttl: Time to live in seconds (default: 30 minutes)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = "trending:posts"
            
            # Store as JSON with metadata
            trending_data = {
                'posts': trending_posts,
                'updated_at': datetime.utcnow().isoformat(),
                'count': len(trending_posts)
            }
            
            json_data = json.dumps(trending_data, default=str)
            result = self.client.set(key, json_data)
            if result:
                self.client.expire(key, ttl)
            
            self.logger.info(f"Cached {len(trending_posts)} trending posts for {ttl}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cache trending posts: {e}")
            return False
    
    def get_trending_posts(self) -> Optional[List[Dict]]:
        """
        Retrieve cached trending posts
        
        Returns:
            List of trending posts or None if not found/expired
        """
        try:
            key = "trending:posts"
            data = self.client.get(key)
            
            if not data:
                self.logger.info("No cached trending posts found")
                return None
            
            trending_data = json.loads(data)
            posts = trending_data.get('posts', [])
            
            self.logger.info(f"Retrieved {len(posts)} cached trending posts")
            return posts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve trending posts: {e}")
            return None

    def set_default_feed(self, default_posts: List[Dict], ttl: int = 86400) -> bool:
        """
        Store default feed for new users
        
        Args:
            default_posts: List of popular post dictionaries
            ttl: Time to live in seconds (default: 24 hours)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = "default:feed"
            
            # Store as JSON with metadata
            feed_data = {
                'posts': default_posts,
                'updated_at': datetime.utcnow().isoformat(),
                'count': len(default_posts)
            }
            
            json_data = json.dumps(feed_data, default=str)
            result = self.client.set(key, json_data)
            if result:
                self.client.expire(key, ttl)
            
            self.logger.info(f"Cached {len(default_posts)} default posts for {ttl}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cache default feed: {e}")
            return False
    
    def get_default_feed(self) -> Optional[List[Dict]]:
        """
        Retrieve cached default feed
        
        Returns:
            List of default posts or None if not found/expired
        """
        try:
            key = "default:feed"
            data = self.client.get(key)
            
            if not data:
                self.logger.info("No cached default feed found")
                return None
            
            feed_data = json.loads(data)
            posts = feed_data.get('posts', [])
            
            self.logger.info(f"Retrieved {len(posts)} cached default posts")
            return posts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve default feed: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            info = self.client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to get Redis stats: {e}")
            return {}