import redis
import json
import logging
import hashlib
from typing import Dict, List, Optional
import os
import time
from datetime import datetime, timedelta

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
        Store ranked posts for a user with compression
        
        Args:
            user_id: User identifier 
            ranked_posts: List of ranked post dictionaries
            ttl: Time to live in seconds (default: 15 minutes)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"feed:{user_id}"
            
            # Compress posts using short field names and URI hashing
            compressed_posts = []
            for post in ranked_posts:
                # Get URIs and compress them
                post_uri = post.get('post_uri', '')
                uri = post.get('uri', post_uri)  # Fallback to post_uri if uri missing
                
                if post_uri:
                    compressed_post = {
                        'u': self._store_uri_mapping(post_uri),  # post_uri -> u (compressed)
                        'r': self._compress_uri(uri),  # uri -> r (reference, for consumption tracking)
                        's': int(float(post.get('score', 0)) * 10000),  # score -> s (as integer)
                        't': post.get('post_type', 'o')[:1],  # post_type -> t (first character)
                        'f': 1 if post.get('followed_user') else 0  # followed_user -> f (boolean flag)
                    }
                    compressed_posts.append(compressed_post)
            
            # Store compressed data with metadata
            feed_data = {
                'posts': compressed_posts,
                'updated_at': datetime.utcnow().isoformat(),
                'count': len(compressed_posts)
            }
            
            json_data = json.dumps(feed_data, default=str)
            result = self.client.set(key, json_data)
            if result:
                self.client.expire(key, ttl)
            
            self.logger.info(f"Stored compressed feed for user {user_id}: {len(compressed_posts)} posts")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to store feed for user {user_id}: {e}")
            return False
    
    def get_user_feed(self, user_id: str) -> Optional[List[Dict]]:
        """
        Retrieve and decompress ranked posts for a user
        
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
            compressed_posts = feed_data.get('posts', [])
            
            # Decompress posts back to original format
            decompressed_posts = []
            for post in compressed_posts:
                if isinstance(post, dict) and 'u' in post:
                    # New compressed format
                    decompressed_post = {
                        'post_uri': self._decompress_uri(post['u']),
                        'uri': self._decompress_uri(post.get('r', post['u'])),
                        'score': float(post.get('s', 0)) / 10000,
                        'post_type': 'repost' if post.get('t', 'o') == 'r' else 'original',
                        'followed_user': 'placeholder' if post.get('f', 0) == 1 else None
                    }
                    decompressed_posts.append(decompressed_post)
                else:
                    # Fallback for old format
                    decompressed_posts.append(post)
            
            self.logger.info(f"Retrieved and decompressed feed for user {user_id}: {len(decompressed_posts)} posts")
            return decompressed_posts
            
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

    def set_value(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set a simple string value with TTL"""
        try:
            self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set value for key {key}: {e}")
            return False
    
    def get_value(self, key: str) -> Optional[str]:
        """Get a simple string value"""
        try:
            return self.client.get(key)
        except Exception as e:
            self.logger.error(f"Failed to get value for key {key}: {e}")
            return None

    def mark_posts_consumed(self, user_id: str, post_uris: List[str], ttl: int = 10800) -> bool:
        """
        Mark posts as consumed by a user with memory-optimized hashing (3 hour TTL)
        
        Args:
            user_id: User identifier
            post_uris: List of post URIs that were served to user
            ttl: Time to live in seconds (default: 3 hours)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not post_uris:
                return True
                
            import hashlib
            key = f"consumed:{user_id}"
            
            # Use Redis set to store hashed post URIs (memory efficient)
            pipeline = self.client.pipeline()
            for uri in post_uris:
                # Hash URI to save memory: 85 bytes -> 12 bytes
                uri_hash = hashlib.md5(uri.encode()).hexdigest()[:12]
                pipeline.sadd(key, uri_hash)
            pipeline.expire(key, ttl)
            pipeline.execute()
            
            self.logger.debug(f"Marked {len(post_uris)} posts as consumed for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to mark posts consumed for user {user_id}: {e}")
            return False
    
    def get_consumed_posts(self, user_id: str) -> set:
        """
        Get set of consumed post URI hashes for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Set of consumed post URI hashes
        """
        try:
            key = f"consumed:{user_id}"
            consumed = self.client.smembers(key)
            return consumed if consumed else set()
        except Exception as e:
            self.logger.error(f"Failed to get consumed posts for user {user_id}: {e}")
            return set()
    
    def filter_unconsumed_posts(self, user_id: str, posts: List[Dict]) -> List[Dict]:
        """
        Filter out already consumed posts for a user using URI hashing
        
        Args:
            user_id: User identifier
            posts: List of post dictionaries with 'uri' key
            
        Returns:
            List of unconsumed posts
        """
        try:
            consumed_hashes = self.get_consumed_posts(user_id)
            
            if not consumed_hashes:
                return posts
            
            unconsumed = []
            for post in posts:
                uri = post.get('uri', '')
                if uri:
                    # Use 12-char hash for consumption tracking (more precision)
                    uri_hash = hashlib.md5(uri.encode()).hexdigest()[:12]
                    if uri_hash not in consumed_hashes:
                        unconsumed.append(post)
            
            consumed_count = len(posts) - len(unconsumed)
            if consumed_count > 0:
                self.logger.info(f"Filtered out {consumed_count} already consumed posts for user {user_id}")
            
            return unconsumed
        except Exception as e:
            self.logger.error(f"Failed to filter consumed posts for user {user_id}: {e}")
            return posts

    def get_consumed_posts_for_feed(self, user_id: str, posts: List[Dict]) -> List[Dict]:
        """
        Get already consumed posts from the provided posts list, in original order
        
        Args:
            user_id: User identifier
            posts: List of post dictionaries with 'uri' key
            
        Returns:
            List of consumed posts in original order
        """
        try:
            consumed_hashes = self.get_consumed_posts(user_id)
            
            if not consumed_hashes:
                return []
            
            consumed_posts = []
            for post in posts:
                uri = post.get('uri', '')
                if uri:
                    # Use 12-char hash for consumption tracking (more precision)
                    uri_hash = hashlib.md5(uri.encode()).hexdigest()[:12]
                    if uri_hash in consumed_hashes:
                        consumed_posts.append(post)
            
            if consumed_posts:
                self.logger.info(f"Found {len(consumed_posts)} consumed posts for flowing feed")
            
            return consumed_posts
        except Exception as e:
            self.logger.error(f"Failed to get consumed posts for user {user_id}: {e}")
            return []

    def track_user_activity(self, user_id: str) -> bool:
        """
        Track user activity using Redis sorted set for efficient querying
        
        Args:
            user_id: User identifier (DID)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_timestamp = time.time()
            # Use the old redis-py 2.x syntax: zadd(key, member1, score1, member2, score2...)
            result = self.client.zadd("user_activity", user_id, current_timestamp)
            
            # Clean up old activity (older than 30 days) periodically  
            thirty_days_ago = current_timestamp - (30 * 24 * 60 * 60)
            self.client.zremrangebyscore("user_activity", 0, thirty_days_ago)
            
            self.logger.debug(f"Tracked activity for user {user_id}")
            return result is not None
        except Exception as e:
            self.logger.error(f"Failed to track activity for user {user_id}: {e}")
            return False
    
    def is_new_user(self, user_id: str) -> bool:
        """
        Check if user is new (not in activity tracking)
        
        Args:
            user_id: User identifier (DID)
            
        Returns:
            True if user is new, False if already tracked
        """
        try:
            score = self.client.zscore("user_activity", user_id)
            is_new = score is None
            self.logger.debug(f"User {user_id} is {'new' if is_new else 'existing'}")
            return is_new
        except Exception as e:
            self.logger.error(f"Failed to check if user {user_id} is new: {e}")
            return True  # Default to new user on error
    
    def get_active_users(self, days: int = 30) -> List[str]:
        """
        Get list of users active in the last N days
        
        Args:
            days: Number of days to look back (default: 30)
            
        Returns:
            List of user IDs active in the specified period
        """
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            active_users = self.client.zrangebyscore("user_activity", cutoff_time, "+inf")
            
            self.logger.info(f"Found {len(active_users)} users active in last {days} days")
            return active_users
        except Exception as e:
            self.logger.error(f"Failed to get active users: {e}")
            return []
    
    def get_user_last_activity(self, user_id: str) -> Optional[datetime]:
        """
        Get last activity timestamp for a user
        
        Args:
            user_id: User identifier (DID)
            
        Returns:
            DateTime of last activity or None if not found
        """
        try:
            timestamp = self.client.zscore("user_activity", user_id)
            if timestamp is None:
                return None
            
            return datetime.fromtimestamp(timestamp)
        except Exception as e:
            self.logger.error(f"Failed to get last activity for user {user_id}: {e}")
            return None

    def _compress_uri(self, uri: str) -> str:
        """
        Compress URI to 8-character hash for memory efficiency
        
        Args:
            uri: Full URI string
            
        Returns:
            8-character hash of the URI
        """
        return hashlib.md5(uri.encode()).hexdigest()[:8]
    
    def _store_uri_mapping(self, uri: str) -> str:
        """
        Store URI mapping and return compressed hash
        
        Args:
            uri: Full URI string
            
        Returns:
            8-character hash that maps to the URI
        """
        uri_hash = self._compress_uri(uri)
        # Store mapping with 24-hour TTL (longer than feed TTL)
        # Use set + expire for compatibility with older redis-py
        key = f"uri:{uri_hash}"
        self.client.set(key, uri)
        self.client.expire(key, 86400)
        return uri_hash
    
    def _decompress_uri(self, uri_hash: str) -> str:
        """
        Decompress URI hash back to full URI
        
        Args:
            uri_hash: 8-character hash
            
        Returns:
            Full URI string or empty string if not found
        """
        try:
            full_uri = self.client.get(f"uri:{uri_hash}")
            return full_uri if full_uri else uri_hash  # Fallback to hash if not found
        except Exception as e:
            self.logger.error(f"Failed to decompress URI {uri_hash}: {e}")
            return uri_hash

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