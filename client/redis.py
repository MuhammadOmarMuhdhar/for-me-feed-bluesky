import redis
import json
import logging
# import hashlib Will add hashing later
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
        Store ranked posts for a user without URI compression for performance
        
        Args:
            user_id: User identifier 
            ranked_posts: List of ranked post dictionaries
            ttl: Time to live in seconds (default: 15 minutes)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"feed:{user_id}"
            
            # Process posts without URI compression
            processed_posts = []
            
            for post in ranked_posts:
                post_uri = post.get('post_uri', '')
                uri = post.get('uri', post_uri)  # Fallback to post_uri if uri missing
                
                if post_uri:
                    processed_post = {
                        'uri': post_uri,
                        'score': float(post.get('score', 0)),
                        'post_type': post.get('post_type', 'original'),
                        'followed_user': post.get('followed_user')
                    }
                    processed_posts.append(processed_post)
            
            # Store the feed data
            feed_data = {
                'posts': processed_posts,
                'updated_at': datetime.utcnow().isoformat(),
                'count': len(processed_posts)
            }
            
            json_data = json.dumps(feed_data, default=str)
            self.client.set(key, json_data)

            self.logger.info(f"Stored feed for user {user_id}: {len(processed_posts)} posts")
            return True
            
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
                
            key = f"consumed:{user_id}"
            
            # Use Redis set to store hashed post URIs (memory efficient)
            pipeline = self.client.pipeline()
            for uri in post_uris:
                pipeline.sadd(key, uri)
            pipeline.expire(key, ttl)
            pipeline.execute()
            
            self.logger.debug(f"Marked {len(post_uris)} posts as consumed for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to mark posts consumed for user {user_id}: {e}")
            return False
    
    def split_posts_by_consumption(self, user_id: str, posts: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """
        Split posts into fresh (unconsumed) and old (consumed) posts
        
        Args:
            user_id: User identifier
            posts: List of post dictionaries with 'uri' key
            
        Returns:
            Tuple of (fresh_posts, old_posts)
        """
        try:
            key = f"consumed:{user_id}"
            consumed_uris = self.client.smembers(key)
            consumed_set = consumed_uris if consumed_uris else set()
            
            if not consumed_set:
                return posts, []
            
            fresh_posts = []
            old_posts = []
            
            for post in posts:
                uri = post.get('uri', '')
                if uri:
                    if uri in consumed_set:
                        old_posts.append(post)
                    else:
                        fresh_posts.append(post)
            
            self.logger.info(f"Split {len(posts)} posts: {len(fresh_posts)} fresh, {len(old_posts)} old")
            return fresh_posts, old_posts
            
        except Exception as e:
            self.logger.error(f"Failed to split posts for user {user_id}: {e}")
            return posts, []

    def track_user_activity(self, user_id: str) -> bool:
        """
        Track user activity using Redis sorted set for efficient querying
        
        Args:
            user_id: User identifier (DID)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_timestamp = int(time.time())
            # Use redis-py 5.x syntax: zadd(key, {member: score})
            result = self.client.zadd("user_activity", {user_id: current_timestamp})
            
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

    def set_user_experiment_group(self, user_id: str, experiment_name: str, group: str, ttl: int = 86400) -> bool:
        """
        Store user's experiment group assignment
        
        Args:
            user_id: User identifier
            experiment_name: Name of the experiment
            group: Assigned group name
            ttl: Time to live in seconds (default: 24 hours)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"experiment:{experiment_name}:{user_id}"
            self.client.setex(key, ttl, group)
            self.logger.debug(f"Set experiment group for user {user_id[:12]}...: {experiment_name} -> {group}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set experiment group for user {user_id}: {e}")
            return False
    
    def get_user_experiment_group(self, user_id: str, experiment_name: str) -> Optional[str]:
        """
        Get user's experiment group assignment
        
        Args:
            user_id: User identifier
            experiment_name: Name of the experiment
            
        Returns:
            Group name if cached, None otherwise
        """
        try:
            key = f"experiment:{experiment_name}:{user_id}"
            group = self.client.get(key)
            if group:
                self.logger.debug(f"Retrieved experiment group for user {user_id[:12]}...: {experiment_name} -> {group}")
            return group
        except Exception as e:
            self.logger.error(f"Failed to get experiment group for user {user_id}: {e}")
            return None
    
    def log_experiment_metrics(self, user_id: str, experiment_data: Dict) -> bool:
        """
        Log experiment metrics to Redis for later analysis
        
        Args:
            user_id: User identifier
            experiment_data: Dictionary with experiment metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            timestamp = int(time.time())
            key = f"experiment_metrics:{datetime.now().strftime('%Y-%m-%d')}"
            
            # Store metrics as JSON with timestamp
            metrics_entry = {
                'user_id': user_id,
                'timestamp': timestamp,
                **experiment_data
            }
            
            # Add to daily sorted set with timestamp as score
            self.client.zadd(key, {json.dumps(metrics_entry): timestamp})
            
            # Expire daily metrics after 30 days
            self.client.expire(key, 30 * 24 * 60 * 60)
            
            self.logger.debug(f"Logged experiment metrics for user {user_id[:12]}...")
            return True
        except Exception as e:
            self.logger.error(f"Failed to log experiment metrics for user {user_id}: {e}")
            return False
    
    def get_experiment_metrics(self, date_str: str = None, limit: int = 100) -> List[Dict]:
        """
        Get experiment metrics for analysis
        
        Args:
            date_str: Date string in YYYY-MM-DD format (default: today)
            limit: Maximum number of entries to return
            
        Returns:
            List of experiment metric dictionaries
        """
        try:
            if not date_str:
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            key = f"experiment_metrics:{date_str}"
            
            # Get entries from sorted set (most recent first)
            entries = self.client.zrevrange(key, 0, limit-1)
            
            metrics = []
            for entry in entries:
                try:
                    metrics.append(json.loads(entry))
                except json.JSONDecodeError:
                    continue
            
            self.logger.debug(f"Retrieved {len(metrics)} experiment metrics for {date_str}")
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get experiment metrics: {e}")
            return []

    def cache_user_following_list(self, user_id: str, following_list: List[Dict], ttl: int = 86400) -> bool:
        """
        Cache user's 1st degree following list
        
        Args:
            user_id: User identifier
            following_list: List of users they follow
            ttl: Time to live in seconds (default: 24 hours)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"network:{user_id}:following"
            
            # Store following data with metadata
            network_data = {
                'following_list': following_list,
                'updated_at': datetime.utcnow().isoformat(),
                'count': len(following_list)
            }
            
            json_data = json.dumps(network_data, default=str)
            result = self.client.setex(key, ttl, json_data)
            
            self.logger.info(f"Cached {len(following_list)} following accounts for user {user_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cache following list for user {user_id}: {e}")
            return False
    
    def get_cached_following_list(self, user_id: str) -> Optional[List[Dict]]:
        """
        Retrieve cached 1st degree following list
        
        Args:
            user_id: User identifier
            
        Returns:
            List of following users or None if not found/expired
        """
        try:
            key = f"network:{user_id}:following"
            data = self.client.get(key)
            
            if not data:
                self.logger.debug(f"No cached following list for user {user_id}")
                return None
            
            network_data = json.loads(data)
            following_list = network_data.get('following_list', [])
            
            self.logger.info(f"Retrieved {len(following_list)} cached following accounts for user {user_id}")
            return following_list
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached following list for user {user_id}: {e}")
            return None

    def cache_network_overlap_analysis(self, user_id: str, overlap_data: Dict, ttl: int = 604800) -> bool:
        """
        Cache 2nd degree network overlap analysis
        
        Args:
            user_id: User identifier
            overlap_data: Dictionary with overlap candidates and scores
            ttl: Time to live in seconds (default: 1 week)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"network:{user_id}:2nd_degree"
            
            # Store overlap analysis with metadata
            analysis_data = {
                'overlap_candidates': overlap_data,
                'updated_at': datetime.utcnow().isoformat(),
                'candidate_count': len(overlap_data),
                'analysis_version': '1.0'
            }
            
            json_data = json.dumps(analysis_data, default=str)
            result = self.client.setex(key, ttl, json_data)
            
            self.logger.info(f"Cached 2nd degree analysis for user {user_id}: {len(overlap_data)} candidates")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cache 2nd degree analysis for user {user_id}: {e}")
            return False
    
    def get_cached_overlap_analysis(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve cached 2nd degree overlap analysis
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with overlap candidates or None if not found/expired
        """
        try:
            key = f"network:{user_id}:2nd_degree"
            data = self.client.get(key)
            
            if not data:
                self.logger.debug(f"No cached 2nd degree analysis for user {user_id}")
                return None
            
            analysis_data = json.loads(data)
            overlap_candidates = analysis_data.get('overlap_candidates', {})
            
            self.logger.info(f"Retrieved cached 2nd degree analysis for user {user_id}: {len(overlap_candidates)} candidates")
            return overlap_candidates
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached 2nd degree analysis for user {user_id}: {e}")
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