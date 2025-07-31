import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from atproto import Client as AtprotoClient, models
from dateutil import parser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Client:
    def __init__(self, service_url: str = "https://bsky.social"):
        self.client = AtprotoClient(service_url)
        self.authenticated = False
    
    def login(self, identifier: str = None, password: str = None):
        """Login to Bluesky using provided credentials or environment variables"""
        # Use environment variables if credentials not provided
        if not identifier:
            identifier = os.getenv('BLUESKY_IDENTIFIER')
        if not password:
            password = os.getenv('BLUESKY_PASSWORD')
            
        if not identifier or not password:
            raise Exception("Credentials required. Provide them as parameters or set BLUESKY_IDENTIFIER and BLUESKY_PASSWORD in .env file")
        
        try:
            self.client.login(identifier, password)
            self.authenticated = True
            print(f"Successfully logged in as {identifier}")
        except Exception as e:
            print(f"Login failed: {e}")
            raise
    
    def get_user_likes(
        self,
        actor: str = None,
        limit: int = 100,
        cursor: str = None
    ) -> List[Dict]:
        """
        Get user's liked posts
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            limit: Number of likes to fetch (max 100 per request)
            cursor: Pagination cursor
            
        Returns:
            List of liked posts with metadata
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        # If no actor specified, use the authenticated user
        if not actor:
            actor = self.client.me.handle
        
        try:
            params = {
                'actor': actor,
                'limit': limit
            }
            if cursor:
                params['cursor'] = cursor
                
            response = self.client.app.bsky.feed.get_actor_likes(params)
            
            likes = []
            for feed_item in response.feed:
                post = feed_item.post
                like_data = {
                    'uri': post.uri,
                    'cid': post.cid,
                    'author': {
                        'did': post.author.did,
                        'handle': post.author.handle,
                        'display_name': getattr(post.author, 'display_name', ''),
                    },
                    'text': post.record.text,
                    'created_at': post.record.created_at,
                    'indexed_at': post.indexed_at,
                    'like_count': getattr(post, 'like_count', 0),
                    'repost_count': getattr(post, 'repost_count', 0),
                    'reply_count': getattr(post, 'reply_count', 0),
                    'liked_at': getattr(feed_item, 'indexed_at', None),  # When user liked it
                    'engagement_score': (
                        getattr(post, 'like_count', 0) + 
                        getattr(post, 'repost_count', 0) * 2 + 
                        getattr(post, 'reply_count', 0)
                    )
                }
                likes.append(like_data)
            
            return likes
            
        except Exception as e:
            print(f"Error fetching likes: {e}")
            return []
    
    def get_user_reposts(
        self,
        actor: str = None,
        limit: int = 100,
        cursor: str = None
    ) -> List[Dict]:
        """
        Get user's reposts/reshares
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            limit: Number of reposts to fetch
            cursor: Pagination cursor
            
        Returns:
            List of reposted content
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        if not actor:
            actor = self.client.me.handle
        
        try:
            # Get user's feed to find reposts
            params = {
                'actor': actor,
                'limit': limit,
                'filter': 'posts_and_author_threads'
            }
            if cursor:
                params['cursor'] = cursor
                
            response = self.client.app.bsky.feed.get_author_feed(params)
            
            reposts = []
            for feed_item in response.feed:
                # Check if this is a repost
                if hasattr(feed_item, 'reason') and feed_item.reason:
                    post = feed_item.post
                    repost_data = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'original_author': {
                            'did': post.author.did,
                            'handle': post.author.handle,
                            'display_name': getattr(post.author, 'display_name', ''),
                        },
                        'text': post.record.text,
                        'created_at': post.record.created_at,
                        'indexed_at': post.indexed_at,
                        'like_count': getattr(post, 'like_count', 0),
                        'repost_count': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'reposted_at': getattr(feed_item.reason, 'indexed_at', None),
                        'reposted_by': actor,
                        'engagement_score': (
                            getattr(post, 'like_count', 0) + 
                            getattr(post, 'repost_count', 0) * 2 + 
                            getattr(post, 'reply_count', 0)
                        )
                    }
                    reposts.append(repost_data)
            
            return reposts
            
        except Exception as e:
            print(f"Error fetching reposts: {e}")
            return []
    
    def get_user_posts(
        self,
        actor: str = None,
        limit: int = 100,
        cursor: str = None
    ) -> List[Dict]:
        """
        Get user's own posts
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            limit: Number of posts to fetch
            cursor: Pagination cursor
            
        Returns:
            List of user's posts
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        if not actor:
            actor = self.client.me.handle
        
        try:
            params = {
                'actor': actor,
                'limit': limit,
                'filter': 'posts_no_replies'
            }
            if cursor:
                params['cursor'] = cursor
                
            response = self.client.app.bsky.feed.get_author_feed(params)
            
            posts = []
            for feed_item in response.feed:
                # Only include original posts (not reposts)
                if not hasattr(feed_item, 'reason') or not feed_item.reason:
                    post = feed_item.post
                    post_data = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'author': {
                            'did': post.author.did,
                            'handle': post.author.handle,
                            'display_name': getattr(post.author, 'display_name', ''),
                        },
                        'text': post.record.text,
                        'created_at': post.record.created_at,
                        'indexed_at': post.indexed_at,
                        'like_count': getattr(post, 'like_count', 0),
                        'repost_count': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'engagement_score': (
                            getattr(post, 'like_count', 0) + 
                            getattr(post, 'repost_count', 0) * 2 + 
                            getattr(post, 'reply_count', 0)
                        )
                    }
                    posts.append(post_data)
            
            return posts
            
        except Exception as e:
            print(f"Error fetching posts: {e}")
            return []
    
    def get_user_replies(
        self,
        actor: str = None,
        limit: int = 100,
        cursor: str = None
    ) -> List[Dict]:
        """
        Get user's replies to other posts
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            limit: Number of replies to fetch
            cursor: Pagination cursor
            
        Returns:
            List of user's replies
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        if not actor:
            actor = self.client.me.handle
        
        try:
            params = {
                'actor': actor,
                'limit': limit,
                'filter': 'posts_with_replies'
            }
            if cursor:
                params['cursor'] = cursor
                
            response = self.client.app.bsky.feed.get_author_feed(params)
            
            replies = []
            for feed_item in response.feed:
                post = feed_item.post
                # Check if this is a reply (has reply data in record)
                if hasattr(post.record, 'reply') and post.record.reply:
                    reply_data = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'author': {
                            'did': post.author.did,
                            'handle': post.author.handle,
                            'display_name': getattr(post.author, 'display_name', ''),
                        },
                        'text': post.record.text,
                        'created_at': post.record.created_at,
                        'indexed_at': post.indexed_at,
                        'like_count': getattr(post, 'like_count', 0),
                        'repost_count': getattr(post, 'repost_count', 0),
                        'reply_count': getattr(post, 'reply_count', 0),
                        'reply_to': {
                            'uri': post.record.reply.parent.uri,
                            'cid': post.record.reply.parent.cid
                        },
                        'engagement_score': (
                            getattr(post, 'like_count', 0) + 
                            getattr(post, 'repost_count', 0) * 2 + 
                            getattr(post, 'reply_count', 0)
                        )
                    }
                    replies.append(reply_data)
            
            return replies
            
        except Exception as e:
            print(f"Error fetching replies: {e}")
            return []
    
    def get_comprehensive_user_data(
        self,
        actor: str = None,
        likes_limit: int = 100,
        posts_limit: int = 100,
        reposts_limit: int = 100,
        replies_limit: int = 100,
        include_likes: bool = True,
        likes_username: str = None,
        likes_password: str = None
    ) -> Dict:
        """
        Get comprehensive user engagement data in one call
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            likes_limit: Number of likes to fetch
            posts_limit: Number of posts to fetch
            reposts_limit: Number of reposts to fetch
            replies_limit: Number of replies to fetch
            include_likes: Whether to fetch likes data (requires user's credentials)
            likes_username: Username for the user whose likes to fetch (required if include_likes=True)
            likes_password: Password for the user whose likes to fetch (required if include_likes=True)
            
        Returns:
            Dictionary containing all user engagement data
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        if not actor:
            actor = self.client.me.handle
        
        print(f"Collecting comprehensive data for {actor}...")
        
        user_data = {
            'actor': actor,
            'collected_at': datetime.now().isoformat(),
            'posts': self.get_user_posts(actor, posts_limit),
            'reposts': self.get_user_reposts(actor, reposts_limit),
            'replies': self.get_user_replies(actor, replies_limit)
        }
        
        # Only fetch likes if requested and credentials provided
        if include_likes:
            if not likes_username or not likes_password:
                raise Exception("likes_username and likes_password are required when include_likes=True")
            
            # Create separate client for likes authentication
            likes_client = BlueskyUserDataClient()
            try:
                likes_client.login(likes_username, likes_password)
                
                # Can only fetch likes for the authenticated user
                if actor and actor != likes_client.client.me.handle:
                    print(f"Warning: Cannot fetch likes for {actor} - can only fetch likes for authenticated user {likes_client.client.me.handle}")
                    user_data['likes'] = []
                else:
                    # Use the likes client to fetch likes for the authenticated user
                    user_data['likes'] = likes_client.get_user_likes(likes_client.client.me.handle, likes_limit)
                    
            except Exception as e:
                print(f"Warning: Failed to authenticate for likes: {e}")
                user_data['likes'] = []
        else:
            user_data['likes'] = []
        
        print(f"Collected: {len(user_data['likes'])} likes, "
              f"{len(user_data['posts'])} posts, "
              f"{len(user_data['reposts'])} reposts, "
              f"{len(user_data['replies'])} replies")
        
        return user_data
    
    def get_user_follows(
        self,
        actor: str = None,
        limit: int = 100,
        cursor: str = None
    ) -> List[Dict]:
        """
        Get list of users that the specified user follows
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            limit: Number of follows to fetch (max 100 per request)
            cursor: Pagination cursor
            
        Returns:
            List of followed users with metadata
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        if not actor:
            actor = self.client.me.handle
        
        try:
            params = {
                'actor': actor,
                'limit': limit
            }
            if cursor:
                params['cursor'] = cursor
                
            response = self.client.app.bsky.graph.get_follows(params)
            
            follows = []
            for follow in response.follows:
                follow_data = {
                    'did': follow.did,
                    'handle': follow.handle,
                    'display_name': getattr(follow, 'display_name', ''),
                    'description': getattr(follow, 'description', ''),
                    'follower_count': getattr(follow, 'followers_count', 0),
                    'follows_count': getattr(follow, 'follows_count', 0),
                    'posts_count': getattr(follow, 'posts_count', 0),
                    'indexed_at': getattr(follow, 'indexed_at', None),
                    'created_at': getattr(follow, 'created_at', None)
                }
                follows.append(follow_data)
            
            return follows
            
        except Exception as e:
            print(f"Error fetching follows: {e}")
            return []
    
    def get_all_user_follows(
        self,
        actor: str = None,
        max_follows: int = 1000
    ) -> List[Dict]:
        """
        Get all users that the specified user follows with pagination
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            max_follows: Maximum number of follows to fetch
            
        Returns:
            Complete list of followed users
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        if not actor:
            actor = self.client.me.handle
        
        all_follows = []
        cursor = None
        
        print(f"Fetching following list for {actor}...")
        
        while len(all_follows) < max_follows:
            batch_limit = min(100, max_follows - len(all_follows))
            follows_batch = self.get_user_follows(actor, limit=batch_limit, cursor=cursor)
            
            if not follows_batch:
                break
            
            all_follows.extend(follows_batch)
            print(f"   Fetched {len(follows_batch)} follows (total: {len(all_follows)})")
            
            # Check if there's more data (implementation depends on API response structure)
            # For now, break after first batch - extend if pagination cursor is available
            break
        
        print(f"Retrieved {len(all_follows)} total follows for {actor}")
        return all_follows
    
    def cache_user_follows(
        self,
        actor: str = None,
        cache_duration_hours: int = 6,
        cache_dir: str = "cache"
    ) -> List[Dict]:
        """
        Get user follows with caching to reduce API calls
        
        Args:
            actor: User handle or DID (defaults to authenticated user)
            cache_duration_hours: How long to cache the following list
            cache_dir: Directory to store cache files
            
        Returns:
            List of followed users (from cache or fresh API call)
        """
        if not actor:
            actor = self.client.me.handle if self.authenticated else None
            
        if not actor:
            raise Exception("No actor specified and not authenticated")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache filename
        safe_actor = actor.replace('.', '_').replace('@', '')
        cache_file = os.path.join(cache_dir, f"follows_{safe_actor}.json")
        
        # Check if cache exists and is fresh
        if os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(hours=cache_duration_hours):
                print(f"Using cached following list (age: {cache_age})")
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Cache read error: {e}, fetching fresh data")
        
        # Fetch fresh data
        print("Fetching fresh following list...")
        follows = self.get_all_user_follows(actor)
        
        # Save to cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(follows, f, indent=2, ensure_ascii=False, default=str)
            print(f"Cached {len(follows)} follows to {cache_file}")
        except Exception as e:
            print(f"Cache write error: {e}")
        
        return follows

    def save_user_data(self, user_data: Dict, filename: str = None):
        """Save user data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_data_{user_data['actor']}_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"User data saved to {filename}")

