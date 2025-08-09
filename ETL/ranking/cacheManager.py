"""
Cache management functions for the ranking ETL system.
"""
import logging
from typing import Dict, List

from ETL.ranking.config import (
    CACHE_TTL_SECONDS, DEFAULT_FEED_TTL_SECONDS, MAX_CACHED_POSTS,
    DEFAULT_FEED_QUERIES, DEFAULT_FEED_TARGET_COUNT, DEFAULT_TIME_HOURS
)

logger = logging.getLogger(__name__)


def cache_user_rankings(redis_client, user_id: str, ranked_posts: List[Dict]) -> bool:
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
                'score': float(post.get('final_score', 0)),  # Use final_score for priority-based ranking
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
        
        final_feed = sorted(merged_posts.values(), key=lambda x: x['score'], reverse=True)[:MAX_CACHED_POSTS]
        
        logger.info(f"Deduplication: {duplicate_count} duplicate posts found and updated with fresh scores")
        
        success = redis_client.set_user_feed(user_id, final_feed, ttl=CACHE_TTL_SECONDS)
        
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


def update_default_feed_if_needed(redis_client):
    """Update default feed if it's expired or missing (once daily)"""
    try:
        # Check if default feed exists and is fresh
        existing_default = redis_client.get_default_feed()
        if existing_default:
            logger.info("Default feed already cached and fresh, skipping update")
            return
        
        logger.info("Default feed expired or missing, generating new default feed...")
        
        # Generate default feed with popular posts from last 24 hours
        from client.bluesky.newPosts import Client as BlueskyClient
        bluesky_client = BlueskyClient()
        bluesky_client.login()
        
        # Get popular posts using multiple search queries
        popular_posts = bluesky_client.get_top_posts_multiple_queries(
            queries=DEFAULT_FEED_QUERIES,
            target_count=DEFAULT_FEED_TARGET_COUNT,
            time_hours=DEFAULT_TIME_HOURS  # Last 24 hours
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
        success = redis_client.set_default_feed(formatted_posts, ttl=DEFAULT_FEED_TTL_SECONDS)
        
        if success:
            logger.info(f"Successfully updated default feed with {len(formatted_posts)} posts")
        else:
            logger.error("Failed to cache default feed")
            
    except Exception as e:
        logger.error(f"Error updating default feed: {e}")
        raise