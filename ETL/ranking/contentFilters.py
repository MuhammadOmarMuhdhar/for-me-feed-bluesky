"""
Content filtering functions for the ranking ETL system.
"""
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Set
from dateutil import parser

from ETL.ranking.config import (
    FAQ_KEYWORDS, ThresholdConfig, EngagementWeights,
    MIN_ENGAGEMENT_FEED, MIN_ENGAGEMENT_NETWORK
)

logger = logging.getLogger(__name__)


def get_post_age_days(post: Dict) -> int:
    """
    Calculate the age of a post in days
    
    Args:
        post: Post dictionary with createdAt or indexed_at field
        
    Returns:
        Age in days, or 0 if date cannot be parsed
    """
    try:
        # Try different timestamp fields
        timestamp_str = None
        for field in ['createdAt', 'indexed_at', 'created_at', 'indexedAt']:
            if field in post:
                timestamp_str = post[field]
                break
        
        if not timestamp_str:
            return 0
        
        # Parse timestamp
        if isinstance(timestamp_str, str):
            post_time = parser.parse(timestamp_str)
        else:
            return 0
        
        # Calculate age in days
        now = datetime.now(timezone.utc)
        if post_time.tzinfo is None:
            post_time = post_time.replace(tzinfo=timezone.utc)
        
        age_delta = now - post_time
        return age_delta.days
        
    except Exception as e:
        logger.debug(f"Could not parse post date: {e}")
        return 0


def is_faq_post(post: Dict, position: int) -> bool:
    """
    Determine if a post is likely a FAQ/instructional post
    
    Args:
        post: Post dictionary
        position: Position in the feed (0-indexed)
        
    Returns:
        True if post should be filtered as FAQ
    """
    # Filter 1: Position + Age - Skip first 2 posts if they're >5 days old
    if position < ThresholdConfig.FAQ_POSITION_THRESHOLD:
        post_age = get_post_age_days(post)
        if post_age > ThresholdConfig.FAQ_POST_AGE_DAYS:
            logger.debug(f"Filtering FAQ: position {position}, age {post_age} days")
            return True
    
    # Filter 2: Keyword pattern matching
    text = post.get('text', '').lower()
    if not text:
        return False
    
    for keyword in FAQ_KEYWORDS:
        if keyword in text:
            logger.debug(f"Filtering FAQ: keyword '{keyword}' found")
            return True
    
    return False


def filter_faq_posts(posts: List[Dict]) -> List[Dict]:
    """
    Filter out FAQ and instructional posts from a list
    
    Args:
        posts: List of posts from a feed
        
    Returns:
        Filtered list with FAQ posts removed
    """
    filtered_posts = []
    faq_count = 0
    
    for i, post in enumerate(posts):
        if is_faq_post(post, position=i):
            faq_count += 1
            continue
        filtered_posts.append(post)
    
    if faq_count > 0:
        logger.info(f"Filtered {faq_count} FAQ posts from {len(posts)} total posts")
    
    return filtered_posts


def calculate_engagement_score(post: Dict) -> float:
    """
    Calculate combined engagement score for a post
    
    Args:
        post: Post dictionary with engagement metrics
        
    Returns:
        Combined engagement score (likes + reposts*2 + replies*0.5)
    """
    likes = post.get('like_count', 0)
    reposts = post.get('repost_count', 0)
    replies = post.get('reply_count', 0)
    
    # Combined engagement: likes + (reposts * 2) + (replies * 0.5)
    return likes + (reposts * EngagementWeights.REPOSTS) + (replies * EngagementWeights.REPLIES_BASIC)


def filter_low_engagement_posts(posts: List[Dict], source: str, min_engagement: float = None) -> List[Dict]:
    """
    Filter posts based on engagement threshold for feeds
    
    Args:
        posts: List of posts to filter
        source: Source type ('feed', 'network')
        min_engagement: Minimum engagement score required (uses defaults if None)
        
    Returns:
        Filtered posts that meet engagement criteria
    """
    if source == 'network':
        # Never filter network posts - always include following posts
        return posts
    
    # Set default thresholds based on source
    if min_engagement is None:
        min_engagement = MIN_ENGAGEMENT_FEED if source == 'feed' else MIN_ENGAGEMENT_NETWORK
    
    filtered_posts = []
    low_engagement_count = 0
    
    for post in posts:
        engagement_score = calculate_engagement_score(post)
        
        if engagement_score >= min_engagement:
            post['engagement_score'] = engagement_score  # Store for potential later use
            filtered_posts.append(post)
        else:
            low_engagement_count += 1
    
    if low_engagement_count > 0:
        logger.info(f"Filtered {low_engagement_count} low-engagement posts from {source} source (< {min_engagement} score)")
    
    return filtered_posts


def load_blocked_users() -> Set[str]:
    """Load blocked users from moderation file"""
    try:
        moderation_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            'moderation', 'moderation.txt'
        )
        
        if not os.path.exists(moderation_file):
            logger.warning(f"Moderation file not found: {moderation_file}")
            return set()
        
        with open(moderation_file, 'r', encoding='utf-8') as f:
            blocked_users = {line.strip() for line in f if line.strip()}
        
        logger.info(f"Loaded {len(blocked_users):,} blocked users for moderation")
        return blocked_users
        
    except Exception as e:
        logger.error(f"Failed to load moderation file: {e}")
        return set()


def filter_blocked_posts(posts: List[Dict], blocked_users: Set[str]) -> List[Dict]:
    """Filter out posts from blocked users"""
    if not blocked_users:
        return posts
    
    original_count = len(posts)
    filtered_posts = []
    
    for post in posts:
        author_handle = post.get('author', {}).get('handle', '')
        if author_handle not in blocked_users:
            filtered_posts.append(post)
    
    blocked_count = original_count - len(filtered_posts)
    if blocked_count > 0:
        filter_rate = blocked_count / original_count * 100
        logger.info(f"Moderation: blocked {blocked_count}/{original_count} posts ({filter_rate:.1f}%)")
    
    return filtered_posts