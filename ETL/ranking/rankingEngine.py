"""
Ranking and scoring functions for the ranking ETL system.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dateutil import parser

from ETL.ranking.config import (
    ThresholdConfig, EngagementWeights, PriorityBoosts, VIRAL_MULTIPLIER,
    TARGET_NETWORK_RATIO
)
from ETL.ranking.contentFilters import load_blocked_users, filter_blocked_posts

logger = logging.getLogger(__name__)


def calculate_post_age_minutes(post: Dict) -> float:
    """
    Calculate post age in minutes from created_at timestamp
    
    Args:
        post: Post dictionary with timestamp fields
        
    Returns:
        Age in minutes, or 0 if timestamp cannot be parsed
    """
    try:
        # Try different timestamp fields
        timestamp_str = None
        for field in ['created_at', 'createdAt', 'indexed_at', 'indexedAt']:
            if field in post:
                timestamp_str = post[field]
                break
        
        if not timestamp_str:
            return 0.0
        
        # Parse timestamp
        if isinstance(timestamp_str, str):
            post_time = parser.parse(timestamp_str)
        else:
            return 0.0
        
        # Calculate age in minutes
        now = datetime.now(timezone.utc)
        if post_time.tzinfo is None:
            post_time = post_time.replace(tzinfo=timezone.utc)
        
        age_delta = now - post_time
        age_minutes = age_delta.total_seconds() / 60.0
        return max(0.0, age_minutes)  # Ensure non-negative
        
    except Exception as e:
        logger.debug(f"Could not parse post timestamp: {e}")
        return 0.0


def calculate_engagement_velocity(post: Dict, age_minutes: float) -> float:
    """
    Calculate total engagement per minute
    
    Args:
        post: Post dictionary with engagement metrics
        age_minutes: Post age in minutes
        
    Returns:
        Engagement velocity (engagements per minute)
    """
    try:
        likes = post.get('like_count', 0)
        reposts = post.get('repost_count', 0) 
        replies = post.get('reply_count', 0)
        
        # Weighted engagement: likes + (reposts * 2) + (replies * 1.5)
        total_engagement = likes + (reposts * EngagementWeights.REPOSTS) + (replies * EngagementWeights.REPLIES_ENHANCED)
        
        # Handle very fresh posts (< 1 minute) to avoid division issues
        if age_minutes < 1.0:
            age_minutes = 1.0
        
        velocity = total_engagement / age_minutes
        return max(0.0, velocity)
        
    except Exception as e:
        logger.debug(f"Could not calculate engagement velocity: {e}")
        return 0.0


def is_viral_post(post: Dict, age_threshold: float = None, velocity_threshold: float = None) -> bool:
    """
    Determine if post qualifies for viral boost
    
    Args:
        post: Post dictionary
        age_threshold: Maximum age in minutes for viral consideration
        velocity_threshold: Minimum engagement velocity required
        
    Returns:
        True if post should receive viral boost
    """
    if age_threshold is None:
        age_threshold = ThresholdConfig.VIRAL_AGE_MINUTES
    if velocity_threshold is None:
        velocity_threshold = ThresholdConfig.VIRAL_VELOCITY
        
    try:
        # Only check feed and network posts
        source = post.get('source', '')
        if source not in ['feed', 'network']:
            return False
        
        # Check age constraint
        age_minutes = calculate_post_age_minutes(post)
        if age_minutes <= 0 or age_minutes > age_threshold:
            return False
        
        # Check engagement velocity
        velocity = calculate_engagement_velocity(post, age_minutes)
        if velocity < velocity_threshold:
            return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Error in viral detection: {e}")
        return False


def calculate_rankings_with_feed_boosting(
    user_terms: List[str], 
    posts: List[Dict], 
    user_sentiment_context: Optional[Dict] = None, 
    user_reading_level: int = 8
) -> List[Dict]:
    """Calculate enhanced BM25 rankings with sentiment analysis, feed similarity boosting and combined scoring"""
    try:
        if not user_terms:
            logger.warning("No user terms provided for ranking")
            return []
        
        # Load blocked users and filter posts before ranking
        blocked_users = load_blocked_users()
        filtered_posts = filter_blocked_posts(posts, blocked_users)
        
        if len(filtered_posts) < len(posts):
            logger.info(f"Moderation filtering: {len(posts)} -> {len(filtered_posts)} posts")
        
        # Deduplicate posts by URI (simple cross-source deduplication)
        seen_uris = set()
        deduplicated_posts = []
        duplicate_count = 0
        
        for post in filtered_posts:
            uri = post.get('post_uri') or post.get('uri', '')
            if uri and uri not in seen_uris:
                seen_uris.add(uri)
                deduplicated_posts.append(post)
            elif uri:
                duplicate_count += 1
        
        if duplicate_count > 0:
            logger.info(f"Deduplication: removed {duplicate_count} duplicate posts from {len(filtered_posts)} posts")
        
        filtered_posts = deduplicated_posts
        
        # Use enhanced BM25 if sentiment context is available, otherwise fall back to basic BM25
        if user_sentiment_context:
            logger.info("Using enhanced BM25 with sentiment analysis")
            from ranking.bm25Similarity import compute_enhanced_bm25_similarity
            ranked_posts = compute_enhanced_bm25_similarity(
                user_terms, 
                filtered_posts, 
                user_sentiment_context=user_sentiment_context,
                user_reading_level=user_reading_level
            )
            # Enhanced BM25 sets enhanced_bm25_score, use that as the base
            for post in ranked_posts:
                post['bm25_score'] = post.get('enhanced_bm25_score', 0.0)
        else:
            logger.info("Using basic BM25 without sentiment analysis")
            from ranking.bm25Similarity import compute_bm25_similarity
            ranked_posts = compute_bm25_similarity(user_terms, filtered_posts)
        
        # Apply priority-based boosting: network > feed > search
        for post in ranked_posts:
            bm25_score = post.get('bm25_score', 0)
            source = post.get('source', 'unknown')
            
            # Priority-based boosting system
            if source == 'network':
                network_degree = post.get('network_degree', 1)
                if network_degree == 1:
                    # 1st degree network: Highest priority boost
                    priority_boost = PriorityBoosts.NETWORK_1ST_DEGREE + (bm25_score * 0.1)
                    post['boost_reason'] = '1st_degree_network'
                else:
                    # Should not happen since network posts are tagged as 1st degree
                    priority_boost = PriorityBoosts.NETWORK_1ST_DEGREE + (bm25_score * 0.1)
                    post['boost_reason'] = 'network_priority'
                post['priority_boost'] = priority_boost
            elif source == '2nd_degree':
                # 2nd degree network: High priority based on overlap count
                overlap_count = post.get('overlap_count', 1)
                base_boost = PriorityBoosts.NETWORK_2ND_DEGREE_BASE + (overlap_count / 3.0)  # 1.0-2.0 range
                priority_boost = base_boost + (bm25_score * 0.15)
                post['priority_boost'] = priority_boost
                post['boost_reason'] = '2nd_degree_network'
            elif source == 'feed':
                # Feed posts: Medium priority boost (feed similarity + BM25)
                feed_similarity = post.get('feed_similarity', 0)
                priority_boost = PriorityBoosts.FEED_PRIORITY + (feed_similarity * 1.0) + (bm25_score * 0.2)
                post['priority_boost'] = priority_boost
                post['boost_reason'] = 'feed_priority'
            else:
                # Unknown source: minimal boost
                priority_boost = PriorityBoosts.UNKNOWN_SOURCE
                post['priority_boost'] = priority_boost
                post['boost_reason'] = 'unknown_source'
            
            # Final score = BM25 + Priority Boost
            final_score = bm25_score + priority_boost
            post['final_score'] = final_score
        
        # Apply viral boost to qualifying posts
        viral_boosted_count = 0
        for post in ranked_posts:
            if is_viral_post(post):
                original_score = post['final_score']
                viral_multiplier = VIRAL_MULTIPLIER  # 80% boost for viral content
                post['final_score'] = original_score * viral_multiplier
                post['viral_boost'] = True
                post['viral_multiplier'] = viral_multiplier
                post['original_score'] = original_score
                viral_boosted_count += 1
            else:
                post['viral_boost'] = False
        
        # Sort by final score (after viral boost applied)
        ranked_posts = sorted(ranked_posts, key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Log scoring statistics
        feed_posts = [p for p in ranked_posts if p.get('source') == 'feed']
        network_posts = [p for p in ranked_posts if p.get('source') == 'network']
        second_degree_posts = [p for p in ranked_posts if p.get('source') == '2nd_degree']
        
        logger.info(f"Calculated network-focused rankings for {len(ranked_posts)} posts using {len(set(user_terms))} unique terms")
        logger.info(f"Source distribution: {len(feed_posts)} feed, {len(network_posts)} 1st degree, {len(second_degree_posts)} 2nd degree")
        
        # Log top 10 post sources for verification
        top_10_sources = [p.get('source', 'unknown') for p in ranked_posts[:10]]
        source_counts_top10 = {src: top_10_sources.count(src) for src in ['network', 'feed', '2nd_degree']}
        logger.info(f"Top 10 posts source breakdown: {source_counts_top10}")
        
        if network_posts:
            avg_network_boost = sum(p.get('priority_boost', 0) for p in network_posts) / len(network_posts)
            logger.info(f"Average 1st degree network boost: {avg_network_boost:.2f}")
        
        if second_degree_posts:
            avg_2nd_degree_boost = sum(p.get('priority_boost', 0) for p in second_degree_posts) / len(second_degree_posts)
            avg_overlap_count = sum(p.get('overlap_count', 0) for p in second_degree_posts) / len(second_degree_posts)
            second_degree_in_top_50 = sum(1 for post in ranked_posts[:50] if post.get('source') == '2nd_degree')
            logger.info(f"Average 2nd degree boost: {avg_2nd_degree_boost:.2f} (avg overlap: {avg_overlap_count:.1f})")
            logger.info(f"2nd degree posts in top 50: {second_degree_in_top_50} ({second_degree_in_top_50/min(50, len(ranked_posts))*100:.1f}%)")
        
        if feed_posts:
            avg_feed_boost = sum(p.get('priority_boost', 0) for p in feed_posts) / len(feed_posts)
            logger.info(f"Average feed priority boost: {avg_feed_boost:.2f}")
        
        
        # Log viral boost statistics
        if viral_boosted_count > 0:
            viral_posts = [p for p in ranked_posts if p.get('viral_boost', False)]
            viral_in_top_20 = sum(1 for post in ranked_posts[:20] if post.get('viral_boost', False))
            avg_viral_multiplier = sum(p.get('viral_multiplier', 0) for p in viral_posts) / len(viral_posts)
            avg_original_score = sum(p.get('original_score', 0) for p in viral_posts) / len(viral_posts)
            avg_boosted_score = sum(p.get('final_score', 0) for p in viral_posts) / len(viral_posts)
            
            logger.info(f"Viral boost applied to {viral_boosted_count} posts (multiplier: {avg_viral_multiplier:.1f}x)")
            logger.info(f"Viral posts in top 20: {viral_in_top_20} ({viral_in_top_20/min(20, len(ranked_posts))*100:.1f}%)")
            logger.info(f"Average viral scores: {avg_original_score:.2f} -> {avg_boosted_score:.2f}")
            
            # Log sample viral post details for debugging
            if viral_posts:
                sample_viral = viral_posts[0]
                age_minutes = calculate_post_age_minutes(sample_viral)
                velocity = calculate_engagement_velocity(sample_viral, age_minutes)
                logger.info(f"Sample viral post: age={age_minutes:.1f}min, velocity={velocity:.2f} eng/min, source={sample_viral.get('source')}")
        else:
            logger.info("No posts qualified for viral boost")
        
        return ranked_posts
        
    except Exception as e:
        logger.error(f"Failed to calculate rankings: {e}")
        return []


def calculate_rankings(user_terms: List[str], posts: List[Dict]) -> List[Dict]:
    """Calculate TF-IDF/BM25 rankings using pre-stored keywords (legacy function)"""
    try:
        if not user_terms:
            logger.warning("No user terms provided for ranking")
            return []
        
        # Load blocked users and filter posts before ranking
        blocked_users = load_blocked_users()
        filtered_posts = filter_blocked_posts(posts, blocked_users)
        
        if len(filtered_posts) < len(posts):
            logger.info(f"Moderation filtering: {len(posts)} -> {len(filtered_posts)} posts")
        
        # Calculate BM25 similarity using stored keywords on filtered posts
        from ranking.bm25Similarity import compute_bm25_similarity
        ranked_posts = compute_bm25_similarity(user_terms, filtered_posts)
        
        # Sort by BM25 score
        ranked_posts = sorted(ranked_posts, key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        logger.info(f"Calculated rankings for {len(ranked_posts)} posts using {len(set(user_terms))} unique keywords")
        return ranked_posts
        
    except Exception as e:
        logger.error(f"Failed to calculate rankings: {e}")
        return []


def distribute_network_posts(ranked_posts: List[Dict], target_network_ratio: float = None) -> List[Dict]:
    """
    Distribute network posts evenly throughout the ranked feed instead of clustering at top
    
    Args:
        ranked_posts: List of posts with BM25 scores and source tags
        target_network_ratio: Target percentage of network posts in feed
        
    Returns:
        Posts with network posts distributed evenly throughout
    """
    if target_network_ratio is None:
        target_network_ratio = TARGET_NETWORK_RATIO
        
    try:
        # Separate posts by source
        network_posts = [p for p in ranked_posts if p.get('from_network', False)]
        other_posts = [p for p in ranked_posts if not p.get('from_network', False)]
        
        if not network_posts:
            logger.info("No network posts to distribute")
            return ranked_posts
        
        # Sort both groups by final score (descending)
        network_posts.sort(key=lambda x: x.get('final_score', x.get('bm25_score', 0)), reverse=True)
        other_posts.sort(key=lambda x: x.get('final_score', x.get('bm25_score', 0)), reverse=True)
        
        total_positions = len(ranked_posts)
        target_network_count = min(len(network_posts), int(total_positions * target_network_ratio))
        
        if target_network_count == 0:
            return other_posts
        
        # Calculate distribution interval
        interval = total_positions // target_network_count if target_network_count > 0 else total_positions
        
        # Distribute network posts evenly
        final_feed = []
        network_index = 0
        other_index = 0
        
        for position in range(total_positions):
            # Insert network post at regular intervals
            if (position % interval == 0 and 
                network_index < target_network_count and 
                network_index < len(network_posts)):
                final_feed.append(network_posts[network_index])
                network_index += 1
            # Fill remaining positions with other posts
            elif other_index < len(other_posts):
                final_feed.append(other_posts[other_index])
                other_index += 1
            # If we run out of other posts, add remaining network posts
            elif network_index < len(network_posts):
                final_feed.append(network_posts[network_index])
                network_index += 1
        
        # Add any remaining posts that didn't fit
        while other_index < len(other_posts):
            final_feed.append(other_posts[other_index])
            other_index += 1
        
        while network_index < len(network_posts):
            final_feed.append(network_posts[network_index])
            network_index += 1
        
        # Log distribution stats
        network_in_top_50 = sum(1 for post in final_feed[:50] if post.get('from_network', False))
        network_in_final = sum(1 for post in final_feed if post.get('from_network', False))
        
        logger.info(f"Network distribution: {network_in_final} network posts distributed evenly")
        logger.info(f"Network in top 50: {network_in_top_50} posts ({network_in_top_50/min(50, len(final_feed))*100:.1f}%)")
        logger.info(f"Target ratio achieved: {network_in_final/len(final_feed)*100:.1f}% (target: {target_network_ratio*100:.1f}%)")
        
        return final_feed
        
    except Exception as e:
        logger.error(f"Failed to distribute network posts: {e}")
        return ranked_posts