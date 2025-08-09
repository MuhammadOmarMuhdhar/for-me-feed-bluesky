"""
Post collection functions for the ranking ETL system.
"""
import logging
import numpy as np
from typing import Dict, List, Optional

from ETL.ranking.config import (
    DEFAULT_TIME_HOURS, MAX_TOP_FEEDS, FEED_SIMILARITY_THRESHOLD,
    MIN_ENGAGEMENT_FEED
)
from ETL.ranking.contentFilters import filter_faq_posts, filter_low_engagement_posts
from ETL.ranking.networkAnalyzer import get_or_update_following_list, collect_2nd_degree_posts

logger = logging.getLogger(__name__)


def collect_comprehensive_posts(
    user_embeddings: Optional[np.ndarray],
    user_keywords: List[str], 
    user_did: str,
    bq_client,
    redis_client
) -> List[Dict]:
    """
    Collect posts from three sources: matching feeds, 1st degree network, and 2nd degree network
    
    Args:
        user_embeddings: User's embedding vector for feed matching
        user_keywords: User's keywords for BM25 ranking (not collection)
        user_did: User's DID for network posts
        bq_client: BigQuery client for feed matching
        redis_client: Redis client for network caching
        
    Returns:
        List of posts from all sources with source tagging
    """
    all_posts = []
    
    try:
        from client.bluesky.newPosts import Client as BlueskyClient
        posts_client = BlueskyClient()
        posts_client.login()
        
        # 1. Get posts from matching feeds (if user has embeddings)
        if user_embeddings is not None:
            try:
                logger.info("Collecting posts from matching feeds...")
                from ranking.cosineSimilarity import compute_user_feed_similarity
                
                matching_feeds = compute_user_feed_similarity(
                    user_embeddings=user_embeddings.tolist(),
                    bq_client=bq_client,
                    threshold=FEED_SIMILARITY_THRESHOLD
                )
                
                feed_posts = []
                for feed_match in matching_feeds[:MAX_TOP_FEEDS]:  # Limit to top feeds
                    try:
                        posts = posts_client.extract_posts_from_feed(
                            feed_url_or_uri=feed_match['feed_uri'],
                            time_hours=DEFAULT_TIME_HOURS
                        )
                        
                        # Filter FAQ posts before processing
                        posts = filter_faq_posts(posts)
                        
                        # Filter low-engagement posts
                        posts = filter_low_engagement_posts(posts, source='feed', min_engagement=MIN_ENGAGEMENT_FEED)
                        
                        # Tag posts with feed info
                        for post in posts:
                            post['source'] = 'feed'
                            post['feed_similarity'] = feed_match['similarity_score']
                            post['feed_uri'] = feed_match['feed_uri']
                        
                        feed_posts.extend(posts)
                        logger.info(f"Collected {len(posts)} posts from feed: {feed_match['feed_uri']}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to collect from feed {feed_match['feed_uri']}: {e}")
                        continue
                
                all_posts.extend(feed_posts)
                logger.info(f"Total feed posts collected: {len(feed_posts)}")
                
            except Exception as e:
                logger.error(f"Failed to collect feed posts: {e}")
        else:
            logger.info("No user embeddings available, skipping feed-based collection")
        
        # 2. Get posts from 1st degree network (following)
        try:
            logger.info("Collecting posts from 1st degree network...")
            
            # Get following list (cached) - using all accounts, no z-score filtering
            following_list = get_or_update_following_list(user_did, redis_client)
            
            if following_list:
                network_posts = posts_client.get_following_timeline(
                    following_list=following_list,
                    time_hours=DEFAULT_TIME_HOURS,
                    include_reposts=True,
                    repost_weight=0.5
                )
                
                # Filter and tag 1st degree network posts  
                network_posts = filter_low_engagement_posts(network_posts, source='network')
                for post in network_posts:
                    post['source'] = 'network'
                    post['from_network'] = True
                    post['network_degree'] = 1
                
                all_posts.extend(network_posts)
                logger.info(f"Collected {len(network_posts)} 1st degree network posts")
            else:
                logger.warning("No following list found for 1st degree network posts")
                
        except Exception as e:
            logger.error(f"Failed to collect 1st degree network posts: {e}")
        
        # 3. Get posts from 2nd degree network (overlap-based discovery)
        try:
            logger.info("Collecting posts from 2nd degree network...")
            
            second_degree_posts = collect_2nd_degree_posts(user_did, redis_client)
            
            # Tag 2nd degree posts
            for post in second_degree_posts:
                post['network_degree'] = 2
            
            all_posts.extend(second_degree_posts)
            logger.info(f"Collected {len(second_degree_posts)} 2nd degree network posts")
                
        except Exception as e:
            logger.error(f"Failed to collect 2nd degree network posts: {e}")
        
        
        # Log collection summary
        source_counts = {}
        for post in all_posts:
            source = post.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info(f"Network-focused collection complete: {len(all_posts)} total posts")
        logger.info(f"Source breakdown: {source_counts}")
        
        return all_posts
        
    except Exception as e:
        logger.error(f"Failed comprehensive post collection: {e}")
        return []


def collect_posts_to_rank(user_did: str = None) -> List[Dict]:
    """Collect posts to rank using network-only system"""
    try:
        from client.bluesky.newPosts import Client as BlueskyClient
        posts_client = BlueskyClient()
        posts_client.login()
        
        # Get user's following list for network posts
        following_list = []
        if user_did:
            try:
                from client.bluesky.userData import Client as UserDataClient
                user_client = UserDataClient()
                user_client.login()
                # Get following 
                following_data = user_client.get_all_user_follows(user_did)
                following_list = following_data if following_data else []  # Keep full objects
                logger.info(f"Retrieved {len(following_list)} following accounts for user {user_did}")
                
                # Remove z-score filtering - use all following accounts
                logger.info(f"Using all {len(following_list)} following accounts (no filtering)")
                    
            except Exception as e:
                logger.error(f"Failed to get following list for {user_did}: {e}")
                import traceback
                logger.error(f"Following list error traceback: {traceback.format_exc()}")
                following_list = []
        
        if not following_list:
            logger.warning("No following list available for network-based collection")
            return []
        
        # Get network posts only
        network_posts = posts_client.get_following_timeline(
            following_list=following_list,
            time_hours=DEFAULT_TIME_HOURS,
            include_reposts=True,
            repost_weight=0.5
        )
        
        # Tag all posts as network posts
        for post in network_posts:
            post['source'] = 'network'
            post['from_network'] = True
        
        logger.info(f"Collected {len(network_posts)} network posts from {len(following_list)} following accounts")
        
        return network_posts
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to collect posts: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []