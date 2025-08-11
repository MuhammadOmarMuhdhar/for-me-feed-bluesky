"""
Network analysis functions for the ranking ETL system.
"""
import logging
import numpy as np
import time
from typing import Dict, List

from ETL.ranking.config import (
    MIN_2ND_DEGREE_OVERLAP, MAX_2ND_DEGREE_CANDIDATES, POSTS_PER_2ND_DEGREE_ACCOUNT,
    NETWORK_BOOST_FACTOR, Z_SCORE_THRESHOLD, PROFILE_BATCH_SIZE,
    MIN_FOLLOWING_USERS, FOLLOWING_PERCENTAGE_FALLBACK, DEFAULT_TIME_HOURS,
    ANALYSIS_API_DELAY_SECONDS, MAX_FOLLOWS_PER_USER_ANALYSIS
)
from ETL.ranking.cacheManager import (
    cache_following_list, get_cached_following_list,
    cache_2nd_degree_network, get_cached_2nd_degree_network
)

logger = logging.getLogger(__name__)


def get_or_update_following_list(user_did: str, redis_client) -> List[Dict]:
    """
    Get user's following list from cache or fetch fresh if expired
    
    Args:
        user_did: User's DID
        redis_client: Redis client for caching
        
    Returns:
        List of users they follow
    """
    try:
        # Try to get from cache first using centralized cache manager
        cached_following = get_cached_following_list(redis_client, user_did)
        
        if cached_following:
            logger.info(f"Using cached following list for {user_did}: {len(cached_following)} accounts")
            return cached_following
        
        # Cache miss - fetch fresh data
        logger.info(f"Fetching fresh following list for {user_did}")
        from client.bluesky.userData import Client as UserDataClient
        user_client = UserDataClient()
        user_client.login()
        
        following_list = user_client.get_all_user_follows(user_did)
        
        if following_list:
            # Cache the results using centralized cache manager
            cache_following_list(redis_client, user_did, following_list)
            logger.info(f"Cached fresh following list for {user_did}: {len(following_list)} accounts")
        
        return following_list or []
        
    except Exception as e:
        logger.error(f"Failed to get following list for {user_did}: {e}")
        return []


def fetch_following_list_temporary(user_did: str) -> List[Dict]:
    """
    Fetch user's following list temporarily without caching (for 2nd degree analysis)
    
    Args:
        user_did: User's DID
        
    Returns:
        List of users they follow (not cached)
    """
    try:
        logger.debug(f"Fetching temporary following list for {user_did}")
        from client.bluesky.userData import Client as UserDataClient
        user_client = UserDataClient()
        user_client.login()
        
        following_list = user_client.get_all_user_follows(user_did)
        
        if following_list:
            logger.debug(f"Fetched {len(following_list)} accounts for {user_did} (temporary, not cached)")
        
        return following_list or []
        
    except Exception as e:
        logger.warning(f"Failed to fetch temporary following list for {user_did}: {e}")
        return []


def get_mutual_connections(user_did: str, following_list: List[Dict]) -> List[Dict]:
    """
    Get mutual connections (users who follow each other) from the following list
    
    Args:
        user_did: User's DID
        following_list: List of users they follow
        
    Returns:
        List of mutual connections (users who follow back)
    """
    try:
        from client.bluesky.userData import Client as UserDataClient
        user_client = UserDataClient()
        user_client.login()
        
        # Get user's followers
        logger.info(f"Fetching followers list for mutual connection analysis...")
        followers_list = user_client.get_all_user_followers(user_did)
        
        if not followers_list:
            logger.warning(f"No followers found for {user_did}")
            return []
        
        # Create set of follower DIDs for fast lookup
        followers_dids = {follower.get('did') for follower in followers_list}
        
        # Find mutual connections
        mutual_connections = [
            user for user in following_list 
            if user.get('did') in followers_dids
        ]
        
        logger.info(f"Found {len(mutual_connections)} mutual connections from {len(following_list)} following")
        return mutual_connections
        
    except Exception as e:
        logger.error(f"Failed to get mutual connections for {user_did}: {e}")
        return []


def calculate_2nd_degree_overlap(user_did: str, redis_client, min_overlap: int = None) -> Dict:
    """
    Calculate 2nd degree network overlap by analyzing who each 1st degree connection follows
    
    Args:
        user_did: User's DID
        redis_client: Redis client for caching
        min_overlap: Minimum overlap count to qualify as 2nd degree candidate
        
    Returns:
        Dictionary of qualified 2nd degree candidates with overlap scores
    """
    if min_overlap is None:
        min_overlap = MIN_2ND_DEGREE_OVERLAP
        
    try:
        # Check cache first using centralized cache manager
        cached_analysis = get_cached_2nd_degree_network(redis_client, user_did)
        if cached_analysis:
            logger.info(f"Using cached 2nd degree analysis for {user_did}: {len(cached_analysis)} candidates")
            return cached_analysis
        
        # Cache miss - perform fresh analysis
        logger.info(f"Calculating fresh 2nd degree overlap analysis for {user_did}")
        
        # Get user's 1st degree network (cached)
        first_degree = get_or_update_following_list(user_did, redis_client)
        
        if not first_degree:
            logger.warning(f"No 1st degree network found for {user_did}")
            return {}
        
        # Get mutual connections (users who follow each other)
        mutual_connections = get_mutual_connections(user_did, first_degree)
        
        if not mutual_connections:
            logger.warning(f"No mutual connections found for {user_did}")
            return {}
        
        logger.info(f"Analyzing {len(mutual_connections)} mutual connections (filtered from {len(first_degree)} total)")
        
        # Use mutual connections instead of all first degree connections
        connections_to_analyze = mutual_connections
        
        # Track overlap candidates
        overlap_candidates = {}
        processed_count = 0
        
        # For each mutual connection, get who they follow
        for followed_user in connections_to_analyze:
            followed_did = followed_user.get('did')
            followed_handle = followed_user.get('handle', 'unknown')
            
            if not followed_did:
                continue
                
            try:
                # Add delay to prevent rate limiting
                if processed_count > 0:
                    time.sleep(ANALYSIS_API_DELAY_SECONDS)
                
                # Get who this mutual connection follows (their 2nd degree candidates) - temporary fetch without caching
                their_following = fetch_following_list_temporary(followed_did)
                
                # Limit follows per user if configured
                if MAX_FOLLOWS_PER_USER_ANALYSIS and len(their_following) > MAX_FOLLOWS_PER_USER_ANALYSIS:
                    their_following = their_following[:MAX_FOLLOWS_PER_USER_ANALYSIS]
                    logger.debug(f"Limited {followed_handle} following list to {MAX_FOLLOWS_PER_USER_ANALYSIS} users")
                
                # Count overlaps
                for candidate in their_following:
                    candidate_did = candidate.get('did')
                    candidate_handle = candidate.get('handle', 'unknown')
                    
                    if not candidate_did or candidate_did == user_did:  # Skip self
                        continue
                    
                    # Initialize or update overlap tracking
                    if candidate_did not in overlap_candidates:
                        overlap_candidates[candidate_did] = {
                            'account': candidate,
                            'overlap_count': 0,
                            'followed_by': []
                        }
                    
                    overlap_candidates[candidate_did]['overlap_count'] += 1
                    overlap_candidates[candidate_did]['followed_by'].append(followed_handle)
                
                processed_count += 1
                logger.info(f"Processed {processed_count}/{len(connections_to_analyze)}: {followed_handle} follows {len(their_following)} accounts")
                
            except Exception as e:
                logger.warning(f"Failed to get following list for {followed_handle}: {e}")
                continue
        
        # Filter candidates by minimum overlap threshold
        qualified_candidates = {}
        for candidate_did, data in overlap_candidates.items():
            if data['overlap_count'] >= min_overlap:
                # Calculate priority score based on overlap count
                priority_score = data['overlap_count']
                if data['overlap_count'] >= 3:
                    priority_score *= 1.5  # Bonus for high overlap
                
                data['priority_score'] = priority_score
                qualified_candidates[candidate_did] = data
        
        # Cache the results using centralized cache manager with relevance filtering
        if qualified_candidates:
            cache_2nd_degree_network(redis_client, user_did, qualified_candidates)
        
        logger.info(f"2nd degree analysis complete for {user_did}: {len(qualified_candidates)} qualified candidates from {len(overlap_candidates)} total")
        logger.info(f"Overlap distribution: {sum(1 for c in qualified_candidates.values() if c['overlap_count'] == 2)} 2-overlap, {sum(1 for c in qualified_candidates.values() if c['overlap_count'] >= 3)} 3+-overlap")
        
        return qualified_candidates
        
    except Exception as e:
        logger.error(f"Failed to calculate 2nd degree overlap for {user_did}: {e}")
        return {}


def collect_2nd_degree_posts(user_did: str, redis_client, posts_limit: int = None) -> List[Dict]:
    """
    Collect posts from 2nd degree network (overlap-based discovery)
    
    Args:
        user_did: User's DID
        redis_client: Redis client for caching
        posts_limit: Number of posts to collect per 2nd degree account
        
    Returns:
        List of posts from 2nd degree network with overlap metadata
    """
    if posts_limit is None:
        posts_limit = POSTS_PER_2ND_DEGREE_ACCOUNT
        
    try:
        # Get 2nd degree overlap analysis
        overlap_candidates = calculate_2nd_degree_overlap(user_did, redis_client)
        
        if not overlap_candidates:
            logger.info(f"No 2nd degree candidates found for {user_did}")
            return []
        
        # Sort candidates by priority score and limit to max candidates
        sorted_candidates = sorted(
            overlap_candidates.values(), 
            key=lambda x: x['priority_score'], 
            reverse=True
        )[:MAX_2ND_DEGREE_CANDIDATES]
        
        logger.info(f"Collecting posts from top {len(sorted_candidates)} 2nd degree candidates")
        
        # Initialize posts client for timeline extraction
        from client.bluesky.newPosts import Client as BlueskyClient
        posts_client = BlueskyClient()
        posts_client.login()
        
        all_2nd_degree_posts = []
        successful_collections = 0
        
        # Collect posts from each qualified 2nd degree account
        for candidate_data in sorted_candidates:
            account = candidate_data['account']
            candidate_did = account.get('did')
            candidate_handle = account.get('handle', 'unknown')
            overlap_count = candidate_data['overlap_count']
            
            try:
                # Create single-user following list and use same timeline method as 1st degree
                single_user_list = [{'did': candidate_did, 'handle': candidate_handle}]
                posts = posts_client.get_following_timeline(
                    following_list=single_user_list,
                    time_hours=DEFAULT_TIME_HOURS,
                    include_reposts=True,
                    repost_weight=0.5
                )
                
                # Tag posts with 2nd degree metadata
                for post in posts:
                    post['source'] = '2nd_degree'
                    post['overlap_count'] = overlap_count
                    post['followed_by'] = candidate_data['followed_by']
                    post['priority_weight'] = min(1.0, overlap_count / 3.0)  # Scale 0.67-1.0
                    post['2nd_degree_account'] = candidate_handle
                
                all_2nd_degree_posts.extend(posts)
                successful_collections += 1
                
                logger.debug(f"Collected {len(posts)} posts from 2nd degree {candidate_handle} (overlap: {overlap_count})")
                
            except Exception as e:
                logger.warning(f"Failed to collect posts from 2nd degree {candidate_handle}: {e}")
                continue
        
        logger.info(f"2nd degree collection complete: {len(all_2nd_degree_posts)} posts from {successful_collections}/{len(sorted_candidates)} accounts")
        
        return all_2nd_degree_posts
        
    except Exception as e:
        logger.error(f"Failed to collect 2nd degree posts for {user_did}: {e}")
        return []


def prioritize_follows_by_ratio(following_list: List[Dict], posts_client, std_dev_multiplier: float = 0.2) -> List[Dict]:
    """
    Filter following list using dynamic percentiles based on postsCount/followersCount ratio
    
    Args:
        following_list: List of user objects from get_all_user_follows()
        posts_client: Client for fetching profile data
        std_dev_multiplier: How many standard deviations above mean to set threshold
        
    Returns:
        Filtered list of users above the dynamic threshold
    """
    if not following_list:
        logger.warning("Empty following list provided to ratio filtering")
        return []
    
    try:
        # Calculate ratios for all users
        user_ratios = []
        valid_users = []
        
        # Extract DIDs for batch profile fetching
        dids = [user.get('did') for user in following_list if user.get('did')]
        if not dids:
            logger.warning("No DIDs found in following list")
            return following_list
        
        # Fetch profiles in batches
        all_profiles = []
        batch_size = PROFILE_BATCH_SIZE
        
        for i in range(0, len(dids), batch_size):
            batch_dids = dids[i:i + batch_size]
            try:
                logger.info(f"Fetching profile batch {i//batch_size + 1}/{(len(dids) + batch_size - 1)//batch_size}: {len(batch_dids)} profiles")
                profiles_batch = posts_client.get_profiles(batch_dids)
                all_profiles.extend(profiles_batch)
            except Exception as e:
                logger.error(f"Error fetching profile batch: {e}")
                continue
        
        if not all_profiles:
            logger.warning("No profiles fetched, falling back to original list")
            return following_list
        
        # Create lookup map from DID to profile
        profile_map = {profile['did']: profile for profile in all_profiles}
        logger.info(f"Successfully fetched {len(all_profiles)} profiles out of {len(dids)} requested")
        
        # Debug: Log first few profiles to see actual field names and scoring
        if all_profiles:
            sample_profile = all_profiles[0]
            logger.info(f"Debug sample profile keys: {list(sample_profile.keys())}")
            posts = sample_profile.get('postsCount', 0)
            followers = sample_profile.get('followersCount', 1) or 1
            follows = sample_profile.get('followsCount', 1) or 1
            activity = posts / followers
            selectivity = followers / follows
            combined = activity * selectivity
            logger.info(f"Debug sample scores: posts={posts}, followers={followers}, follows={follows}")
            logger.info(f"Debug sample ratios: activity={activity:.3f}, selectivity={selectivity:.3f}, combined={combined:.3f}")
        
        for user in following_list:
            user_did = user.get('did')
            profile = profile_map.get(user_did)
            
            if not profile:
                continue
                
            posts_count = profile.get('postsCount', 0)
            followers_count = profile.get('followersCount', 1)
            follows_count = profile.get('followsCount', 1)
            
            # Avoid division by zero
            if followers_count <= 0:
                followers_count = 1
            if follows_count <= 0:
                follows_count = 1
                
            # Two-factor scoring: activity × selectivity
            activity = posts_count / followers_count      # How much they post relative to followers
            selectivity = followers_count / follows_count # How selective they are (followers/following)
            combined_score = activity * selectivity
            
            user_ratios.append(combined_score)
            
            # Add profile data to user dict
            enhanced_user = user.copy()
            enhanced_user.update({
                'postsCount': posts_count,
                'followersCount': followers_count,
                'followsCount': follows_count,
                'activity_score': activity,
                'selectivity_score': selectivity,
                'calculated_ratio': combined_score
            })
            valid_users.append(enhanced_user)
        
        if not user_ratios:
            logger.warning("No valid ratios calculated, returning original list")
            return following_list
        
        # Calculate statistics
        mean_ratio = np.mean(user_ratios)
        std_ratio = np.std(user_ratios)
        
        # Normalize ratios using z-score: (ratio - mean) / std_dev
        if std_ratio > 0:
            z_scores = [(ratio - mean_ratio) / std_ratio for ratio in user_ratios]
        else:
            # If std_dev is 0 (all ratios are the same), keep all users
            z_scores = [0.0] * len(user_ratios)
        
        # Threshold: keep users above threshold
        z_threshold = Z_SCORE_THRESHOLD
        
        # Filter users above z-score threshold
        filtered_users = []
        for user, z_score in zip(valid_users, z_scores):
            if z_score >= z_threshold:
                # Add z-score to user data for debugging
                user['z_score'] = z_score
                filtered_users.append(user)
        
        # Log filtering results
        logger.info(f"Ratio filtering results (Z-score normalized):")
        logger.info(f"  Original users: {len(following_list)}")
        logger.info(f"  Profiles fetched: {len(all_profiles)}")
        logger.info(f"  Valid ratios: {len(user_ratios)}")
        logger.info(f"  Mean ratio: {mean_ratio:.3f}")
        logger.info(f"  Std dev: {std_ratio:.3f}")
        logger.info(f"  Z-score threshold: {z_threshold}")
        logger.info(f"  Users above threshold: {len(filtered_users)}")
        logger.info(f"  Filter rate: {len(filtered_users)/len(user_ratios)*100:.1f}%")
        
        # Always return at least minimum users if available
        min_users = max(MIN_FOLLOWING_USERS, len(valid_users) // (100 // FOLLOWING_PERCENTAGE_FALLBACK))
        if len(filtered_users) < min_users:
            logger.info(f"Filter too aggressive ({len(filtered_users)} users), taking top {min_users} users by ratio")
            valid_users.sort(key=lambda x: x['calculated_ratio'], reverse=True)
            filtered_users = valid_users[:min_users]
        
        return filtered_users
        
    except Exception as e:
        logger.error(f"Error in ratio-based filtering: {e}")
        logger.error(f"Returning original following list")
        return following_list


def boost_network_posts(ranked_posts: List[Dict], user_did: str) -> List[Dict]:
    """
    Boost posts from user's network to ensure visibility throughout the feed
    
    Args:
        ranked_posts: List of posts with BM25 scores
        user_did: User's DID to get their following list
        
    Returns:
        Re-ranked posts with network posts boosted
    """
    try:
        # Get user's following list for boost identification
        following_dids = set()
        try:
            from client.bluesky.userData import Client as UserDataClient
            user_client = UserDataClient()
            user_client.login()
            following_data = user_client.get_all_user_follows(user_did)
            following_dids = {f.get('did', '') for f in following_data if isinstance(f, dict)}
            logger.info(f"Retrieved {len(following_dids)} following DIDs for network boost")
        except Exception as e:
            logger.warning(f"Could not get following list for network boost: {e}")
            return ranked_posts
        
        if not following_dids:
            return ranked_posts
        
        network_boosted = 0
        boost_factor = NETWORK_BOOST_FACTOR
        
        for post in ranked_posts:
            author_did = post.get('author', {}).get('did', '')
            if author_did in following_dids:
                # Apply strong boost to network posts
                original_score = post.get('bm25_score', 0)
                post['bm25_score'] = original_score * boost_factor
                post['network_boosted'] = True
                network_boosted += 1
            else:
                post['network_boosted'] = False
        
        # Re-sort after boosting
        ranked_posts.sort(key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        logger.info(f"Network boost applied: {network_boosted} posts boosted by {boost_factor}x")
        
        # Log distribution in top positions
        network_in_top_50 = sum(1 for post in ranked_posts[:50] if post.get('network_boosted', False))
        logger.info(f"Network representation in top 50: {network_in_top_50} posts ({network_in_top_50/50*100:.1f}%)")
        
        return ranked_posts
        
    except Exception as e:
        logger.error(f"Failed to boost network posts: {e}")
        return ranked_posts