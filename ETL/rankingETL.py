import os
import sys
import argparse
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.bluesky.newPosts import Client as BlueskyClient
from client.bigQuery import Client as BigQueryClient
from client.redis import Client as RedisClient
from ranking.bm25Similarity import compute_bm25_similarity, compute_enhanced_bm25_similarity
from ranking.cosineSimilarity import compute_user_feed_similarity
from experiments.abTesting import get_ab_test_manager
from datetime import datetime, timezone
from dateutil import parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_active_users_with_keywords(redis_client: RedisClient, bq_client: BigQueryClient, test_mode: bool = False) -> List[Dict]:
    """Get active users with their stored keywords from Redis + BigQuery"""
    try:
        # Get active users from Redis (fast, real-time activity)
        active_user_ids = redis_client.get_active_users(days=30)
        
        if not active_user_ids:
            logger.warning("No active users found in Redis, falling back to BigQuery")
            return get_users_with_keywords_from_bigquery_fallback(bq_client, test_mode)
        
        # Limit in test mode
        if test_mode:
            active_user_ids = active_user_ids[:5]
        
        logger.info(f"Found {len(active_user_ids)} active users from Redis")
        
        # Get keywords for active users from BigQuery
        if not active_user_ids:
            return []
        
        # Create parameterized query for active users
        user_ids_str = "', '".join(active_user_ids)
        query = f"""
        SELECT 
            user_id,
            handle,
            keywords
        FROM `{bq_client.project_id}.data.users`
        WHERE user_id IN ('{user_ids_str}')
        AND keywords IS NOT NULL
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved keywords for {len(users)} active users from BigQuery")
        return users
        
    except Exception as e:
        logger.error(f"Error getting active users: {e}")
        logger.warning("Falling back to BigQuery-only approach")
        return get_users_with_keywords_from_bigquery_fallback(bq_client, test_mode)

def get_users_with_keywords_from_bigquery_fallback(bq_client: BigQueryClient, test_mode: bool = False) -> List[Dict]:
    """Fallback: Get active users with their stored keywords from BigQuery only"""
    try:
        # Only limit in test mode for development
        limit_clause = "LIMIT 5" if test_mode else ""
        
        query = f"""
        SELECT 
            user_id,
            handle,
            keywords
        FROM `{bq_client.project_id}.data.users`
        WHERE last_request_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY last_request_at DESC
        {limit_clause}
        """
        
        result = bq_client.query(query)
        users = result.to_dict('records') if not result.empty else []
        
        logger.info(f"Retrieved {len(users)} active users with keywords from BigQuery (fallback)")
        return users
        
    except Exception as e:
        logger.warning(f"Could not get users from BigQuery: {e}")
        return []

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
    if position < 2:
        post_age = get_post_age_days(post)
        if post_age > 5:
            logger.debug(f"Filtering FAQ: position {position}, age {post_age} days")
            return True
    
    # Filter 2: Keyword pattern matching
    text = post.get('text', '').lower()
    if not text:
        return False
    
    faq_keywords = [
        "faq", "frequently asked", "welcome to", "how to use",
        "getting started", "please read", "rules", "guidelines", 
        "this feed", "submit to", "curated by", "about this feed",
        "before posting", "read first", "pinned", "instructions",
        "how this works", "feed description", "what is this feed"
    ]
    
    for keyword in faq_keywords:
        if keyword in text:
            logger.debug(f"Filtering FAQ: keyword '{keyword}' found")
            return True
    
    return False


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
    return likes + (reposts * 2) + (replies * 0.5)


def filter_low_engagement_posts(posts: List[Dict], source: str, min_engagement: float = 8.0) -> List[Dict]:
    """
    Filter posts based on engagement threshold for feeds
    
    Args:
        posts: List of posts to filter
        source: Source type ('feed', 'network')
        min_engagement: Minimum engagement score required
        
    Returns:
        Filtered posts that meet engagement criteria
    """
    if source == 'network':
        # Never filter network posts - always include following posts
        return posts
    
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


def get_user_embeddings(user_id: str, bq_client: BigQueryClient) -> Optional[np.ndarray]:
    """
    Retrieve user embeddings from BigQuery users table
    
    Args:
        user_id: User's DID
        bq_client: BigQuery client instance
        
    Returns:
        User embedding as numpy array, or None if not found
    """
    try:
        query = f"""
        SELECT embeddings
        FROM `{bq_client.project_id}.data.users`
        WHERE user_id = '{user_id}'
        AND embeddings IS NOT NULL
        LIMIT 1
        """
        
        result = bq_client.query(query)
        
        if result.empty:
            logger.warning(f"No embeddings found for user {user_id}")
            return None
        
        embeddings_data = result.iloc[0]['embeddings']
        
        # Parse JSON embeddings
        if isinstance(embeddings_data, str):
            embeddings = json.loads(embeddings_data)
        else:
            embeddings = embeddings_data
        
        if not embeddings or not isinstance(embeddings, list):
            logger.warning(f"Invalid embeddings format for user {user_id}")
            return None
        
        embeddings_array = np.array(embeddings)
        logger.info(f"Retrieved embeddings for user {user_id}: shape {embeddings_array.shape}")
        return embeddings_array
        
    except Exception as e:
        logger.error(f"Failed to retrieve embeddings for user {user_id}: {e}")
        return None


def get_user_reading_level(user_id: str, bq_client: BigQueryClient) -> int:
    """
    Retrieve user's reading level from BigQuery users table
    
    Args:
        user_id: User's DID
        bq_client: BigQuery client instance
        
    Returns:
        User's reading level (1-20 scale), defaults to 8 if not found
    """
    try:
        query = f"""
        SELECT reading_level
        FROM `{bq_client.project_id}.data.users`
        WHERE user_id = '{user_id}'
        AND reading_level IS NOT NULL
        LIMIT 1
        """
        
        result = bq_client.query(query)
        
        if result.empty:
            logger.debug(f"No reading level found for user {user_id}, using default")
            return 8  # Default reading level
        
        reading_level = result.iloc[0]['reading_level']
        
        # Ensure reasonable bounds
        reading_level = max(1, min(20, int(reading_level)))
        
        logger.debug(f"Retrieved reading level {reading_level} for user {user_id}")
        return reading_level
        
    except Exception as e:
        logger.error(f"Failed to retrieve reading level for user {user_id}: {e}")
        return 8  # Default reading level


def detect_keywords_format(user_keywords) -> str:
    """
    Detect the format of user keywords data
    
    Returns:
        'enhanced', 'basic', or 'invalid'
    """
    try:
        import json
        
        if user_keywords is None:
            return 'invalid'
        
        # Parse JSON string if needed
        if isinstance(user_keywords, str):
            try:
                keywords_data = json.loads(user_keywords)
            except json.JSONDecodeError:
                return 'invalid'
        else:
            keywords_data = user_keywords
        
        # Check for enhanced format (dict with sentiment/emotions)
        if isinstance(keywords_data, dict):
            # Look for enhanced keyword structure
            for keyword, data in keywords_data.items():
                if isinstance(data, dict) and 'sentiment' in data and 'emotions' in data:
                    return 'enhanced'
            return 'basic'  # Dict but not enhanced format
        
        # Check for basic format (list of strings)
        elif isinstance(keywords_data, list):
            return 'basic'
        
        return 'invalid'
        
    except Exception as e:
        logger.error(f"Error detecting keywords format: {e}")
        return 'invalid'


def get_enhanced_user_terms(enhanced_keywords) -> tuple:
    """
    Convert enhanced keywords to weighted terms + sentiment context
    
    Returns:
        (weighted_terms_list, sentiment_context_dict)
    """
    try:
        import json
        
        # Parse JSON string if needed
        if isinstance(enhanced_keywords, str):
            keywords_data = json.loads(enhanced_keywords)
        else:
            keywords_data = enhanced_keywords
        
        if not isinstance(keywords_data, dict):
            logger.warning("Enhanced keywords should be a dictionary")
            return [], {}
        
        terms = []
        sentiment_context = {}
        
        for keyword, data in keywords_data.items():
            if not isinstance(data, dict):
                continue
                
            # Extract frequency for weighting
            frequency = data.get('frequency', 1)
            weight = min(5, max(1, frequency))  # Clamp weight between 1-5
            
            # Add terms weighted by frequency
            terms.extend([keyword] * weight)
            
            # Store sentiment context
            sentiment_context[keyword] = data
        
        logger.info(f"Converted {len(sentiment_context)} enhanced keywords to {len(terms)} weighted terms")
        return terms, sentiment_context
        
    except Exception as e:
        logger.error(f"Failed to process enhanced keywords: {e}")
        return [], {}


def get_user_keywords_as_terms(user_keywords) -> List[str]:
    """Convert stored keywords to term list for BM25 (legacy function for backward compatibility)"""
    try:
        import json
        terms = []
        
        # Handle different formats of keywords from BigQuery JSON
        if isinstance(user_keywords, list):
            keywords_list = user_keywords
        elif isinstance(user_keywords, str):
            # JSON string that needs parsing
            try:
                keywords_list = json.loads(user_keywords)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in keywords: {user_keywords}")
                return []
        elif user_keywords is None:
            return []
        else:
            logger.warning(f"Unexpected keywords format: {type(user_keywords)}")
            return []
        
        # Keywords are stored as simple string arrays
        for kw in keywords_list:
            if isinstance(kw, str):
                # Simple string format - add multiple times for weighting
                terms.extend([kw] * 3)  # Default weight of 3
            else:
                logger.warning(f"Unexpected keyword type: {type(kw)}")
        
        logger.info(f"Converted {len(set(terms))} unique keywords to {len(terms)} weighted terms")
        return terms
        
    except Exception as e:
        logger.error(f"Failed to process user keywords: {e}")
        return []


def process_user_keywords(user_keywords):
    """
    Process user keywords detecting format and returning appropriate data
    
    Returns:
        For enhanced format: (terms_list, sentiment_context_dict, 'enhanced')
        For basic format: (terms_list, None, 'basic')  
        For invalid: ([], None, 'invalid')
    """
    keywords_format = detect_keywords_format(user_keywords)
    
    if keywords_format == 'enhanced':
        terms, sentiment_context = get_enhanced_user_terms(user_keywords)
        return terms, sentiment_context, 'enhanced'
    elif keywords_format == 'basic':
        terms = get_user_keywords_as_terms(user_keywords)
        return terms, None, 'basic'
    else:
        logger.warning("Invalid keywords format detected")
        return [], None, 'invalid'


def get_or_update_following_list(user_did: str, redis_client: RedisClient) -> List[Dict]:
    """
    Get user's following list from cache or fetch fresh if expired
    
    Args:
        user_did: User's DID
        redis_client: Redis client for caching
        
    Returns:
        List of users they follow
    """
    try:
        # Try to get from cache first
        cached_following = redis_client.get_cached_following_list(user_did)
        
        if cached_following is not None:
            logger.info(f"Using cached following list for {user_did}: {len(cached_following)} accounts")
            return cached_following
        
        # Cache miss - fetch fresh data
        logger.info(f"Fetching fresh following list for {user_did}")
        from client.bluesky.userData import Client as UserDataClient
        user_client = UserDataClient()
        user_client.login()
        
        following_list = user_client.get_all_user_follows(user_did)
        
        if following_list:
            # Cache the results
            redis_client.cache_user_following_list(user_did, following_list)
            logger.info(f"Cached fresh following list for {user_did}: {len(following_list)} accounts")
        
        return following_list or []
        
    except Exception as e:
        logger.error(f"Failed to get following list for {user_did}: {e}")
        return []


def calculate_2nd_degree_overlap(user_did: str, redis_client: RedisClient, min_overlap: int = 2) -> Dict:
    """
    Calculate 2nd degree network overlap by analyzing who each 1st degree connection follows
    
    Args:
        user_did: User's DID
        redis_client: Redis client for caching
        min_overlap: Minimum overlap count to qualify as 2nd degree candidate
        
    Returns:
        Dictionary of qualified 2nd degree candidates with overlap scores
    """
    try:
        # Check cache first
        cached_analysis = redis_client.get_cached_overlap_analysis(user_did)
        if cached_analysis is not None:
            logger.info(f"Using cached 2nd degree analysis for {user_did}: {len(cached_analysis)} candidates")
            return cached_analysis
        
        # Cache miss - perform fresh analysis
        logger.info(f"Calculating fresh 2nd degree overlap analysis for {user_did}")
        
        # Get user's 1st degree network (cached)
        first_degree = get_or_update_following_list(user_did, redis_client)
        
        if not first_degree:
            logger.warning(f"No 1st degree network found for {user_did}")
            return {}
        
        # Track overlap candidates
        overlap_candidates = {}
        processed_count = 0
        
        # For each 1st degree connection, get who they follow
        for followed_user in first_degree:
            followed_did = followed_user.get('did')
            followed_handle = followed_user.get('handle', 'unknown')
            
            if not followed_did:
                continue
                
            try:
                # Get who this 1st degree connection follows (their 2nd degree candidates)
                their_following = get_or_update_following_list(followed_did, redis_client)
                
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
                logger.info(f"Processed {processed_count}/{len(first_degree)}: {followed_handle} follows {len(their_following)} accounts")
                
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
        
        # Cache the results for 1 week
        if qualified_candidates:
            redis_client.cache_network_overlap_analysis(user_did, qualified_candidates)
        
        logger.info(f"2nd degree analysis complete for {user_did}: {len(qualified_candidates)} qualified candidates from {len(overlap_candidates)} total")
        logger.info(f"Overlap distribution: {sum(1 for c in qualified_candidates.values() if c['overlap_count'] == 2)} 2-overlap, {sum(1 for c in qualified_candidates.values() if c['overlap_count'] >= 3)} 3+-overlap")
        
        return qualified_candidates
        
    except Exception as e:
        logger.error(f"Failed to calculate 2nd degree overlap for {user_did}: {e}")
        return {}


def collect_2nd_degree_posts(user_did: str, redis_client: RedisClient, posts_limit: int = 20) -> List[Dict]:
    """
    Collect posts from 2nd degree network (overlap-based discovery)
    
    Args:
        user_did: User's DID
        redis_client: Redis client for caching
        posts_limit: Number of posts to collect per 2nd degree account
        
    Returns:
        List of posts from 2nd degree network with overlap metadata
    """
    try:
        # Get 2nd degree overlap analysis
        overlap_candidates = calculate_2nd_degree_overlap(user_did, redis_client)
        
        if not overlap_candidates:
            logger.info(f"No 2nd degree candidates found for {user_did}")
            return []
        
        # Sort candidates by priority score and limit to top 50
        sorted_candidates = sorted(
            overlap_candidates.values(), 
            key=lambda x: x['priority_score'], 
            reverse=True
        )[:50]
        
        logger.info(f"Collecting posts from top {len(sorted_candidates)} 2nd degree candidates")
        
        # Initialize posts client
        from client.bluesky.userData import Client as UserDataClient
        user_client = UserDataClient()
        user_client.login()
        
        all_2nd_degree_posts = []
        successful_collections = 0
        
        # Collect posts from each qualified 2nd degree account
        for candidate_data in sorted_candidates:
            account = candidate_data['account']
            candidate_did = account.get('did')
            candidate_handle = account.get('handle', 'unknown')
            overlap_count = candidate_data['overlap_count']
            
            try:
                # Get recent posts from this 2nd degree account
                posts = user_client.get_user_posts(
                    actor=candidate_did,
                    limit=posts_limit
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
        
        # Fetch profiles in batches of 25 (API limit)
        all_profiles = []
        batch_size = 25
        
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
        
        # Threshold: keep users above +0.0 standard deviations
        z_threshold = -0.30
        
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
        
        # Always return at least 30% of users if available
        min_users = max(20, len(valid_users) // 3)  # At least 20 users or 30% of total
        if len(filtered_users) < min_users:
            logger.info(f"Filter too aggressive ({len(filtered_users)} users), taking top {min_users} users by ratio")
            valid_users.sort(key=lambda x: x['calculated_ratio'], reverse=True)
            filtered_users = valid_users[:min_users]
        
        return filtered_users
        
    except Exception as e:
        logger.error(f"Error in ratio-based filtering: {e}")
        logger.error(f"Returning original following list")
        return following_list

def collect_comprehensive_posts(
    user_embeddings: Optional[np.ndarray],
    user_keywords: List[str], 
    user_did: str,
    bq_client: BigQueryClient,
    redis_client: RedisClient
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
        posts_client = BlueskyClient()
        posts_client.login()
        
        # 1. Get posts from matching feeds (if user has embeddings)
        if user_embeddings is not None:
            try:
                logger.info("Collecting posts from matching feeds...")
                matching_feeds = compute_user_feed_similarity(
                    user_embeddings=user_embeddings.tolist(),
                    bq_client=bq_client,
                    threshold=0.4
                )
                
                feed_posts = []
                for feed_match in matching_feeds[:10]:  # Limit to top 10 feeds
                    try:
                        posts = posts_client.extract_posts_from_feed(
                            feed_url_or_uri=feed_match['feed_uri'],
                            time_hours=6
                        )
                        
                        # Filter FAQ posts before processing
                        posts = filter_faq_posts(posts)
                        
                        # Filter low-engagement posts
                        posts = filter_low_engagement_posts(posts, source='feed', min_engagement=15.0)
                        
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
                    time_hours=6,
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
            
            second_degree_posts = collect_2nd_degree_posts(user_did, redis_client, posts_limit=15)
            
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
            time_hours=6,
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

def load_blocked_users() -> set:
    """Load blocked users from moderation file"""
    try:
        moderation_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'moderation', 'moderation.txt')
        
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

def filter_blocked_posts(posts: List[Dict], blocked_users: set) -> List[Dict]:
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
        total_engagement = likes + (reposts * 2) + (replies * 1.5)
        
        # Handle very fresh posts (< 1 minute) to avoid division issues
        if age_minutes < 1.0:
            age_minutes = 1.0
        
        velocity = total_engagement / age_minutes
        return max(0.0, velocity)
        
    except Exception as e:
        logger.debug(f"Could not calculate engagement velocity: {e}")
        return 0.0


def is_viral_post(post: Dict, age_threshold: float = 45.0, velocity_threshold: float = 1.0) -> bool:
    """
    Determine if post qualifies for viral boost
    
    Args:
        post: Post dictionary
        age_threshold: Maximum age in minutes for viral consideration
        velocity_threshold: Minimum engagement velocity required
        
    Returns:
        True if post should receive viral boost
    """
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


def calculate_rankings_with_feed_boosting(user_terms: List[str], posts: List[Dict], user_sentiment_context: Dict = None, user_reading_level: int = 8) -> List[Dict]:
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
        
        # Use enhanced BM25 if sentiment context is available, otherwise fall back to basic BM25
        if user_sentiment_context:
            logger.info("Using enhanced BM25 with sentiment analysis")
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
                    priority_boost = 3.0 + (bm25_score * 0.1)
                    post['boost_reason'] = '1st_degree_network'
                else:
                    # Should not happen since network posts are tagged as 1st degree
                    priority_boost = 3.0 + (bm25_score * 0.1)
                    post['boost_reason'] = 'network_priority'
                post['priority_boost'] = priority_boost
            elif source == '2nd_degree':
                # 2nd degree network: High priority based on overlap count
                overlap_count = post.get('overlap_count', 1)
                base_boost = 1.0 + (overlap_count / 3.0)  # 1.0-2.0 range
                priority_boost = base_boost + (bm25_score * 0.15)
                post['priority_boost'] = priority_boost
                post['boost_reason'] = '2nd_degree_network'
            elif source == 'feed':
                # Feed posts: Medium priority boost (feed similarity + BM25)
                feed_similarity = post.get('feed_similarity', 0)
                priority_boost = 1.5 + (feed_similarity * 1.0) + (bm25_score * 0.2)
                post['priority_boost'] = priority_boost
                post['boost_reason'] = 'feed_priority'
            else:
                # Unknown source: minimal boost
                priority_boost = 0.1
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
                viral_multiplier = 1.8  # 80% boost for viral content
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
        ranked_posts = compute_bm25_similarity(user_terms, filtered_posts)
        
        # Sort by BM25 score
        ranked_posts = sorted(ranked_posts, key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        logger.info(f"Calculated rankings for {len(ranked_posts)} posts using {len(set(user_terms))} unique keywords")
        return ranked_posts
        
    except Exception as e:
        logger.error(f"Failed to calculate rankings: {e}")
        return []

def distribute_network_posts(ranked_posts: List[Dict], target_network_ratio: float = 0.3) -> List[Dict]:
    """
    Distribute network posts evenly throughout the ranked feed instead of clustering at top
    
    Args:
        ranked_posts: List of posts with BM25 scores and source tags
        target_network_ratio: Target percentage of network posts in feed (default 30%)
        
    Returns:
        Posts with network posts distributed evenly throughout
    """
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
        boost_factor = 2.5  # Strong boost to ensure network visibility
        
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

def cache_user_rankings(redis_client: RedisClient, user_id: str, ranked_posts: List[Dict]) -> bool:
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
        
        final_feed = sorted(merged_posts.values(), key=lambda x: x['score'], reverse=True)[:1500]
        
        logger.info(f"Deduplication: {duplicate_count} duplicate posts found and updated with fresh scores")
        
        success = redis_client.set_user_feed(user_id, final_feed, ttl=900)  # 15 minutes TTL
        
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

def update_default_feed_if_needed(redis_client: RedisClient):
    """Update default feed if it's expired or missing (once daily)"""
    try:
        # Check if default feed exists and is fresh
        existing_default = redis_client.get_default_feed()
        if existing_default:
            logger.info("Default feed already cached and fresh, skipping update")
            return
        
        logger.info("Default feed expired or missing, generating new default feed...")
        
        # Generate default feed with popular posts from last 24 hours
        bluesky_client = BlueskyClient()
        bluesky_client.login()
        
        # Get popular posts using multiple search queries
        popular_posts = bluesky_client.get_top_posts_multiple_queries(
            queries=["the", "a", "I", "you", "this", "today", "new", "just", "really", "good"],
            target_count=100,
            time_hours=24  # Last 24 hours
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
        success = redis_client.set_default_feed(formatted_posts, ttl=86400)
        
        if success:
            logger.info(f"Successfully updated default feed with {len(formatted_posts)} posts")
        else:
            logger.error("Failed to cache default feed")
            
    except Exception as e:
        logger.error(f"Error updating default feed: {e}")
        raise

# Keyword extraction removed - keywords are now pre-stored in BigQuery
# This will be handled by the separate user discovery process

def main():
    """Main ETL process"""
    parser = argparse.ArgumentParser(description='Feed Ranking ETL')
    parser.add_argument('--test-mode', default='false', help='Run in test mode with limited users')
    args = parser.parse_args()
    
    test_mode = args.test_mode.lower() == 'true'
    batch_id = f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting Feed Ranking ETL (test_mode={test_mode}, batch_id={batch_id})")
    
    try:
        # Initialize clients
        credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
        bq_client = BigQueryClient(credentials_json, os.environ['BIGQUERY_PROJECT_ID'])
        redis_client = RedisClient()
        
        
        # Get users with their stored keywords (Redis + BigQuery hybrid)
        users = get_active_users_with_keywords(redis_client, bq_client, test_mode)
        
        if not users:
            logger.warning("No users found to process, but will still update default feed")
            success_count, error_count = 0, 0
        else:
            logger.info(f"Processing {len(users)} users")
            success_count = 0
            error_count = 0
        
        for user in users:
            user_handle = user.get('handle', '')
            user_id = user.get('user_id', user_handle)
            
            try:
                logger.info(f"Processing user: {user_handle}")
                
                # Step 1: Get user embeddings, keywords, and reading level
                user_embeddings = get_user_embeddings(user_id, bq_client)
                user_keywords = user.get('keywords')
                user_reading_level = get_user_reading_level(user_id, bq_client)
                
                # Process keywords using enhanced format detection
                user_terms, user_sentiment_context, keywords_format = process_user_keywords(user_keywords)
                
                # Users can now get feeds based on network alone, even without keywords/embeddings
                if not user_terms and user_embeddings is None:
                    logger.info(f"No keywords or embeddings for {user_handle}, using network-only approach")
                
                # Log keyword format for monitoring
                logger.info(f"User {user_handle} using {keywords_format} keywords format with {len(user_terms)} terms")
                
                # A/B Testing: Determine if user should get enhanced ranking
                ab_test_manager = get_ab_test_manager()
                use_enhanced, experiment_group = ab_test_manager.should_use_enhanced_ranking(user_id)
                
                # Override user_sentiment_context based on A/B test group
                if not use_enhanced:
                    logger.info(f"User {user_handle} in A/B test group '{experiment_group}' - using basic ranking")
                    user_sentiment_context = None  # Force basic ranking
                else:
                    logger.info(f"User {user_handle} in A/B test group '{experiment_group}' - using enhanced ranking")
                
                # Cache experiment group assignment in Redis for consistency
                redis_client.set_user_experiment_group(user_id, 'enhanced_keywords_experiment', experiment_group)
                
                # Step 2: Comprehensive post collection (feeds + 1st/2nd degree network)
                posts_to_rank = collect_comprehensive_posts(
                    user_embeddings=user_embeddings,
                    user_keywords=user_terms,
                    user_did=user_id,
                    bq_client=bq_client,
                    redis_client=redis_client
                )
                
                if not posts_to_rank:
                    logger.warning(f"No posts collected for {user_handle}, skipping")
                    continue
                
                # Step 3: Calculate rankings with enhanced BM25 and feed boosting
                if user_terms:
                    ranked_posts = calculate_rankings_with_feed_boosting(
                        user_terms, 
                        posts_to_rank,
                        user_sentiment_context=user_sentiment_context,
                        user_reading_level=user_reading_level
                    )
                else:
                    # Fallback for users with embeddings but no keywords
                    logger.info(f"No keywords for {user_handle}, using basic scoring")
                    for post in posts_to_rank:
                        post['bm25_score'] = 1.0  # Base score
                        source = post.get('source', 'unknown')
                        if source == 'feed':
                            feed_similarity = post.get('feed_similarity', 0)
                            feed_boost = 1.0 + (feed_similarity * 2.0)
                            post['final_score'] = 1.0 * feed_boost
                        else:
                            post['final_score'] = 1.0
                    ranked_posts = sorted(posts_to_rank, key=lambda x: x.get('final_score', 0), reverse=True)
                
                if not ranked_posts:
                    logger.warning(f"No rankings calculated for {user_handle}, skipping")
                    continue
                
                # Step 4: Posts already ranked by priority (network > feed > search)
                logger.info(f"Applied priority-based ranking for user {user_handle}")
                
                # Step 5: Cache results
                cache_success = cache_user_rankings(redis_client, user_id, ranked_posts)
                
                if cache_success:
                    success_count += 1
                    embeddings_status = "with embeddings" if user_embeddings is not None else "no embeddings"
                    keywords_count = len(user_terms) if user_terms else 0
                    
                    # Log experiment metrics for A/B testing analysis
                    experiment_metrics = {
                        'experiment_group': experiment_group,
                        'keywords_format': keywords_format,
                        'user_reading_level': user_reading_level,
                        'posts_ranked': len(ranked_posts),
                        'keywords_count': keywords_count,
                        'has_embeddings': user_embeddings is not None,
                        'has_sentiment_context': user_sentiment_context is not None,
                        'ranking_method': 'enhanced' if user_sentiment_context else 'basic'
                    }
                    
                    # Add sentiment analysis stats if available
                    if user_sentiment_context and ranked_posts:
                        sentiment_boosted = len([p for p in ranked_posts[:50] if p.get('sentiment_multiplier', 1.0) > 1.05])
                        sentiment_penalized = len([p for p in ranked_posts[:50] if p.get('sentiment_multiplier', 1.0) < 0.95])
                        experiment_metrics.update({
                            'sentiment_boosted_posts': sentiment_boosted,
                            'sentiment_penalized_posts': sentiment_penalized
                        })
                    
                    ab_test_manager.log_ranking_experiment(user_id, experiment_metrics)
                    redis_client.log_experiment_metrics(user_id, experiment_metrics)
                    
                    logger.info(f"Successfully processed {user_handle} ({embeddings_status}, {keywords_count} keywords, {experiment_group} group)")
                else:
                    error_count += 1
                    logger.error(f"Failed to cache results for {user_handle}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {user_handle}: {e}")
                continue
        
        # Update default feed for new users (once daily)
        try:
            update_default_feed_if_needed(redis_client)
        except Exception as e:
            logger.error(f"Failed to update default feed: {e}")
        
        # Final summary
        logger.info(f"ETL Complete! Success: {success_count}, Errors: {error_count}")
        
        # Show cache stats
        stats = redis_client.get_stats()
        cached_users = redis_client.get_cached_users()
        logger.info(f"Redis Stats: {stats}")
        logger.info(f"Total cached feeds: {len(cached_users)}")
        logger.info(f"ETL Performance: Network-focused collection (feeds + network) with embedding-based feed filtering")
        
    except Exception as e:
        logger.error(f"ETL failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()