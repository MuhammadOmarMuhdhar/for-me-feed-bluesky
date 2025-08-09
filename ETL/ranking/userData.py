"""
User data retrieval and processing functions for the ranking ETL system.
"""
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from ETL.ranking.config import (
    ACTIVE_USERS_DAYS, TEST_MODE_USER_LIMIT, DEFAULT_READING_LEVEL,
    MIN_READING_LEVEL, MAX_READING_LEVEL, DEFAULT_KEYWORD_WEIGHT
)

logger = logging.getLogger(__name__)


def get_active_users_with_keywords(redis_client, bq_client, test_mode: bool = False) -> List[Dict]:
    """Get active users with their stored keywords from Redis + BigQuery"""
    try:
        # Get active users from Redis (fast, real-time activity)
        active_user_ids = redis_client.get_active_users(days=ACTIVE_USERS_DAYS)
        
        if not active_user_ids:
            logger.warning("No active users found in Redis, falling back to BigQuery")
            return get_users_with_keywords_from_bigquery_fallback(bq_client, test_mode)
        
        # Limit in test mode
        if test_mode:
            active_user_ids = active_user_ids[:TEST_MODE_USER_LIMIT]
        
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


def get_users_with_keywords_from_bigquery_fallback(bq_client, test_mode: bool = False) -> List[Dict]:
    """Fallback: Get active users with their stored keywords from BigQuery only"""
    try:
        # Only limit in test mode for development
        limit_clause = f"LIMIT {TEST_MODE_USER_LIMIT}" if test_mode else ""
        
        query = f"""
        SELECT 
            user_id,
            handle,
            keywords
        FROM `{bq_client.project_id}.data.users`
        WHERE last_request_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {ACTIVE_USERS_DAYS} DAY)
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


def get_user_embeddings(user_id: str, bq_client) -> Optional[np.ndarray]:
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


def get_user_reading_level(user_id: str, bq_client) -> int:
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
            return DEFAULT_READING_LEVEL
        
        reading_level = result.iloc[0]['reading_level']
        
        # Ensure reasonable bounds
        reading_level = max(MIN_READING_LEVEL, min(MAX_READING_LEVEL, int(reading_level)))
        
        logger.debug(f"Retrieved reading level {reading_level} for user {user_id}")
        return reading_level
        
    except Exception as e:
        logger.error(f"Failed to retrieve reading level for user {user_id}: {e}")
        return DEFAULT_READING_LEVEL


def detect_keywords_format(user_keywords) -> str:
    """
    Detect the format of user keywords data
    
    Returns:
        'enhanced', 'basic', or 'invalid'
    """
    try:
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


def get_enhanced_user_terms(enhanced_keywords) -> Tuple[List[str], Dict]:
    """
    Convert enhanced keywords to weighted terms + sentiment context
    
    Returns:
        (weighted_terms_list, sentiment_context_dict)
    """
    try:
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
                terms.extend([kw] * DEFAULT_KEYWORD_WEIGHT)
            else:
                logger.warning(f"Unexpected keyword type: {type(kw)}")
        
        logger.info(f"Converted {len(set(terms))} unique keywords to {len(terms)} weighted terms")
        return terms
        
    except Exception as e:
        logger.error(f"Failed to process user keywords: {e}")
        return []


def process_user_keywords(user_keywords) -> Tuple[List[str], Optional[Dict], str]:
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