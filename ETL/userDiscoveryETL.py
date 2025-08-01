import os
import sys
import argparse
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.bluesky.userData import Client as BlueskyUserDataClient
from client.bigQuery import Client as BigQueryClient
from featureEngineering.userKeywords import extract_user_keywords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def decode_jwt_user_did(jwt_token: str) -> Optional[str]:
    """
    Extract user DID from JWT token (simplified version)
    In production, use proper JWT library with signature verification
    """
    try:
        import base64
        
        # Split JWT token
        parts = jwt_token.split('.')
        if len(parts) != 3:
            return None
            
        # Decode payload (middle part)
        payload = parts[1]
        # Add padding if needed
        payload += '=' * (4 - len(payload) % 4)
        
        decoded = base64.b64decode(payload)
        payload_data = json.loads(decoded)
        
        # Extract user DID from 'iss' field
        user_did = payload_data.get('iss')
        return user_did
        
    except Exception as e:
        logger.error(f"Failed to decode JWT: {e}")
        return None

def get_users_needing_profile_data(bq_client: BigQueryClient, since_timestamp: datetime) -> List[Dict]:
    """
    Get users from BigQuery who need their profile data populated
    
    Args:
        bq_client: BigQuery client
        since_timestamp: Only process users who made requests after this time
        
    Returns:
        List of user data that needs profile information
    """
    try:
        query = f"""
        SELECT user_id, last_request_at, request_count
        FROM `{bq_client.project_id}.data.users`
        WHERE (handle = '' OR handle IS NULL)
        AND last_request_at >= '{since_timestamp.isoformat()}'
        AND user_id LIKE 'did:plc:%'
        ORDER BY last_request_at DESC
        """
        
        result = bq_client.query(query)
        
        if result.empty:
            logger.info("No users needing profile data found")
            return []
        
        users_needing_data = []
        for _, row in result.iterrows():
            users_needing_data.append({
                'user_did': row['user_id'],
                'request_count': row['request_count']
            })
        
        logger.info(f"Found {len(users_needing_data)} users needing profile data")
        return users_needing_data
        
    except Exception as e:
        logger.error(f"Failed to get users needing profile data: {e}")
        return []

def get_existing_users_from_bigquery(bq_client: BigQueryClient) -> set:
    """Get set of existing user DIDs from BigQuery"""
    try:
        query = f"""
        SELECT user_id
        FROM `{bq_client.project_id}.data.users`
        """
        
        result = bq_client.query(query)
        existing_dids = set(result['user_id'].tolist()) if not result.empty else set()
        
        logger.info(f"Found {len(existing_dids)} existing users in BigQuery")
        return existing_dids
        
    except Exception as e:
        logger.warning(f"Failed to get existing users from BigQuery: {e}")
        return set()

def get_user_profile_from_did(client: BlueskyUserDataClient, user_did: str) -> Optional[Dict]:
    """Get user profile information from their DID"""
    try:
        # Use DID to get profile - this requires resolving DID to handle first
        # For now, we'll use the DID directly (Bluesky client should handle this)
        profile = client.client.app.bsky.actor.get_profile({'actor': user_did})
        
        return {
            'did': profile.did,
            'handle': profile.handle or ''
        }
        
    except Exception as e:
        logger.error(f"Failed to get profile for DID {user_did}: {e}")
        return None

def collect_and_process_user_data(client: BlueskyUserDataClient, user_did: str, user_profile: Dict) -> Optional[Dict]:
    """Collect user posts and extract keywords"""
    try:
        logger.info(f"Processing user data for {user_profile.get('handle', user_did)}")
        
        # Collect user engagement data
        user_data = client.get_comprehensive_user_data(
            actor=user_did,
            include_likes=False,  # Skip likes for performance
            posts_limit=100,      # Get recent posts for keyword extraction
            reposts_limit=50,
            replies_limit=50
        )
        
        # Check if we have enough content for keyword extraction
        total_items = len(user_data['posts']) + len(user_data['reposts']) + len(user_data['replies'])
        if total_items < 5:
            logger.warning(f"User {user_profile.get('handle')} has insufficient content ({total_items} items)")
            return None
        
        # Extract keywords from user content
        keywords = extract_user_keywords(user_data, top_k=20, min_freq=1)
        
        if not keywords:
            logger.warning(f"No keywords extracted for user {user_profile.get('handle')}")
            return None
        
        # Prepare user data for BigQuery
        processed_user = {
            'user_id': user_did,
            'handle': user_profile.get('handle', ''),
            'keywords': keywords,  # Store as JSON array
            'updated_at': datetime.utcnow()
        }
        
        logger.info(f"Extracted {len(keywords)} keywords for user {user_profile.get('handle')}")
        return processed_user
        
    except Exception as e:
        logger.error(f"Failed to process user data for {user_did}: {e}")
        return None

def update_user_profile_in_bigquery(bq_client: BigQueryClient, user_data: Dict, batch_id: str) -> bool:
    """Update existing user record with profile data in BigQuery"""
    try:
        # Update the existing user record with profile information
        keywords_str = "', '".join(user_data['keywords']) if user_data['keywords'] else ""
        keywords_array = f"['{keywords_str}']" if keywords_str else "[]"
        
        update_query = f"""
        UPDATE `{bq_client.project_id}.data.users`
        SET handle = '{user_data['handle']}',
            keywords = {keywords_array},
            updated_at = '{datetime.utcnow().isoformat()}'
        WHERE user_id = '{user_data['user_id']}'
        """
        
        bq_client.query(update_query)
        
        logger.info(f"Updated user profile for {user_data['handle']} in BigQuery")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update user profile in BigQuery: {e}")
        return False

def main():
    """Main user discovery ETL process"""
    parser = argparse.ArgumentParser(description='User Discovery ETL')
    parser.add_argument('--log-file', default='/var/log/feed-server.log', help='Path to feed server log file')
    parser.add_argument('--hours-back', type=int, default=1, help='Hours to look back in logs')
    parser.add_argument('--dry-run', action='store_true', help='Run without storing to BigQuery')
    
    args = parser.parse_args()
    
    batch_id = f"user_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    since_timestamp = datetime.now() - timedelta(hours=args.hours_back)
    
    logger.info(f"Starting User Discovery ETL (batch_id={batch_id})")
    logger.info(f"Processing logs since: {since_timestamp}")
    
    try:
        # Initialize clients
        credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
        bq_client = BigQueryClient(credentials_json, os.environ['BIGQUERY_PROJECT_ID'])
        
        bluesky_client = BlueskyUserDataClient()
        bluesky_client.login()
        
        # Get users needing profile data from BigQuery
        users_needing_data = get_users_needing_profile_data(bq_client, since_timestamp)
        
        if not users_needing_data:
            logger.info("No users needing profile data found")
            return
        
        logger.info(f"Found {len(users_needing_data)} users needing profile data")
        
        success_count = 0
        error_count = 0
        
        for user_request in users_needing_data:
            user_did = user_request['user_did']
            
            try:
                logger.info(f"Processing new user: {user_did}")
                
                # Get user profile
                user_profile = get_user_profile_from_did(bluesky_client, user_did)
                if not user_profile:
                    logger.warning(f"Could not get profile for user {user_did}")
                    error_count += 1
                    continue
                
                # Collect and process user data
                processed_user = collect_and_process_user_data(bluesky_client, user_did, user_profile)
                if not processed_user:
                    logger.warning(f"Could not process user data for {user_profile.get('handle', user_did)}")
                    error_count += 1
                    continue
                
                # Update user profile in BigQuery
                if not args.dry_run:
                    if update_user_profile_in_bigquery(bq_client, processed_user, batch_id):
                        success_count += 1
                        logger.info(f"Successfully updated user profile for {processed_user['handle']}")
                    else:
                        error_count += 1
                        logger.error(f"Failed to update user profile for {processed_user['handle']}")
                else:
                    logger.info(f"DRY RUN: Would update user {processed_user['handle']} with {len(processed_user['keywords'])} keywords")
                    success_count += 1
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing user {user_did}: {e}")
                continue
        
        # Final summary
        logger.info(f"User Discovery ETL Complete!")
        logger.info(f"Success: {success_count}, Errors: {error_count}")
        logger.info(f"New active users ready for feed ranking")
        
    except Exception as e:
        logger.error(f"User Discovery ETL failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()