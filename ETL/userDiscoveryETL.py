#!/usr/bin/env python3
"""
User Discovery ETL for Feed Request Tracking
Detects new users from feed server logs, extracts their keywords, and stores in BigQuery
"""

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

def parse_feed_server_logs(log_file_path: str, since_timestamp: datetime) -> List[Dict]:
    """
    Parse feed server logs to extract new user requests
    
    Args:
        log_file_path: Path to feed server log file
        since_timestamp: Only process logs after this time
        
    Returns:
        List of new user request data
    """
    new_users = []
    seen_dids = set()
    
    try:
        # Mock log parsing - replace with actual log format
        # Expected log format: "TIMESTAMP - GET /xrpc/app.bsky.feed.getFeedSkeleton - Authorization: Bearer JWT_TOKEN"
        
        if not os.path.exists(log_file_path):
            logger.warning(f"Feed server log file not found: {log_file_path}")
            # Return mock data for testing
            return [
                {
                    'user_did': 'did:plc:test_user_1',
                    'first_seen': datetime.now(),
                    'request_count': 1
                }
            ]
        
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    # Parse log line (adjust regex for your actual log format)
                    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*getFeedSkeleton.*Bearer\s+([^\\s]+)', line)
                    if not match:
                        continue
                    
                    timestamp_str, jwt_token = match.groups()
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    
                    # Only process recent logs
                    if timestamp < since_timestamp:
                        continue
                    
                    # Extract user DID from JWT
                    user_did = decode_jwt_user_did(jwt_token)
                    if not user_did or user_did in seen_dids:
                        continue
                    
                    seen_dids.add(user_did)
                    new_users.append({
                        'user_did': user_did,
                        'first_seen': timestamp,
                        'request_count': 1
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse log line: {e}")
                    continue
        
        logger.info(f"Parsed {len(new_users)} new users from feed logs")
        return new_users
        
    except Exception as e:
        logger.error(f"Failed to parse feed server logs: {e}")
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
            'handle': profile.handle,
            'display_name': profile.display_name or '',
            'description': profile.description or '',
            'followers_count': profile.followers_count or 0,
            'following_count': profile.following_count or 0,
            'posts_count': profile.posts_count or 0
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
            'display_name': user_profile.get('display_name', ''),
            'description': user_profile.get('description', ''),
            'followers_count': user_profile.get('followers_count', 0),
            'following_count': user_profile.get('following_count', 0),
            'posts_count': user_profile.get('posts_count', 0),
            'keywords': keywords,  # Store as JSON array
            'is_active': True,
            'discovered_via': 'feed_request',
            'first_discovered_at': datetime.utcnow(),
            'last_seen_at': datetime.utcnow(),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        logger.info(f"Extracted {len(keywords)} keywords for user {user_profile.get('handle')}")
        return processed_user
        
    except Exception as e:
        logger.error(f"Failed to process user data for {user_did}: {e}")
        return None

def store_user_in_bigquery(bq_client: BigQueryClient, user_data: Dict, batch_id: str) -> bool:
    """Store processed user data in BigQuery"""
    try:
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame([user_data])
        
        # Store in BigQuery users table
        bq_client.append(df, 'data', 'users', create_if_not_exists=True)
        
        logger.info(f"Stored user {user_data['handle']} in BigQuery")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store user in BigQuery: {e}")
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
        if not args.dry_run:
            credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
            bq_client = BigQueryClient(credentials_json, os.environ['BIGQUERY_PROJECT_ID'])
            existing_users = get_existing_users_from_bigquery(bq_client)
        else:
            existing_users = set()
        
        bluesky_client = BlueskyUserDataClient()
        bluesky_client.login()
        
        # Parse feed server logs for new users
        new_user_requests = parse_feed_server_logs(args.log_file, since_timestamp)
        
        if not new_user_requests:
            logger.info("No new users found in feed logs")
            return
        
        # Filter out existing users
        truly_new_users = [
            user for user in new_user_requests 
            if user['user_did'] not in existing_users
        ]
        
        logger.info(f"Found {len(truly_new_users)} truly new users (out of {len(new_user_requests)} log entries)")
        
        success_count = 0
        error_count = 0
        
        for user_request in truly_new_users:
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
                
                # Store in BigQuery
                if not args.dry_run:
                    if store_user_in_bigquery(bq_client, processed_user, batch_id):
                        success_count += 1
                        logger.info(f"Successfully onboarded user {processed_user['handle']}")
                    else:
                        error_count += 1
                        logger.error(f"Failed to store user {processed_user['handle']}")
                else:
                    logger.info(f"DRY RUN: Would store user {processed_user['handle']} with {len(processed_user['keywords'])} keywords")
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