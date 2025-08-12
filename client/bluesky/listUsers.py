import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from atproto import Client as AtprotoClient
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
    
    def get_list_members(
        self,
        list_uri: str,
        limit: int = 100,
        cursor: str = None
    ) -> List[Dict]:
        """
        Get members from a Bluesky list
        
        Args:
            list_uri: URI of the list (e.g., "at://did:plc:xyz/app.bsky.graph.list/abc123")
            limit: Number of members to fetch per request (max 100)
            cursor: Pagination cursor
            
        Returns:
            List of list members with metadata
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        try:
            params = {
                'list': list_uri,
                'limit': min(limit, 100)
            }
            if cursor:
                params['cursor'] = cursor
                
            response = self.client.app.bsky.graph.get_list(params)
            
            members = []
            for item in response.items:
                member_data = {
                    'did': item.subject.did,
                    'handle': item.subject.handle,
                    'display_name': getattr(item.subject, 'display_name', ''),
                    'description': getattr(item.subject, 'description', ''),
                    'follower_count': getattr(item.subject, 'followers_count', 0),
                    'follows_count': getattr(item.subject, 'follows_count', 0),
                    'posts_count': getattr(item.subject, 'posts_count', 0),
                    'indexed_at': getattr(item.subject, 'indexed_at', None),
                    'created_at': getattr(item.subject, 'created_at', None),
                    'avatar': getattr(item.subject, 'avatar', None),
                    'list_uri': list_uri,
                    'added_at': getattr(response, 'indexed_at', None)
                }
                members.append(member_data)
            
            return members, getattr(response, 'cursor', None)
            
        except Exception as e:
            print(f"Error fetching list members: {e}")
            return [], None
    
    def get_all_list_members(
        self,
        list_uri: str,
        max_members: int = 10000,
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Get all members from a Bluesky list with pagination
        
        Args:
            list_uri: URI of the list
            max_members: Maximum number of members to fetch
            batch_size: Number of members to fetch per API request
            
        Returns:
            Complete list of all members
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        all_members = []
        cursor = None
        
        print(f"Fetching all members from list: {list_uri}")
        
        while len(all_members) < max_members:
            batch_limit = min(batch_size, max_members - len(all_members))
            members_batch, cursor = self.get_list_members(
                list_uri=list_uri,
                limit=batch_limit,
                cursor=cursor
            )
            
            if not members_batch:
                break
            
            all_members.extend(members_batch)
            print(f"   Fetched {len(members_batch)} members (total: {len(all_members)})")
            
            if not cursor:
                break
        
        print(f"Retrieved {len(all_members)} total members from list")
        return all_members
    
    def get_list_info(self, list_uri: str) -> Dict:
        """
        Get basic information about a list
        
        Args:
            list_uri: URI of the list
            
        Returns:
            Dictionary containing list metadata
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        try:
            params = {'list': list_uri, 'limit': 1}
            response = self.client.app.bsky.graph.get_list(params)
            
            list_info = {
                'uri': list_uri,
                'name': getattr(response.list, 'name', ''),
                'description': getattr(response.list, 'description', ''),
                'creator': {
                    'did': getattr(response.list.creator, 'did', ''),
                    'handle': getattr(response.list.creator, 'handle', ''),
                    'display_name': getattr(response.list.creator, 'display_name', '')
                },
                'created_at': getattr(response.list, 'indexed_at', None),
                'purpose': getattr(response.list, 'purpose', ''),
                'total_members': len(response.items) if hasattr(response, 'items') else 0
            }
            
            return list_info
            
        except Exception as e:
            print(f"Error fetching list info: {e}")
            return {}
    
    def get_multiple_lists_members(
        self,
        list_uris: List[str],
        max_members_per_list: int = 10000
    ) -> Dict[str, List[Dict]]:
        """
        Get members from multiple lists efficiently
        
        Args:
            list_uris: List of list URIs to fetch members from
            max_members_per_list: Maximum members to fetch per list
            
        Returns:
            Dictionary mapping list URIs to their member lists
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        results = {}
        
        print(f"Fetching members from {len(list_uris)} lists...")
        
        for i, list_uri in enumerate(list_uris):
            print(f"\n({i+1}/{len(list_uris)}) Processing list: {list_uri}")
            
            try:
                # Get list info first
                list_info = self.get_list_info(list_uri)
                print(f"   List: '{list_info.get('name', 'Unknown')}' by @{list_info.get('creator', {}).get('handle', 'unknown')}")
                
                # Get all members
                members = self.get_all_list_members(
                    list_uri=list_uri,
                    max_members=max_members_per_list
                )
                
                results[list_uri] = {
                    'list_info': list_info,
                    'members': members,
                    'member_count': len(members)
                }
                
            except Exception as e:
                print(f"   Error processing list {list_uri}: {e}")
                results[list_uri] = {
                    'list_info': {},
                    'members': [],
                    'member_count': 0,
                    'error': str(e)
                }
        
        total_members = sum(result['member_count'] for result in results.values())
        print(f"\nCompleted: {total_members} total members across {len(list_uris)} lists")
        
        return results
    
    def extract_unique_users(
        self,
        lists_data: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """
        Extract unique users from multiple lists data
        
        Args:
            lists_data: Output from get_multiple_lists_members()
            
        Returns:
            Deduplicated list of unique users across all lists
        """
        unique_users = {}
        
        for list_uri, list_data in lists_data.items():
            members = list_data.get('members', [])
            list_name = list_data.get('list_info', {}).get('name', 'Unknown')
            
            for member in members:
                did = member['did']
                
                if did not in unique_users:
                    # First time seeing this user
                    unique_users[did] = member.copy()
                    unique_users[did]['found_in_lists'] = [{'uri': list_uri, 'name': list_name}]
                else:
                    # User already exists, add this list to their found_in_lists
                    unique_users[did]['found_in_lists'].append({'uri': list_uri, 'name': list_name})
        
        # Convert to list and sort by follower count
        users_list = list(unique_users.values())
        users_list.sort(key=lambda x: x.get('follower_count', 0), reverse=True)
        
        print(f"Extracted {len(users_list)} unique users from {len(lists_data)} lists")
        
        return users_list
    
    def cache_list_members(
        self,
        list_uri: str,
        cache_duration_hours: int = 24,
        cache_dir: str = "cache"
    ) -> List[Dict]:
        """
        Get list members with caching to reduce API calls
        
        Args:
            list_uri: URI of the list
            cache_duration_hours: How long to cache the member list
            cache_dir: Directory to store cache files
            
        Returns:
            List of members (from cache or fresh API call)
        """
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache filename based on list URI
        safe_uri = list_uri.replace(':', '_').replace('/', '_').replace('.', '_')
        cache_file = os.path.join(cache_dir, f"list_members_{safe_uri}.json")
        
        # Check if cache exists and is fresh
        if os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(hours=cache_duration_hours):
                print(f"Using cached list members (age: {cache_age})")
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Cache read error: {e}, fetching fresh data")
        
        # Fetch fresh data
        print("Fetching fresh list members...")
        members = self.get_all_list_members(list_uri)
        
        # Save to cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(members, f, indent=2, ensure_ascii=False, default=str)
            print(f"Cached {len(members)} members to {cache_file}")
        except Exception as e:
            print(f"Cache write error: {e}")
        
        return members
    
    def save_members_data(
        self,
        members_data: Dict,
        filename: str = None
    ):
        """
        Save list members data to JSON file
        
        Args:
            members_data: Data from get_multiple_lists_members() or similar
            filename: Optional filename, will auto-generate if not provided
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"list_members_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(members_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"List members data saved to {filename}")
    
    def extract_users_for_moderation(
        self,
        list_uris: List[str],
        max_members_per_list: int = 50000
    ) -> List[str]:
        """
        Extract user handles from multiple lists for moderation (minimal data)
        
        Args:
            list_uris: List of AT Protocol URIs for the lists
            max_members_per_list: Maximum members to extract per list
            
        Returns:
            Deduplicated list of user handles
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        all_handles = set()
        
        print(f"Extracting users for moderation from {len(list_uris)} lists...")
        
        for i, list_uri in enumerate(list_uris):
            print(f"\n({i+1}/{len(list_uris)}) Processing list: {list_uri}")
            
            try:
                # Get all members but only store handles
                members = self.get_all_list_members(
                    list_uri=list_uri,
                    max_members=max_members_per_list
                )
                
                # Extract just the handles
                for member in members:
                    handle = member.get('handle', '')
                    if handle:
                        all_handles.add(handle)
                
                print(f"   Added {len(members)} users (total unique: {len(all_handles)})")
                
            except Exception as e:
                print(f"   Error processing list {list_uri}: {e}")
                continue
        
        handles_list = sorted(list(all_handles))
        print(f"\nExtracted {len(handles_list)} unique users for moderation")
        
        return handles_list
    
    def save_moderation_users(
        self,
        handles: List[str],
        filename: str
    ):
        """
        Save user handles to text file for moderation
        
        Args:
            handles: List of user handles
            filename: Path to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for handle in handles:
                f.write(f"{handle}\n")
        
        print(f"Saved {len(handles)} user handles to {filename}")
        print(f"File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
    
    def get_user_follows(
        self,
        actor: str,
        limit: int = 100,
        cursor: str = None
    ) -> tuple[List[Dict], str]:
        """
        Get users that the specified actor follows using getFollows endpoint
        
        Args:
            actor: User handle or DID
            limit: Number of follows to fetch (max 100)
            cursor: Pagination cursor
            
        Returns:
            Tuple of (follows list, next cursor)
        """
        if not self.authenticated:
            raise Exception("Must login first. Call client.login(identifier, password)")
        
        try:
            params = {
                'actor': actor,
                'limit': min(limit, 100)
            }
            if cursor:
                params['cursor'] = cursor
                
            response = self.client.app.bsky.graph.get_follows(params)
            
            follows = []
            for follow in response.follows:
                follow_data = {
                    'did': follow.did,
                    'handle': follow.handle,
                    'displayName': getattr(follow, 'displayName', ''),
                    'description': getattr(follow, 'description', ''),
                    'avatar': getattr(follow, 'avatar', ''),
                    'indexedAt': getattr(follow, 'indexedAt', None),
                    'createdAt': getattr(follow, 'createdAt', None),
                    
                }
                follows.append(follow_data)
            
            next_cursor = getattr(response, 'cursor', None)
            return follows, next_cursor
            
        except Exception as e:
            print(f"Error fetching follows: {e}")
            return [], None

    def print_members_summary(
        self,
        members: List[Dict],
        limit: int = 10
    ):
        """Print a summary of list members"""
        print(f"\n=== Top {min(limit, len(members))} List Members ===\n")
        
        for i, member in enumerate(members[:limit]):
            print(f"{i+1}. @{member['handle']}")
            if member.get('display_name'):
                print(f"   Name: {member['display_name']}")
            print(f"   Followers: {member.get('follower_count', 0):,}")
            print(f"   Following: {member.get('follows_count', 0):,}")
            print(f"   Posts: {member.get('posts_count', 0):,}")
            if member.get('description'):
                print(f"   Bio: {member['description'][:100]}...")
            if member.get('found_in_lists'):
                list_names = [lst['name'] for lst in member['found_in_lists']]
                print(f"   Found in lists: {', '.join(list_names)}")
            print()