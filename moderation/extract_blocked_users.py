#!/usr/bin/env python3
import sys
import os
import json
from datetime import datetime
from typing import List, Set

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.bluesky.listUsers import Client


def convert_bsky_url_to_uri(bsky_url: str) -> str:
    """Convert a bsky.app list URL to AT Protocol URI format"""
    if "/profile/" in bsky_url and "/lists/" in bsky_url:
        parts = bsky_url.split("/profile/")[1]
        did, list_id = parts.split("/lists/")
        return f"at://{did}/app.bsky.graph.list/{list_id}"
    else:
        raise ValueError(f"Invalid Bluesky list URL format: {bsky_url}")


def extract_users_from_list(client: Client, list_uri: str, max_members: int = 50000) -> Set[str]:
    """Extract user handles from a single list"""
    print(f"Processing list: {list_uri}")
    
    try:
        # Get all members
        members = client.get_all_list_members(
            list_uri=list_uri,
            max_members=max_members
        )
        
        # Extract handles
        handles = set()
        for member in members:
            handle = member.get('handle', '')
            if handle:
                handles.add(handle)
        
        print(f"   Extracted {len(handles)} users from this list")
        return handles
        
    except Exception as e:
        print(f"   Error processing list {list_uri}: {e}")
        return set()


def main():
    """Extract users from all moderation lists and save to moderation.txt"""
    
    print("=== Bluesky Moderation User Extraction ===\n")
    start_time = datetime.now()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load list URLs
    lists_file = os.path.join(script_dir, 'lists.json')
    with open(lists_file, 'r') as f:
        list_urls = json.load(f)
    
    print(f"Loading {len(list_urls)} moderation lists from {lists_file}")
    
    # Convert URLs to AT Protocol URIs
    list_uris = []
    for url in list_urls:
        uri = convert_bsky_url_to_uri(url)
        list_uris.append(uri)
    
    # Initialize client and login
    print(f"\nInitializing Bluesky client...")
    client = Client()
    client.login()
    
    # Extract users from all lists
    print(f"\nExtracting users from {len(list_uris)} lists...")
    all_blocked_users = set()
    
    for i, list_uri in enumerate(list_uris):
        print(f"\n({i+1}/{len(list_uris)}) {list_urls[i]}")
        
        # Extract users from this list
        list_users = extract_users_from_list(client, list_uri)
        
        # Add to global set (automatically deduplicates)
        before_count = len(all_blocked_users)
        all_blocked_users.update(list_users)
        after_count = len(all_blocked_users)
        
        print(f"   Added {after_count - before_count} new unique users")
        print(f"   Total unique users so far: {len(all_blocked_users):,}")
    
    # Sort users for consistent output
    sorted_users = sorted(list(all_blocked_users))
    
    # Save to moderation.txt
    output_file = os.path.join(script_dir, 'moderation.txt')
    print(f"\nSaving {len(sorted_users):,} users to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for handle in sorted_users:
            f.write(f"{handle}\n")
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    
    # Calculate time taken
    end_time = datetime.now()
    extraction_time = end_time - start_time
    
    # Final summary
    print(f"\n=== Extraction Complete ===")
    print(f"Total unique users: {len(sorted_users):,}")
    print(f"File: {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Extraction time: {extraction_time}")
    print(f"Lists processed: {len(list_urls)}")
    
    # Verify file
    print(f"\nVerifying saved file...")
    with open(output_file, 'r') as f:
        loaded_count = sum(1 for line in f)
    
    if loaded_count == len(sorted_users):
        print(f"✅ File verification passed: {loaded_count:,} users")
    else:
        print(f"❌ File verification failed: expected {len(sorted_users):,}, got {loaded_count:,}")
        return False
    
    print(f"\n✅ Moderation list extraction completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n❌ Extraction cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)