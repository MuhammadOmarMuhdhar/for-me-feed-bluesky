import os
import sys
from datetime import datetime
from atproto import Client, models
from dotenv import load_dotenv

load_dotenv()

def publish_feed():
    """Publish feed generator to AT Protocol"""
    
    # Configuration
    service_did = os.getenv('FEEDGEN_SERVICE_DID')
    hostname = os.getenv('FEEDGEN_HOSTNAME')
    
    if not service_did or not hostname:
        print("Error: FEEDGEN_SERVICE_DID and FEEDGEN_HOSTNAME must be set")
        return
    
    # Prompt for feed details
    print("Publishing your Bluesky feed generator...")
    print(f"Service DID: {service_did}")
    print(f"Hostname: {hostname}")
    
    handle = input("Enter your Bluesky handle (e.g., alice.bsky.social): ")
    password = input("Enter your Bluesky password: ")
    record_name = input("Enter feed record name (e.g., 'my-feed'): ")
    display_name = input("Enter feed display name (e.g., 'My Personalized Feed'): ")
    description = input("Enter feed description (optional): ")
    
    try:
        # Login to Bluesky
        print("Logging in to Bluesky...")
        client = Client()
        client.login(handle, password)
        
        # Create feed generator record
        print("Creating feed generator record...")
        
        feed_record = models.AppBskyFeedGenerator.Record(
            did=service_did,
            display_name=display_name,
            description=description if description else None,
            created_at=datetime.now().isoformat() + 'Z'
        )
        
        # Publish the record
        uri = client.com.atproto.repo.put_record(
            models.ComAtprotoRepoPutRecord.Data(
                repo=client.me.did,
                collection=models.ids.AppBskyFeedGenerator,
                rkey=record_name,
                record=feed_record
            )
        )
        
        feed_uri = f"at://{client.me.did}/{models.ids.AppBskyFeedGenerator}/{record_name}"
        
        print(f"✅ Feed published successfully!")
        print(f"Feed URI: {feed_uri}")
        print(f"Record URI: {uri.uri}")
        print(f"CID: {uri.cid}")
        
        print("\nNow update your Cloud Run environment variables:")
        print(f"FEED_DID={service_did}")
        print(f"FEED_URI={feed_uri}")
        print(f"FEED_CID={uri.cid}")
        
        print(f"\nYour feed should now be discoverable in the Bluesky app!")
        
    except Exception as e:
        print(f"❌ Error publishing feed: {e}")
        return

if __name__ == "__main__":
    publish_feed()