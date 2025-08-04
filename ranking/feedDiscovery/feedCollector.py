import json
import os
import sys
import logging
import pandas as pd
from typing import List, Dict, Optional
from urllib.parse import urlparse

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from client.bluesky.newPosts import Client as BlueskyClient
from client.bigQuery import Client as BigQueryClient
from featureEngineering.encoder import embed_posts


class FeedCollector:
    def __init__(self):
        """Initialize feed collector with Bluesky client"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bluesky_client = None
        self.bq_client = None
        self.config_path = os.path.join(
            os.path.dirname(__file__), 
            'config', 
            'feed_urls.json'
        )
    
    def connect_client(self) -> bool:
        """
        Initialize and authenticate Bluesky client
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.bluesky_client = BlueskyClient()
            self.bluesky_client.login()
            self.logger.info("Successfully connected to Bluesky client")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Bluesky client: {e}")
            return False
    
    def connect_bigquery(self) -> bool:
        """
        Initialize BigQuery client
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
            project_id = os.environ['BIGQUERY_PROJECT_ID']
            self.bq_client = BigQueryClient(credentials_json, project_id)
            self.logger.info("Successfully connected to BigQuery client")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to BigQuery client: {e}")
            return False
    
    
    def load_feed_config(self) -> List[str]:
        """
        Load feed URLs from config file
        
        Returns:
            List of feed URLs
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            feed_urls = config.get('feeds', [])
            self.logger.info(f"Loaded {len(feed_urls)} feed URLs from config")
            return feed_urls
            
        except Exception as e:
            self.logger.error(f"Failed to load feed config from {self.config_path}: {e}")
            return []
    
    def collect_posts_from_feeds(
        self, 
        limit: int = 100,
        # max_requests_per_feed: int = 2
    ) -> List[str]:
        """
        Collect posts from all feeds configured in feed_urls.json
        
        Args:
            limit: Number of posts to collect from each feed
            
        Returns:
            List of post texts from all feeds
        """
        if not self.bluesky_client:
            self.logger.error("Bluesky client not connected. Call connect_client() first.")
            return []
        
        # Load feed URLs from config
        feed_urls = self.load_feed_config()
        if not feed_urls:
            self.logger.warning("No feed URLs found in config")
            return []
        
        try: 
            self.logger.info(f"Collecting posts from {len(feed_urls)} feeds...")
            all_posts = []        
            
            for url in feed_urls:
                self.logger.info(f"Collecting from feed: {url}")
                feed_data = self.bluesky_client.extract_posts_from_feed(
                    feed_url_or_uri=url, 
                    limit=limit
                )
                
                # Extract text from each post
                for post in feed_data:
                    if 'text' in post:
                        all_posts.append(post['text'])
                
                self.logger.info(f"Collected {len(feed_data)} posts from this feed")
            
            self.logger.info(f"Total posts collected: {len(all_posts)}")
            return all_posts
            
        except Exception as e:
            self.logger.error(f"Error collecting posts from feeds: {e}")
            return []

    def collect_and_embed_posts(
        self,
        limit: int = 100,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Collect posts from feeds and generate embeddings, keeping only feed_url and embedding
        
        Args:
            limit: Number of posts to collect from each feed
            model_name: SentenceTransformer model to use
            batch_size: Number of texts to embed per batch
            
        Returns:
            List of dicts with 'feed_url' and 'embedding'
        """
        if not self.bluesky_client:
            self.logger.error("Bluesky client not connected. Call connect_client() first.")
            return []
        
        # Load feed URLs from config
        feed_urls = self.load_feed_config()
        if not feed_urls:
            self.logger.warning("No feed URLs found in config")
            return []
        
        try:
            self.logger.info(f"Collecting and embedding posts from {len(feed_urls)} feeds...")
            result = []
            
            for url in feed_urls:
                self.logger.info(f"Processing feed: {url}")
                feed_data = self.bluesky_client.extract_posts_from_feed(
                    feed_url_or_uri=url,
                    limit=limit
                )
                
                # Prepare posts for embedding
                posts_for_embedding = []
                for post in feed_data:
                    if 'text' in post and post['text'].strip():
                        posts_for_embedding.append({'text': post['text']})
                
                if posts_for_embedding:
                    # Create user_data format expected by embed_posts
                    user_data = {'posts': posts_for_embedding}
                    
                    # Generate embeddings
                    embeddings = embed_posts(
                        user_data=user_data,
                        model_name=model_name,
                        batch_size=batch_size
                    )
                    
                    # Add feed_url to each embedding and keep only feed_url and embedding
                    for embedding_data in embeddings:
                        result.append({
                            'feed_url': url,
                            'embedding': embedding_data['embedding']
                        })
                    
                    self.logger.info(f"Generated {len(embeddings)} embeddings for feed: {url}")
                else:
                    self.logger.warning(f"No valid posts found for feed: {url}")
            
            self.logger.info(f"Total embeddings generated: {len(result)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error collecting and embedding posts: {e}")
            return []

    def replace_feeds_with_embeddings(
        self,
        dataset_id: str = 'data',
        table_id: str = 'feeds',
        limit_per_feed: int = 100,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        batch_size: int = 100
    ) -> bool:
        """
        Replace all data in feeds table with URIs and their embeddings
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID  
            limit_per_feed: Number of posts to collect from each feed
            model_name: SentenceTransformer model to use
            batch_size: Number of texts to embed per batch
            
        Returns:
            bool: Success status
        """
        if not self.bluesky_client:
            self.logger.error("Bluesky client not connected. Call connect_client() first.")
            return False
            
        if not self.bq_client:
            self.logger.error("BigQuery client not connected. Call connect_bigquery() first.")
            return False
        
        try:
            self.logger.info("Starting feeds table replacement with embeddings...")
            
            # Generate embeddings for all feeds
            embedding_data = self.collect_and_embed_posts(
                limit=limit_per_feed,
                model_name=model_name,
                batch_size=batch_size
            )
            
            if not embedding_data:
                self.logger.error("No embedding data generated")
                return False
            
            # Prepare DataFrame for BigQuery
            # Group embeddings by feed_url to create one row per feed
            feed_embeddings = {}
            for item in embedding_data:
                feed_url = item['feed_url']
                if feed_url not in feed_embeddings:
                    feed_embeddings[feed_url] = []
                feed_embeddings[feed_url].append(item['embedding'])
            
            # Create DataFrame with schema: uri (STRING), embeddings (JSON)
            df_data = []
            for feed_url, embeddings_list in feed_embeddings.items():
                # Average all post embeddings into one feed embedding
                import numpy as np
                avg_embedding = np.mean(embeddings_list, axis=0).tolist()
                # Convert to JSON format for BigQuery
                embeddings_json = json.dumps(avg_embedding)
                df_data.append({
                    'uri': feed_url,
                    'embeddings': embeddings_json
                })
            
            df = pd.DataFrame(df_data)
            self.logger.info(f"Created DataFrame with {len(df)} feed records")
            
            # Replace table data using existing BigQuery client
            self.bq_client.replace(df, dataset_id, table_id)
            
            # If we get here, the replace was successful (no exception raised)
            self.logger.info(f"Successfully replaced {dataset_id}.{table_id} table with {len(df)} feed records")
            return True
                
        except Exception as e:
            self.logger.error(f"Error replacing feeds table: {e}")
            return False


def main():
    """
    Production entry point for feed collection and BigQuery replacement
    """
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('feed_collector.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting feed collection and BigQuery replacement process...")
    
    try:
        # Initialize collector
        collector = FeedCollector()
        
        # Connect to clients
        logger.info("Connecting to Bluesky client...")
        if not collector.connect_client():
            logger.error("Failed to connect to Bluesky client")
            sys.exit(1)
        
        logger.info("Connecting to BigQuery client...")
        if not collector.connect_bigquery():
            logger.error("Failed to connect to BigQuery client")
            sys.exit(1)
        
        # Replace feeds table with embeddings
        logger.info("Starting feeds table replacement with embeddings...")
        success = collector.replace_feeds_with_embeddings(
            dataset_id='data',
            table_id='feeds',
            limit_per_feed=100
        )
        
        if success:
            logger.info("Successfully completed feeds table replacement")
            sys.exit(0)
        else:
            logger.error("Failed to replace feeds table")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}")
        sys.exit(1)


# Example usage
if __name__ == "__main__":
    main()
