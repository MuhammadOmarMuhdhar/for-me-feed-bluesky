import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.bigQuery import Client as BigQueryClient


def get_user_embedding(user_embeddings: List[float]) -> np.ndarray:
    """
    Convert user embeddings to numpy array
    
    Args:
        user_embeddings: User's embedding vector as list
        
    Returns:
        User embedding as numpy array
    """
    if not user_embeddings:
        return np.array([])
    
    return np.array(user_embeddings)


def get_feed_embeddings_from_bigquery(bq_client: BigQueryClient) -> Dict[str, np.ndarray]:
    """
    Fetch feed embeddings from BigQuery feeds table
    
    Args:
        bq_client: BigQuery client instance
        
    Returns:
        Dictionary mapping feed URIs to embedding arrays
    """
    print("Fetching feed embeddings from BigQuery...")
    
    query = f"""
    SELECT uri, embeddings
    FROM `{bq_client.project_id}.data.feeds`
    """
    
    try:
        results = bq_client.execute_query(query)
        feed_embeddings = {}
        
        # BigQuery client returns pandas DataFrame
        if results.empty:
            print("No feed embeddings found in BigQuery")
            return {}
        
        for index, row in results.iterrows():
            uri = row['uri']
            embeddings_json = row['embeddings']
            
            # Parse JSON embeddings
            if isinstance(embeddings_json, str):
                try:
                    embeddings = json.loads(embeddings_json)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON for feed {uri}: {e}")
                    continue
            else:
                embeddings = embeddings_json
            
            if embeddings and isinstance(embeddings, list):
                feed_embeddings[uri] = np.array(embeddings)
                print(f"Loaded embeddings for feed: {uri} (shape: {len(embeddings)})")
            else:
                print(f"Warning: Invalid embeddings format for feed {uri}: {type(embeddings)}")
        
        print(f"Successfully loaded {len(feed_embeddings)} feed embeddings")
        return feed_embeddings
        
    except Exception as e:
        print(f"Error fetching feed embeddings: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}


def compute_feed_similarity_scores(user_embedding: np.ndarray, feed_embeddings: Dict[str, np.ndarray], threshold: float = 0.7) -> List[Dict]:
    """
    Compute cosine similarity between user embedding and feed embeddings
    
    Args:
        user_embedding: User's embedding vector
        feed_embeddings: Dictionary of feed URI -> embedding mappings
        threshold: Minimum similarity threshold for feeds
        
    Returns:
        List of feed similarity results with scores above threshold
    """
    print(f"Computing cosine similarity for {len(feed_embeddings)} feeds...")
    
    if len(user_embedding) == 0:
        print("Warning: Empty user embedding - returning no matches")
        return []
    
    results = []
    similarities = []
    
    for feed_uri, feed_emb in feed_embeddings.items():
        if len(feed_emb) == 0:
            continue
            
        # Compute cosine similarity
        similarity = cosine_similarity(
            user_embedding.reshape(1, -1),
            feed_emb.reshape(1, -1)
        )[0][0]
        
        similarities.append(similarity)
        
        # Only include feeds above threshold
        if similarity >= threshold:
            results.append({
                'feed_uri': feed_uri,
                'similarity_score': float(similarity)
            })
    
    # Sort by similarity score (highest first)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Print stats
    if similarities:
        print(f"Computed {len(similarities)} similarity scores")
        print(f"Max similarity: {max(similarities):.3f}")
        print(f"Min similarity: {min(similarities):.3f}")
        print(f"Feeds above threshold ({threshold}): {len(results)}")
    else:
        print("Warning: No valid feed embeddings to compare")
    
    return results


def compute_user_feed_similarity(user_embeddings: List[float], bq_client: BigQueryClient, threshold: float = 0.7) -> List[Dict]:
    """
    Convenience function that gets user embedding and computes similarity with all feeds
    
    Args:
        user_embeddings: User's embedding vector as list
        bq_client: BigQuery client instance
        threshold: Minimum similarity threshold
        
    Returns:
        List of matching feeds with similarity scores
    """
    print("Computing user-feed similarity scores...")
    
    # Convert user embeddings to numpy array
    user_embedding = get_user_embedding(user_embeddings)
    
    if len(user_embedding) == 0:
        print("Warning: No user embeddings provided")
        return []
    
    # Get feed embeddings from BigQuery
    feed_embeddings = get_feed_embeddings_from_bigquery(bq_client)
    
    if not feed_embeddings:
        print("Warning: No feed embeddings found")
        return []
    
    # Compute similarities
    return compute_feed_similarity_scores(user_embedding, feed_embeddings, threshold)


if __name__ == "__main__":
    # Example usage
    import json
    import os
    
    # Initialize BigQuery client
    credentials_json = json.loads(os.environ['BIGQUERY_CREDENTIALS_JSON'])
    project_id = os.environ['BIGQUERY_PROJECT_ID']
    bq_client = BigQueryClient(credentials_json, project_id)
    
    # Example user embeddings (replace with actual user embeddings)
    user_embeddings = [0.1, -0.2, 0.3, 0.4]  # Example embedding
    
    # Compute similarities
    matching_feeds = compute_user_feed_similarity(user_embeddings, bq_client, threshold=0.5)
    
    print(f"\nFound {len(matching_feeds)} matching feeds:")
    for feed in matching_feeds:
        print(f"  {feed['feed_uri']}: {feed['similarity_score']:.3f}")