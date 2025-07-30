import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


def extract_umap_embedding(item: Dict, umap_components: int) -> np.ndarray:
    """Extract UMAP embedding from item"""
    embedding = []
    for i in range(1, umap_components + 1):
        umap_key = f'UMAP{i}'
        if umap_key in item:
            embedding.append(item[umap_key])
        else:
            return None
    return np.array(embedding)


def compute_cosine_similarity(user_profile_vector: np.ndarray, posts_with_embeddings: List[Dict], umap_components: int = 32) -> List[Dict]:
    """
    Compute cosine similarity between user profile vector and posts
    
    Args:
        user_profile_vector: Pre-computed user preference vector
        posts_with_embeddings: Posts with UMAP embeddings  
        umap_components: UMAP dimensions
        
    Returns:
        Posts with similarity_score field added
    """
    print(f"Computing cosine similarity for {len(posts_with_embeddings)} posts...")
    
    if np.all(user_profile_vector == 0):
        print("Warning: Zero user profile vector - setting all similarities to 0")
        for post in posts_with_embeddings:
            post['similarity_score'] = 0.0
        return posts_with_embeddings
    
    valid_posts = []
    post_embeddings = []
    
    # Extract embeddings from posts
    for post in posts_with_embeddings:
        embedding = extract_umap_embedding(post, umap_components)
        if embedding is not None:
            post_embeddings.append(embedding)
            valid_posts.append(post)
        else:
            # Set similarity to 0 for posts without embeddings
            post['similarity_score'] = 0.0
    
    if not post_embeddings:
        print("Warning: No valid post embeddings")
        return posts_with_embeddings
    
    # Calculate cosine similarities
    post_embeddings = np.array(post_embeddings)
    similarities = cosine_similarity(
        post_embeddings, 
        user_profile_vector.reshape(1, -1)
    ).flatten()
    
    # Add similarity scores to valid posts
    for i, post in enumerate(valid_posts):
        post['similarity_score'] = float(similarities[i])
    
    # Print stats
    valid_scores = [p['similarity_score'] for p in valid_posts]
    if valid_scores:
        print(f"Success: Computed {len(valid_scores)} similarity scores")
        print(f"Max similarity: {max(valid_scores):.3f}")
        print(f"Min similarity: {min(valid_scores):.3f}")
    
    return posts_with_embeddings


if __name__ == "__main__":
    # Example usage
    user_vector = np.random.random(32)  # Example 32-dim user profile
    posts = [
        {'UMAP1': 0.1, 'UMAP2': 0.2, 'text': 'Example post'},
        # ... more posts with UMAP embeddings
    ]
    
    posts_with_scores = compute_cosine_similarity(user_vector, posts)
    print(f"Computed similarity for {len(posts_with_scores)} posts")