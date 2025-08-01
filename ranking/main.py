import numpy as np
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking.cosineSimilarity import compute_cosine_similarity
from ranking.bm25Similarity import compute_bm25_similarity
from featureEngineering.textPreprocessing import preprocess_for_bm25


def build_cosine_user_profile(user_data_with_embeddings: Dict, weights: Dict, umap_components: int = 32) -> np.ndarray:
    """
    Build weighted user profile vector for cosine similarity
    
    Args:
        user_data_with_embeddings: User data with UMAP embeddings
        weights: Weight multipliers for each content type
        umap_components: Number of UMAP dimensions
        
    Returns:
        Weighted user profile vector
    """
    user_embeddings = []
    
    for content_type in ['posts', 'reposts', 'replies', 'likes']:
        weight = weights.get(content_type, 1.0)
        content_items = user_data_with_embeddings.get(content_type, [])
        
        for item in content_items:
            # Extract UMAP embedding
            embedding = []
            for i in range(1, umap_components + 1):
                umap_key = f'UMAP{i}'
                if umap_key in item:
                    embedding.append(item[umap_key])
                else:
                    break
            
            if len(embedding) == umap_components:
                embedding_array = np.array(embedding)
                # Add multiple copies for weighting
                for _ in range(int(weight)):
                    user_embeddings.append(embedding_array)
                # Handle fractional weights
                if weight % 1 != 0:
                    user_embeddings.append(embedding_array)
    
    if not user_embeddings:
        return np.zeros(umap_components)
    
    # Average all embeddings to create profile
    return np.mean(user_embeddings, axis=0)


def build_bm25_user_terms(user_data_with_embeddings: Dict, weights: Dict) -> List[str]:
    """
    Build weighted user query terms for BM25
    
    Args:
        user_data_with_embeddings: User data with text content
        weights: Weight multipliers for each content type
        
    Returns:
        List of weighted terms for BM25 query
    """
    user_terms = []
    
    for content_type in ['posts', 'reposts', 'replies', 'likes']:
        weight = weights.get(content_type, 1.0)
        content_items = user_data_with_embeddings.get(content_type, [])
        
        for item in content_items:
            text = item.get('text', '')
            if text:
                terms = preprocess_for_bm25(text, remove_stops=False)
                # Add terms multiple times based on weight
                for _ in range(int(weight)):
                    user_terms.extend(terms)
                # Handle fractional weights
                if weight % 1 != 0:
                    user_terms.extend(terms)
    
    return user_terms


def normalize_and_combine_scores(posts: List[Dict], cosine_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Dict]:
    """
    Normalize BM25 scores and combine with cosine similarity
    
    Args:
        posts: Posts with similarity_score and bm25_score
        cosine_weight: Weight for cosine similarity in combination
        bm25_weight: Weight for BM25 in combination
        
    Returns:
        Posts with normalized scores and combined_score, sorted by combined score
    """
    # Get all BM25 scores for normalization
    bm25_scores = [post.get('bm25_score', 0) for post in posts]
    max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1
    
    # Normalize and combine scores
    for post in posts:
        cosine_score = post.get('similarity_score', 0)
        raw_bm25_score = post.get('bm25_score', 0)
        
        # Normalize BM25 to [0,1] range
        normalized_bm25 = raw_bm25_score / max_bm25 if max_bm25 > 0 else 0
        post['normalized_bm25_score'] = normalized_bm25
        
        # Weighted combination
        combined_score = cosine_weight * cosine_score + bm25_weight * normalized_bm25
        post['combined_score'] = combined_score
    
    # Sort by combined score
    return sorted(posts, key=lambda x: x.get('combined_score', 0), reverse=True)


def rank_posts(
    user_data_with_embeddings: Dict,
    posts_with_embeddings: List[Dict], 
    weights: Dict = None,
    cosine_weight: float = 0.7,
    bm25_weight: float = 0.3,
    umap_components: int = 32,
    methods: List[str] = ['cosine', 'bm25']
) -> List[Dict]:
    """
    Main ranking function - combines multiple similarity methods with configurable weights
    
    Args:
        user_data_with_embeddings: User engagement data with embeddings
        posts_with_embeddings: Posts to rank with embeddings
        weights: Content type weights (posts, reposts, replies, likes)
        cosine_weight: Weight for cosine similarity in final combination
        bm25_weight: Weight for BM25 in final combination  
        umap_components: Number of UMAP dimensions
        methods: List of similarity methods to use
        
    Returns:
        Posts ranked by combined similarity score
    """
    if weights is None:
        weights = {'posts': 2.0, 'reposts': 3.0, 'replies': 1.5, 'likes': 1.0}
    
    print(f"Ranking {len(posts_with_embeddings)} posts using methods: {methods}")
    print(f"Content weights: {weights}")
    print(f"Similarity weights: cosine={cosine_weight}, bm25={bm25_weight}")
    
    # Initialize posts with zero scores
    for post in posts_with_embeddings:
        post['similarity_score'] = 0.0
        post['bm25_score'] = 0.0
    
    # Compute cosine similarity if requested
    if 'cosine' in methods:
        print("Computing cosine similarity...")
        user_profile = build_cosine_user_profile(user_data_with_embeddings, weights, umap_components)
        posts_with_embeddings = compute_cosine_similarity(user_profile, posts_with_embeddings, umap_components)
        print("Cosine similarity computed")
    
    # Compute BM25 similarity if requested  
    if 'bm25' in methods:
        print("Computing BM25 similarity...")
        user_terms = build_bm25_user_terms(user_data_with_embeddings, weights)
        posts_with_embeddings = compute_bm25_similarity(user_terms, posts_with_embeddings)
        print("BM25 similarity computed")
    
    # Normalize and combine scores
    print("Normalizing and combining scores...")
    ranked_posts = normalize_and_combine_scores(posts_with_embeddings, cosine_weight, bm25_weight)
    
    print(f"Ranking completed: {len(ranked_posts)} posts ranked")
    if ranked_posts:
        top_score = ranked_posts[0].get('combined_score', 0)
        bottom_score = ranked_posts[-1].get('combined_score', 0)
        print(f"Score range: {bottom_score:.3f} to {top_score:.3f}")
    
    return ranked_posts


def rank_posts_simple(user_data_with_embeddings: Dict, posts_with_embeddings: List[Dict], top_k: int = 50) -> List[Dict]:
    """
    Simple ranking with default weights and methods
    
    Args:
        user_data_with_embeddings: User engagement data with embeddings
        posts_with_embeddings: Posts to rank with embeddings
        top_k: Number of top posts to return
        
    Returns:
        Top-k ranked posts
    """
    ranked_posts = rank_posts(user_data_with_embeddings, posts_with_embeddings)
    return ranked_posts[:top_k]

