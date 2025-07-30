import numpy as np
from collections import Counter, defaultdict
import math
from typing import List, Dict, Set
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from featureEngineering.textPreprocessing import preprocess_for_bm25


def build_user_query_terms(user_data: Dict) -> List[str]:
    """
    Extract all terms from user's content to build query
    
    Args:
        user_data: User's posts/reposts/replies/likes with text
        
    Returns:
        List of terms representing user's interests
    """
    all_terms = []
    
    # Extract terms with weights
    content_types = [
        ('posts', 2),     # Posts weight: 2
        ('reposts', 3),   # Reposts weight: 3 (highest)
        ('replies', 1.5), # Replies weight: 1.5
        ('likes', 1)      # Likes weight: 1
    ]
    
    for content_type, weight in content_types:
        for item in user_data.get(content_type, []):
            text = item.get('text', '')
            if text:
                terms = preprocess_for_bm25(text, remove_stops=False)
                # Add terms multiple times based on weight
                for _ in range(int(weight)):
                    all_terms.extend(terms)
                # Handle fractional weights
                if weight % 1 != 0:
                    all_terms.extend(terms)
    
    return all_terms


def compute_bm25_similarity(user_query_terms: List[str], posts_with_text: List[Dict], k1: float = 1.2, b: float = 0.75) -> List[Dict]:
    """
    Compute BM25 similarity between user query terms and posts
    
    Args:
        user_query_terms: List of terms representing user's interests
        posts_with_text: Posts with text content
        k1: BM25 parameter controlling term frequency saturation (default: 1.2)
        b: BM25 parameter controlling document length normalization (default: 0.75)
        
    Returns:
        Posts with bm25_score field added
    """
    print(f"Computing BM25 similarity for {len(posts_with_text)} posts...")
    
    if not user_query_terms:
        print("Warning: No user query terms - setting all BM25 scores to 0")
        for post in posts_with_text:
            post['bm25_score'] = 0.0
        return posts_with_text
    
    # Tokenize all posts
    post_tokens = []
    valid_posts = []
    
    for post in posts_with_text:
        text = post.get('text', '')
        if text:
            tokens = preprocess_for_bm25(text, remove_stops=False)
            post_tokens.append(tokens)
            valid_posts.append(post)
        else:
            post['bm25_score'] = 0.0
    
    if not post_tokens:
        print("Warning: No valid post texts")
        return posts_with_text
    
    # Calculate document statistics
    N = len(post_tokens)  # Total number of documents
    doc_lengths = [len(tokens) for tokens in post_tokens]
    avg_doc_length = sum(doc_lengths) / N
    
    # Build vocabulary and document frequency
    vocab = set()
    for tokens in post_tokens:
        vocab.update(tokens)
    
    # Count document frequency for each term
    df = defaultdict(int)  # document frequency
    for tokens in post_tokens:
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1
    
    # Get unique query terms and their frequencies
    query_term_counts = Counter(user_query_terms)
    unique_query_terms = list(query_term_counts.keys())
    
    print(f"Query has {len(unique_query_terms)} unique terms")
    print(f"Corpus has {N} documents, avg length: {avg_doc_length:.1f}")
    
    # Calculate BM25 scores
    bm25_scores = []
    
    for i, tokens in enumerate(post_tokens):
        doc_length = doc_lengths[i]
        term_counts = Counter(tokens)
        
        score = 0.0
        
        for query_term in unique_query_terms:
            if query_term in term_counts:
                # Term frequency in document
                tf = term_counts[query_term]
                
                # Document frequency and IDF
                doc_freq = df[query_term]
                idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5))
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                term_score = idf * (numerator / denominator)
                score += term_score
        
        bm25_scores.append(score)
    
    # Add BM25 scores to valid posts
    for i, post in enumerate(valid_posts):
        post['bm25_score'] = float(bm25_scores[i])
    
    # Print stats
    valid_scores = [score for score in bm25_scores if score > 0]
    if valid_scores:
        print(f"Success: Computed {len(bm25_scores)} BM25 scores")
        print(f"Max BM25 score: {max(bm25_scores):.3f}")
        print(f"Min BM25 score: {min(bm25_scores):.3f}")
        print(f"Posts with score > 0: {len(valid_scores)}")
    else:
        print("Warning: No posts matched query terms")
    
    return posts_with_text


def compute_bm25_from_user_data(user_data: Dict, posts_with_text: List[Dict], k1: float = 1.2, b: float = 0.75) -> List[Dict]:
    """
    Convenience function that builds query terms from user data and computes BM25
    
    Args:
        user_data: User's engagement data with text content
        posts_with_text: Posts with text content
        k1: BM25 k1 parameter
        b: BM25 b parameter
        
    Returns:
        Posts with bm25_score field added
    """
    print("Building user query terms from engagement data...")
    
    # Build query terms from user data
    query_terms = build_user_query_terms(user_data)
    
    print(f"Built query with {len(query_terms)} total terms")
    unique_terms = len(set(query_terms))
    print(f"Query has {unique_terms} unique terms")
    
    # Compute BM25 similarity
    return compute_bm25_similarity(query_terms, posts_with_text, k1, b)


if __name__ == "__main__":
    # Example usage
    user_data = {
        'posts': [{'text': 'I love machine learning and AI'}],
        'reposts': [{'text': 'Great paper on neural networks'}],
        'replies': [{'text': 'Fascinating AI research'}],
        'likes': []
    }
    
    posts = [
        {'text': 'New breakthrough in AI and machine learning', 'author': {'handle': 'ai_researcher'}},
        {'text': 'Beautiful sunset today', 'author': {'handle': 'photographer'}},
        {'text': 'Deep learning neural networks are amazing', 'author': {'handle': 'ml_expert'}}
    ]
    
    posts_with_scores = compute_bm25_from_user_data(user_data, posts)
    
    # Sort by BM25 score to see results
    sorted_posts = sorted(posts_with_scores, key=lambda x: x.get('bm25_score', 0), reverse=True)
    
    print("\nTop posts by BM25 score:")
    for i, post in enumerate(sorted_posts[:3]):
        print(f"{i+1}. Score: {post.get('bm25_score', 0):.3f} - {post.get('text', '')[:50]}...")