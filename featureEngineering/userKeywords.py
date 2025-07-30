import sys
import os
from collections import Counter
from typing import Dict, List, Set
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from featureEngineering.textPreprocessing import preprocess_for_bm25, remove_stopwords


def extract_user_keywords(user_data: Dict, top_k: int = 10, min_freq: int = 2) -> List[str]:
    """
    Extract distinctive keywords from user's engagement data using scikit-learn TF-IDF
    
    Args:
        user_data: User's posts, reposts, replies, likes data
        top_k: Number of top keywords to return
        min_freq: Minimum frequency for a term to be considered
        
    Returns:
        List of top user keywords for personalized search
    """
    print(f"Extracting top {top_k} user keywords using TF-IDF...")
    
    # Collect user text with engagement weights
    documents = []
    engagement_weights = []
    
    # Posts - what they create (weight: 2)
    for post in user_data.get('posts', []):
        text = post.get('text', '')
        if text:
            documents.append(text)
            engagement_weights.append(2.0)
    
    # Reposts - what they amplify (weight: 3) 
    for repost in user_data.get('reposts', []):
        text = repost.get('text', '')
        if text:
            documents.append(text)
            engagement_weights.append(3.0)
    
    # Replies - what they engage with (weight: 1.5)
    for reply in user_data.get('replies', []):
        text = reply.get('text', '')
        if text:
            documents.append(text)
            engagement_weights.append(1.5)
    
    # Likes - what they consume (weight: 1)
    for like in user_data.get('likes', []):
        text = like.get('text', '')
        if text:
            documents.append(text)
            engagement_weights.append(1.0)
    
    if not documents:
        print("No user text found for keyword extraction")
        return []
    
    print(f"Processing {len(documents)} documents with engagement weights")
    
    # Custom preprocessing function to use existing text preprocessing
    def preprocess_text(text):
        terms = preprocess_for_bm25(text, remove_stops=False)
        return ' '.join(terms)
    
    # Apply preprocessing
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    
    # Initialize TF-IDF vectorizer with custom parameters
    tfidf = TfidfVectorizer(
        max_features=1000,  # Limit vocabulary size
        min_df=min_freq,    # Minimum document frequency
        max_df=0.8,         # Remove terms that appear in >80% of documents
        stop_words='english',  # Remove English stopwords
        token_pattern=r'\b[a-zA-Z]{3,}\b',  # Only alphabetic terms, min 3 chars
        lowercase=True,
        ngram_range=(1, 1)  # Only unigrams
    )
    
    try:
        # Fit and transform documents
        tfidf_matrix = tfidf.fit_transform(preprocessed_docs)
        feature_names = tfidf.get_feature_names_out()
        
        if tfidf_matrix.shape[1] == 0:
            print("No valid features found after TF-IDF processing")
            return []
        
        # Apply engagement weights to TF-IDF scores
        engagement_weights_array = np.array(engagement_weights).reshape(-1, 1)
        weighted_tfidf = tfidf_matrix.multiply(engagement_weights_array)
        
        # Sum TF-IDF scores across all documents for each term
        term_scores = np.array(weighted_tfidf.sum(axis=0)).flatten()
        
        # Create (term, score) pairs and sort by score
        term_score_pairs = list(zip(feature_names, term_scores))
        term_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top K terms
        keywords = [term for term, score in term_score_pairs[:top_k] if score > 0]
        
        print(f"Top user keywords: {keywords}")
        return keywords
        
    except ValueError as e:
        print(f"TF-IDF processing failed: {e}")
        # Fallback to simple frequency counting
        return extract_user_keywords_simple(user_data, top_k)


def extract_user_keywords_simple(user_data: Dict, top_k: int = 10) -> List[str]:
    """
    Simple keyword extraction using just frequency counting
    
    Args:
        user_data: User's engagement data
        top_k: Number of keywords to return
        
    Returns:
        List of most frequent user terms
    """
    # Collect all user text
    all_text = []
    
    for content_type in ['posts', 'reposts', 'replies', 'likes']:
        for item in user_data.get(content_type, []):
            text = item.get('text', '')
            if text:
                all_text.append(text)
    
    if not all_text:
        return []
    
    # Extract terms
    all_terms = []
    for text in all_text:
        terms = preprocess_for_bm25(text, remove_stops=True)
        all_terms.extend(terms)
    
    # Count and return most frequent
    term_counts = Counter(all_terms)
    
    # Filter short terms and numbers
    filtered_counts = {
        term: count for term, count in term_counts.items() 
        if len(term) > 2 and term.isalpha()
    }
    
    # Return top K most frequent
    top_terms = [term for term, count in Counter(filtered_counts).most_common(top_k)]
    
    return top_terms


def get_fallback_keywords() -> List[str]:
    """
    Return fallback keywords for users with insufficient data
    
    Returns:
        List of generic but useful search terms
    """
    return [
        'interesting', 'important', 'great', 'amazing', 'cool', 'awesome',
        'news', 'update', 'thoughts', 'opinion', 'discussion', 'question',
        'today', 'new', 'people', 'time', 'work', 'life'
    ]


def extract_keywords_with_fallback(user_data: Dict, top_k: int = 10, min_posts: int = 3) -> List[str]:
    """
    Extract user keywords with fallback for users with insufficient data
    
    Args:
        user_data: User's engagement data
        top_k: Number of keywords to return
        min_posts: Minimum posts required for personalized extraction
        
    Returns:
        List of keywords (personalized or fallback)
    """
    # Check if user has enough content
    total_content = (
        len(user_data.get('posts', [])) + 
        len(user_data.get('reposts', [])) + 
        len(user_data.get('replies', []))
    )
    
    if total_content < min_posts:
        print(f"User has only {total_content} posts - using fallback keywords")
        return get_fallback_keywords()[:top_k]
    
    # Try personalized extraction
    keywords = extract_user_keywords(user_data, top_k)
    
    if len(keywords) < top_k // 2:
        print("Insufficient personalized keywords - mixing with fallback")
        fallback = get_fallback_keywords()
        # Mix personalized + fallback
        keywords.extend(fallback)
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates while preserving order
    
    return keywords[:top_k]


if __name__ == "__main__":
    # Example usage
    sample_user_data = {
        'posts': [
            {'text': 'Working on machine learning algorithms today'},
            {'text': 'Python is great for data science projects'},
            {'text': 'Neural networks are fascinating to study'}
        ],
        'reposts': [
            {'text': 'Excellent paper on deep learning architectures'}
        ],
        'replies': [
            {'text': 'Agree that transformers changed everything in NLP'},
            {'text': 'TensorFlow vs PyTorch is always a good debate'}
        ],
        'likes': []
    }
    
    print("Testing keyword extraction:")
    print("=" * 50)
    
    # Test main extraction
    keywords = extract_user_keywords(sample_user_data, top_k=8)
    print(f"Main extraction: {keywords}")
    
    # Test simple extraction
    simple_keywords = extract_user_keywords_simple(sample_user_data, top_k=8)
    print(f"Simple extraction: {simple_keywords}")
    
    # Test with fallback
    fallback_keywords = extract_keywords_with_fallback(sample_user_data, top_k=8)
    print(f"With fallback: {fallback_keywords}")
    
    # Test with minimal data
    minimal_data = {'posts': [{'text': 'Hello world'}], 'reposts': [], 'replies': [], 'likes': []}
    minimal_keywords = extract_keywords_with_fallback(minimal_data, top_k=8)
    print(f"Minimal data result: {minimal_keywords}")