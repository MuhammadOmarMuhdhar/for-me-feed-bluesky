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
        ('posts', 1),     
        ('reposts', 1),   
        ('replies', 1), 
        ('likes', 1)     
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


def analyze_post_sentiment_and_emotions(text: str) -> Dict:
    """
    Analyze post sentiment and emotions for compatibility matching
    
    Args:
        text: Post text to analyze
        
    Returns:
        Dict with sentiment and emotion scores
    """
    try:
        from textblob import TextBlob
        from nrclex import NRCLex
        
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
        
        # Emotion analysis
        emotion = NRCLex(text)
        emotions = emotion.raw_emotion_scores
        
        # Normalize emotions by text length
        text_words = len(text.split())
        normalization_factor = max(1, text_words)
        
        normalized_emotions = {
            'joy': min(1.0, emotions.get('joy', 0) / normalization_factor),
            'anger': min(1.0, emotions.get('anger', 0) / normalization_factor),
            'trust': min(1.0, emotions.get('trust', 0) / normalization_factor),
            'fear': min(1.0, emotions.get('fear', 0) / normalization_factor),
            'surprise': min(1.0, emotions.get('surprise', 0) / normalization_factor)
        }
        
        return {
            'sentiment': sentiment,
            'emotions': normalized_emotions
        }
        
    except Exception as e:
        print(f"Warning: Post sentiment analysis failed: {e}")
        # Return neutral values on failure
        return {
            'sentiment': {'polarity': 0.0, 'subjectivity': 0.5},
            'emotions': {'joy': 0.3, 'anger': 0.1, 'trust': 0.5, 'fear': 0.2, 'surprise': 0.3}
        }


def calculate_sentiment_compatibility_multiplier(post_analysis: Dict, user_sentiment_context: Dict, matched_keywords: List[str]) -> float:
    """
    Calculate sentiment compatibility multiplier for enhanced BM25 scoring
    
    Args:
        post_analysis: Post's sentiment/emotion analysis
        user_sentiment_context: User's sentiment preferences per keyword
        matched_keywords: Keywords that matched between user and post
        
    Returns:
        Multiplier for BM25 score (0.5 to 1.5 range)
    """
    try:
        if not matched_keywords or not user_sentiment_context:
            return 1.0  # Neutral multiplier
        
        post_sentiment = post_analysis['sentiment']
        post_emotions = post_analysis['emotions']
        
        total_compatibility = 0.0
        keyword_count = 0
        
        # Calculate compatibility for each matched keyword
        for keyword in matched_keywords:
            if keyword in user_sentiment_context:
                user_data = user_sentiment_context[keyword]
                user_sentiment = user_data['sentiment']
                user_emotions = user_data['emotions']
                
                # Sentiment compatibility (weight: 40%)
                sentiment_diff = abs(post_sentiment['polarity'] - user_sentiment['avg_polarity'])
                sentiment_compatibility = max(0.0, 1.0 - sentiment_diff)  # 1.0 = perfect match, 0.0 = opposite
                
                # Emotion compatibility (weight: 30%)  
                emotion_compatibility = 0.0
                for emotion_type in ['joy', 'anger', 'trust', 'fear', 'surprise']:
                    post_emotion = post_emotions.get(emotion_type, 0.0)
                    user_emotion = user_emotions.get(emotion_type, 0.0)
                    emotion_diff = abs(post_emotion - user_emotion)
                    emotion_compatibility += max(0.0, 1.0 - emotion_diff)
                emotion_compatibility /= 5  # Average across emotions
                
                # Subjectivity alignment (weight: 30%)
                subjectivity_diff = abs(post_sentiment['subjectivity'] - user_sentiment['avg_subjectivity'])
                subjectivity_compatibility = max(0.0, 1.0 - subjectivity_diff)
                
                # Weighted combination
                keyword_compatibility = (
                    sentiment_compatibility * 0.4 +
                    emotion_compatibility * 0.3 +
                    subjectivity_compatibility * 0.3
                )
                
                total_compatibility += keyword_compatibility
                keyword_count += 1
        
        if keyword_count == 0:
            return 1.0
        
        # Average compatibility across matched keywords
        avg_compatibility = total_compatibility / keyword_count
        
        # Convert to multiplier: 0.0 compatibility → 0.6x, 1.0 compatibility → 1.4x
        multiplier = 0.6 + (avg_compatibility * 0.8)
        
        return max(0.5, min(1.5, multiplier))  # Clamp to reasonable bounds
        
    except Exception as e:
        print(f"Warning: Sentiment compatibility calculation failed: {e}")
        return 1.0  # Neutral multiplier on error


def calculate_reading_level_multiplier(post_text: str, user_reading_level: int) -> float:
    """
    Calculate reading level compatibility multiplier
    
    Args:
        post_text: Post text to analyze
        user_reading_level: User's preferred reading level (1-20 scale)
        
    Returns:
        Multiplier for BM25 score (0.7 to 1.0 range)
    """
    try:
        import textstat
        
        # Calculate post reading level
        post_reading_level = textstat.flesch_kincaid_grade(post_text)
        
        # Ensure reasonable bounds
        post_reading_level = max(1, min(20, post_reading_level))
        
        # Calculate difference
        level_diff = abs(post_reading_level - user_reading_level)
        
        # Apply penalty for large differences
        if level_diff <= 1:
            return 1.0      # Perfect/close match
        elif level_diff <= 2:
            return 0.95     # Minor penalty
        elif level_diff <= 3:
            return 0.9      # Moderate penalty
        else:
            return 0.8      # Significant penalty
            
    except Exception as e:
        print(f"Warning: Reading level calculation failed: {e}")
        return 1.0  # No penalty on error


def compute_enhanced_bm25_similarity(
    user_query_terms: List[str], 
    posts_with_text: List[Dict], 
    user_sentiment_context: Dict = None, 
    user_reading_level: int = 8,
    k1: float = 1.2, 
    b: float = 0.75
) -> List[Dict]:
    """
    Enhanced BM25 similarity with sentiment compatibility and reading level filtering
    
    Args:
        user_query_terms: List of terms representing user's interests
        posts_with_text: Posts with text content
        user_sentiment_context: User's sentiment/emotion preferences per keyword
        user_reading_level: User's preferred reading level (1-20)
        k1: BM25 parameter controlling term frequency saturation
        b: BM25 parameter controlling document length normalization
        
    Returns:
        Posts with enhanced_bm25_score field added
    """
    print(f"Computing Enhanced BM25 similarity for {len(posts_with_text)} posts with sentiment analysis...")
    
    # First compute standard BM25 scores
    posts_with_bm25 = compute_bm25_similarity(user_query_terms, posts_with_text, k1, b)
    
    if not user_sentiment_context:
        print("No sentiment context provided, using basic BM25 scores")
        for post in posts_with_bm25:
            post['enhanced_bm25_score'] = post.get('bm25_score', 0.0)
            post['sentiment_multiplier'] = 1.0
            post['reading_level_multiplier'] = 1.0
        return posts_with_bm25
    
    # Extract unique keywords from query terms for matching
    unique_query_terms = list(set(user_query_terms))
    
    enhanced_posts = []
    sentiment_analysis_count = 0
    
    for post in posts_with_bm25:
        text = post.get('text', '')
        base_bm25_score = post.get('bm25_score', 0.0)
        
        if not text or base_bm25_score == 0:
            post['enhanced_bm25_score'] = base_bm25_score
            post['sentiment_multiplier'] = 1.0
            post['reading_level_multiplier'] = 1.0
            enhanced_posts.append(post)
            continue
        
        # Analyze post sentiment and emotions
        post_analysis = analyze_post_sentiment_and_emotions(text)
        sentiment_analysis_count += 1
        
        # Find keywords that appear in this post
        text_lower = text.lower()
        matched_keywords = [term for term in unique_query_terms if term.lower() in text_lower]
        
        # Calculate sentiment compatibility multiplier
        sentiment_multiplier = calculate_sentiment_compatibility_multiplier(
            post_analysis, user_sentiment_context, matched_keywords
        )
        
        # Calculate reading level compatibility multiplier
        reading_level_multiplier = calculate_reading_level_multiplier(text, user_reading_level)
        
        # Calculate enhanced score
        enhanced_score = base_bm25_score * sentiment_multiplier * reading_level_multiplier
        
        # Store analysis results
        post['enhanced_bm25_score'] = float(enhanced_score)
        post['base_bm25_score'] = float(base_bm25_score)
        post['sentiment_multiplier'] = float(sentiment_multiplier)
        post['reading_level_multiplier'] = float(reading_level_multiplier)
        post['matched_keywords'] = matched_keywords
        post['post_sentiment_analysis'] = post_analysis
        
        enhanced_posts.append(post)
    
    # Log enhancement statistics
    if sentiment_analysis_count > 0:
        sentiment_multipliers = [p.get('sentiment_multiplier', 1.0) for p in enhanced_posts if p.get('sentiment_multiplier')]
        reading_multipliers = [p.get('reading_level_multiplier', 1.0) for p in enhanced_posts if p.get('reading_level_multiplier')]
        
        avg_sentiment_mult = sum(sentiment_multipliers) / len(sentiment_multipliers) if sentiment_multipliers else 1.0
        avg_reading_mult = sum(reading_multipliers) / len(reading_multipliers) if reading_multipliers else 1.0
        
        boosted_posts = len([m for m in sentiment_multipliers if m > 1.05])
        penalized_posts = len([m for m in sentiment_multipliers if m < 0.95])
        
        print(f"Enhanced BM25 complete: analyzed {sentiment_analysis_count} posts")
        print(f"Average sentiment multiplier: {avg_sentiment_mult:.3f}")
        print(f"Average reading level multiplier: {avg_reading_mult:.3f}")
        print(f"Posts boosted by sentiment: {boosted_posts}, Posts penalized: {penalized_posts}")
    
    return enhanced_posts

