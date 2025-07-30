import re
from typing import List, Set


def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization - lowercase, remove punctuation, split on whitespace
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Split on whitespace and filter empty strings
    tokens = [token for token in text.split() if token]
    
    return tokens


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text


def remove_stopwords(tokens: List[str], stopwords: Set[str] = None) -> List[str]:
    """
    Remove common stopwords from token list
    
    Args:
        tokens: List of tokens
        stopwords: Set of stopwords to remove (uses default if None)
        
    Returns:
        Filtered tokens without stopwords
    """
    if stopwords is None:
        # Basic English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'but', 'or', 'not', 'this', 'they',
            'i', 'you', 'we', 'she', 'him', 'her', 'his', 'them', 'their'
        }
    
    return [token for token in tokens if token not in stopwords]


def extract_hashtags(text: str) -> List[str]:
    """
    Extract hashtags from text
    
    Args:
        text: Input text
        
    Returns:
        List of hashtags (without # symbol)
    """
    if not text:
        return []
    
    hashtags = re.findall(r'#(\w+)', text.lower())
    return hashtags


def extract_mentions(text: str) -> List[str]:
    """
    Extract @mentions from text
    
    Args:
        text: Input text
        
    Returns:
        List of mentions (without @ symbol)
    """
    if not text:
        return []
    
    mentions = re.findall(r'@(\w+)', text.lower())
    return mentions


def preprocess_for_bm25(text: str, remove_stops: bool = False) -> List[str]:
    """
    Complete preprocessing pipeline for BM25
    
    Args:
        text: Input text
        remove_stops: Whether to remove stopwords
        
    Returns:
        List of processed tokens
    """
    # Clean and tokenize
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    
    # Optionally remove stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)
    
    return tokens


if __name__ == "__main__":
    # Example usage
    sample_text = "Hello World! This is a #test post with @mentions and some punctuation..."
    
    print("Original text:", sample_text)
    print("Tokens:", tokenize_text(sample_text))
    print("Hashtags:", extract_hashtags(sample_text))
    print("Mentions:", extract_mentions(sample_text))
    print("BM25 preprocessing:", preprocess_for_bm25(sample_text, remove_stops=True))