"""
Configuration and constants for the ranking ETL system.
"""
import logging
from typing import List

# Time-based constants
DEFAULT_TIME_HOURS = 6
ACTIVE_USERS_DAYS = 30
CACHE_TTL_SECONDS = 900  # 15 minutes
DEFAULT_FEED_TTL_SECONDS = 86400  # 24 hours

# Engagement thresholds
MIN_ENGAGEMENT_FEED = 15.0
MIN_ENGAGEMENT_NETWORK = 8.0
VIRAL_AGE_THRESHOLD_MINUTES = 45.0
VIRAL_VELOCITY_THRESHOLD = 1.0

# Network analysis constants
MIN_2ND_DEGREE_OVERLAP = 2
MAX_2ND_DEGREE_CANDIDATES = 100
MAX_2ND_DEGREE_CACHE_CANDIDATES = 100  # Only cache most relevant candidates
POSTS_PER_2ND_DEGREE_ACCOUNT = None  # No limit - collect all posts within time window
NETWORK_BOOST_FACTOR = 2.5
Z_SCORE_THRESHOLD = -0.30

# Cache TTL values
FOLLOWING_LIST_CACHE_TTL = 86400  # 24 hours
NETWORK_OVERLAP_CACHE_TTL = 604800  # 1 week

# Scoring weights and multipliers
REPOST_WEIGHT = 2.0
REPLY_WEIGHT = 0.5
ENHANCED_REPLY_WEIGHT = 1.5
VIRAL_MULTIPLIER = 1.8
DEFAULT_KEYWORD_WEIGHT = 3

# Priority boost values
NETWORK_1ST_DEGREE_BOOST = 3.0
NETWORK_2ND_DEGREE_BASE_BOOST = 1.0
FEED_PRIORITY_BOOST = 1.5
UNKNOWN_SOURCE_BOOST = 0.1

# Feed limits
MAX_POSTS_PER_FEED = 10
MAX_CACHED_POSTS = 1500
MAX_TOP_FEEDS = 10
TARGET_NETWORK_RATIO = 0.3

# User limits
TEST_MODE_USER_LIMIT = 5
MIN_FOLLOWING_USERS = 20
FOLLOWING_PERCENTAGE_FALLBACK = 30  # 30% of users if filtering too aggressive

# Reading level bounds
MIN_READING_LEVEL = 1
MAX_READING_LEVEL = 20
DEFAULT_READING_LEVEL = 8

# FAQ filtering keywords
FAQ_KEYWORDS: List[str] = [
    "faq", "frequently asked", "welcome to", "how to use",
    "getting started", "please read", "rules", "guidelines", 
    "this feed", "submit to", "curated by", "about this feed",
    "before posting", "read first", "pinned", "instructions",
    "how this works", "feed description", "what is this feed"
]

# Feed similarity threshold
FEED_SIMILARITY_THRESHOLD = 0.6

# Profile batch size for API calls
PROFILE_BATCH_SIZE = 25

# Search queries for default feed
DEFAULT_FEED_QUERIES: List[str] = [
    "the", "a", "I", "you", "this", "today", "new", "just", "really", "good"
]
DEFAULT_FEED_TARGET_COUNT = 100

class LoggingConfig:
    """Logging configuration for the ranking system."""
    
    LEVEL = logging.INFO
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @staticmethod
    def configure_logging():
        """Configure logging for the ranking system."""
        logging.basicConfig(
            level=LoggingConfig.LEVEL,
            format=LoggingConfig.FORMAT
        )

class EngagementWeights:
    """Engagement scoring weights."""
    
    LIKES = 1.0
    REPOSTS = 2.0
    REPLIES_BASIC = 0.5
    REPLIES_ENHANCED = 1.5

class PriorityBoosts:
    """Priority boost values for different sources."""
    
    NETWORK_1ST_DEGREE = 3.0
    NETWORK_2ND_DEGREE_BASE = 1.0
    FEED_PRIORITY = 1.5
    UNKNOWN_SOURCE = 0.1

class ThresholdConfig:
    """Dynamic threshold configuration."""
    
    VIRAL_AGE_MINUTES = 45.0
    VIRAL_VELOCITY = 1.0
    FAQ_POST_AGE_DAYS = 5
    FAQ_POSITION_THRESHOLD = 2