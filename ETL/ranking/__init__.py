"""
Ranking ETL system - A modular feed ranking and caching system.

This package provides a complete ETL system for generating personalized feeds
by collecting posts from various sources, ranking them using BM25 and sentiment
analysis, and caching the results.

Main modules:
- feed_orchestrator: Main ETL coordination
- user_data: User data retrieval and keyword processing  
- post_collector: Post collection from feeds and networks
- ranking_engine: BM25 scoring, sentiment analysis, viral detection
- network_analyzer: Network analysis and 2nd degree connections
- content_filters: FAQ filtering, engagement filtering, moderation
- cache_manager: Redis caching operations
- config: Constants and configuration
"""

from ETL.ranking.feedOrchestrator import main
from ETL.ranking.config import LoggingConfig

# User data functions
from ETL.ranking.userData import (
    get_active_users_with_keywords,
    get_user_embeddings, 
    get_user_reading_level,
    process_user_keywords
)

# Post collection functions
from ETL.ranking.postCollector import (
    collect_comprehensive_posts,
    collect_posts_to_rank
)

# Ranking functions
from ETL.ranking.rankingEngine import (
    calculate_rankings_with_feed_boosting,
    calculate_rankings,
    distribute_network_posts,
    is_viral_post
)

# Network analysis functions
from ETL.ranking.networkAnalyzer import (
    get_or_update_following_list,
    calculate_2nd_degree_overlap,
    prioritize_follows_by_ratio
)

# Content filtering functions
from ETL.ranking.contentFilters import (
    filter_faq_posts,
    filter_low_engagement_posts,
    load_blocked_users,
    filter_blocked_posts
)

# Cache management functions
from ETL.ranking.cacheManager import (
    cache_user_rankings,
    update_default_feed_if_needed
)

__version__ = "1.0.0"
__author__ = "Feed Ranking Team"

# Public API
__all__ = [
    # Main entry point
    'main',
    
    # User data
    'get_active_users_with_keywords',
    'get_user_embeddings',
    'get_user_reading_level', 
    'process_user_keywords',
    
    # Post collection
    'collect_comprehensive_posts',
    'collect_posts_to_rank',
    
    # Ranking
    'calculate_rankings_with_feed_boosting',
    'calculate_rankings',
    'distribute_network_posts',
    'is_viral_post',
    
    # Network analysis
    'get_or_update_following_list',
    'calculate_2nd_degree_overlap',
    'prioritize_follows_by_ratio',
    
    # Content filtering
    'filter_faq_posts',
    'filter_low_engagement_posts',
    'load_blocked_users',
    'filter_blocked_posts',
    
    # Cache management
    'cache_user_rankings',
    'update_default_feed_if_needed',
    
    # Configuration
    'LoggingConfig',
]