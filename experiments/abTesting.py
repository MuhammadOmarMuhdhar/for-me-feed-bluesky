import json
import hashlib
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ABTestManager:
    """
    A/B Testing Manager for feed ranking experiments
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize A/B Testing Manager
        
        Args:
            config_path: Path to experiment configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load experiment configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded A/B test configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"A/B test config file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in A/B test config: {e}")
            return {}
    
    def _hash_user_id(self, user_id: str, experiment_name: str) -> float:
        """
        Generate consistent hash for user assignment
        
        Returns:
            Float between 0.0 and 1.0
        """
        # Combine user ID and experiment name for consistent but experiment-specific hashing
        combined = f"{user_id}:{experiment_name}"
        hash_bytes = hashlib.md5(combined.encode()).digest()
        # Convert to float between 0 and 1
        hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
        return hash_int / (2**32)
    
    def get_user_experiment_group(self, user_id: str, experiment_name: str) -> str:
        """
        Determine which experiment group a user belongs to
        
        Args:
            user_id: User's DID or identifier
            experiment_name: Name of the experiment
            
        Returns:
            Group name (e.g., 'enhanced', 'basic', 'control')
        """
        if experiment_name not in self.config:
            self.logger.warning(f"Experiment '{experiment_name}' not found in config")
            return 'control'
        
        experiment = self.config[experiment_name]
        
        # Check if experiment is enabled
        if not experiment.get('enabled', False):
            return experiment.get('default_group', 'control')
        
        # Get user hash for consistent assignment
        user_hash = self._hash_user_id(user_id, experiment_name)
        
        # Determine group based on percentage splits
        groups = experiment.get('groups', {})
        cumulative_percentage = 0.0
        
        for group_name, group_config in groups.items():
            cumulative_percentage += group_config.get('percentage', 0) / 100.0
            if user_hash <= cumulative_percentage:
                self.logger.debug(f"User {user_id[:12]}... assigned to '{group_name}' group for experiment '{experiment_name}'")
                return group_name
        
        # Fallback to default group
        default_group = experiment.get('default_group', 'control')
        self.logger.debug(f"User {user_id[:12]}... assigned to default group '{default_group}'")
        return default_group
    
    def should_use_enhanced_ranking(self, user_id: str) -> Tuple[bool, str]:
        """
        Determine if user should receive enhanced ranking
        
        Args:
            user_id: User's DID
            
        Returns:
            (should_use_enhanced, group_name)
        """
        group = self.get_user_experiment_group(user_id, 'enhanced_keywords_experiment')
        
        use_enhanced = group == 'enhanced'
        return use_enhanced, group
    
    def should_use_reading_level_filtering(self, user_id: str) -> Tuple[bool, str]:
        """
        Determine if user should receive reading level filtering
        
        Args:
            user_id: User's DID
            
        Returns:
            (should_use_filtering, group_name)
        """
        group = self.get_user_experiment_group(user_id, 'reading_level_experiment')
        
        use_filtering = group == 'filtered'
        return use_filtering, group
    
    def log_ranking_experiment(self, user_id: str, experiment_data: Dict):
        """
        Log experiment data for analysis
        
        Args:
            user_id: User's DID
            experiment_data: Dict with experiment metrics
        """
        try:
            # Add timestamp and user info
            log_entry = {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                **experiment_data
            }
            
            # For now, just log to logger - could be enhanced to write to database/file
            self.logger.info(f"AB_TEST_LOG: {json.dumps(log_entry)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log experiment data: {e}")
    
    def get_experiment_info(self, experiment_name: str) -> Optional[Dict]:
        """
        Get experiment configuration
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Experiment configuration dict or None
        """
        return self.config.get(experiment_name)
    
    def get_active_experiments(self) -> List[str]:
        """
        Get list of currently active experiments
        
        Returns:
            List of experiment names that are enabled
        """
        active = []
        for exp_name, exp_config in self.config.items():
            if exp_config.get('enabled', False):
                active.append(exp_name)
        return active
    
    def reload_config(self):
        """Reload configuration from file (for dynamic updates)"""
        self.config = self._load_config()
        self.logger.info("A/B test configuration reloaded")


# Singleton instance for easy access
_ab_test_manager = None

def get_ab_test_manager() -> ABTestManager:
    """Get singleton A/B test manager instance"""
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager