"""
Cache management utilities for the Chicago Housing Pipeline project.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages data caching and freshness checks."""
    
    def __init__(self, cache_dir):
        """
        Initialize cache manager.
        
        Args:
            cache_dir (str or Path): Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def is_cache_fresh(self, cache_file, max_age_hours=24):
        """
        Check if cache file is fresh enough to use.
        
        Args:
            cache_file (str or Path): Path to cache file
            max_age_hours (int): Maximum age in hours before cache is stale
            
        Returns:
            bool: True if cache is fresh, False if stale or missing
        """
        cache_path = Path(cache_file)
        
        if not cache_path.exists():
            logger.info(f"Cache file does not exist: {cache_path}")
            return False
        
        # Check file modification time
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time
        
        if age > timedelta(hours=max_age_hours):
            logger.warning(f"Cache file is stale ({age.total_seconds()/3600:.1f} hours old): {cache_path}")
            return False
        
        logger.info(f"Cache file is fresh ({age.total_seconds()/3600:.1f} hours old): {cache_path}")
        return True
    
    def clear_cache(self, pattern="*", confirm=True):
        """
        Clear cache files matching pattern.
        
        Args:
            pattern (str): Glob pattern for files to delete
            confirm (bool): Whether to log before deleting
            
        Returns:
            int: Number of files deleted
        """
        cache_files = list(self.cache_dir.glob(pattern))
        
        if not cache_files:
            logger.info(f"No cache files found matching pattern: {pattern}")
            return 0
        
        if confirm:
            logger.info(f"Clearing {len(cache_files)} cache files matching pattern: {pattern}")
        
        deleted_count = 0
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted cache file: {cache_file}")
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {str(e)}")
        
        logger.info(f"Cleared {deleted_count} cache files")
        return deleted_count
    
    def get_cache_info(self):
        """
        Get information about cache contents.
        
        Returns:
            dict: Cache statistics and file info
        """
        cache_files = list(self.cache_dir.rglob("*"))
        cache_files = [f for f in cache_files if f.is_file()]
        
        if not cache_files:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_file': None,
                'newest_file': None,
                'files': []
            }
        
        total_size = sum(f.stat().st_size for f in cache_files)
        file_times = [datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files]
        
        file_info = []
        for f in cache_files:
            stat = f.stat()
            file_info.append({
                'path': str(f),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'age_hours': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600
            })
        
        return {
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_file': min(file_times).isoformat() if file_times else None,
            'newest_file': max(file_times).isoformat() if file_times else None,
            'files': sorted(file_info, key=lambda x: x['age_hours'])
        }
    
    def force_refresh_datasets(self, datasets=None):
        """
        Force refresh of specific datasets by clearing their cache.
        
        Args:
            datasets (list, optional): List of dataset names to refresh.
                                     If None, refreshes all datasets.
                                     Options: ['census', 'fred', 'chicago']
        """
        if datasets is None:
            datasets = ['census', 'fred', 'chicago']
        
        if not isinstance(datasets, list):
            datasets = [datasets]
        
        logger.info(f"Forcing refresh of datasets: {datasets}")
        
        total_cleared = 0
        for dataset in datasets:
            if dataset == 'census':
                cleared = self.clear_cache('census/*', confirm=False)
            elif dataset == 'fred':
                cleared = self.clear_cache('fred/*', confirm=False)
            elif dataset == 'chicago':
                cleared = self.clear_cache('chicago/*', confirm=False)
            else:
                logger.warning(f"Unknown dataset for refresh: {dataset}")
                continue
            
            total_cleared += cleared
            logger.info(f"Cleared {cleared} cache files for {dataset} dataset")
        
        logger.info(f"Total cache files cleared: {total_cleared}")
        return total_cleared
    
    def clear_cache_by_pattern(self, pattern):
        """
        Clear cache files matching a pattern.
        
        Args:
            pattern (str): Glob pattern to match files
        
        Returns:
            int: Number of files cleared
        """
        return self.clear_cache(pattern, confirm=True) 