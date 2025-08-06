"""
Automatic data refresh system for the Chicago Housing Pipeline project.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import json
from typing import Dict, List, Optional
import daemon
from daemon import pidfile
import signal

from .cache_manager import CacheManager
from .census_collector import CensusCollector
from .fred_collector import FREDCollector
from .chicago_collector import ChicagoCollector
from ..config import settings

logger = logging.getLogger(__name__)

class AutoRefresher:
    """Automatically refreshes data based on schedules and staleness detection."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the auto refresher.
        
        Args:
            cache_dir (str): Directory for cache files
        """
        self.cache_manager = CacheManager(cache_dir)
        self.pid_file_path = Path(cache_dir) / "auto_refresher.pid"
        
        # Default refresh intervals (hours)
        self.refresh_intervals = {
            'census': 24 * 7,  # Weekly (Census data changes infrequently)
            'fred': 24,        # Daily (Economic indicators update daily)
            'chicago': 12      # Twice daily (City data can update frequently)
        }
        
        # Data staleness thresholds (hours)
        self.staleness_thresholds = {
            'census': 24 * 7,    # 1 week
            'fred': 36,          # 1.5 days  
            'chicago': 18        # 18 hours
        }
        
        # Initialize collectors
        self.collectors = {}
        self._init_collectors()
        
        # Track last refresh times
        self.last_refresh_file = Path(cache_dir) / "last_refresh.json"
        self.last_refresh_times = self._load_last_refresh_times()
    
    def _init_collectors(self):
        """Initialize data collectors."""
        try:
            if hasattr(settings, 'CENSUS_API_KEY') and settings.CENSUS_API_KEY:
                self.collectors['census'] = CensusCollector()
                logger.info("Census collector initialized")
            
            if hasattr(settings, 'FRED_API_KEY') and settings.FRED_API_KEY:
                self.collectors['fred'] = FREDCollector()
                logger.info("FRED collector initialized")
            
            if hasattr(settings, 'CHICAGO_DATA_TOKEN') and settings.CHICAGO_DATA_TOKEN:
                self.collectors['chicago'] = ChicagoCollector()
                logger.info("Chicago collector initialized")
                
        except Exception as e:
            logger.error(f"Error initializing collectors: {e}")
    
    def _load_last_refresh_times(self) -> Dict[str, str]:
        """Load last refresh times from file."""
        if self.last_refresh_file.exists():
            try:
                with open(self.last_refresh_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load last refresh times: {e}")
        return {}
    
    def _save_last_refresh_times(self):
        """Save last refresh times to file."""
        try:
            self.last_refresh_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.last_refresh_file, 'w') as f:
                json.dump(self.last_refresh_times, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save last refresh times: {e}")
    
    def configure_refresh_intervals(self, intervals: Dict[str, int]):
        """
        Configure refresh intervals for data sources.
        
        Args:
            intervals (dict): Dictionary mapping data source to refresh interval in hours
        """
        self.refresh_intervals.update(intervals)
        logger.info(f"Updated refresh intervals: {self.refresh_intervals}")
    
    def configure_staleness_thresholds(self, thresholds: Dict[str, int]):
        """
        Configure staleness thresholds for data sources.
        
        Args:
            thresholds (dict): Dictionary mapping data source to staleness threshold in hours
        """
        self.staleness_thresholds.update(thresholds)
        logger.info(f"Updated staleness thresholds: {self.staleness_thresholds}")
    
    def is_data_stale(self, data_source: str) -> bool:
        """
        Check if data source is stale based on cache age.
        
        Args:
            data_source (str): Name of data source
            
        Returns:
            bool: True if data is stale
        """
        threshold_hours = self.staleness_thresholds.get(data_source, 24)
        
        # Check cache files for this data source
        cache_files = list(Path(self.cache_manager.cache_dir).glob(f"*{data_source}*"))
        
        if not cache_files:
            logger.info(f"No cache files found for {data_source}, considering stale")
            return True
        
        # Check the newest file
        newest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
        file_age = datetime.now() - datetime.fromtimestamp(newest_file.stat().st_mtime)
        
        is_stale = file_age > timedelta(hours=threshold_hours)
        if is_stale:
            logger.info(f"{data_source} data is stale (age: {file_age}, threshold: {threshold_hours}h)")
        
        return is_stale
    
    def should_refresh(self, data_source: str) -> bool:
        """
        Determine if a data source should be refreshed based on schedule and staleness.
        
        Args:
            data_source (str): Name of data source
            
        Returns:
            bool: True if should refresh
        """
        # Check if data is stale
        if self.is_data_stale(data_source):
            return True
        
        # Check if it's time for scheduled refresh
        last_refresh_str = self.last_refresh_times.get(data_source)
        if not last_refresh_str:
            logger.info(f"No last refresh time for {data_source}, should refresh")
            return True
        
        try:
            last_refresh = datetime.fromisoformat(last_refresh_str)
            time_since_refresh = datetime.now() - last_refresh
            refresh_interval = timedelta(hours=self.refresh_intervals.get(data_source, 24))
            
            should_refresh = time_since_refresh >= refresh_interval
            if should_refresh:
                logger.info(f"{data_source} scheduled for refresh (last: {time_since_refresh} ago)")
            
            return should_refresh
            
        except Exception as e:
            logger.warning(f"Error checking refresh schedule for {data_source}: {e}")
            return True
    
    def refresh_data_source(self, data_source: str) -> bool:
        """
        Refresh a specific data source.
        
        Args:
            data_source (str): Name of data source
            
        Returns:
            bool: True if refresh successful
        """
        if data_source not in self.collectors:
            logger.warning(f"No collector available for {data_source}")
            return False
        
        logger.info(f"Starting refresh for {data_source}")
        start_time = time.time()
        
        try:
            collector = self.collectors[data_source]
            
            # Clear relevant cache files
            self.cache_manager.clear_cache_by_pattern(f"*{data_source}*")
            
            # Collect fresh data based on data source
            success = False
            if data_source == 'census':
                # Get Chicago ZIP codes for Census data
                chicago_zips = ['60601', '60602', '60603', '60604', '60605', 
                               '60606', '60607', '60608', '60609', '60610']
                data = collector.collect_demographics_data(chicago_zips)
                success = data is not None and len(data) > 0
                
            elif data_source == 'fred':
                # Collect FRED economic indicators
                data = collector.collect_economic_indicators(['2020', '2021', '2022', '2023', '2024'])
                success = data is not None and len(data) > 0
                
            elif data_source == 'chicago':
                # Collect various Chicago datasets
                years = ['2020', '2021', '2022', '2023', '2024']
                datasets = ['building_permits', 'business_licenses', 'zoning_changes']
                success = True
                
                for dataset in datasets:
                    try:
                        if dataset == 'building_permits':
                            data = collector.collect_building_permits(years)
                        elif dataset == 'business_licenses':
                            data = collector.collect_business_licenses(years)
                        elif dataset == 'zoning_changes':
                            data = collector.collect_zoning_changes(years)
                        
                        if data is None or len(data) == 0:
                            success = False
                            logger.warning(f"No data collected for {dataset}")
                    except Exception as e:
                        logger.error(f"Error collecting {dataset}: {e}")
                        success = False
            
            if success:
                # Update last refresh time
                self.last_refresh_times[data_source] = datetime.now().isoformat()
                self._save_last_refresh_times()
                
                elapsed_time = time.time() - start_time
                logger.info(f"Successfully refreshed {data_source} in {elapsed_time:.2f} seconds")
                return True
            else:
                logger.error(f"Failed to collect data for {data_source}")
                return False
                
        except Exception as e:
            logger.error(f"Error refreshing {data_source}: {e}")
            return False
    
    def check_and_refresh_all(self):
        """Check all data sources and refresh if needed."""
        logger.info("Checking all data sources for refresh needs")
        
        for data_source in self.collectors.keys():
            if self.should_refresh(data_source):
                try:
                    self.refresh_data_source(data_source)
                except Exception as e:
                    logger.error(f"Error during automatic refresh of {data_source}: {e}")
            else:
                logger.debug(f"{data_source} data is fresh, no refresh needed")
    
    def _refresh_worker(self):
        """Background worker loop for the daemon."""
        logger.info("Auto-refresh daemon worker started")
        
        # Schedule jobs
        schedule.every(30).minutes.do(self.check_and_refresh_all)
        schedule.every().day.at("06:00").do(lambda: self.refresh_data_source('fred'))
        schedule.every().day.at("18:00").do(lambda: self.refresh_data_source('chicago'))
        schedule.every().sunday.at("02:00").do(lambda: self.refresh_data_source('census'))

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in refresh worker: {e}")
                time.sleep(300) # Wait 5 minutes on error
        
        logger.info("Auto-refresh daemon worker stopped")
    
    def start(self):
        """Start the automatic refresh system as a daemon."""
        if self.is_running():
            logger.warning("Auto-refresh daemon is already running")
            return
        
        logger.info("Starting automatic data refresh daemon")
        
        # Create a daemon context
        daemon_context = daemon.DaemonContext(
            working_directory=os.getcwd(),
            umask=0o002,
            pidfile=pidfile.TimeoutPIDLockFile(self.pid_file_path),
        )
        
        try:
            with daemon_context:
                self._refresh_worker()
        except Exception as e:
            logger.error(f"Error starting daemon: {e}")

    def stop(self):
        """Stop the automatic refresh daemon."""
        if not self.is_running():
            logger.warning("Auto-refresh daemon is not running")
            return
        
        logger.info("Stopping automatic data refresh daemon")
        
        try:
            with open(self.pid_file_path, 'r') as f:
                pid = int(f.read().strip())

            # Send termination signal
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process {pid}")

            # Wait for process to terminate
            time.sleep(2)

            # Clean up pid file
            if os.path.exists(self.pid_file_path):
                os.remove(self.pid_file_path)

        except FileNotFoundError:
            logger.warning("PID file not found. Daemon may not be running.")
        except ProcessLookupError:
            logger.warning(f"Process with PID {pid} not found. May have already stopped.")
            if os.path.exists(self.pid_file_path):
                os.remove(self.pid_file_path)
        except Exception as e:
            logger.error(f"Error stopping daemon: {e}")
    
    def is_running(self) -> bool:
        """Check if the daemon is currently running."""
        if not os.path.exists(self.pid_file_path):
            return False

        try:
            with open(self.pid_file_path, 'r') as f:
                pid = int(f.read().strip())

            # Check if process exists
            os.kill(pid, 0)
            return True
        except (IOError, ValueError, ProcessLookupError):
            return False

    def get_status(self) -> Dict:
        """
        Get current status of the auto-refresh system.
        
        Returns:
            dict: Status information
        """
        status = {
            'is_running': self.is_running(),
            'pid': None,
            'collectors': list(self.collectors.keys()),
            'refresh_intervals': self.refresh_intervals,
            'staleness_thresholds': self.staleness_thresholds,
            'last_refresh_times': self.last_refresh_times,
            'data_staleness': {}
        }
        
        if status['is_running']:
            try:
                with open(self.pid_file_path, 'r') as f:
                    status['pid'] = int(f.read().strip())
            except (IOError, ValueError):
                pass

        # Check staleness for each data source
        for data_source in self.collectors.keys():
            status['data_staleness'][data_source] = self.is_data_stale(data_source)
        
        return status 