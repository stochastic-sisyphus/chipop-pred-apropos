"""
Main script to run the Chicago Housing Pipeline.
"""

import logging
import sys
import os
from pathlib import Path
import argparse
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.pipeline.pipeline import Pipeline
from src.config import settings
from src.data_collection.cache_manager import CacheManager
from src.data_collection.auto_refresher import AutoRefresher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

def check_api_keys_available():
    """
    Check if API keys are available for production data collection.
    
    Returns:
        bool: True if at least some API keys are available, False otherwise
    """
    api_keys = {
        'CENSUS_API_KEY': os.environ.get('CENSUS_API_KEY') or settings.CENSUS_API_KEY,
        'FRED_API_KEY': os.environ.get('FRED_API_KEY') or settings.FRED_API_KEY,
        'CHICAGO_DATA_TOKEN': os.environ.get('CHICAGO_DATA_TOKEN') or settings.CHICAGO_DATA_TOKEN,
    }
    
    available_keys = []
    for key_name, key_value in api_keys.items():
        if key_value and key_value not in ['', 'your_census_api_key', 'your_fred_api_key', 'your_chicago_data_token']:
            available_keys.append(key_name)
    
    if not available_keys:
        logger.warning("No API keys found. Pipeline will use sample data.")
        logger.warning("To use production data, set environment variables:")
        logger.warning("  export CENSUS_API_KEY='your_actual_census_key'")
        logger.warning("  export FRED_API_KEY='your_actual_fred_key'")
        logger.warning("  export CHICAGO_DATA_TOKEN='your_actual_chicago_token'")
        return False
    else:
        logger.info(f"Found API keys for: {', '.join(available_keys)}")
        return True

def clear_cache():
    """
    Clear all cached data to force fresh data collection.
    """
    import shutil
    
    cache_dir = Path('data/cache')
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    else:
        logger.info("No cache directory found")

def check_api_configuration():
    """
    Check and display API configuration status.
    """
    print("\n=== API Configuration Status ===")
    
    api_configs = [
        {
            'name': 'Census API Key',
            'env_var': 'CENSUS_API_KEY',
            'value': os.environ.get('CENSUS_API_KEY') or settings.CENSUS_API_KEY,
            'description': 'Required for demographic and housing data'
        },
        {
            'name': 'FRED API Key',
            'env_var': 'FRED_API_KEY', 
            'value': os.environ.get('FRED_API_KEY') or settings.FRED_API_KEY,
            'description': 'Required for economic indicators'
        },
        {
            'name': 'Chicago Data Portal Token',
            'env_var': 'CHICAGO_DATA_TOKEN',
            'value': os.environ.get('CHICAGO_DATA_TOKEN') or settings.CHICAGO_DATA_TOKEN,
            'description': 'Required for building permits and business licenses'
        },
        {
            'name': 'BEA API Key',
            'env_var': 'BEA_API_KEY',
            'value': os.environ.get('BEA_API_KEY') or settings.BEA_API_KEY,
            'description': 'Optional for additional economic data'
        }
    ]
    
    for config in api_configs:
        status = "✓ SET" if config['value'] and config['value'] not in ['', f"your_{config['env_var'].lower()}"] else "✗ NOT SET"
        print(f"{config['name']:<25} ({config['env_var']:<20}): {status}")
        print(f"  {config['description']}")
        if status == "✗ NOT SET":
            print(f"  To set: export {config['env_var']}='your_actual_key'")
        print()
    
    print("=== Setup Instructions ===")
    print("1. Get API keys:")
    print("   - Census: https://api.census.gov/data/key_signup.html")
    print("   - FRED: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("   - Chicago Data Portal: https://data.cityofchicago.org/profile/app_tokens")
    print("   - BEA (optional): https://apps.bea.gov/API/signup/")
    print()
    print("2. Set environment variables:")
    print("   export CENSUS_API_KEY='your_actual_census_key'")
    print("   export FRED_API_KEY='your_actual_fred_key'")
    print("   export CHICAGO_DATA_TOKEN='your_actual_chicago_token'")
    print()
    print("3. Run the pipeline:")
    print("   python main.py                    # Uses production data if keys available")
    print("   python main.py --use-sample-data  # Forces sample data usage")

def main():
    """
    Main function to run the Chicago Housing Pipeline.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Run the Chicago Housing Pipeline')
        parser.add_argument('--output-dir', type=str, help='Directory to save outputs')
        parser.add_argument('--use-sample-data', action='store_true', 
                          help='Use sample data instead of collecting from APIs')
        parser.add_argument('--check-api-keys', action='store_true',
                          help='Check API key configuration and exit')
        parser.add_argument('--clear-cache', action='store_true',
                          help='Clear cached data and force fresh data collection')
        parser.add_argument('--refresh-datasets', nargs='+', 
                          choices=['census', 'fred', 'chicago'],
                          help='Force refresh specific datasets only')
        parser.add_argument('--cache-info', action='store_true',
                          help='Display cache information and exit')
        parser.add_argument('--start-auto-refresh', action='store_true',
                          help='Start automatic data refresh daemon')
        parser.add_argument('--stop-auto-refresh', action='store_true',
                          help='Stop automatic data refresh daemon')
        parser.add_argument('--refresh-status', action='store_true',
                          help='Show automatic refresh status and exit')
        parser.add_argument('--refresh-now', nargs='*', 
                          choices=['census', 'fred', 'chicago'],
                          help='Immediately refresh specific datasets (all if none specified)')
        args = parser.parse_args()

        # Check API keys if requested
        if args.check_api_keys:
            check_api_configuration()
            return 0

        # Initialize cache manager and auto refresher
        cache_manager = CacheManager('data/cache')
        auto_refresher = AutoRefresher('data/cache')

        # Display cache info if requested
        if args.cache_info:
            cache_info = cache_manager.get_cache_info()
            print("\n=== Cache Information ===")
            print(f"Total files: {cache_info['total_files']}")
            print(f"Total size: {cache_info['total_size_mb']:.2f} MB")
            if cache_info['total_files'] > 0:
                print(f"Oldest file: {cache_info['oldest_file']}")
                print(f"Newest file: {cache_info['newest_file']}")
                print("\nFiles by age:")
                for file_info in cache_info['files'][:10]:  # Show top 10
                    print(f"  {file_info['age_hours']:.1f}h old: {Path(file_info['path']).name} ({file_info['size_mb']:.2f} MB)")
            return 0

        # Handle auto-refresh commands
        if args.start_auto_refresh:
            logger.info("Starting automatic data refresh daemon...")
            auto_refresher.start()
            print("✓ Automatic data refresh daemon started")
            print("  - FRED data: Daily at 6:00 AM")
            print("  - Chicago data: Daily at 6:00 PM") 
            print("  - Census data: Weekly on Sunday at 2:00 AM")
            print("  - Staleness checks: Every 30 minutes")
            return 0

        if args.stop_auto_refresh:
            logger.info("Stopping automatic data refresh daemon...")
            auto_refresher.stop()
            print("✓ Automatic data refresh daemon stopped")
            return 0

        if args.refresh_status:
            status = auto_refresher.get_status()
            print("\n=== Auto-Refresh Status ===")
            print(f"Status: {'RUNNING' if status['is_running'] else 'STOPPED'}")
            print(f"Available collectors: {', '.join(status['collectors'])}")
            print(f"\nRefresh intervals (hours):")
            for source, interval in status['refresh_intervals'].items():
                print(f"  {source}: {interval}")
            print(f"\nData staleness:")
            for source, is_stale in status['data_staleness'].items():
                stale_text = "STALE" if is_stale else "FRESH"
                print(f"  {source}: {stale_text}")
            if status['last_refresh_times']:
                print(f"\nLast refresh times:")
                for source, time_str in status['last_refresh_times'].items():
                    print(f"  {source}: {time_str}")
            return 0

        if args.refresh_now is not None:
            datasets = args.refresh_now if args.refresh_now else ['census', 'fred', 'chicago']
            print(f"Refreshing datasets: {', '.join(datasets)}")
            
            for dataset in datasets:
                logger.info(f"Manually refreshing {dataset} dataset...")
                success = auto_refresher.refresh_data_source(dataset)
                status = "✓" if success else "✗"
                print(f"  {status} {dataset}: {'SUCCESS' if success else 'FAILED'}")
            
            return 0

        # Clear cache if requested
        if args.clear_cache:
            cache_manager.clear_cache()
            logger.info("Cache cleared successfully. Pipeline will collect fresh data.")

        # Refresh specific datasets if requested
        if args.refresh_datasets:
            cache_manager.force_refresh_datasets(args.refresh_datasets)
            logger.info(f"Refreshed datasets: {args.refresh_datasets}. Pipeline will collect fresh data for these.")

        # Set output directory
        output_dir = args.output_dir or settings.OUTPUT_DIR

        # Check API keys and warn if missing
        use_sample_data = args.use_sample_data or not check_api_keys_available()

        # Initialize pipeline
        logger.info(f"Initializing pipeline with output directory: {output_dir}")
        pipeline = Pipeline(output_dir=output_dir)

        # Run pipeline
        if use_sample_data:
            logger.info("Starting pipeline execution with sample data")
        else:
            logger.info("Starting pipeline execution with production data from APIs")
        results = pipeline.run(use_sample_data=use_sample_data)
        
        if isinstance(results, bool):
            if results:
                logger.info("Pipeline execution completed successfully")
                return 0
            else:
                logger.error("Pipeline execution failed")
                return 1
        elif isinstance(results, dict):
            if results.get("status") == "completed":
                logger.info("Pipeline execution completed successfully")
                return 0
            else:
                error_msg = results.get("error", "Unknown error")
                logger.error(f"Pipeline execution failed: {error_msg}")
                if "traceback" in results:
                    logger.error(results["traceback"])
                return 1
        else:
            logger.error(f"Unexpected results type: {type(results)}")
            return 1

    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
