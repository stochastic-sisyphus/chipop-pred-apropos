#!/usr/bin/env python3
"""
ChiPop: Chicago Population Analysis Runner
This script provides a convenient way to run the Chicago population analysis pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chipop.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure necessary directories exist"""
    for directory in ['data', 'output', 'output/models', 'output/visualizations', 'output/reports']:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Ensured all required directories exist")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ChiPop: Chicago Population Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--steps", 
        choices=["all", "data", "process", "population", "analysis", "model", "visualize", "reports"],
        default="all",
        help="Specify which steps of the pipeline to run"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the Chicago population analysis pipeline"""
    args = parse_args()
    
    # Print welcome message
    print("="*80)
    print("ChiPop: Chicago Population Analysis Tool")
    print("="*80)
    
    # Ensure directories exist
    ensure_directories()
    
    # Import here to avoid circular imports
    from src.pipeline import run_pipeline_step, run_full_pipeline
    
    if args.steps == "all":
        success = run_full_pipeline()
    else:
        logger.info(f"Running {args.steps} step")
        success = run_pipeline_step(args.steps)
    
    if success:
        logger.info(f"ChiPop {'pipeline' if args.steps == 'all' else args.steps} completed successfully")
    else:
        logger.error(f"ChiPop {'pipeline' if args.steps == 'all' else args.steps} failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())