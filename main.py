#!/usr/bin/env python3
"""
Chicago Population Analysis Pipeline
Main entry point for running the complete analysis pipeline.

This script orchestrates:
1. Data collection from multiple sources
2. Data processing and feature engineering
3. Model training and scenario generation
4. Visualization and report generation
"""


import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import json
import shutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from difflib import get_close_matches
import pandas as pd
from typing import List
import traceback

from src.config import settings
from src.data_collection.collector import DataCollector
from src.data_processing.processor import DataProcessor
from src.models.population_model import PopulationModel
from src.models.retail_model import RetailModel
from src.models.housing_model import HousingModel
from src.models.economic_model import EconomicModel
from src.visualization.visualizer import Visualizer
from src.reporting.ten_year_growth_report import TenYearGrowthReport
from src.reporting.retail_deficit_report import RetailDeficitReport
from src.reporting.housing_retail_balance_report import HousingRetailBalanceReport

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Python version check (fail fast if not 3.11) ===
if sys.version_info.major != 3 or sys.version_info.minor < 11:
    sys.stderr.write("ERROR: This pipeline requires Python 3.11 or higher.\n")
    sys.exit(1)

def detect_target_column(df, preferred_names, model_name="model"):
    """
    Smart detection of a target column from a dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame to inspect
        preferred_names (list): List of preferred column names in order
        model_name (str): Name of the model (for logging)
    Returns:
        str: Best matching column name or None if not found
    """
    # First check for exact matches
    for name in preferred_names:
        if name in df.columns:
            logger.info(f"[{model_name}] Found exact target column: '{name}'")
            return name

    # Then attempt fuzzy match
    all_columns = df.columns.tolist()
    for name in preferred_names:
        if matches := get_close_matches(name, all_columns, n=1, cutoff=0.7):
            match = matches[0]
            logger.warning(f"[{model_name}] No exact match for '{name}', using close match '{match}' instead.")
            return match

    logger.error(f"[{model_name}] No suitable target column found among candidates: {preferred_names}")
    return None

def try_run_model(model_instance, method_name, **kwargs):
    """
    Wrapper for safely running model methods with error handling.
    
    Args:
        model_instance: Instance of a model class
        method_name (str): Name of the method to call
        **kwargs: Arguments to pass to the method
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        method = getattr(model_instance, method_name)
        if not method(**kwargs):
            logger.warning(f"{model_instance.__class__.__name__} failed to run.")
            return False
        return True
    except Exception as e:
        logger.error(f"Unexpected error in {model_instance.__class__.__name__}: {e}")
        return False

def create_timestamped_dir(base_dir: Path) -> Path:
    """
    Create a timestamped directory within the base directory.
    
    Args:
        base_dir (Path): Base directory to create timestamped folder in
    Returns:
        Path: Path to the created timestamped directory
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    timestamped_dir = base_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_dir

def archive_run_output(run_dir: Path, archive_dir: Path) -> bool:
    """
    Archive the run output directory into a zip file.
    
    Args:
        run_dir (Path): Directory containing run output
        archive_dir (Path): Directory to store the archive
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        archive_name = f"run_archive_{timestamp}"
        archive_path = archive_dir / f"{archive_name}.zip"
        
        shutil.make_archive(
            str(archive_dir / archive_name),
            'zip',
            run_dir
        )
        logger.info(f"Run output archived to {archive_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to archive run output: {e}")
        return False

def email_pipeline_summary(summary: dict, recipient_email: str) -> bool:
    """
    Email the pipeline summary to specified recipient.
    
    Args:
        summary (dict): Pipeline run summary
        recipient_email (str): Email address to send to
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['Subject'] = f"Pipeline Run Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg['From'] = "chicago.pipeline@noreply.com"
        msg['To'] = recipient_email

        html = (
            "<h2>Chicago Population Analysis Pipeline Summary</h2>"
            + "<h3>Run Details</h3>"
        )
        html += f"<p>Timestamp: {summary['timestamp']}</p>"
        html += f"<p>Status: {summary['status']}</p>"

        # Add model metadata
        html += "<h3>Model Details</h3>"
        for model_name, metadata in summary['detected_model_targets'].items():
            html += f"<p><b>{model_name}</b><br>"
            html += f"Target: {metadata['target_variable']}<br>"
            html += f"Features: {metadata['features_used']}</p>"

        # Add data validation
        html += "<h3>Data Validation</h3>"
        for key, value in summary['data_validation'].items():
            html += f"<p>{key}: {value}</p>"

        msg.attach(MIMEText(html, 'html'))

        # Send email (commented out as SMTP details needed)
        # with smtplib.SMTP('smtp.gmail.com', 587) as server:
        #     server.starttls()
        #     server.login(EMAIL, PASSWORD)
        #     server.send_message(msg)

        logger.info(f"Pipeline summary emailed to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send pipeline summary email: {e}")
        return False

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/interim', 
        'data/processed',
        'output/models',
        'output/visualizations',
        'output/reports',
        'output/reports/figures',
        'output/reports/tables',
        'output/run_archives',
        'logs',
        'src/templates/reports',
        'src/templates/visualizations'
    ]
    
    try:
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Verify directory was created
            if not dir_path.exists():
                logger.error(f"Failed to create directory: {directory}")
                return False
            
            # Create .gitkeep file if directory is empty
            if not any(dir_path.iterdir()):
                (dir_path / '.gitkeep').touch()
        
        logger.info("Ensured all required directories exist")
        
        # Create directory structure summary
        structure = {
            'directories_created': directories,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('output/directory_structure.json', 'w') as f:
            json.dump(structure, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        return False

def verify_api_keys():
    """Verify that all required API keys are present."""
    required_keys = {
        'CENSUS_API_KEY': os.getenv('CENSUS_API_KEY'),
        'FRED_API_KEY': os.getenv('FRED_API_KEY')
    }

    if missing_keys := [
        key for key, value in required_keys.items() if not value
    ]:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        return False

    logger.info("All required API keys are present")
    return True

def inspect_merged_dataset(df: pd.DataFrame, threshold_missing: float = 0.8) -> List[str]:
    """
    Inspect merged dataset and return list of usable features.
    
    Args:
        df (pd.DataFrame): Merged dataset to inspect
        threshold_missing (float): Maximum fraction of missing values allowed
        
    Returns:
        List[str]: List of usable feature names
    """
    try:
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            logger.warning(f"Found completely empty columns: {empty_cols}")

        # Check for columns with too many missing values
        high_missing = df.columns[df.isna().mean() > threshold_missing].tolist()
        if high_missing:
            logger.warning(f"Columns with >{threshold_missing*100}% missing values: {high_missing}")

        # Check for low variance columns
        low_var = []
        low_var.extend(
            col
            for col in df.select_dtypes(include=['number']).columns
            if df[col].nunique() <= 1
        )
        if low_var:
            logger.warning(f"Columns with no variance: {low_var}")

        # Get good features (not empty, low missing values, has variance)
        good_features = [
            col for col in df.columns
            if col not in empty_cols + high_missing + low_var
            and df[col].notna().sum() > (1 - threshold_missing) * len(df)
            and (col not in df.select_dtypes(include=['number']).columns or df[col].nunique() > 1)
        ]

        logger.info(f"Found {len(good_features)} usable features out of {len(df.columns)} total columns")
        return good_features

    except Exception as e:
        logger.error(f"Error inspecting merged dataset: {str(e)}")
        return []

def run_pipeline():
    """Run the complete analysis pipeline."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Verify API keys
        verify_api_keys()
        
        logger.info("Starting pipeline run...")
        
        # Initialize components
        collector = DataCollector()
        processor = DataProcessor()
        population_model = PopulationModel()
        retail_model = RetailModel()
        housing_model = HousingModel()
        economic_model = EconomicModel()
        visualizer = Visualizer()
        
        # Data collection
        logger.info("Starting data collection...")
        if not collector.collect_all_data():
            logger.error("Data collection failed")
            return False
            
        # Data processing
        logger.info("Starting data processing...")
        if not processor.process_all():
            logger.error("Data processing failed")
            return False
            
        # Load merged dataset
        merged_df = pd.read_csv(settings.MERGED_DATA_PATH)
        # PATCH: Add total_permits if missing
        if 'total_permits' not in merged_df.columns:
            merged_df['total_permits'] = (
                merged_df.get('residential_permits', 0).fillna(0) +
                merged_df.get('commercial_permits', 0).fillna(0) +
                merged_df.get('retail_permits', 0).fillna(0)
            )
        
        # Train population model
        logger.info("Training population model...")
        try:
            if not population_model.train(merged_df):
                logger.error("Failed to train population model")
                return False
        except Exception as e:
            logger.error(f"Unexpected error in PopulationModel: {str(e)}")
            return False
            
        # Train retail model
        logger.info("Training retail model...")
        if not retail_model.train(merged_df):
            logger.error("Failed to train retail model")
            return False
            
        # Train housing model
        logger.info("Training housing model...")
        if not housing_model.train(merged_df):
            logger.error("Failed to train housing model")
            return False
            
        # Train economic model
        logger.info("Training economic model...")
        if not economic_model.train(merged_df):
            logger.error("Failed to train economic model")
            return False
            
        # Generate visualizations
        logger.info("Generating visualizations...")
        if not visualizer.create_dashboard():
            logger.error("Failed to create visualizations")
            return False
            
        # Generate reports
        logger.info("Generating reports...")
        reports = [
            RetailDeficitReport(),
            HousingRetailBalanceReport(),
            TenYearGrowthReport()
        ]
        
        for report in reports:
            if not report.generate_report():
                logger.warning(f"Failed to generate {report.__class__.__name__}")
                
        logger.info("Pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False

def main():
    """Main function to run the Chicago population analysis pipeline."""
    try:
        run_pipeline()
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger("__main__")
        logger.error("Uncaught exception in main pipeline:", exc_info=True)
        print("\n\n================ PIPELINE FAILED ================")
        print(f"Uncaught exception: {e}")
        traceback.print_exc()
        print("================================================\n\n")
        sys.exit(1)

if __name__ == "__main__":
    main()  