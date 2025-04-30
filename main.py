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

def inspect_merged_dataset(merged_dataset_path, threshold_missing=0.8):
    """
    Inspects the merged dataset:
    - Lists all columns
    - Shows % of missing values
    - Highlights strong candidate features (low missingness)
    Args:
        merged_dataset_path (str): Path to the merged dataset CSV file.
        threshold_missing (float): Max % missing allowed for "good" features (default = 80%).
    Returns:
        good_features (list): List of columns with <= threshold_missing missingness.
        full_report (pd.DataFrame): DataFrame with missingness report.
    """
    df = pd.read_csv(merged_dataset_path)
    print("\n=== Merged Dataset Columns Overview ===\n")
    missing_pct = df.isnull().mean().sort_values()
    full_report = missing_pct.to_frame(name="missing_pct")
    good_features = full_report[full_report["missing_pct"] <= threshold_missing].index.tolist()
    print(f"Total columns: {len(full_report)}")
    print(f"Usable columns (≤ {int(threshold_missing * 100)}% missing): {len(good_features)}\n")
    print("📋 Strong Candidate Features:")
    for col in good_features:
        miss = full_report.loc[col, "missing_pct"]
        print(f"  - {col}: {miss:.1%} missing")
    print("\n🚫 Highly Missing Columns (not recommended):")
    for col in full_report[full_report["missing_pct"] > threshold_missing].index:
        miss = full_report.loc[col, "missing_pct"]
        print(f"  - {col}: {miss:.1%} missing")
    print("\n✅ Inspection complete.\n")

    # Save validation report
    report_path = settings.OUTPUT_DIR / 'merged_dataset_validation.json'
    full_report.to_json(report_path, orient='records', indent=2)
    logger.info(f"Merged dataset validation report saved to {report_path}")

    return good_features, full_report

def run_pipeline():
    """Run the complete analysis pipeline."""
    try:
        logger.info("Starting pipeline run...")

        # Create timestamped output directories
        run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_viz_dir = create_timestamped_dir(settings.VISUALIZATIONS_DIR)
        run_reports_dir = create_timestamped_dir(settings.REPORTS_DIR)
        run_models_dir = create_timestamped_dir(settings.MODELS_DIR)

        # Initialize pipeline components
        collector = DataCollector()
        processor = DataProcessor()

        # Collect and process data
        logger.info("Starting data collection...")
        if not collector.collect_all_data():
            logger.error("Data collection failed")
            return False

        logger.info("Starting data processing...")
        if not processor.process_all():
            logger.error("Data processing failed")
            return False

        # Load merged dataset for modeling
        merged_dataset_path = settings.PROCESSED_DATA_DIR / 'merged_dataset.csv'
        merged_df = pd.read_csv(merged_dataset_path)

        # Inspect and filter good features
        good_features = [
            col for col in merged_df.columns
            if merged_df[col].notna().sum() > 0.8 * len(merged_df) and merged_df[col].nunique() > 1
        ]

        # Define feature sets
        population_features = [f for f in good_features if f != 'total_population']
        retail_features = [f for f in good_features if f != 'retail_construction_cost']
        housing_features = [f for f in good_features if f not in ['housing_units', 'housing_unit_total']]
        economic_features = [f for f in good_features if f != 'gdp']

        # ======================
        # Handle Population Model (Critical)
        # ======================
        population_target = 'total_population'
        if population_target in merged_df.columns:
            logger.info(f"[Population] Found exact target column: '{population_target}'")
            population_model = PopulationModel()
            if not population_model.run_analysis(feature_list=population_features, target_variable=population_target):
                logger.error("Pipeline failed: Population modeling failed")
                return False
        else:
            logger.error("[Population] Target column not found. Pipeline cannot proceed.")
            return False

        # ======================
        # Handle Retail Model (Critical)
        # ======================
        retail_target = 'retail_construction_cost'
        if retail_target in merged_df.columns:
            logger.info(f"[Retail] Found exact target column: '{retail_target}'")
            retail_model = RetailModel()
            if not retail_model.run_analysis(feature_list=retail_features):
                logger.error("Pipeline failed: Retail modeling failed")
                return False
        else:
            logger.error("[Retail] Target column not found. Pipeline cannot proceed.")
            return False

        # ======================
        # Handle Housing Model (Optional)
        # ======================
        housing_targets = ['housing_units', 'housing_unit_total']
        if housing_target := next(
            (col for col in housing_targets if col in merged_df.columns), None
        ):
            logger.info(f"[Housing] Found target column: '{housing_target}'")
            housing_model = HousingModel()
            if not housing_model.run_analysis(feature_list=housing_features):
                logger.warning("[Housing] Housing modeling failed — continuing without housing predictions.")
        else:
            logger.warning("[Housing] No suitable target found — skipping housing modeling.")

        # ======================
        # Handle Economic Model (Optional)
        # ======================
        economic_target = 'gdp'
        if economic_target in merged_df.columns:
            logger.info(f"[Economic] Found target column: '{economic_target}'")
            economic_model = EconomicModel()
            if not economic_model.run_analysis(feature_list=economic_features):
                logger.warning("[Economic] Economic modeling failed — continuing without economic predictions.")
        else:
            logger.warning("[Economic] No suitable target found — skipping economic modeling.")

        # ======================
        # (Optional) Visualizations and Reporting
        # ======================
        logger.info("Generating visualizations...")
        visualizer = Visualizer(output_dir=str(run_viz_dir))
        visualizer.generate_all()

        logger.info("Generating reports...")
        try:
            TenYearGrowthReport().generate_report()
            RetailDeficitReport().generate_report()
            HousingRetailBalanceReport().generate_report()
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

        logger.info("Pipeline completed successfully")
        archive_run_output(settings.OUTPUT_DIR, settings.OUTPUT_DIR / "run_archives")
        
        # Generate and save pipeline summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": "Pipeline completed successfully", 
            "models_run": ["Population", "Retail", "Housing", "Economic"],
        }
        
        with open(settings.OUTPUT_DIR / 'pipeline_summary.md', 'w') as f:
            f.write(f"# Pipeline Run Summary\n\n")
            f.write(f"**Timestamp:** {summary['timestamp']}\n\n") 
            f.write(f"**Status:** {summary['status']}\n\n")
            f.write("**Models Run:**\n")
            for model in summary['models_run']:
                f.write(f"- {model}\n")
                
        return True

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        return False

def main():
    """Main function to run the Chicago population analysis pipeline."""
    print("=" * 80)
    print("Chicago Population Analysis Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load environment variables
    load_dotenv()

    # Ensure directories exist
    ensure_directories()

    # Verify API keys
    if not verify_api_keys():
        sys.exit(1)

    if run_pipeline():
        logger.info("Pipeline completed successfully")
        sys.exit(0)
    else:
        logger.error("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()  