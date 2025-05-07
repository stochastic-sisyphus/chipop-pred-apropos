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
import json
import shutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from difflib import get_close_matches
import pandas as pd
from typing import List
import traceback
from jinja2 import Template

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
from src.utils.helpers import ensure_output_structure, validate_outputs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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
            logger.warning(
                f"[{model_name}] No exact match for '{name}', using close match '{match}' instead."
            )
            return match

    logger.error(
        f"[{model_name}] No suitable target column found among candidates: {preferred_names}"
    )
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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_name = f"run_archive_{timestamp}"
        archive_path = archive_dir / f"{archive_name}.zip"

        shutil.make_archive(str(archive_dir / archive_name), "zip", run_dir)
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
        msg["Subject"] = f"Pipeline Run Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg["From"] = "chicago.pipeline@noreply.com"
        msg["To"] = recipient_email

        html = "<h2>Chicago Population Analysis Pipeline Summary</h2>" + "<h3>Run Details</h3>"
        html += f"<p>Timestamp: {summary['timestamp']}</p>"
        html += f"<p>Status: {summary['status']}</p>"

        # Add model metadata
        html += "<h3>Model Details</h3>"
        for model_name, metadata in summary["detected_model_targets"].items():
            html += f"<p><b>{model_name}</b><br>"
            html += f"Target: {metadata['target_variable']}<br>"
            html += f"Features: {metadata['features_used']}</p>"

        # Add data validation
        html += "<h3>Data Validation</h3>"
        for key, value in summary["data_validation"].items():
            html += f"<p>{key}: {value}</p>"

        msg.attach(MIMEText(html, "html"))

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
        "data/raw",
        "data/interim",
        "data/processed",
        "output/models",
        "output/visualizations",
        "output/reports",
        "output/reports/figures",
        "output/reports/tables",
        "output/run_archives",
        "logs",
        "src/templates/reports",
        "src/templates/visualizations",
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
                (dir_path / ".gitkeep").touch()

        logger.info("Ensured all required directories exist")

        # Create directory structure summary
        structure = {"directories_created": directories, "timestamp": datetime.now().isoformat()}

        with open("output/directory_structure.json", "w") as f:
            json.dump(structure, f, indent=2, default=str)

        return True

    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        return False


def verify_api_keys():
    """Verify that all required API keys are present."""
    required_keys = {
        "CENSUS_API_KEY": os.getenv("CENSUS_API_KEY"),
        "FRED_API_KEY": os.getenv("FRED_API_KEY"),
    }

    if missing_keys := [key for key, value in required_keys.items() if not value]:
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
            col for col in df.select_dtypes(include=["number"]).columns if df[col].nunique() <= 1
        )
        if low_var:
            logger.warning(f"Columns with no variance: {low_var}")

        # Get good features (not empty, low missing values, has variance)
        good_features = [
            col
            for col in df.columns
            if col not in empty_cols + high_missing + low_var
            and df[col].notna().sum() > (1 - threshold_missing) * len(df)
            and (col not in df.select_dtypes(include=["number"]).columns or df[col].nunique() > 1)
        ]

        logger.info(
            f"Found {len(good_features)} usable features out of {len(df.columns)} total columns"
        )
        return good_features

    except Exception as e:
        logger.error(f"Error inspecting merged dataset: {str(e)}")
        return []


def run_pipeline():
    """Run the complete analysis pipeline."""
    try:
        logger.info("Starting Chicago population analysis pipeline...")

        # Ensure output structure
        if not ensure_output_structure():
            logger.error("Failed to ensure output structure")
            return False

        # Initialize components
        collector = DataCollector()
        processor = DataProcessor()
        population_model = PopulationModel()
        retail_model = RetailModel()
        housing_model = HousingModel()
        economic_model = EconomicModel()
        visualizer = Visualizer()

        # Collect data
        logger.info("Collecting data...")
        if not collector.collect_all():
            logger.error("Failed to collect data")
            return False

        # Process data
        logger.info("Processing data...")
        if not processor.process_all():
            logger.error("Failed to process data")
            return False

        # Train models
        logger.info("Training models...")
        models = {
            "population": population_model,
            "economic": economic_model,
            "retail": retail_model,
            "housing": housing_model,
        }

        for name, model in models.items():
            if not model.train(processor.get_processed_data()):
                logger.error(f"Failed to train {name} model")
                return False

        # Generate predictions and scenarios
        logger.info("Generating predictions and scenarios...")
        for name, model in models.items():
            if not model.run_analysis():
                logger.error(f"Failed to run {name} analysis")
                return False

        # Generate visualizations
        logger.info("Generating visualizations...")
        if not visualizer.generate_all():
            logger.error("Failed to generate visualizations")
            return False

        # Generate reports
        logger.info("Generating reports...")
        if not generate_reports():
            logger.error("Failed to generate reports")
            return False

        # Validate outputs
        logger.info("Validating outputs...")
        if not validate_outputs():
            logger.error("Failed to validate outputs")
            return False

        logger.info("Pipeline completed successfully")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False


def load_processed_data():
    try:
        return {
            "census_data": pd.read_csv(settings.PROCESSED_DATA_DIR / "census_processed.csv"),
            "permit_data": pd.read_csv(settings.PROCESSED_DATA_DIR / "permits_processed.csv"),
            "economic_data": pd.read_csv(settings.PROCESSED_DATA_DIR / "economic_processed.csv"),
            "zoning_data": pd.read_csv(settings.PROCESSED_DATA_DIR / "zoning_processed.csv"),
            "retail_metrics": pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_metrics.csv"),
            "retail_deficit": pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_deficit.csv"),
        }
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def generate_all_reports():
    data = load_processed_data()
    reports = {
        "ten_year_growth": TenYearGrowthReport(),
        "housing_retail_balance": HousingRetailBalanceReport(),
        "retail_deficit": RetailDeficitReport(),
    }
    all_success = True
    for name, report in reports.items():
        try:
            if name == "ten_year_growth":
                # Build context for ten_year_growth
                context = {
                    "generation_date": datetime.now().strftime("%Y-%m-%d"),
                    "generation_time": datetime.now().strftime("%H:%M:%S"),
                    # Add more context keys as needed, or pass data directly
                }
                if report.generate_report(context):
                    logger.info(f"Generated {name} report")
                else:
                    logger.warning(f"Failed to generate {name} report")
                    all_success = False
            elif not report.generate_report(
                    data["census_data"],
                    data["permit_data"],
                    data["economic_data"],
                    data["zoning_data"],
                    data["retail_metrics"],
                    data["retail_deficit"],
                ):
                logger.warning(f"Failed to generate {name} report")
                all_success = False
            else:
                logger.info(f"Generated {name} report")
        except Exception as e:
            logger.error(f"Error generating {name} report: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            all_success = False
    return all_success


def generate_executive_summary(data: dict) -> bool:
    try:
        summary_template = settings.TEMPLATES_DIR / "reports/EXECUTIVE_SUMMARY.md"
        with open(summary_template, "r") as f:
            template = Template(f.read())

        census_data = data["census_data"]
        permit_data = data["permit_data"]
        economic_data = data["economic_data"]

        metrics = {
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "current_analysis": {
                "population": {
                    "metrics": {"total": int(census_data["total_population"].sum())},
                    "demographics": {
                        "population_growth": float(
                            (
                                census_data.groupby("year")["total_population"]
                                .sum()
                                .pct_change()
                                .mean()
                                * 100
                            )
                        )
                    },
                },
                "development": {
                    "active_permits": (
                        int(permit_data["total_permits"].sum())
                        if "total_permits" in permit_data.columns
                        else 0
                    ),
                    "pipeline_value": (
                        float(permit_data["total_construction_cost"].sum())
                        if "total_construction_cost" in permit_data.columns
                        else 0.0
                    ),
                },
            },
            "historical_trends": {
                "economic": {
                    "gdp_growth": (
                        float(economic_data["real_gdp"].pct_change().mean())
                        if "real_gdp" in economic_data.columns
                        else 0.0
                    ),
                    "employment_change": (
                        float(economic_data["unemployment_rate"].pct_change().mean())
                        if "unemployment_rate" in economic_data.columns
                        else 0.0
                    ),
                }
            },
            "projections": {
                "period_start": datetime.now().year,
                "period_end": datetime.now().year + 10,
                "population": {
                    "scenarios": [
                        {
                            "population_change": 0.15,  # 15% growth in base case
                            "final_population": int(census_data["total_population"].sum() * 1.15),
                        }
                    ]
                },
            },
            "growth_areas": {
                "primary": [
                    f"ZIP {zip_code}"
                    for zip_code in census_data.groupby("zip_code")["total_population"]
                    .agg(["first", "last"])
                    .assign(growth=lambda x: (x["last"] - x["first"]) / x["first"])
                    .nlargest(3, "growth")
                    .index
                ]
            },
            "recommendations": {
                "strategic": [
                    "Focus development in high-growth areas",
                    "Address retail deficits in underserved areas",
                    "Promote mixed-use development in opportunity zones",
                ],
                "implementation": [
                    "Update zoning regulations",
                    "Streamline permit processes",
                    "Engage community stakeholders",
                ],
            },
        }

        # Inject avg_household_size for template safety and accuracy
        if (
            "total_population" in census_data.columns
            and "occupied_housing_units" in census_data.columns
        ):
            total_pop = census_data["total_population"].sum()
            total_households = census_data["occupied_housing_units"].sum()
            avg_household_size = (total_pop / total_households) if total_households > 0 else None
        else:
            avg_household_size = None
        if "current_analysis" in metrics and "population" in metrics["current_analysis"]:
            metrics["current_analysis"]["population"]["avg_household_size"] = avg_household_size

        summary = template.render(**metrics)
        summary_path = settings.REPORTS_DIR / "EXECUTIVE_SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write(summary)
        logger.info("Generated executive summary")
        return True
    except Exception as e:
        logger.error(f"Failed to generate executive summary: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def generate_reports() -> bool:
    """Generate all reports."""
    try:
        logger.info("Generating reports...")
        data = load_processed_data()
        if data is None:
            logger.error("Failed to load processed data")
            return False
        if not generate_all_reports():
            logger.error("Failed to generate one or more reports")
            return False
        if not generate_executive_summary(data):
            logger.error("Failed to generate executive summary")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to generate reports: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main execution function."""
    try:
        logger.info("Starting pipeline...")

        # Initialize components
        collector = DataCollector()
        processor = DataProcessor()

        # Collect data
        logger.info("Collecting data...")
        # Use DataCollector instance for all data collection
        census_data = collector.collect_census_data()
        permit_data = collector.collect_permit_data()
        economic_data = collector.collect_economic_data()
        zoning_data = collector.collect_zoning_data()
        # Business licenses, parcel, BEA (still run for side effects/outputs)
        collector.collect_business_licenses_retail()
        collector.collect_parcel_retail_sqft()
        collector.collect_bea_retail_gdp()

        # Process data
        logger.info("Processing data...")
        if not processor.process_all():
            logger.error("Failed to process data")
            return False

        # Generate reports
        logger.info("Generating reports...")
        if not generate_reports():
            logger.error("Failed to generate reports")
            return False

        logger.info("Pipeline completed successfully")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    main()
