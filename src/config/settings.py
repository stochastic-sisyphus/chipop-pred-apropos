"""
Configuration settings for the Chicago Population Analysis Pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Output directories
OUTPUT_DIR = BASE_DIR / 'output'
MODELS_DIR = OUTPUT_DIR / 'models'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
REPORTS_DIR = OUTPUT_DIR / 'reports'
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / 'analysis_results'
MODEL_METRICS_DIR = OUTPUT_DIR / 'model_metrics'
PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
TRAINED_MODELS_DIR = OUTPUT_DIR / 'trained_models'

# Logs directory
LOGS_DIR = BASE_DIR / 'logs'

# Data file paths
CENSUS_DATA_PATH = RAW_DATA_DIR / "census_data.csv"
PERMITS_DATA_PATH = RAW_DATA_DIR / "building_permits.csv"
BUSINESS_LICENSES_PATH = RAW_DATA_DIR / "business_licenses.csv"
ECONOMIC_DATA_PATH = RAW_DATA_DIR / "economic_indicators.csv"

# Processed data paths
CENSUS_PROCESSED_PATH = PROCESSED_DATA_DIR / "census_processed.csv"
PERMITS_PROCESSED_PATH = PROCESSED_DATA_DIR / "permits_processed.csv"
ECONOMIC_PROCESSED_PATH = PROCESSED_DATA_DIR / "economic_processed.csv"
BUSINESS_LICENSES_PROCESSED_PATH = PROCESSED_DATA_DIR / "business_licenses_processed.csv"
PROPERTY_PROCESSED_PATH = PROCESSED_DATA_DIR / "property_processed.csv"
ZONING_PROCESSED_PATH = PROCESSED_DATA_DIR / "zoning_processed.csv"
RETAIL_DEFICIT_PROCESSED_PATH = PROCESSED_DATA_DIR / "retail_deficit_processed.csv"
MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_dataset.csv"

# API Keys
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
CHICAGO_DATA_TOKEN = os.getenv("CHICAGO_DATA_TOKEN")

# Create directories if they don't exist
REQUIRED_DIRS = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
    ANALYSIS_RESULTS_DIR,
    MODEL_METRICS_DIR,
    TRAINED_MODELS_DIR,
    LOGS_DIR
]

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# Model paths
POPULATION_MODEL_PATH = MODELS_DIR / "population_model.joblib"
ECONOMIC_MODEL_PATH = MODELS_DIR / "economic_model.joblib"
HOUSING_MODEL_PATH = MODELS_DIR / "housing_model.joblib"
RETAIL_MODEL_PATH = MODELS_DIR / "retail_model.joblib"

# Analysis output paths
RETAIL_DEFICIT_PATH = ANALYSIS_RESULTS_DIR / "retail_deficit_areas.csv"
HIGH_GROWTH_PATH = ANALYSIS_RESULTS_DIR / "high_growth_areas.csv"
RETAIL_LEAKAGE_PATH = ANALYSIS_RESULTS_DIR / "retail_leakage_areas.csv"

# Report templates
REPORT_TEMPLATES_DIR = BASE_DIR / "src" / "templates" / "reports"

# Chicago Data Portal IDs
CHICAGO_ZIP_CODES = list(range(60601, 60827))  # Chicago ZIP code range
FRED_SERIES = {
    "CHIC917URN": "unemployment_rate",
    "NGMP16980": "gdp",
    "PCPI17031": "per_capita_income",
    "CHIC917PCPI": "personal_income"
}

ZONING_DATASET = "8v9j-bter"  # Chicago zoning dataset ID
PROPERTY_TRANSACTIONS_DATASET = "wvjz-ec8w"  # Property transactions dataset ID
BUSINESS_LICENSES_DATASET = "r5kz-chrr"  # Business licenses dataset ID

# Analysis parameters
DEFAULT_TRAIN_YEARS = list(range(2015, 2024))  # Training years from 2015-2023
FORECAST_YEARS = range(2024, 2034)

# Model parameters
POPULATION_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

RETAIL_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "random_state": 42
}

HOUSING_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "random_state": 42
}

ECONOMIC_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "random_state": 42
}

# Scenario parameters
SCENARIOS = {
    "optimistic": {
        "growth_factor": 1.2,
        "confidence": 0.9
    },
    "neutral": {
        "growth_factor": 1.0,
        "confidence": 0.8
    },
    "pessimistic": {
        "growth_factor": 0.8,
        "confidence": 0.7
    }
}

# Visualization settings
VIZ_SETTINGS = {
    "width": 1200,
    "height": 800,
    "template": "plotly_white"
}

# Report settings
REPORT_SETTINGS = {
    "date_format": "%Y-%m-%d",
    "confidence_level": 0.95
}

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Plot settings
PLOT_STYLE = "seaborn"
COLOR_PALETTE = "viridis"
FIGURE_DPI = 300

# Report settings
REPORT_TEMPLATE_DIR = BASE_DIR / "src" / "templates" / "reports"
REPORT_DATE_FORMAT = "%Y-%m-%d"

# Logging settings
LOG_FILE = LOGS_DIR / "chipop.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Raw data file paths
CENSUS_DATA_PATH = RAW_DATA_DIR / 'census_data.csv'
PERMITS_DATA_PATH = RAW_DATA_DIR / 'permits.csv'
ECONOMIC_DATA_PATH = RAW_DATA_DIR / 'economic_indicators.csv'
RETAIL_DEFICIT_PATH = RAW_DATA_DIR / 'retail_deficit.csv'
BUSINESS_LICENSES_PATH = RAW_DATA_DIR / 'business_licenses.csv'
ZONING_DATA_PATH = RAW_DATA_DIR / 'zoning.csv'
PROPERTY_DATA_PATH = RAW_DATA_DIR / 'property.csv'

# Processed data file paths
CENSUS_PROCESSED_PATH = PROCESSED_DATA_DIR / 'census_processed.csv'
PERMITS_PROCESSED_PATH = PROCESSED_DATA_DIR / 'permits_processed.csv'
ECONOMIC_PROCESSED_PATH = PROCESSED_DATA_DIR / 'economic_processed.csv'
RETAIL_DEFICIT_PROCESSED_PATH = PROCESSED_DATA_DIR / 'retail_deficit_processed.csv'
BUSINESS_LICENSES_PROCESSED_PATH = PROCESSED_DATA_DIR / 'business_licenses_processed.csv'
ZONING_PROCESSED_PATH = PROCESSED_DATA_DIR / 'zoning_processed.csv'
PROPERTY_PROCESSED_PATH = PROCESSED_DATA_DIR / 'property_processed.csv'
ZONING_PROPERTY_MERGED_PATH = PROCESSED_DATA_DIR / 'zoning_property_merged.csv'

# Retail Analysis Settings
RETAIL_SALES_PATH = RAW_DATA_DIR / 'retail_sales.csv'
DEMOGRAPHIC_DATA_PATH = RAW_DATA_DIR / 'demographics.csv'
RETAIL_DEFICIT_PROCESSED_PATH = PROCESSED_DATA_DIR / 'retail_deficit.csv'
RETAIL_LEAKAGE_SUMMARY_PATH = PROCESSED_DATA_DIR / 'retail_leakage_summary.csv'

# Retail Analysis Constants
RETAIL_SPENDING_FACTOR = 0.30  # Average percentage of income spent on retail

# Export all settings
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'OUTPUT_DIR',
    'RAW_DATA_DIR', 'INTERIM_DATA_DIR', 'PROCESSED_DATA_DIR',
    'MODELS_DIR', 'VISUALIZATIONS_DIR', 'REPORTS_DIR',
    'ANALYSIS_RESULTS_DIR', 'MODEL_METRICS_DIR', 'PREDICTIONS_DIR',
    'TRAINED_MODELS_DIR', 'LOGS_DIR',
    'CENSUS_DATA_PATH', 'PERMITS_DATA_PATH', 'BUSINESS_LICENSES_PATH', 'ECONOMIC_DATA_PATH',
    'CENSUS_PROCESSED_PATH', 'PERMITS_PROCESSED_PATH', 'ECONOMIC_PROCESSED_PATH', 'BUSINESS_LICENSES_PROCESSED_PATH', 'MERGED_DATA_PATH',
    'POPULATION_MODEL_PATH', 'ECONOMIC_MODEL_PATH', 'HOUSING_MODEL_PATH', 'RETAIL_MODEL_PATH',
    'RETAIL_DEFICIT_PATH', 'HIGH_GROWTH_PATH', 'RETAIL_LEAKAGE_PATH',
    'REPORT_TEMPLATES_DIR',
    'CHICAGO_ZIP_CODES', 'FRED_SERIES', 'ZONING_DATASET', 'PROPERTY_TRANSACTIONS_DATASET', 'BUSINESS_LICENSES_DATASET',
    'DEFAULT_TRAIN_YEARS', 'FORECAST_YEARS',
    'POPULATION_MODEL_PARAMS', 'RETAIL_MODEL_PARAMS', 'HOUSING_MODEL_PARAMS', 'ECONOMIC_MODEL_PARAMS',
    'SCENARIOS', 'VIZ_SETTINGS', 'REPORT_SETTINGS',
    'RANDOM_STATE', 'TEST_SIZE', 'CV_FOLDS',
    'PLOT_STYLE', 'COLOR_PALETTE', 'FIGURE_DPI',
    'REPORT_TEMPLATE_DIR', 'REPORT_DATE_FORMAT', 'LOG_FILE', 'LOG_FORMAT', 'LOG_DATE_FORMAT',
    'RETAIL_SALES_PATH', 'DEMOGRAPHIC_DATA_PATH', 'RETAIL_DEFICIT_PROCESSED_PATH', 'RETAIL_LEAKAGE_SUMMARY_PATH', 'RETAIL_SPENDING_FACTOR'
] 