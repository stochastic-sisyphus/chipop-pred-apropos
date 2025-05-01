"""
Configuration settings for the Chicago Population Analysis Pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
LOGS_DIR = BASE_DIR / 'logs'

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Output subdirectories
MODEL_METRICS_DIR = OUTPUT_DIR / 'model_metrics'
TRAINED_MODELS_DIR = OUTPUT_DIR / 'trained_models'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
REPORTS_DIR = OUTPUT_DIR / 'reports'
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / 'analysis_results'

# Raw data paths
CENSUS_RAW_PATH = RAW_DATA_DIR / 'census_data.csv'
PERMITS_RAW_PATH = RAW_DATA_DIR / 'building_permits.csv'
ECONOMIC_RAW_PATH = RAW_DATA_DIR / 'economic_indicators.csv'
BUSINESS_RAW_PATH = RAW_DATA_DIR / 'business_licenses.csv'

# Processed data paths
CENSUS_PROCESSED_PATH = PROCESSED_DATA_DIR / 'census_processed.csv'
PERMITS_PROCESSED_PATH = PROCESSED_DATA_DIR / 'permits_processed.csv'
ECONOMIC_PROCESSED_PATH = PROCESSED_DATA_DIR / 'economic_processed.csv'
BUSINESS_PROCESSED_PATH = PROCESSED_DATA_DIR / 'business_processed.csv'
MERGED_DATA_PATH = PROCESSED_DATA_DIR / 'merged_dataset.csv'
RETAIL_METRICS_PATH = PROCESSED_DATA_DIR / 'retail_metrics.csv'

# API keys
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
CHICAGO_DATA_TOKEN = os.getenv('CHICAGO_DATA_TOKEN')

# Analysis parameters
RETAIL_SPENDING_FACTOR = 0.3
POPULATION_GROWTH_THRESHOLD = 0.2
RETAIL_DEFICIT_THRESHOLD = 1000000

# Create all required directories
REQUIRED_DIRS = [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_METRICS_DIR,
    TRAINED_MODELS_DIR,
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
    ANALYSIS_RESULTS_DIR,
    LOGS_DIR
]

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# Directory structure
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_INTERIM_DIR = DATA_DIR / 'interim'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = OUTPUT_DIR / 'trained_models'
TRAINED_MODELS_DIR = MODELS_DIR  # Alias for backward compatibility
REPORTS_DIR = OUTPUT_DIR / 'reports'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / 'analysis_results'

# Aliases for backward compatibility
VIZ_DIR = VISUALIZATIONS_DIR
PROCESSED_DATA_DIR = DATA_PROCESSED_DIR
RAW_DATA_DIR = DATA_RAW_DIR

# Raw data file paths
CENSUS_DATA_PATH = DATA_RAW_DIR / 'census_data.csv'
PERMITS_DATA_PATH = DATA_RAW_DIR / 'building_permits.csv'
BUSINESS_LICENSES_PATH = DATA_RAW_DIR / 'business_licenses.csv'
ECONOMIC_DATA_PATH = DATA_RAW_DIR / 'economic_indicators.csv'
PROPERTY_DATA_PATH = DATA_RAW_DIR / 'property.csv'
ZONING_DATA_PATH = DATA_RAW_DIR / 'zoning.csv'
RETAIL_SALES_PATH = DATA_RAW_DIR / 'retail_sales.csv'
DEMOGRAPHIC_DATA_PATH = DATA_RAW_DIR / 'demographics.csv'

# Processed data file paths
CENSUS_PROCESSED_PATH = DATA_PROCESSED_DIR / 'census_processed.csv'
PERMITS_PROCESSED_PATH = DATA_PROCESSED_DIR / 'permits_processed.csv'
ECONOMIC_PROCESSED_PATH = DATA_PROCESSED_DIR / 'economic_processed.csv'
BUSINESS_LICENSES_PROCESSED_PATH = DATA_PROCESSED_DIR / 'business_licenses_processed.csv'
PROPERTY_PROCESSED_PATH = DATA_PROCESSED_DIR / 'property_processed.csv'
ZONING_PROCESSED_PATH = DATA_PROCESSED_DIR / 'zoning_processed.csv'
RETAIL_DEFICIT_PROCESSED_PATH = DATA_PROCESSED_DIR / 'retail_deficit.csv'
RETAIL_LEAKAGE_SUMMARY_PATH = DATA_PROCESSED_DIR / 'retail_leakage_summary.csv'
ZONING_PROPERTY_MERGED_PATH = DATA_PROCESSED_DIR / 'zoning_property_merged.csv'
MERGED_DATA_PATH = DATA_PROCESSED_DIR / 'merged_dataset.csv'

# Analysis output paths
RETAIL_DEFICIT_PATH = DATA_PROCESSED_DIR / 'retail_deficit_processed.csv'
HIGH_GROWTH_PATH = ANALYSIS_RESULTS_DIR / 'high_growth_areas.csv'
RETAIL_LEAKAGE_PATH = DATA_PROCESSED_DIR / 'retail_leakage.csv'

# Model paths
POPULATION_MODEL_PATH = MODELS_DIR / 'population_model.joblib'
ECONOMIC_MODEL_PATH = MODELS_DIR / 'economic_model.joblib'
HOUSING_MODEL_PATH = MODELS_DIR / 'housing_model.joblib'
RETAIL_MODEL_PATH = MODELS_DIR / 'retail_model.joblib'

# Report templates
REPORT_TEMPLATES_DIR = BASE_DIR / 'src' / 'templates' / 'reports'

# Report paths
EXECUTIVE_SUMMARY_PATH = REPORTS_DIR / 'EXECUTIVE_SUMMARY.md'
RETAIL_DEFICIT_REPORT_PATH = REPORTS_DIR / 'retail_deficit_analysis.md'
HOUSING_RETAIL_BALANCE_REPORT_PATH = REPORTS_DIR / 'housing_retail_balance_report.md'
TEN_YEAR_GROWTH_REPORT_PATH = REPORTS_DIR / 'ten_year_growth_analysis.md'

# Visualization paths
PERMITS_DISTRIBUTION_PATH = VISUALIZATIONS_DIR / 'permits_distribution.html'
RETAIL_DEFICIT_MAP_PATH = VISUALIZATIONS_DIR / 'retail_deficit_map.html'
ECONOMIC_INDICATORS_PATH = VISUALIZATIONS_DIR / 'economic_indicators.html'

# Create required directories
REQUIRED_DIRS = [
    DATA_RAW_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    ANALYSIS_RESULTS_DIR,
    REPORTS_DIR,
    REPORTS_DIR
]

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
CHICAGO_DATA_TOKEN = os.getenv('CHICAGO_DATA_TOKEN')

# Chicago Data Portal IDs
CHICAGO_ZIP_CODES = list(range(60601, 60827))  # Chicago ZIP code range
FRED_SERIES = {
    'CHIC917URN': 'unemployment_rate',
    'NGMP16980': 'gdp',
    'PCPI17031': 'per_capita_income',
    'CHIC917PCPI': 'personal_income'
}

ZONING_DATASET = 'pzjm-v2g4'  # Chicago zoning dataset ID
PROPERTY_TRANSACTIONS_DATASET = 'wvjz-ec8w'  # Property transactions dataset ID
BUSINESS_LICENSES_DATASET = 'r5kz-chrr'  # Business licenses dataset ID

# Analysis parameters
DEFAULT_TRAIN_YEARS = list(range(2015, 2024))  # Training years from 2015-2023
FORECAST_YEARS = range(2024, 2034)
RETAIL_SPENDING_FACTOR = 0.3  # Retail spending as proportion of income

# Model parameters
POPULATION_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

RETAIL_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'random_state': 42
}

HOUSING_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'random_state': 42
}

ECONOMIC_MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'random_state': 42
}

# Scenario parameters
SCENARIOS = {
    'base': {
        'growth_factor': 1.0,
        'confidence': 1.0
    },
    'optimistic': {
        'growth_factor': 1.2,
        'confidence': 0.9
    },
    'neutral': {
        'growth_factor': 1.0,
        'confidence': 0.8
    },
    'pessimistic': {
        'growth_factor': 0.8,
        'confidence': 0.7
    }
}

# Visualization settings
VIZ_SETTINGS = {
    'width': 1200,
    'height': 800,
    'template': 'plotly_white'
}

# Report settings
REPORT_SETTINGS = {
    'date_format': '%Y-%m-%d',
    'confidence_level': 0.95
}

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Plot settings
PLOT_STYLE = 'seaborn'
COLOR_PALETTE = 'viridis'
FIGURE_DPI = 300

# Logging settings
LOG_FILE = LOGS_DIR / 'chipop.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Export all settings
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'OUTPUT_DIR',
    'DATA_RAW_DIR', 'DATA_INTERIM_DIR', 'DATA_PROCESSED_DIR',
    'MODELS_DIR', 'TRAINED_MODELS_DIR', 'VISUALIZATIONS_DIR', 'REPORTS_DIR',
    'ANALYSIS_RESULTS_DIR', 'RAW_DATA_DIR',
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
    'RETAIL_SALES_PATH', 'DEMOGRAPHIC_DATA_PATH', 'RETAIL_DEFICIT_PROCESSED_PATH', 'RETAIL_LEAKAGE_SUMMARY_PATH', 'RETAIL_SPENDING_FACTOR',
    'EXECUTIVE_SUMMARY_PATH', 'RETAIL_DEFICIT_REPORT_PATH', 'HOUSING_RETAIL_BALANCE_REPORT_PATH', 'TEN_YEAR_GROWTH_REPORT_PATH',
    'PERMITS_DISTRIBUTION_PATH', 'RETAIL_DEFICIT_MAP_PATH', 'ECONOMIC_INDICATORS_PATH'
] 