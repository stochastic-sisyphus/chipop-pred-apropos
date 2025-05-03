"""
Configuration settings for the Chicago Population Analysis Pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
MODELS_DIR = BASE_DIR / 'models'

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Output subdirectories
PREDICTIONS_DIR = OUTPUT_DIR / 'predictions'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
REPORTS_DIR = OUTPUT_DIR / 'reports'
MODEL_METRICS_DIR = OUTPUT_DIR / 'model_metrics'
TRAINED_MODELS_DIR = MODELS_DIR / 'trained'

# Required CSV outputs
REQUIRED_OUTPUTS = {
    'feature_importance.csv': PREDICTIONS_DIR,
    'population_shift_patterns.csv': PREDICTIONS_DIR,
    'retail_deficit_predictions.csv': PREDICTIONS_DIR,
    'scenario_predictions.csv': PREDICTIONS_DIR,
    'scenario_summary.csv': PREDICTIONS_DIR,
    'retail_spending_leakage.csv': PREDICTIONS_DIR,
    'ten_year_growth_areas.csv': PREDICTIONS_DIR,
    'emerging_housing_areas.csv': PREDICTIONS_DIR,
    'retail_housing_opportunity.csv': PREDICTIONS_DIR,
    'downtown_comparison.csv': PREDICTIONS_DIR,
    'zip_summary.csv': PREDICTIONS_DIR,
    'high_leakage_areas.csv': PREDICTIONS_DIR,
    'lowest_retail_provision.csv': PREDICTIONS_DIR,
    'top_impacted_areas.csv': PREDICTIONS_DIR,
    'model_metrics.csv': MODEL_METRICS_DIR,
    'retail_deficit_feature_importance.csv': MODEL_METRICS_DIR
}

# Required reports
REQUIRED_REPORTS = {
    'EXECUTIVE_SUMMARY.md': REPORTS_DIR,
    'chicago_population_analysis_report.md': REPORTS_DIR,
    'economic_impact_analysis.md': REPORTS_DIR,
    'housing_retail_balance_report.md': REPORTS_DIR,
    'chicago_zip_summary.md': REPORTS_DIR,
    'ten_year_growth_analysis.md': REPORTS_DIR
}

# Required visualizations
REQUIRED_VISUALIZATIONS = {
    'permits_by_year.png': VISUALIZATIONS_DIR,
    'permits_by_community.png': VISUALIZATIONS_DIR,
    'permit_costs_distribution.png': VISUALIZATIONS_DIR,
    'population_changes_distribution.png': VISUALIZATIONS_DIR,
    'median_income_trend.png': VISUALIZATIONS_DIR,
    'total_population_trend.png': VISUALIZATIONS_DIR,
    'retail_deficit_feature_importance.png': VISUALIZATIONS_DIR,
    'top_retail_deficit_areas.png': VISUALIZATIONS_DIR,
    'housing_vs_retail_scatter.png': VISUALIZATIONS_DIR
}

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
    PREDICTIONS_DIR,
    MODELS_DIR
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
PROPERTY_DATA_PATH = DATA_RAW_DIR / 'property_transactions.csv'
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
CHICAGO_ZIP_CODES = [
    '60601', '60602', '60603', '60604', '60605', '60606', '60607', '60608',
    '60609', '60610', '60611', '60612', '60613', '60614', '60615', '60616',
    '60617', '60618', '60619', '60620', '60621', '60622', '60623', '60624',
    '60625', '60626', '60628', '60629', '60630', '60631', '60632', '60633',
    '60634', '60636', '60637', '60638', '60639', '60640', '60641', '60642',
    '60643', '60644', '60645', '60646', '60647', '60649', '60651', '60652',
    '60653', '60654', '60655', '60656', '60657', '60659', '60660', '60661',
    '60666', '60707', '60827'
]

FRED_SERIES = {
    'CHIC917URN': 'unemployment_rate',
    'NGMP16980': 'real_gdp',
    'PCPI17031': 'per_capita_income',
    'CHIC917PCPI': 'personal_income'
}

ZONING_DATASET = 'dj47-wfun'
PROPERTY_TRANSACTIONS_DATASET = 'wvjp-8m67'
BUSINESS_LICENSES_DATASET = 'r5kz-chrr'

# Analysis parameters
DEFAULT_TRAIN_YEARS = list(range(2015, 2026))
FORECAST_YEARS = range(2024, 2034)
RETAIL_SPENDING_FACTOR = 0.3

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
LOG_FILE = MODELS_DIR / 'chipop.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Add TEMPLATES_DIR after the other directory settings
TEMPLATES_DIR = BASE_DIR / 'src' / 'templates'

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
    'PERMITS_DISTRIBUTION_PATH', 'RETAIL_DEFICIT_MAP_PATH', 'ECONOMIC_INDICATORS_PATH',
    'PREDICTIONS_DIR',
    'TEMPLATES_DIR'
] 