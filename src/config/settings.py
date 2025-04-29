"""
Configuration settings for the Chicago population analysis pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
CHICAGO_DATA_TOKEN = os.getenv('CHICAGO_DATA_TOKEN')

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
SRC_DIR = BASE_DIR / 'src'

# Data directories
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Output directories
MODELS_DIR = OUTPUT_DIR / 'models'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
REPORTS_DIR = OUTPUT_DIR / 'reports'
PREDICTIONS_DIR = MODELS_DIR / 'predictions'

# Template directories
REPORT_TEMPLATES_DIR = SRC_DIR / 'templates' / 'reports'
TEMPLATE_DIR = SRC_DIR / 'templates'

# Model directories
TRAINED_MODELS_DIR = MODELS_DIR / 'trained'
MODEL_METRICS_DIR = MODELS_DIR / 'metrics'

# Report subdirectories
REPORT_FIGURES_DIR = REPORTS_DIR / 'figures'
REPORT_TABLES_DIR = REPORTS_DIR / 'tables'

# Analysis results directory
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / 'analysis_results'

# Ensure all directories exist
for directory in [
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR,
    TEMPLATE_DIR, REPORT_TEMPLATES_DIR,
    TRAINED_MODELS_DIR, MODEL_METRICS_DIR, PREDICTIONS_DIR,
    REPORT_FIGURES_DIR, REPORT_TABLES_DIR, ANALYSIS_RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Data collection settings
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

# FRED API series mappings
FRED_SERIES = {
    'CHIC917URN': 'unemployment_rate',
    'NGMP16980': 'gdp',
    'PCPI17031': 'per_capita_income',
    'CHIC917PCPI': 'personal_income'
}

# Chicago Data Portal dataset IDs
PERMITS_DATASET = "ydr8-5enu"
ZONING_DATASET = "8v9j-bter"
BUSINESS_LICENSES_DATASET = "r5kz-chrr"
PROPERTY_TRANSACTIONS_DATASET = "wvjz-am8w"

# Analysis settings
DEFAULT_TRAIN_YEARS = range(2015, 2024)  # Training data from 2015-2023
FORECAST_YEARS = range(2024, 2034)  # 10-year forecast period

# Model parameters
POPULATION_MODEL_PARAMS = {
    "forecast_periods": 10,
    "confidence_level": 0.95,
    "scenarios": ["base", "high_growth", "low_growth"],
    "features": [
        "population_density",
        "median_income",
        "employment_rate",
        "housing_units",
        "retail_space",
        "business_licenses"
    ]
}

RETAIL_MODEL_PARAMS = {
    "min_retail_space": 500,  # square feet
    "max_vacancy_rate": 0.15,
    "target_retail_per_capita": 23.5,  # square feet per person
    "retail_categories": [
        "grocery",
        "restaurant",
        "general_merchandise",
        "clothing",
        "health_personal_care",
        "home_furnishing"
    ]
}

HOUSING_MODEL_PARAMS = {
    "min_density": 10,  # units per acre
    "max_density": 100,
    "target_vacancy_rate": 0.05,
    "housing_types": [
        "single_family",
        "multi_family",
        "mixed_use"
    ]
}

ECONOMIC_MODEL_PARAMS = {
    "indicators": [
        "gdp",
        "employment",
        "income",
        "retail_sales"
    ],
    "forecast_horizon": 10,
    "seasonality": True
}

# Scenario definitions
SCENARIOS = {
    'base': {
        'gdp_growth': 1.0,
        'population_growth': 1.0,
        'employment_growth': 1.0,
        'income_growth': 1.0
    },
    'high_growth': {
        'gdp_growth': 1.15,  # 15% higher growth
        'population_growth': 1.10,  # 10% higher growth
        'employment_growth': 1.12,  # 12% higher growth
        'income_growth': 1.08  # 8% higher growth
    },
    'low_growth': {
        'gdp_growth': 0.85,  # 15% lower growth
        'population_growth': 0.90,  # 10% lower growth
        'employment_growth': 0.88,  # 12% lower growth
        'income_growth': 0.92  # 8% lower growth
    }
}

# Visualization settings
VIZ_SETTINGS = {
    "style": "seaborn-v0_8-whitegrid",
    "palette": "viridis",
    "dpi": 300,
    "figsize": (12, 8)
}

# Report settings
REPORT_SETTINGS = {
    "formats": ["md", "pdf", "html"],
    "include_executive_summary": True,
    "include_methodology": True,
    "include_data_quality": True
}

# Export all settings
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'OUTPUT_DIR', 'SRC_DIR',
    'RAW_DATA_DIR', 'INTERIM_DATA_DIR', 'PROCESSED_DATA_DIR',
    'MODELS_DIR', 'VISUALIZATIONS_DIR', 'REPORTS_DIR',
    'TEMPLATE_DIR', 'REPORT_TEMPLATES_DIR',
    'TRAINED_MODELS_DIR', 'MODEL_METRICS_DIR', 'PREDICTIONS_DIR',
    'REPORT_FIGURES_DIR', 'REPORT_TABLES_DIR', 'ANALYSIS_RESULTS_DIR',
    'CENSUS_API_KEY', 'FRED_API_KEY', 'CHICAGO_DATA_TOKEN',
    'PERMITS_DATASET', 'ZONING_DATASET', 'BUSINESS_LICENSES_DATASET', 'PROPERTY_TRANSACTIONS_DATASET',
    'FRED_SERIES', 'DEFAULT_TRAIN_YEARS', 'FORECAST_YEARS',
    'POPULATION_MODEL_PARAMS', 'RETAIL_MODEL_PARAMS', 'HOUSING_MODEL_PARAMS', 'ECONOMIC_MODEL_PARAMS',
    'SCENARIOS', 'VIZ_SETTINGS', 'REPORT_SETTINGS'
] 