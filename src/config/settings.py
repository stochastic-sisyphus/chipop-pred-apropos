"""
Settings configuration for the Chicago Population Analysis project.
"""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
MODELS_DIR = OUTPUT_DIR / "models"
ANALYSIS_RESULTS_DIR = OUTPUT_DIR / "analysis"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, REPORTS_DIR, VISUALIZATIONS_DIR, MODELS_DIR, ANALYSIS_RESULTS_DIR, SAMPLE_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
CENSUS_DATA_PATH = RAW_DATA_DIR / "census_data.csv"
PERMITS_DATA_PATH = RAW_DATA_DIR / "building_permits.csv"
BUSINESS_LICENSES_PATH = RAW_DATA_DIR / "business_licenses.csv"
ECONOMIC_DATA_PATH = RAW_DATA_DIR / "economic_data.csv"
VACANCY_DATA_PATH = RAW_DATA_DIR / "vacancy_data.csv"
MIGRATION_DATA_PATH = RAW_DATA_DIR / "migration_data.csv"

# Processed data paths
CENSUS_PROCESSED_PATH = PROCESSED_DATA_DIR / "census_processed.csv"
PERMITS_PROCESSED_PATH = PROCESSED_DATA_DIR / "permits_processed.csv"
BUSINESS_LICENSES_PROCESSED_PATH = PROCESSED_DATA_DIR / "business_licenses_processed.csv"
ECONOMIC_PROCESSED_PATH = PROCESSED_DATA_DIR / "economic_processed.csv"
MERGED_DATA_PATH = PROCESSED_DATA_DIR / "merged_dataset.csv"

# API credentials
# These would typically be loaded from environment variables or a secure config file
# For development, placeholder values are provided with clear instructions
CENSUS_API_KEY = os.environ.get('CENSUS_API_KEY', '')
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
CHICAGO_DATA_TOKEN = os.environ.get('CHICAGO_DATA_TOKEN', '')
SOCRATA_APP_TOKEN = os.environ.get('SOCRATA_APP_TOKEN', '')
SOCRATA_USERNAME = os.environ.get('SOCRATA_USERNAME', '')
SOCRATA_PASSWORD = os.environ.get('SOCRATA_PASSWORD', '')
HUD_API_KEY = os.environ.get('HUD_API_KEY', '')
BEA_API_KEY = os.environ.get('BEA_API_KEY', '')

# API configuration
CENSUS_STATE_FIPS = '17'  # Illinois
CENSUS_YEAR = 2020  # Latest decennial census
CENSUS_ACS_YEAR = 2022  # Latest ACS 5-year estimates

# FRED API Series IDs - Updated with correct Chicago-specific series
# Corrected series IDs based on FRED API availability
FRED_SERIES_IDS = {
    # National indicators (available)
    'MORTGAGE30US': 'MORTGAGE30US',  # 30-Year Fixed Rate Mortgage Average
    'CPIAUCSL': 'CPIAUCSL',          # Consumer Price Index for All Urban Consumers
    'HOUST': 'HOUST',                # Housing Starts
    'RRVRUSQ156N': 'RRVRUSQ156N',    # Rental Vacancy Rate
    'MSPUS': 'MSPUS',                # Median Sales Price of Houses
    
    # Chicago-specific indicators (corrected)
    'CHIC917URN': 'CHIC917URN',      # Unemployment Rate in Chicago-Naperville-Elgin, IL-IN-WI (MSA)
    'NGMP16980': 'NGMP16980',        # Total GDP for Chicago-Naperville-Elgin, IL-IN-WI (MSA)
    'CUURA207SA0': 'CUURA207SA0',    # CPI for Chicago-Naperville-Elgin, IL-IN-WI
    'ATNHPIUS16980Q': 'ATNHPIUS16980Q', # House Price Index for Chicago-Naperville-Elgin
    
    # Corrected series IDs for previously failing ones
    'CHICAGO_HOUSING_PRICE': 'CHXRSA',  # S&P CoreLogic Case-Shiller IL-Chicago Home Price Index
    'CHICAGO_RETAIL_SALES': 'RSAFS',    # Advance Retail Sales: Retail and Food Services, Total (national data)
    
    # Alternative series for Chicago housing prices
    'CHICAGO_HOUSING_PRICE_ALT1': 'ATNHPIUS16980Q',  # FHFA House Price Index for Chicago
    'CHICAGO_HOUSING_PRICE_ALT2': 'CHXRHTSA',        # Home Price Index (High Tier) for Chicago
    'CHICAGO_HOUSING_PRICE_ALT3': 'CHXRCSA',         # Condo Price Index for Chicago
    
    # Alternative retail sales indicators
    'CHICAGO_RETAIL_SALES_ALT': 'RETSCHUS',  # Retail Sales: Total (Chain-Type) in United States
}

# Census API endpoints for vacancy data
CENSUS_API_ENDPOINTS = {
    'vacancy': {
        'acs5': 'https://api.census.gov/data/{year}/acs/acs5',
        'table': 'B25002',  # Occupancy Status table
        'variables': ['B25002_001E', 'B25002_002E', 'B25002_003E'],  # Total, Occupied, Vacant
        'for': 'zip code tabulation area:*',
        'in': f'state:{CENSUS_STATE_FIPS}'
    }
}

# Chicago Data Portal Dataset IDs
CHICAGO_DATASETS = {
    'building_permits': 'ydr8-5enu',
    'business_licenses': 'r5kz-chrr',
    'zoning_changes': '7cve-jgbp',
    'affordable_housing': 'uahe-iimk',
    'vacant_buildings': '7nii-7srd',
}

# Validate API keys and log warnings
def validate_api_keys():
    """Validate API keys and log warnings for missing keys."""
    api_keys = {
        'CENSUS_API_KEY': CENSUS_API_KEY,
        'FRED_API_KEY': FRED_API_KEY,
        'CHICAGO_DATA_TOKEN': CHICAGO_DATA_TOKEN,
        'HUD_API_KEY': HUD_API_KEY,
        'BEA_API_KEY': BEA_API_KEY
    }
    
    for key_name, key_value in api_keys.items():
        if not key_value:
            logger.warning(f"{key_name} not set. Set the {key_name} environment variable for real data.")

# Run validation
validate_api_keys()

# Chicago ZIP codes
CHICAGO_ZIP_CODES = [
    '60601', '60602', '60603', '60604', '60605', '60606', '60607', '60608',
    '60609', '60610', '60611', '60612', '60613', '60614', '60615', '60616',
    '60617', '60618', '60619', '60620', '60621', '60622', '60623', '60624',
    '60625', '60626', '60628', '60629', '60630', '60631', '60632', '60633',
    '60634', '60636', '60637', '60638', '60639', '60640', '60641', '60642',
    '60643', '60644', '60645', '60646', '60647', '60649', '60651', '60652',
    '60653', '60654', '60655', '60656', '60657', '60659', '60660', '60661'
]

# South and West side ZIP codes
SOUTH_WEST_ZIP_CODES = [
    # South side
    '60609', '60615', '60616', '60617', '60619', '60620', '60621', '60628', 
    '60629', '60633', '60636', '60637', '60643', '60649', '60653', '60655',
    # West side
    '60608', '60612', '60622', '60623', '60624', '60644', '60651'
]

# Downtown ZIP codes
DOWNTOWN_ZIP_CODES = ['60601', '60602', '60603', '60604', '60605', '60606', '60607', '60611']

# Analysis settings
HISTORICAL_PERIOD_START = 2010
HISTORICAL_PERIOD_END = 2015
RECENT_PERIOD_START = 2018
RECENT_PERIOD_END = 2023
FORECAST_PERIOD_START = 2024
FORECAST_PERIOD_END = 2034

# Retail gap settings
HOUSING_GROWTH_THRESHOLD = 20  # 20% growth threshold
RETAIL_GAP_THRESHOLD = 10  # 10% gap threshold

# Visualization settings
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
VISUALIZATION_FORMAT = 'png'

# Report settings
REPORT_FORMAT = 'markdown'  # 'markdown' or 'html'

# Schema settings - Ensuring consistent data types across pipeline
SCHEMA_DEFINITIONS = {
    'zip_code': {'type': 'string', 'format': r'^\d{5}$', 'required': True},
    'population': {'type': 'numeric', 'min': 0, 'required': True},
    'median_income': {'type': 'numeric', 'min': 0, 'required': True},
    'housing_units': {'type': 'numeric', 'min': 0, 'required': True},
    'year': {'type': 'integer', 'min': 1900, 'max': 2100, 'required': True},
    'retail_sales': {'type': 'numeric', 'min': 0},
    'consumer_spending': {'type': 'numeric', 'min': 0},
}

# Default values for missing required fields
DEFAULT_VALUES = {
    'population': 0,
    'median_income': 0,
    'housing_units': 0,
    'year': 2023,
    'retail_sales': 0,
    'consumer_spending': 0,
}

# API retry settings
API_RETRY_SETTINGS = {
    'max_retries': 5,
    'initial_delay': 2,  # seconds
    'backoff_factor': 2,  # exponential backoff
    'max_delay': 60,     # maximum delay between retries in seconds
    'jitter': 0.1,       # add randomness to delay to prevent thundering herd
}

# Data source documentation
DATA_SOURCES = {
    'housing_price_index': {
        'primary': {
            'source': 'FRED',
            'series_id': 'CHXRSA',
            'description': 'S&P CoreLogic Case-Shiller IL-Chicago Home Price Index',
            'url': 'https://fred.stlouisfed.org/series/CHXRSA',
            'frequency': 'Monthly',
            'last_verified': '2025-06-04'
        },
        'alternatives': [
            {
                'source': 'FRED',
                'series_id': 'ATNHPIUS16980Q',
                'description': 'FHFA House Price Index for Chicago-Naperville-Elgin, IL-IN-WI (MSA)',
                'url': 'https://fred.stlouisfed.org/series/ATNHPIUS16980Q',
                'frequency': 'Quarterly',
                'last_verified': '2025-06-04'
            },
            {
                'source': 'FRED',
                'series_id': 'CHXRHTSA',
                'description': 'Home Price Index (High Tier) for Chicago, Illinois',
                'url': 'https://fred.stlouisfed.org/series/CHXRHTSA',
                'frequency': 'Monthly',
                'last_verified': '2025-06-04'
            }
        ]
    },
    'retail_sales': {
        'primary': {
            'source': 'FRED',
            'series_id': 'RSAFS',
            'description': 'Advance Retail Sales: Retail and Food Services, Total',
            'url': 'https://fred.stlouisfed.org/series/RSAFS',
            'frequency': 'Monthly',
            'last_verified': '2025-06-04',
            'notes': 'National data used as proxy for Chicago retail sales'
        },
        'alternatives': [
            {
                'source': 'FRED',
                'series_id': 'RETSCHUS',
                'description': 'Retail Sales: Total (Chain-Type) in United States',
                'url': 'https://fred.stlouisfed.org/series/RETSCHUS',
                'frequency': 'Monthly',
                'last_verified': '2025-06-04'
            },
            {
                'source': 'Census Bureau',
                'endpoint': 'https://api.census.gov/data/timeseries/eits/marts',
                'description': 'Monthly Retail Trade Survey',
                'url': 'https://www.census.gov/retail/index.html',
                'frequency': 'Monthly',
                'last_verified': '2025-06-04'
            }
        ]
    },
    'vacancy_data': {
        'primary': {
            'source': 'Census Bureau ACS',
            'table': 'B25002',
            'description': 'Occupancy Status (includes vacancy data)',
            'url': 'https://data.census.gov/table/ACSDT1Y2022.B25002',
            'endpoint': 'https://api.census.gov/data/{year}/acs/acs5',
            'frequency': 'Annual',
            'last_verified': '2025-06-04'
        },
        'alternatives': [
            {
                'source': 'FRED',
                'series_id': 'RRVRUSQ156N',
                'description': 'Rental Vacancy Rate in the United States',
                'url': 'https://fred.stlouisfed.org/series/RRVRUSQ156N',
                'frequency': 'Quarterly',
                'last_verified': '2025-06-04',
                'notes': 'National data, not Chicago-specific'
            },
            {
                'source': 'Chicago Data Portal',
                'dataset_id': '7nii-7srd',
                'description': 'Vacant and Abandoned Buildings',
                'url': 'https://data.cityofchicago.org/Buildings/Vacant-and-Abandoned-Buildings-Violations/7nii-7srd',
                'frequency': 'Daily',
                'last_verified': '2025-06-04',
                'notes': 'Provides data on vacant buildings with violations, not overall vacancy rates'
            }
        ]
    }
}
