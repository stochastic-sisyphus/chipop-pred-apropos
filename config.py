"""
ChiPop: Chicago Population Analysis Configuration
This module contains configuration settings for the Chicago population analysis.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Data paths
DATA_DIR = os.getenv('DATA_DIR', 'data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

# Chicago Data Portal Settings
CHICAGO_DATA_PORTAL = os.getenv('CHICAGO_DATA_PORTAL', 'data.cityofchicago.org')
CHICAGO_SOCRATA_APP_TOKEN = os.getenv('CHICAGO_SOCRATA_APP_TOKEN', None)

# Dataset IDs
BUILDING_PERMITS_DATASET = os.getenv('BUILDING_PERMITS_DATASET', 'ydr8-5enu')
ZONING_DATASET = os.getenv('ZONING_DATASET', 'dj47-wfun')

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

# Analysis settings
ANALYSIS_START_YEAR = 2013
ANALYSIS_END_YEAR = 2023
FORECAST_YEARS = 2  # Number of years to forecast

# Model settings
MODEL_RANDOM_STATE = 42
MODEL_TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5

# Economic scenario settings
SCENARIOS = {
    'optimistic': {
        'mortgage_30y': -0.9,  # 10% decrease from current
        'treasury_10y': -0.5,  # 0.5 percentage point decrease
        'consumer_sentiment': 1.2,  # 20% increase
        'housing_starts': 1.1,  # 10% increase
    },
    'neutral': {
        'mortgage_30y': 0.0,  # No change
        'treasury_10y': 0.0,  # No change
        'consumer_sentiment': 1.0,  # No change
        'housing_starts': 1.0,  # No change
    },
    'pessimistic': {
        'mortgage_30y': 0.7,  # 7% increase from current
        'treasury_10y': 0.3,  # 0.3 percentage point increase
        'consumer_sentiment': 0.8,  # 20% decrease
        'housing_starts': 0.9,  # 10% decrease
    }
}

# Visualization settings
FIGURE_DPI = 300
FIGURE_SIZE_LARGE = (15, 8)
FIGURE_SIZE_MEDIUM = (12, 6)
FIGURE_SIZE_SMALL = (8, 6)
COLOR_PALETTE = 'viridis'