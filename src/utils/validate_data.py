"""
Data validation utilities for the Chicago Population Analysis project.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

def validate_data_files():
    """
    Validate that all required data files exist and have the expected structure.
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    try:
        # Check if required data files exist
        required_files = [
            settings.CENSUS_DATA_PATH,
            settings.PERMITS_DATA_PATH,
            settings.BUSINESS_LICENSES_PATH
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.error(f"Missing required data files: {', '.join(missing_files)}")
            return False
        
        # Validate census data
        if settings.CENSUS_DATA_PATH.exists():
            try:
                census_df = pd.read_csv(settings.CENSUS_DATA_PATH, dtype={'zip_code': str})
                if 'zip_code' not in census_df.columns:
                    logger.error("Census data missing zip_code column")
                    return False
                if 'population' not in census_df.columns:
                    logger.error("Census data missing population column")
                    return False
            except Exception as e:
                logger.error(f"Error validating census data: {str(e)}")
                return False
        
        # Validate permit data
        if settings.PERMITS_DATA_PATH.exists():
            try:
                permits_df = pd.read_csv(settings.PERMITS_DATA_PATH, dtype={'zip_code': str})
                if 'zip_code' not in permits_df.columns:
                    logger.error("Permit data missing zip_code column")
                    return False
                if 'permit_type' not in permits_df.columns:
                    logger.error("Permit data missing permit_type column")
                    return False
            except Exception as e:
                logger.error(f"Error validating permit data: {str(e)}")
                return False
        
        # Validate business license data
        if settings.BUSINESS_LICENSES_PATH.exists():
            try:
                licenses_df = pd.read_csv(settings.BUSINESS_LICENSES_PATH, dtype={'zip_code': str})
                if 'zip_code' not in licenses_df.columns:
                    logger.error("Business license data missing zip_code column")
                    return False
                if 'business_type' not in licenses_df.columns:
                    logger.error("Business license data missing business_type column")
                    return False
            except Exception as e:
                logger.error(f"Error validating business license data: {str(e)}")
                return False
        
        logger.info("All data files validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data files: {str(e)}")
        return False

def validate_merged_dataset(df):
    """
    Validate the merged dataset.
    
    Args:
        df (pd.DataFrame): Merged dataset to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Check if required columns exist
        required_columns = [
            'zip_code', 'population', 'median_income', 'housing_units'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Merged dataset missing required columns: {', '.join(missing_columns)}")
            return False
        
        # Check for valid ZIP codes
        df['zip_code'] = df['zip_code'].astype(str)
        chicago_zips = settings.CHICAGO_ZIP_CODES
        invalid_zips = set(df['zip_code']) - set(chicago_zips)
        
        if invalid_zips:
            logger.warning(f"Merged dataset contains {len(invalid_zips)} invalid ZIP codes")
            # Don't fail validation for this
        
        # Check for missing values in key columns
        for col in required_columns:
            if col in df.columns and df[col].isna().any():
                logger.warning(f"Merged dataset contains missing values in {col}")
                # Don't fail validation for this
        
        logger.info("Merged dataset validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating merged dataset: {str(e)}")
        return False

def validate_zip_code(zip_code):
    """
    Validate a ZIP code.
    
    Args:
        zip_code (str): ZIP code to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Ensure it's a string
        zip_str = str(zip_code).strip()
        
        # Check length
        if len(zip_str) != 5:
            return False
            
        # Check if numeric
        if not zip_str.isdigit():
            return False
            
        # Check if in Chicago ZIP codes
        if zip_str not in settings.CHICAGO_ZIP_CODES:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating ZIP code: {str(e)}")
        return False

def flag_insufficient_data(df):
    """
    Flag rows with insufficient data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with data_status column
    """
    try:
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Define required columns for different analyses
        retail_cols = ['retail_businesses', 'retail_space', 'retail_demand', 'retail_supply']
        housing_cols = ['housing_units', 'median_home_value', 'pct_owner_occupied']
        economic_cols = ['median_income', 'employment_rate', 'labor_force_participation']
        
        # Flag rows with insufficient retail data
        def has_retail_data(row):
            return all(col in row.index and not pd.isna(row[col]) and row[col] != 0 for col in retail_cols)
        
        # Flag rows with insufficient housing data
        def has_housing_data(row):
            return all(col in row.index and not pd.isna(row[col]) and row[col] != 0 for col in housing_cols)
        
        # Flag rows with insufficient economic data
        def has_economic_data(row):
            return all(col in row.index and not pd.isna(row[col]) and row[col] != 0 for col in economic_cols)
        
        # Create status column
        result['retail_data_status'] = result.apply(lambda row: 'ok' if has_retail_data(row) else 'insufficient', axis=1)
        result['housing_data_status'] = result.apply(lambda row: 'ok' if has_housing_data(row) else 'insufficient', axis=1)
        result['economic_data_status'] = result.apply(lambda row: 'ok' if has_economic_data(row) else 'insufficient', axis=1)
        
        # Overall status
        result['data_status'] = 'ok'
        result.loc[(result['retail_data_status'] == 'insufficient') | 
                  (result['housing_data_status'] == 'insufficient') | 
                  (result['economic_data_status'] == 'insufficient'), 'data_status'] = 'insufficient'
        
        return result
        
    except Exception as e:
        logger.error(f"Error flagging insufficient data: {str(e)}")
        return df
