"""
Data cleaning and validation module for Chicago Housing Pipeline.

This module handles data cleaning, validation, and feature engineering for all collected data.
"""

import os
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import re

from src.config import settings

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Data cleaner for Chicago Housing Pipeline.
    
    Handles data cleaning, validation, and feature engineering for all collected data.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the data cleaner.
        
        Args:
            output_dir (Path, optional): Directory to save cleaned data
        """
        self.output_dir = Path(output_dir) if output_dir else Path(settings.DATA_DIR) / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_data(self, data):
        """
        Clean and validate the input data.
        
        Args:
            data (pd.DataFrame or dict): Input data to clean
            
        Returns:
            pd.DataFrame or dict: Cleaned data
        """
        try:
            # Handle dictionary input (from Chicago collector)
            if isinstance(data, dict):
                cleaned_data = {}
                total_records = 0
                for key, df in data.items():
                    if df is not None and hasattr(df, 'empty') and not df.empty:
                        cleaned_df = self._clean_single_dataframe(df)
                        cleaned_data[key] = cleaned_df
                        total_records += len(cleaned_df)
                    else:
                        cleaned_data[key] = pd.DataFrame()
                
                logger.info(f"Cleaned data: {total_records} records")
                return cleaned_data
            
            # Handle DataFrame input (from FRED collector)
            elif isinstance(data, pd.DataFrame):
                logger.info(f"Cleaning data with {len(data)} records")
                
                if data is None or data.empty:
                    logger.warning("Empty DataFrame received")
                    return pd.DataFrame()
                
                return self._clean_single_dataframe(data)
            
            else:
                logger.warning(f"Unknown data type: {type(data)}")
                return data
                
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            logger.error(traceback.format_exc())
            return data if isinstance(data, dict) else pd.DataFrame()
    
    def _clean_single_dataframe(self, df):
        """
        Clean a single DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Clean ZIP codes
            if 'zip_code' in df.columns:
                df = self._normalize_zip_codes(df, 'zip_code')
            
            # Clean numeric columns
            numeric_columns = ['population', 'median_income', 'housing_units', 'unit_count', 
                              'retail_sales', 'consumer_spending', 'vacancy_rate']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Clean date columns
            date_columns = ['issue_date', 'license_start_date', 'date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Clean string columns
            string_columns = ['business_name', 'retail_category', 'permit_type', 'series_id']
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str).str.strip()
            
            # Ensure year column exists
            if 'year' not in df.columns:
                # Try to derive from date columns
                for date_col in date_columns:
                    if date_col in df.columns and not df[date_col].isna().all():
                        df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
                        break
                else:
                    # If no date column available, use current year
                    df['year'] = datetime.now().year
            
            # Ensure year is integer
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(datetime.now().year).astype(int)
            
            logger.info(f"Cleaned data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _normalize_zip_codes(self, df, zip_column):
        """
        Normalize ZIP codes to 5-digit string format.
        
        Args:
            df (pd.DataFrame): DataFrame containing ZIP codes
            zip_column (str): Name of the ZIP code column
            
        Returns:
            pd.DataFrame: DataFrame with normalized ZIP codes
        """
        if zip_column not in df.columns:
            return df
        
        # Convert to string and handle NA/NaN values
        df[zip_column] = df[zip_column].fillna('')
        df[zip_column] = df[zip_column].astype(str).str.strip()
        
        # Extract 5-digit ZIP code if embedded in longer string (e.g., ZIP+4 format)
        zip_extracted = df[zip_column].str.extract(r'(\d{5})')
        if zip_extracted is not None and not zip_extracted.empty:
            df[zip_column] = zip_extracted.iloc[:, 0]
        
        # Ensure 5-digit format for non-empty values
        mask = df[zip_column].str.len() > 0
        df.loc[mask, zip_column] = df.loc[mask, zip_column].str.zfill(5)
        
        # Validate ZIP codes
        valid_zip_mask = df[zip_column].str.match(r'^\d{5}$')
        valid_zip_mask = valid_zip_mask.fillna(False).infer_objects(copy=False)  # Replace NaN with False
        
        if not valid_zip_mask.all():
            logger.warning(f"Found {(~valid_zip_mask).sum()} invalid ZIP codes")
            # Try to fix invalid ZIPs - handle both string and numeric types
            def fix_zip_code(x):
                if pd.isna(x):
                    return ''
                # Convert to string first
                x_str = str(x).strip()
                if len(x_str) > 5 and x_str[:5].isdigit():
                    return x_str[:5]
                elif x_str.replace('.', '').isdigit():
                    # Handle numeric ZIP codes (remove decimal point if present)
                    return x_str.split('.')[0].zfill(5)
                else:
                    return ''
            
            df.loc[~valid_zip_mask, zip_column] = df.loc[~valid_zip_mask, zip_column].apply(fix_zip_code)
            # Final validation
            valid_zip_mask = df[zip_column].str.match(r'^\d{5}$')
            valid_zip_mask = valid_zip_mask.fillna(False).infer_objects(copy=False)
            if not valid_zip_mask.all():
                logger.warning("Some ZIP codes could not be normalized, dropping those records")
                df = df[valid_zip_mask]
        
        # Filter to Chicago ZIP codes if specified and not empty
        if hasattr(settings, 'CHICAGO_ZIP_CODES') and settings.CHICAGO_ZIP_CODES and not df.empty:
            chicago_zips = [str(z).zfill(5) for z in settings.CHICAGO_ZIP_CODES]
            # Create a boolean mask without NaN values
            chicago_mask = df[zip_column].isin(chicago_zips)
            if not chicago_mask.any():
                logger.warning(f"No records found with Chicago ZIP codes")
            else:
                df = df[chicago_mask]
        
        return df
    
    def clean_census_data(self, census_df):
        """
        Clean and validate Census data.
        
        Args:
            census_df (pd.DataFrame): Raw Census data
            
        Returns:
            pd.DataFrame: Cleaned Census data
        """
        try:
            logger.info(f"Cleaning Census data with {len(census_df) if census_df is not None else 0} records")
            
            if census_df is None or census_df.empty:
                logger.warning("Empty Census DataFrame received")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            df = census_df.copy()
            
            # Validate required columns
            required_columns = ['zip_code', 'population', 'median_income', 'housing_units']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in Census data: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    df[col] = settings.DEFAULT_VALUES.get(col, 0)
            
            # Normalize ZIP codes
            df = self._normalize_zip_codes(df, 'zip_code')
            
            # Convert numeric columns
            numeric_columns = [
                'population', 'median_income', 'housing_units', 
                'occupied_housing_units', 'renter_occupied_units'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Add year column if missing
            if 'year' not in df.columns:
                df['year'] = datetime.now().year
            else:
                df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(datetime.now().year).astype(int)
            
            # Calculate derived features
            if all(col in df.columns for col in ['housing_units', 'occupied_housing_units']):
                # Handle division by zero and NaN values
                housing_units = df['housing_units'].replace(0, np.nan)
                df['vacancy_rate'] = 1 - (df['occupied_housing_units'] / housing_units)
                df['vacancy_rate'] = df['vacancy_rate'].fillna(0).clip(0, 1) * 100  # Convert to percentage
            
            if all(col in df.columns for col in ['occupied_housing_units', 'renter_occupied_units']):
                # Handle division by zero and NaN values
                occupied_units = df['occupied_housing_units'].replace(0, np.nan)
                df['renter_rate'] = df['renter_occupied_units'] / occupied_units
                df['renter_rate'] = df['renter_rate'].fillna(0).clip(0, 1) * 100  # Convert to percentage
            
            # Handle missing values with proper NA/NaN masking
            for col in numeric_columns:
                if col in df.columns:
                    # First identify NaN values
                    nan_mask = df[col].isna() | (df[col] == 0)
                    
                    if nan_mask.any():
                        # Calculate medians by zip_code for filling
                        zip_medians = df.groupby('zip_code')[col].median()
                        
                        # Apply the filling logic with robust NA handling
                        for zip_code in df['zip_code'].unique():
                            if zip_code in zip_medians and not pd.isna(zip_medians[zip_code]) and zip_medians[zip_code] > 0:
                                # Create safe boolean mask for this ZIP code with NaN values
                                zip_mask = df['zip_code'] == zip_code
                                combined_mask = zip_mask & nan_mask
                                # Apply median value
                                df.loc[combined_mask, col] = zip_medians[zip_code]
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['zip_code', 'year'])
            
            # Sort by ZIP code and year
            df = df.sort_values(['zip_code', 'year'])
            
            logger.info(f"Cleaned Census data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning Census data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def clean_economic_data(self, economic_df):
        """
        Clean and validate economic data from FRED and BEA.
        
        Args:
            economic_df (pd.DataFrame): Raw economic data
            
        Returns:
            pd.DataFrame: Cleaned economic data
        """
        try:
            logger.info(f"Cleaning economic data with {len(economic_df) if economic_df is not None else 0} records")
            
            if economic_df is None or economic_df.empty:
                logger.warning("Empty economic DataFrame received")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            df = economic_df.copy()
            
            # Validate required columns
            required_columns = ['date', 'value', 'series_id']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in economic data: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    if col == 'date':
                        df[col] = pd.to_datetime('today')
                    elif col == 'value':
                        df[col] = 0
                    elif col == 'series_id':
                        df[col] = 'UNKNOWN'
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Filter out invalid dates - use boolean indexing without NaN values
                valid_date_mask = ~df['date'].isna()
                df = df[valid_date_mask]
                # Extract year
                df['year'] = df['date'].dt.year
            
            # Convert value column to numeric
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Handle missing values with proper NA/NaN masking
            if 'value' in df.columns:
                # Identify NaN values
                nan_mask = df['value'].isna()
                
                if nan_mask.any():
                    # Apply more sophisticated filling by series_id if available
                    if 'series_id' in df.columns:
                        for series in df['series_id'].unique():
                            # Create safe boolean mask
                            series_mask = df['series_id'] == series
                            # Get values for this series
                            series_values = df.loc[series_mask, 'value'].copy()
                            
                            # Only process if we have some non-NaN values
                            if not series_values.isna().all():
                                # Calculate non-NaN mean for this series
                                non_nan_mean = series_values.mean(skipna=True)
                                # Replace NaNs with non-NaN mean
                                if not pd.isna(non_nan_mean):
                                    # Create a boolean mask for NaN values in this series
                                    combined_mask = series_mask & nan_mask
                                    # Apply mean value
                                    df.loc[combined_mask, 'value'] = non_nan_mean
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['series_id', 'date'])
            
            # Sort by series_id and date
            df = df.sort_values(['series_id', 'date'])
            
            logger.info(f"Cleaned economic data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning economic data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def clean_building_permits(self, permits_df):
        """
        Clean and validate building permit data.
        
        Args:
            permits_df (pd.DataFrame): Raw building permit data
            
        Returns:
            pd.DataFrame: Cleaned building permit data
        """
        try:
            logger.info(f"Cleaning building permit data with {len(permits_df) if permits_df is not None else 0} records")
            
            if permits_df is None or permits_df.empty:
                logger.warning("Empty permits DataFrame received")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            df = permits_df.copy()
            
            # Normalize ZIP codes
            df = self._normalize_zip_codes(df, 'zip_code')
            
            # Convert issue_date to datetime
            if 'issue_date' in df.columns:
                df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
                # Filter out invalid dates - use boolean indexing without NaN values
                valid_date_mask = ~df['issue_date'].isna()
                df = df[valid_date_mask]
                # Extract year
                df['permit_year'] = df['issue_date'].dt.year
                # Ensure year column exists
                if 'year' not in df.columns:
                    df['year'] = df['permit_year']
            elif 'year' not in df.columns:
                df['year'] = datetime.now().year
            
            # Ensure unit_count is numeric
            if 'unit_count' in df.columns:
                df['unit_count'] = pd.to_numeric(df['unit_count'], errors='coerce')
                # Set minimum unit count to 1
                df['unit_count'] = df['unit_count'].fillna(1).clip(lower=1)
            else:
                df['unit_count'] = 1  # Default to 1 unit if missing
            
            # Identify multifamily permits
            if 'permit_type' in df.columns and 'is_multifamily' not in df.columns:
                # Create a boolean mask without NaN values
                multifamily_mask = df['permit_type'].str.contains(
                    'MULTI-FAMILY|APARTMENT|CONDO', case=False, na=False
                )
                df['is_multifamily'] = multifamily_mask
            elif 'is_multifamily' not in df.columns:
                df['is_multifamily'] = False  # Default if permit_type is missing
            
            # Add permit_type if missing (required by models)
            if 'permit_type' not in df.columns:
                df['permit_type'] = 'UNKNOWN'
            
            # Ensure estimated_cost is numeric
            if 'estimated_cost' in df.columns:
                df['estimated_cost'] = pd.to_numeric(df['estimated_cost'], errors='coerce').fillna(0)
            else:
                df['estimated_cost'] = 0  # Default if missing
            
            # Aggregate permits by ZIP code and year
            if len(df) > 0:
                agg_df = df.groupby(['zip_code', 'year']).agg({
                    'permit_type': 'count',
                    'unit_count': 'sum',
                    'estimated_cost': 'sum',
                    'is_multifamily': 'sum'
                }).reset_index()
                
                # Rename columns
                agg_df = agg_df.rename(columns={
                    'permit_type': 'permit_count',
                    'is_multifamily': 'multifamily_permit_count'
                })
                
                # Calculate multifamily unit percentage
                agg_df['multifamily_permit_pct'] = (agg_df['multifamily_permit_count'] / agg_df['permit_count'] * 100).fillna(0)
                
                logger.info(f"Aggregated permits: {len(agg_df)} records")
                return agg_df
            else:
                logger.warning("No valid permits to aggregate")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error cleaning building permits: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def clean_business_licenses(self, licenses_df):
        """
        Clean and validate business license data.
        
        Args:
            licenses_df (pd.DataFrame): Raw business license data
            
        Returns:
            pd.DataFrame: Cleaned business license data
        """
        try:
            logger.info(f"Cleaning business license data with {len(licenses_df) if licenses_df is not None else 0} records")
            
            if licenses_df is None or licenses_df.empty:
                logger.warning("Empty licenses DataFrame received")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying the original
            df = licenses_df.copy()
            
            # Normalize ZIP codes
            df = self._normalize_zip_codes(df, 'zip_code')
            
            # Convert license_start_date to datetime
            if 'license_start_date' in df.columns:
                df['license_start_date'] = pd.to_datetime(df['license_start_date'], errors='coerce')
                # Filter out invalid dates - use boolean indexing without NaN values
                valid_date_mask = ~df['license_start_date'].isna()
                df = df[valid_date_mask]
                # Extract year
                df['license_year'] = df['license_start_date'].dt.year
                # Ensure year column exists
                if 'year' not in df.columns:
                    df['year'] = df['license_year']
            elif 'year' not in df.columns:
                df['year'] = datetime.now().year
            
            # Clean business_name
            if 'business_name' in df.columns:
                df['business_name'] = df['business_name'].fillna('').astype(str).str.strip()
            else:
                df['business_name'] = 'UNKNOWN'
            
            # Clean license_description
            if 'license_description' in df.columns:
                df['license_description'] = df['license_description'].fillna('').astype(str).str.strip()
            else:
                df['license_description'] = 'UNKNOWN'
            
            # Identify retail businesses
            if 'is_retail' not in df.columns:
                # Create a boolean mask without NaN values
                retail_mask = df['license_description'].str.contains(
                    'RETAIL|STORE|SHOP|FOOD|RESTAURANT|GROCERY|MARKET|BAKERY|CAFE', 
                    case=False, na=False
                )
                df['is_retail'] = retail_mask
            
            # Categorize retail businesses
            if 'retail_category' not in df.columns and 'license_description' in df.columns:
                # Initialize with unknown
                df['retail_category'] = 'OTHER'
                
                # Define retail categories and their keywords
                retail_categories = {
                    'FOOD_SERVICE': ['RESTAURANT', 'CAFE', 'COFFEE', 'BAKERY', 'CATERING', 'FOOD'],
                    'GROCERY': ['GROCERY', 'SUPERMARKET', 'FOOD STORE', 'CONVENIENCE'],
                    'APPAREL': ['APPAREL', 'CLOTHING', 'FASHION', 'GARMENT'],
                    'ELECTRONICS': ['ELECTRONICS', 'COMPUTER', 'PHONE', 'DEVICE'],
                    'HOME_GOODS': ['FURNITURE', 'HOME', 'DECOR', 'APPLIANCE'],
                    'HEALTH_BEAUTY': ['PHARMACY', 'BEAUTY', 'COSMETIC', 'SALON', 'SPA'],
                    'SPECIALTY': ['SPECIALTY', 'GIFT', 'CRAFT', 'BOOK', 'JEWELRY']
                }
                
                # Assign categories based on keywords
                for category, keywords in retail_categories.items():
                    category_mask = df['license_description'].str.contains(
                        '|'.join(keywords), case=False, na=False
                    )
                    df.loc[category_mask, 'retail_category'] = category
            
            # Aggregate licenses by ZIP code, year, and retail category
            if len(df) > 0:
                # First aggregate all licenses
                all_agg = df.groupby(['zip_code', 'year']).agg({
                    'business_name': 'count',
                    'is_retail': 'sum'
                }).reset_index()
                
                # Rename columns
                all_agg = all_agg.rename(columns={
                    'business_name': 'license_count',
                    'is_retail': 'retail_license_count'
                })
                
                # Calculate retail percentage
                all_agg['retail_license_pct'] = (all_agg['retail_license_count'] / all_agg['license_count'] * 100).fillna(0)
                
                # Then aggregate retail licenses by category
                if 'retail_category' in df.columns and df['is_retail'].any():
                    retail_df = df[df['is_retail']]
                    
                    if len(retail_df) > 0:
                        retail_agg = retail_df.groupby(['zip_code', 'year', 'retail_category']).size().reset_index(name='category_count')
                        
                        # Pivot to get categories as columns
                        retail_pivot = retail_agg.pivot_table(
                            index=['zip_code', 'year'],
                            columns='retail_category',
                            values='category_count',
                            fill_value=0
                        ).reset_index()
                        
                        # Flatten column names
                        retail_pivot.columns = [f'retail_{col.lower()}' if col not in ['zip_code', 'year'] else col for col in retail_pivot.columns]
                        
                        # Merge with all_agg
                        result = pd.merge(all_agg, retail_pivot, on=['zip_code', 'year'], how='left')
                        
                        # Fill NaN values with 0 for retail category columns
                        for col in result.columns:
                            if col.startswith('retail_') and col not in ['retail_license_count', 'retail_license_pct']:
                                result[col] = result[col].fillna(0)
                        
                        logger.info(f"Aggregated licenses: {len(result)} records")
                        return result
                
                logger.info(f"Aggregated licenses: {len(all_agg)} records")
                return all_agg
            else:
                logger.warning("No valid licenses to aggregate")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error cleaning business licenses: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def ensure_data_types(self, df):
        """
        Ensure consistent data types across all columns.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            
        Returns:
            pd.DataFrame: DataFrame with consistent data types
        """
        if df is None or df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Apply schema definitions from settings
        if hasattr(settings, 'SCHEMA_DEFINITIONS'):
            for column, schema in settings.SCHEMA_DEFINITIONS.items():
                if column in df.columns:
                    # Handle different data types
                    if schema['type'] in ['numeric', 'integer', 'float']:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                        
                        # Apply min/max constraints
                        if 'min' in schema:
                            df[column] = df[column].clip(lower=schema['min'])
                        if 'max' in schema:
                            df[column] = df[column].clip(upper=schema['max'])
                        
                        # Convert to integer if specified
                        if schema['type'] == 'integer':
                            df[column] = df[column].fillna(0).astype(int)
                    
                    elif schema['type'] == 'string':
                        df[column] = df[column].fillna('').astype(str)
                        
                        # Apply format validation if specified
                        if 'format' in schema:
                            format_regex = re.compile(schema['format'])
                            valid_format = df[column].str.match(format_regex)
                            valid_format = valid_format.fillna(False)
                            
                            if not valid_format.all():
                                logger.warning(f"Found {(~valid_format).sum()} values not matching format for column {column}")
                                
                                # For ZIP codes, attempt to fix
                                if column == 'zip_code':
                                    df = self._normalize_zip_codes(df, column)
                    
                    elif schema['type'] == 'date':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                
                # Handle required columns
                if schema.get('required', False) and df[column].isna().any():
                    # Use default value from settings if available
                    if hasattr(settings, 'DEFAULT_VALUES') and column in settings.DEFAULT_VALUES:
                        df[column] = df[column].fillna(settings.DEFAULT_VALUES[column])
                    else:
                        # Use sensible defaults based on type
                        if schema['type'] in ['numeric', 'integer', 'float']:
                            df[column] = df[column].fillna(0)
                        elif schema['type'] == 'string':
                            df[column] = df[column].fillna('')
                        elif schema['type'] == 'date':
                            df[column] = df[column].fillna(pd.to_datetime('today'))
        
        return df
