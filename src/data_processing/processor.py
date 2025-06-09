"""
Data processing module for the Chicago Housing Pipeline project.

This module ensures consistent data processing, normalization, and integration
across all pipeline components.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import os
import traceback

from src.config import settings
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processor for the Chicago Housing Pipeline project."""
    
    def __init__(self, input_dir=None, output_dir=None):
        """
        Initialize the data processor.
        
        Args:
            input_dir (Path, optional): Directory containing raw data files
            output_dir (Path, optional): Directory to save processed data files
        """
        # Set input directory
        if input_dir is None:
            self.input_dir = Path(settings.RAW_DATA_DIR)
        else:
            self.input_dir = Path(input_dir)
        
        # Set output directory
        if output_dir is None:
            self.output_dir = Path(settings.PROCESSED_DATA_DIR)
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data cleaner
        self.data_cleaner = DataCleaner()
        
        # Initialize schema validator
        self.schema_validator = SchemaValidator()
    
    def normalize_zip_codes(self, data):
        """
        Normalize ZIP codes to 5-digit string format.
        
        Args:
            data (pd.DataFrame): Data with zip_code column
            
        Returns:
            pd.DataFrame: Data with normalized ZIP codes
        """
        try:
            if data is None or len(data) == 0:
                logger.warning("Empty dataframe received for ZIP code normalization")
                return data
                
            df = data.copy()
            
            # Check if zip_code column exists
            if 'zip_code' not in df.columns:
                # Check if this is a dataset that inherently doesn't have ZIP codes (like FRED, economic data)
                is_national_data = any(col in df.columns for col in ['series_id', 'series_name', 'value']) or \
                                 any(keyword in str(df.columns).lower() for keyword in ['fred', 'economic', 'national'])
                
                if is_national_data:
                    logger.info("National/regional data detected (no ZIP codes by nature) - adding placeholder ZIP code for aggregation")
                    # Use a special ZIP code that indicates this is national data
                    df['zip_code'] = '99999'  # National data marker
                    df['data_scope'] = 'national'
                else:
                    logger.warning("No zip_code column found in dataframe, adding default ZIP code")
                    # Add zip_code column with default values if missing
                    df['zip_code'] = '00000'
                    df['data_scope'] = 'unknown'
                return df
            
            # Handle NaN values first
            df['zip_code'] = df['zip_code'].fillna('00000')
            
            # Convert to string and handle various formats
            df['zip_code'] = df['zip_code'].astype(str)
            
            # Remove any non-numeric characters
            df['zip_code'] = df['zip_code'].str.replace(r'\D', '', regex=True)
            
            # Handle empty strings
            df.loc[df['zip_code'] == '', 'zip_code'] = '00000'
            
            # Pad with zeros to 5 digits
            df['zip_code'] = df['zip_code'].str.zfill(5)
            
            # Truncate to 5 digits if longer
            df['zip_code'] = df['zip_code'].str.slice(0, 5)
            
            # More lenient validation - allow any 5-digit number
            valid_mask = df['zip_code'].str.match(r'^\d{5}$')
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                logger.warning(f"Found {invalid_count} invalid ZIP codes after normalization")
                # Instead of dropping invalid ZIPs, try to fix them
                df.loc[~valid_mask, 'zip_code'] = df.loc[~valid_mask, 'zip_code'].apply(
                    lambda x: x[:5] if len(x) > 5 else x.zfill(5)
                )
                # Final validation
                valid_mask = df['zip_code'].str.match(r'^\d{5}$')
                if not valid_mask.all():
                    logger.warning("Some ZIP codes could not be normalized, setting to default")
                    df.loc[~valid_mask, 'zip_code'] = '00000'
            
            # Filter to Chicago ZIP codes if specified
            if hasattr(settings, 'CHICAGO_ZIP_CODES') and settings.CHICAGO_ZIP_CODES:
                chicago_zips = [str(z).zfill(5) for z in settings.CHICAGO_ZIP_CODES]
                non_chicago = ~df['zip_code'].isin(chicago_zips)
                if non_chicago.any():
                    logger.info(f"Found {non_chicago.sum()} non-Chicago ZIP codes")
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing ZIP codes: {str(e)}")
            logger.error(traceback.format_exc())
            if data is not None:
                return data
            return None
    
    def ensure_data_types(self, data):
        """
        Ensure consistent data types across all columns.
        
        Args:
            data (pd.DataFrame): Data to process
            
        Returns:
            pd.DataFrame: Data with consistent types
        """
        try:
            if data is None or len(data) == 0:
                logger.warning("Empty dataframe received for type enforcement")
                return data
                
            df = data.copy()
            
            # Ensure ZIP code is string
            if 'zip_code' in df.columns:
                df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
            
            # Ensure numeric columns are float
            numeric_cols = [
                'population', 'median_income', 'housing_units', 'retail_sales', 
                'consumer_spending', 'unit_count', 'value', 'estimated_cost'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with 0
                    df[col] = df[col].fillna(0).astype(float)
            
            # Ensure date columns are datetime
            date_cols = ['date', 'issue_date', 'license_start_date', 'expiration_date']
            for col in date_cols:
                if col in df.columns:
                    # Convert to datetime, coercing errors to NaT
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Ensure year columns are int
            year_cols = ['year', 'permit_year', 'census_year']
            for col in year_cols:
                if col in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with current year
                    current_year = pd.Timestamp.now().year
                    df[col] = df[col].fillna(current_year).astype(int)
            
            # Ensure boolean columns are boolean
            bool_cols = ['is_multifamily']
            for col in bool_cols:
                if col in df.columns:
                    # Convert various values to boolean
                    if df[col].dtype != bool:
                        # Convert string representations to boolean
                        if df[col].dtype == object:
                            true_values = ['true', 'yes', '1', 't', 'y']
                            df[col] = df[col].astype(str).str.lower().isin(true_values)
                        else:
                            # For numeric columns, treat non-zero as True
                            df[col] = df[col].astype(bool)
            
            # Ensure string columns are string
            string_cols = [
                'business_name', 'retail_category', 'permit_type', 
                'description', 'series_id', 'series_name'
            ]
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
            
            return df
            
        except Exception as e:
            logger.error(f"Error ensuring data types: {str(e)}")
            logger.error(traceback.format_exc())
            if data is not None:
                return data
            return None
    
    def process_data(self, data):
        """
        Process and clean the collected data.
        
        Args:
            data (dict or pd.DataFrame): Raw data to process. Can be a dictionary of dataframes or a single dataframe.
            
        Returns:
            dict: Dictionary of processed dataframes
        """
        try:
            logger.info("Processing data...")
            
            if data is None:
                logger.warning("No data received for processing")
                return None
            
            # Handle both dictionary of dataframes and single dataframe
            if isinstance(data, dict):
                logger.info(f"Processing dictionary with {len(data)} dataframes")
                processed_data = {}
                
                # **FIXED: Better handling of nested Chicago data structure**
                # Process each dataframe in the dictionary
                for key, df in data.items():
                    if df is not None:
                        # **FIXED: Handle nested dictionary structure (like Chicago data)**
                        if isinstance(df, dict):
                            logger.info(f"Processing nested {key} data dictionary with {len(df)} sub-datasets")
                            # Process each sub-dataset in the nested dictionary
                            for sub_key, sub_df in df.items():
                                if sub_df is not None and isinstance(sub_df, pd.DataFrame) and len(sub_df) > 0:
                                    logger.info(f"Processing {key}/{sub_key} data with {len(sub_df)} records")
                                    
                                    # Normalize ZIP codes
                                    sub_df = self.normalize_zip_codes(sub_df)
                                    
                                    # Ensure consistent data types
                                    sub_df = self.ensure_data_types(sub_df)
                                    
                                    # Add year column if missing
                                    if 'year' not in sub_df.columns:
                                        if 'date' in sub_df.columns and pd.api.types.is_datetime64_dtype(sub_df['date']):
                                            sub_df['year'] = sub_df['date'].dt.year
                                        else:
                                            logger.warning(f"No date column in {key}/{sub_key} to extract year, using current year")
                                            sub_df['year'] = pd.Timestamp.now().year
                                    
                                    # Store with combined key name
                                    processed_data[f"{key}_{sub_key}"] = sub_df
                                else:
                                    logger.info(f"Skipping {key}/{sub_key} data: empty or invalid dataframe")
                        elif isinstance(df, pd.DataFrame) and len(df) > 0:
                            logger.info(f"Processing {key} data with {len(df)} records")
                            
                            # Normalize ZIP codes
                            df = self.normalize_zip_codes(df)
                            
                            # Ensure consistent data types
                            df = self.ensure_data_types(df)
                            
                            # Add year column if missing
                            if 'year' not in df.columns:
                                if 'date' in df.columns and pd.api.types.is_datetime64_dtype(df['date']):
                                    df['year'] = df['date'].dt.year
                                else:
                                    logger.warning(f"No date column in {key} to extract year, using current year")
                                    df['year'] = pd.Timestamp.now().year
                            
                            processed_data[key] = df
                        else:
                            logger.info(f"Skipping {key} data: empty or invalid dataframe")
                
                # Merge dataframes if needed
                if len(processed_data) > 1:
                    logger.info("Merging processed dataframes")
                    # Create a merged dataset for convenience
                    merged_df = self.merge_dataframes(list(processed_data.values()))
                    if merged_df is not None:
                        processed_data['merged'] = merged_df
                
                return processed_data
                
            elif isinstance(data, pd.DataFrame):
                logger.info(f"Processing single dataframe with {len(data)} records")
                
                # Make a copy to avoid modifying original
                df = data.copy()
                
                # Ensure required columns exist
                required_columns = ['zip_code']
                for col in required_columns:
                    if col not in df.columns:
                        logger.warning(f"Required column {col} missing, adding with default values")
                        if col == 'zip_code':
                            df[col] = '00000'
                        else:
                            df[col] = 0
                
                # Normalize ZIP codes
                df = self.normalize_zip_codes(df)
                
                # Ensure consistent data types
                df = self.ensure_data_types(df)
                
                # Add year column if missing
                if 'year' not in df.columns:
                    if 'date' in df.columns and pd.api.types.is_datetime64_dtype(df['date']):
                        df['year'] = df['date'].dt.year
                    else:
                        logger.warning("No date column to extract year, using current year")
                        df['year'] = pd.Timestamp.now().year
                
                # Handle missing values more conservatively
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_cols:
                    # Only fill missing values if more than 30% of data is present (less aggressive)
                    if df[col].notna().mean() > 0.3:
                        # Use median for numeric columns
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                    else:
                        logger.warning(f"Column {col} has too many missing values (>70%), filling with zeros")
                        df[col] = df[col].fillna(0)
                
                # Remove duplicates but keep first occurrence
                if 'zip_code' in df.columns:
                    df = df.drop_duplicates(subset=['zip_code'], keep='first')
                    logger.info(f"After removing duplicates: {len(df)} records")
                
                # Return as a dictionary with a single key for consistency
                return {'data': df}
            
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                return None
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def merge_dataframes(self, dataframes):
        """
        Merge multiple dataframes on ZIP code.
        
        Args:
            dataframes (list): List of dataframes to merge
            
        Returns:
            pd.DataFrame: Merged dataframe
        """
        try:
            logger.info(f"Merging {len(dataframes)} dataframes...")
            
            if not dataframes:
                logger.warning("No dataframes to merge")
                return None
            
            # Normalize ZIP codes and aggregate data to prevent Cartesian products
            processed_dfs = []
            for i, df in enumerate(dataframes):
                if df is not None and len(df) > 0:
                    # Normalize ZIP codes
                    normalized_df = self.normalize_zip_codes(df)
                    # Ensure consistent data types
                    normalized_df = self.ensure_data_types(normalized_df)
                    
                    # Aggregate by ZIP code to prevent duplicates and memory explosion
                    if 'zip_code' in normalized_df.columns:
                        # Group by ZIP code and aggregate columns properly
                        numeric_cols = normalized_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        string_cols = normalized_df.select_dtypes(include=['object']).columns.tolist()
                        
                        # Remove zip_code from string_cols as it's the grouping key
                        if 'zip_code' in string_cols:
                            string_cols.remove('zip_code')
                        
                        if numeric_cols or string_cols:
                            # Create aggregation dict
                            agg_dict = {}
                            
                            # Handle numeric columns
                            for col in numeric_cols:
                                if col != 'year':  # Keep year as is
                                    agg_dict[col] = 'sum'  # Sum values per ZIP
                            
                            # Handle string columns properly (use 'first' to avoid concatenation)
                            for col in string_cols:
                                agg_dict[col] = 'first'  # Take first non-null value to prevent concatenation
                            
                            # Add year with max (most recent)
                            if 'year' in normalized_df.columns:
                                agg_dict['year'] = 'max'
                            
                            # Aggregate using proper functions
                            try:
                                aggregated_df = normalized_df.groupby('zip_code').agg(agg_dict).reset_index()
                            except Exception as e:
                                logger.warning(f"Error in aggregation, using simpler approach: {str(e)}")
                                # Fallback: just sum numeric and take first for strings
                                agg_dict_simple = {}
                                for col in numeric_cols:
                                    agg_dict_simple[col] = 'sum'
                                for col in string_cols:
                                    agg_dict_simple[col] = 'first'
                                if 'year' in normalized_df.columns:
                                    agg_dict_simple['year'] = 'max'
                                aggregated_df = normalized_df.groupby('zip_code').agg(agg_dict_simple).reset_index()
                            
                            # Add source identifier
                            aggregated_df[f'source_{i}'] = True
                            
                            processed_dfs.append(aggregated_df)
                            logger.info(f"Aggregated dataframe {i}: {len(aggregated_df)} records")
                        else:
                            # If no numeric columns, just get unique ZIP codes
                            unique_df = normalized_df[['zip_code']].drop_duplicates()
                            unique_df[f'source_{i}'] = True
                            processed_dfs.append(unique_df)
                            logger.info(f"Unique ZIP dataframe {i}: {len(unique_df)} records")
                    else:
                        logger.warning(f"Dataframe {i} missing zip_code column after processing, skipping merge but preserving data")
                        # Store separately for national/regional data that can't be merged by ZIP
                        if hasattr(self, '_non_zip_data'):
                            self._non_zip_data[f'dataset_{i}'] = df
                        else:
                            self._non_zip_data = {f'dataset_{i}': df}
            
            if not processed_dfs:
                logger.warning("No valid dataframes after processing")
                return None
            
            # Start with the first dataframe
            merged_df = processed_dfs[0].copy()
            
            # Merge with remaining dataframes using outer join
            for i, df in enumerate(processed_dfs[1:], 1):
                if 'zip_code' not in df.columns:
                    logger.warning(f"Processed dataframe {i} missing zip_code column during merge, skipping")
                    continue
                
                # Merge on ZIP code with outer join
                merged_df = pd.merge(
                    merged_df, 
                    df, 
                    on='zip_code', 
                    how='outer', 
                    suffixes=('', f'_{i}')
                )
                logger.info(f"After merging dataframe {i}: {len(merged_df)} records")
            
            # Fill NaN values with 0 for numeric columns
            numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
            with pd.option_context('future.no_silent_downcasting', True):
                merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
            
            # Fill NaN values with False for boolean columns
            bool_cols = [col for col in merged_df.columns if col.startswith('source_')]
            # Fix pandas FutureWarning by using context manager
            with pd.option_context('future.no_silent_downcasting', True):
                merged_df[bool_cols] = merged_df[bool_cols].fillna(False).infer_objects(copy=False)
            
            # Log information about preserved non-ZIP data
            if hasattr(self, '_non_zip_data') and self._non_zip_data:
                logger.info(f"Preserved {len(self._non_zip_data)} datasets without ZIP codes for separate analysis")
                for key, data in self._non_zip_data.items():
                    logger.info(f"  - {key}: {len(data)} records")
            
            logger.info(f"Final merged dataframe has {len(merged_df)} records and {len(merged_df.columns)} columns")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging dataframes: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def get_non_zip_data(self):
        """
        Get data that was preserved separately due to missing ZIP codes.
        
        Returns:
            dict: Dictionary of dataframes without ZIP codes (typically national/regional data)
        """
        if hasattr(self, '_non_zip_data'):
            return self._non_zip_data
        return {}
    
    def process_all(self, use_sample=False):
        """
        Process all raw data files.
        
        Args:
            use_sample (bool): Whether to use sample data instead of raw data
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        try:
            logger.info("Processing all raw data files...")
            
            # Determine which directory to use
            input_dir = Path(settings.SAMPLE_DATA_DIR) if use_sample else self.input_dir
            
            # Check if directory exists
            if not input_dir.exists():
                logger.error(f"Data directory not found: {input_dir}")
                if use_sample:
                    logger.error("Sample data directory not found, cannot proceed")
                    return False
                else:
                    logger.warning("Raw data directory not found, falling back to sample data")
                    input_dir = Path(settings.SAMPLE_DATA_DIR)
                    if not input_dir.exists():
                        logger.error("Sample data directory not found, cannot proceed")
                        return False
            
            # Check for data files
            csv_files = list(input_dir.glob("*.csv"))
            if not csv_files:
                logger.error(f"No CSV files found in {input_dir}")
                return False
            
            # Process each file
            processed_dfs = []
            for file_path in csv_files:
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Process data
                    processed_df = self.process_data(df)
                    
                    if processed_df:
                        processed_dfs.append(processed_df)
                        
                        # Save processed data
                        output_path = self.output_dir / file_path.name
                        processed_df.to_csv(output_path, index=False)
                        logger.info(f"Saved processed data to {output_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            if not processed_dfs:
                logger.error("No files were processed successfully")
                return False
            
            logger.info(f"Successfully processed {len(processed_dfs)} files")
            return True
            
        except Exception as e:
            logger.error(f"Error processing all files: {str(e)}")
            logger.error(traceback.format_exc())
            return False
