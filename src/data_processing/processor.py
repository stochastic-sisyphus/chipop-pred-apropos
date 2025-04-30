"""
Data processing module for Chicago population analysis.
Handles data cleaning, transformation, and feature engineering.
"""

import logging
from typing import Dict, Optional, Tuple, List, Union
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import geopandas as gpd

from src.config import settings
from .zoning import ZoningProcessor

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processor that coordinates all data processing pipelines."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.zoning_processor = ZoningProcessor()
        
    def process_population_data(self) -> bool:
        """Process population data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_csv(settings.POPULATION_DATA_PATH)
            
            # Clean and process population data
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df = df.groupby(['zip_code', 'year']).agg({
                'population': 'sum',
                'households': 'sum',
                'median_income': 'mean'
            }).reset_index()
            
            # Save processed data
            df.to_csv(settings.POPULATION_PROCESSED_PATH, index=False)
            logger.info("Successfully processed population data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing population data: {str(e)}")
            return False
    
    def process_economic_data(self) -> bool:
        """Process economic data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_csv(settings.ECONOMIC_DATA_PATH)
            
            # Clean and process economic data
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df = df.groupby(['zip_code', 'year']).agg({
                'gdp': 'sum',
                'employment': 'sum',
                'retail_sales': 'sum'
            }).reset_index()
            
            # Save processed data
            df.to_csv(settings.ECONOMIC_PROCESSED_PATH, index=False)
            logger.info("Successfully processed economic data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing economic data: {str(e)}")
            return False
    
    def process_permit_data(self) -> bool:
        """Process building permit data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_csv(settings.PERMIT_DATA_PATH)
            
            # Clean and process permit data
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df = df.groupby(['zip_code', 'year']).agg({
                'permit_count': 'sum',
                'construction_value': 'sum',
                'square_footage': 'sum'
            }).reset_index()
            
            # Save processed data
            df.to_csv(settings.PERMIT_PROCESSED_PATH, index=False)
            logger.info("Successfully processed permit data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing permit data: {str(e)}")
            return False
    
    def process_all(self) -> bool:
        """
        Process all data sources and merge results.
        
        Returns:
            bool: True if all processing steps were successful, False otherwise.
        """
        try:
            logger.info("Starting data processing pipeline...")
            
            # Process each data source
            census_success = self.process_census_data()
            permits_success = self.process_permits_data()
            business_success = self.process_business_licenses()
            property_success = self.process_property_data()
            zoning_success = self.process_zoning_data()
            
            # Log processing results
            results = {
                'Census Data': census_success,
                'Permits Data': permits_success,
                'Business Licenses': business_success,
                'Property Data': property_success,
                'Zoning Data': zoning_success
            }
            
            for source, success in results.items():
                status = "successful" if success else "failed"
                logger.info(f"{source} processing {status}")
            
            # Check if all processing steps were successful
            all_successful = all(results.values())
            
            if all_successful:
                logger.info("All data processing steps completed successfully")
                
                # Load processed datasets
                processed_files = {
                    'census': pd.read_csv(settings.CENSUS_PROCESSED_PATH),
                    'permits': pd.read_csv(settings.PERMITS_PROCESSED_PATH),
                    'business': pd.read_csv(settings.BUSINESS_LICENSES_PROCESSED_PATH),
                    'property': pd.read_csv(settings.PROPERTY_PROCESSED_PATH),
                    'zoning': pd.read_csv(settings.ZONING_PROCESSED_PATH)
                }
                
                # Merge all datasets on zip_code and year where applicable
                merged_data = processed_files['census']
                
                for name, df in processed_files.items():
                    if name != 'census':
                        if 'year' in df.columns:
                            merged_data = pd.merge(
                                merged_data,
                                df,
                                on=['zip_code', 'year'],
                                how='outer'
                            )
                        else:
                            merged_data = pd.merge(
                                merged_data,
                                df,
                                on='zip_code',
                                how='outer'
                            )
                
                # Sort by ZIP code and year
                merged_data = merged_data.sort_values(['zip_code', 'year'])
                
                # Save merged dataset
                merged_data.to_csv(settings.MERGED_DATA_PATH, index=False)
                logger.info(f"Merged dataset saved to {settings.MERGED_DATA_PATH}")
                
                return True
            else:
                failed_steps = [source for source, success in results.items() if not success]
                logger.error(f"Data processing pipeline failed. Failed steps: {', '.join(failed_steps)}")
                return False
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            return False
    
    def process_census_data(self) -> bool:
        """Process census population data."""
        try:
            # Load census data with specified dtypes
            dtypes = {
                'zip_code': str,
                'year': int,
                'total_population': float,
                'median_age': float,
                'median_income': float
            }
            
            df = pd.read_csv(
                settings.CENSUS_DATA_PATH,
                dtype=dtypes,
                low_memory=False
            )
            
            # Clean column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Handle missing values
            df = df.fillna({
                'total_population': df['total_population'].mean(),
                'median_age': df['median_age'].mean(),
                'median_income': df['median_income'].mean()
            })
            
            # Save processed data
            df.to_csv(settings.CENSUS_PROCESSED_PATH, index=False)
            logger.info(f"Census data processed and saved to {settings.CENSUS_PROCESSED_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing census data: {str(e)}")
            return False
    
    def process_permits_data(self) -> bool:
        """Process building permits data."""
        try:
            # Load permits data
            df = pd.read_csv(settings.PERMITS_DATA_PATH, low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Convert date columns
            date_cols = ['issue_date', 'completion_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df[f'{col}_year'] = df[col].dt.year
            
            # Group by ZIP code and year
            agg_dict = {
                'permit_number': 'count',
                'estimated_cost': 'sum',
                'reported_cost': 'sum'
            }
            
            df_grouped = df.groupby(['zip_code', 'issue_date_year']).agg(agg_dict).reset_index()
            
            # Rename columns
            df_grouped = df_grouped.rename(columns={
                'permit_number': 'total_permits',
                'estimated_cost': 'total_estimated_cost',
                'reported_cost': 'total_reported_cost',
                'issue_date_year': 'year'
            })
            
            # Save processed data
            df_grouped.to_csv(settings.PERMITS_PROCESSED_PATH, index=False)
            logger.info(f"Permits data processed and saved to {settings.PERMITS_PROCESSED_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing permits data: {str(e)}")
            return False
    
    def process_retail_deficit(self) -> bool:
        """Process retail deficit data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load retail sales and demographic data
            retail_df = pd.read_csv(settings.RETAIL_SALES_PATH)
            demo_df = pd.read_csv(settings.DEMOGRAPHIC_DATA_PATH)
            
            # Clean column names
            retail_df.columns = retail_df.columns.str.lower().str.replace(' ', '_')
            demo_df.columns = demo_df.columns.str.lower().str.replace(' ', '_')
            
            # Calculate retail potential based on demographics
            demo_df['retail_potential'] = (
                demo_df['population'] * 
                demo_df['median_income'] * 
                settings.RETAIL_SPENDING_FACTOR
            )
            
            # Merge retail sales with potential
            merged_df = pd.merge(
                retail_df,
                demo_df[['zip_code', 'year', 'retail_potential']],
                on=['zip_code', 'year'],
                how='left'
            )
            
            # Calculate retail deficit
            merged_df['retail_deficit'] = (
                merged_df['retail_potential'] - 
                merged_df['retail_sales']
            )
            
            # Calculate additional metrics
            merged_df['retail_deficit_per_capita'] = (
                merged_df['retail_deficit'] / 
                merged_df['population']
            )
            
            merged_df['retail_capture_rate'] = (
                (merged_df['retail_sales'] / 
                merged_df['retail_potential'] * 100)
                .round(2)
            )
            
            # Identify retail leakage areas
            merged_df['is_leakage_area'] = merged_df['retail_deficit'] > 0
            
            # Save processed data
            merged_df.to_csv(settings.RETAIL_DEFICIT_PROCESSED_PATH, index=False)
            logger.info("Successfully processed retail deficit data")
            
            # Generate summary of retail leakage areas
            leakage_summary = merged_df[merged_df['is_leakage_area']].groupby('zip_code').agg({
                'retail_deficit': 'mean',
                'retail_deficit_per_capita': 'mean',
                'retail_capture_rate': 'mean'
            }).round(2)
            
            leakage_summary.to_csv(settings.RETAIL_LEAKAGE_SUMMARY_PATH, index=True)
            logger.info("Generated retail leakage areas summary")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Required file not found: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error processing retail deficit data: {str(e)}")
            return False
    
    def process_business_licenses(self) -> bool:
        """Process business license data."""
        try:
            # Load business license data
            df = pd.read_csv(settings.BUSINESS_LICENSES_PATH, low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Convert date columns
            date_cols = ['date_issued', 'expiration_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df[f'{col}_year'] = df[col].dt.year
            
            # Calculate license duration
            df['license_duration'] = (df['expiration_date'] - df['date_issued']).dt.days / 365.25
            
            # Flag retail and restaurant licenses
            df['is_retail'] = df['license_description'].str.contains('retail', case=False, na=False)
            df['is_restaurant'] = df['license_description'].str.contains('restaurant|food', case=False, na=False)
            
            # Group by ZIP code and year
            df_grouped = df.groupby(['zip_code', 'date_issued_year']).agg({
                'license_id': 'count',
                'license_duration': 'mean',
                'is_retail': 'sum',
                'is_restaurant': 'sum'
            }).reset_index()
            
            # Rename columns
            df_grouped = df_grouped.rename(columns={
                'license_id': 'total_licenses',
                'date_issued_year': 'year',
                'is_retail': 'retail_licenses',
                'is_restaurant': 'restaurant_licenses'
            })
            
            # Save processed data
            df_grouped.to_csv(settings.BUSINESS_LICENSES_PROCESSED_PATH, index=False)
            logger.info(f"Business licenses processed and saved to {settings.BUSINESS_LICENSES_PROCESSED_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing business licenses: {str(e)}")
            return False

    def clean_zoning_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform zoning data.
        
        Args:
            df: Raw zoning DataFrame
            
        Returns:
            Cleaned zoning DataFrame
        """
        try:
            logger.info("Cleaning zoning data...")
            
            # Extract zoning categories
            df['zone_class'] = df['zone_class'].str.extract('([A-Z]+)')
            
            # Create indicator columns for zone types
            df['is_residential'] = df['zone_class'].isin(['RS', 'RT', 'RM'])
            df['is_business'] = df['zone_class'].isin(['B', 'C'])
            df['is_manufacturing'] = df['zone_class'].isin(['M'])
            df['is_planned_development'] = df['zone_class'].isin(['PD'])
            
            # Clean address data
            df['zip_code'] = df['zip_code'].astype(str).str.extract('(\d{5})')
            
            logger.info("Successfully cleaned zoning data")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning zoning data: {str(e)}")
            return None

    def clean_business_licenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform business license data.
        
        Args:
            df: Raw business licenses DataFrame
            
        Returns:
            Cleaned business licenses DataFrame
        """
        try:
            logger.info("Cleaning business license data...")
            
            # Extract year and month
            df['start_year'] = df['license_start_date'].dt.year
            df['start_month'] = df['license_start_date'].dt.month
            
            # Calculate license duration
            df['license_duration'] = (df['license_expiration_date'] - df['license_start_date']).dt.days
            
            # Clean address data
            df['zip_code'] = df['zip_code'].astype(str).str.extract('(\d{5})')
            
            # Group licenses by type
            df['is_retail'] = df['license_description'].str.contains('RETAIL', case=False, na=False)
            df['is_restaurant'] = df['license_description'].str.contains('FOOD|RESTAURANT', case=False, na=False)
            
            logger.info("Successfully cleaned business license data")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning business license data: {str(e)}")
            return None

    def clean_property_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform property transaction data.
        
        Args:
            df: Raw property transactions DataFrame
            
        Returns:
            Cleaned property transactions DataFrame
        """
        try:
            logger.info("Cleaning property transaction data...")
            
            # Convert numeric columns
            df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
            
            # Extract year and month
            df['sale_year'] = df['sale_date'].dt.year
            df['sale_month'] = df['sale_date'].dt.month
            
            # Remove outliers
            df = df[df['sale_price'] > 1000]  # Remove likely errors
            
            # Calculate price per square foot where possible
            if 'square_feet' in df.columns:
                df['price_per_sqft'] = df['sale_price'] / pd.to_numeric(df['square_feet'], errors='coerce')
            
            # Clean address data
            df['zip_code'] = df['zip_code'].astype(str).str.extract('(\d{5})')
            
            logger.info("Successfully cleaned property transaction data")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning property transaction data: {str(e)}")
            return None

    def aggregate_by_zip(self, df: pd.DataFrame, value_cols: List[str], 
                        agg_funcs: Dict[str, str]) -> pd.DataFrame:
        """
        Aggregate data by ZIP code using specified aggregation functions.
        
        Args:
            df: DataFrame to aggregate
            value_cols: Columns to aggregate
            agg_funcs: Dictionary mapping columns to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        try:
            logger.info("Aggregating data by ZIP code...")
            
            # Ensure ZIP code column exists
            if 'zip_code' not in df.columns:
                raise ValueError("DataFrame must contain 'zip_code' column")
            
            # Perform aggregation
            agg_df = df.groupby('zip_code')[value_cols].agg(agg_funcs).reset_index()
            
            logger.info("Successfully aggregated data")
            return agg_df
            
        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            return None

    def generate_retail_metrics(self) -> Optional[pd.DataFrame]:
        """Generate retail metrics from processed data."""
        try:
            logger.info("Generating retail metrics...")

            # Load required datasets
            permits = pd.read_csv(self.processed_dir / 'permits_processed.csv')
            census = pd.read_csv(self.processed_dir / 'census_processed.csv')
            economic = pd.read_csv(self.processed_dir / 'economic_processed.csv')

            # Create retail metrics DataFrame
            retail_metrics = permits[['year', 'zip_code', 'retail_permits', 
                                   'retail_construction_cost', 'retail_permit_ratio',
                                   'retail_cost_ratio']].copy()

            # Merge with census data for demographic context
            retail_metrics = retail_metrics.merge(
                census[['year', 'zip_code', 'total_population', 'median_household_income']],
                on=['year', 'zip_code'],
                how='left'
            )

            # Calculate retail metrics
            retail_metrics['retail_space'] = retail_metrics['retail_permits'] * 2000  # Assume 2000 sq ft per permit
            retail_metrics['retail_space_per_capita'] = retail_metrics['retail_space'] / retail_metrics['total_population']
            retail_metrics['retail_spending_potential'] = retail_metrics['total_population'] * retail_metrics['median_household_income'] * 0.3
            retail_metrics['retail_gap'] = retail_metrics['retail_spending_potential'] - retail_metrics['retail_construction_cost']
            retail_metrics['retail_opportunity_score'] = (
                (retail_metrics['retail_gap'] > 0).astype(int) * 
                (retail_metrics['median_household_income'] > retail_metrics['median_household_income'].median()).astype(int) *
                (retail_metrics['total_population'] > retail_metrics['total_population'].median()).astype(int)
            )

            # Calculate year-over-year changes
            for col in ['retail_space', 'retail_space_per_capita', 'retail_spending_potential', 'retail_gap']:
                retail_metrics[f'{col}_change'] = retail_metrics.groupby('zip_code')[col].pct_change(fill_method=None)

            # Fill NaN values
            retail_metrics = retail_metrics.fillna(0)

            return self._extracted_from_generate_retail_metrics_30(
                'retail_metrics.csv', retail_metrics, 'Retail metrics saved to '
            )
        except Exception as e:
            logger.error(f"Error generating retail metrics: {str(e)}")
            return None

    def _extracted_from_generate_retail_metrics_30(self, arg0, arg1, arg2):
        output_file = self.processed_dir / arg0
        arg1.to_csv(output_file, index=False)
        logger.info(f"{arg2}{output_file}")
        return arg1

    def process_property_data(self) -> bool:
        """Process property transaction data."""
        try:
            # Load property data
            df = pd.read_csv(settings.PROPERTY_DATA_PATH, low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Convert numeric columns
            numeric_cols = ['sale_price', 'square_feet', 'year_built']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date columns
            if 'sale_date' in df.columns:
                df['sale_date'] = pd.to_datetime(df['sale_date'])
                df['sale_year'] = df['sale_date'].dt.year
            
            # Calculate property age
            current_year = datetime.now().year
            df['property_age'] = current_year - df['year_built']
            
            # Group by ZIP code and year
            df_grouped = df.groupby(['zip_code', 'sale_year']).agg({
                'sale_price': ['mean', 'median', 'count'],
                'square_feet': 'mean',
                'property_age': 'mean'
            }).reset_index()
            
            # Flatten column names
            df_grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_grouped.columns]
            
            # Save processed data
            df_grouped.to_csv(settings.PROPERTY_PROCESSED_PATH, index=False)
            logger.info(f"Property data processed and saved to {settings.PROPERTY_PROCESSED_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing property data: {str(e)}")
            return False

    def process_zoning_data(self) -> bool:
        """Process zoning data."""
        try:
            # Load zoning data
            df = pd.read_csv(settings.ZONING_DATA_PATH, low_memory=False)
            
            # Clean column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Convert numeric columns
            numeric_cols = ['lot_area', 'floor_area_ratio']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Group by ZIP code and zoning type
            df_grouped = df.groupby(['zip_code', 'zoning_type']).agg({
                'lot_area': 'sum',
                'floor_area_ratio': 'mean',
                'parcel_count': 'count'
            }).reset_index()
            
            # Calculate area percentages
            total_area = df_grouped.groupby('zip_code')['lot_area'].transform('sum')
            df_grouped['area_percentage'] = (df_grouped['lot_area'] / total_area) * 100
            
            # Save processed data
            df_grouped.to_csv(settings.ZONING_PROCESSED_PATH, index=False)
            logger.info(f"Zoning data processed and saved to {settings.ZONING_PROCESSED_PATH}")
            
            # Merge with property data if available
            if os.path.exists(settings.PROPERTY_PROCESSED_PATH):
                property_df = pd.read_csv(settings.PROPERTY_PROCESSED_PATH)
                merged_df = pd.merge(df_grouped, property_df, on='zip_code', how='outer')
                merged_df.to_csv(settings.ZONING_PROPERTY_MERGED_PATH, index=False)
                logger.info(f"Merged zoning and property data saved to {settings.ZONING_PROPERTY_MERGED_PATH}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing zoning data: {str(e)}")
            return False 