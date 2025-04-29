"""
Data processing module for Chicago population analysis.
Handles data cleaning, transformation, and feature engineering.
"""

import logging
from typing import Dict, Optional, Tuple, List, Union
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import geopandas as gpd

from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing and feature engineering."""
    
    def __init__(self):
        """Initialize data processor."""
        self.raw_dir = settings.RAW_DATA_DIR
        self.processed_dir = settings.PROCESSED_DATA_DIR
        self.interim_dir = settings.INTERIM_DATA_DIR
        self.scaler = StandardScaler()
        
    def process_census_data(self):
        """Process Census demographic data."""
        try:
            logger.info("Processing Census data...")

            # Load raw data
            df = pd.read_csv(self.raw_dir / 'census_data.csv')

            # Rename columns to meaningful names
            column_map = {
                'B01003_001E': 'total_population',
                'B19013_001E': 'median_household_income',
                'B25077_001E': 'median_home_value',
                'B23025_002E': 'labor_force'
            }
            df = df.rename(columns=column_map)

            # Convert numeric columns
            numeric_cols = ['total_population', 'median_household_income', 
                          'median_home_value', 'labor_force']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate year-over-year changes
            df = df.sort_values(['zip_code', 'year'])
            for col in numeric_cols:
                df[f'{col}_change'] = df.groupby('zip_code')[col].pct_change()

            return self._extracted_from_generate_retail_metrics_30(
                'census_processed.csv', df, 'Processed Census data saved to '
            )
        except Exception as e:
            logger.error(f"Error processing Census data: {str(e)}")
            return None
            
    def process_permit_data(self):
        """Process building permit data."""
        try:
            logger.info("Processing building permit data...")

            # Load raw data
            df = pd.read_csv(self.raw_dir / 'building_permits.csv')

            # Convert dates
            df['issue_date'] = pd.to_datetime(df['issue_date'])
            df['year'] = df['issue_date'].dt.year

            # Clean numeric columns
            numeric_cols = [
                'reported_cost', 'total_fee', 
                'residential_construction_cost', 
                'commercial_construction_cost',
                'retail_construction_cost'
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Ensure ZIP code is in correct format
            df['zip_code'] = df['contact_1_zipcode'].astype(str).str.extract('(\d{5})').fillna('00000')

            # Group by year and ZIP code
            grouped = df.groupby(['year', 'zip_code']).agg({
                'permit_type': 'count',  # Total permits
                'residential_permits': 'sum',  # Residential permits
                'commercial_permits': 'sum',  # Commercial permits
                'retail_permits': 'sum',  # Retail permits
                'reported_cost': 'sum',  # Total construction cost
                'residential_construction_cost': 'sum',  # Residential construction cost
                'commercial_construction_cost': 'sum',  # Commercial construction cost
                'retail_construction_cost': 'sum',  # Retail construction cost
                'permit_category': lambda x: x.value_counts().to_dict()  # Permit type distribution
            }).reset_index()

            # Rename columns
            grouped = grouped.rename(columns={
                'permit_type': 'total_permits',
                'reported_cost': 'total_construction_cost'
            })

            # Calculate ratios
            grouped['residential_permit_ratio'] = (grouped['residential_permits'] / grouped['total_permits']).fillna(0)
            grouped['commercial_permit_ratio'] = (grouped['commercial_permits'] / grouped['total_permits']).fillna(0)
            grouped['retail_permit_ratio'] = (grouped['retail_permits'] / grouped['total_permits']).fillna(0)

            grouped['residential_cost_ratio'] = (grouped['residential_construction_cost'] / grouped['total_construction_cost']).fillna(0)
            grouped['commercial_cost_ratio'] = (grouped['commercial_construction_cost'] / grouped['total_construction_cost']).fillna(0)
            grouped['retail_cost_ratio'] = (grouped['retail_construction_cost'] / grouped['total_construction_cost']).fillna(0)

            # Add permit category columns
            for category in ['new_construction', 'renovation', 'addition', 'other']:
                grouped[f'{category}_permits'] = grouped['permit_category'].apply(
                    lambda x: x.get(category, 0) if isinstance(x, dict) else 0
                )

            # Drop the permit_category dictionary column
            grouped = grouped.drop('permit_category', axis=1)

            # Calculate housing units (assume 1.5 units per residential permit)
            grouped['housing_units'] = grouped['residential_permits'] * 1.5

            # Calculate year-over-year changes
            for col in ['total_permits', 'residential_permits', 'commercial_permits', 'retail_permits', 'housing_units']:
                grouped[f'{col}_change'] = grouped.groupby('zip_code')[col].pct_change(fill_method=None)

            for col in ['total_construction_cost', 'residential_construction_cost', 'commercial_construction_cost', 'retail_construction_cost']:
                grouped[f'{col}_change'] = grouped.groupby('zip_code')[col].pct_change(fill_method=None)

            # Fill NaN values with 0
            grouped = grouped.fillna(0)

            # Log summary statistics
            logger.info(f"Processed {len(grouped)} permit records across {grouped['zip_code'].nunique()} ZIP codes")
            logger.info(f"Total permits: {grouped['total_permits'].sum():,.0f}")
            logger.info(f"- Residential permits: {grouped['residential_permits'].sum():,.0f}")
            logger.info(f"- Commercial permits: {grouped['commercial_permits'].sum():,.0f}")
            logger.info(f"- Retail permits: {grouped['retail_permits'].sum():,.0f}")
            logger.info(f"Total construction cost: ${grouped['total_construction_cost'].sum():,.2f}")
            logger.info(f"- Residential: ${grouped['residential_construction_cost'].sum():,.2f}")
            logger.info(f"- Commercial: ${grouped['commercial_construction_cost'].sum():,.2f}")
            logger.info(f"- Retail: ${grouped['retail_construction_cost'].sum():,.2f}")
            logger.info(f"Total housing units: {grouped['housing_units'].sum():,.0f}")

            return self._extracted_from_generate_retail_metrics_30(
                'permits_processed.csv', grouped, 'Processed permit data saved to '
            )
        except Exception as e:
            logger.error(f"Error processing permit data: {str(e)}")
            return None
            
    def process_economic_data(self):
        """Process economic indicators."""
        try:
            logger.info("Processing economic indicators...")

            # Load raw data
            df = pd.read_csv(self.raw_dir / 'economic_indicators.csv')

            # Ensure year is a column
            if df.index.name == 'year':
                df = df.reset_index()
            elif 'year' not in df.columns:
                logger.error("No year column found in economic data")
                return None

            # Convert year to integer if it's not already
            df['year'] = df['year'].astype(int)

            # Fill missing values using forward fill
            df = df.ffill()

            # Calculate year-over-year changes for each indicator
            indicator_cols = ['unemployment_rate', 'gdp', 'per_capita_income', 'personal_income']
            for col in indicator_cols:
                if col in df.columns:
                    df[f'{col}_change'] = df[col].pct_change()

            # Sort by year
            df = df.sort_values('year')

            return self._extracted_from_generate_retail_metrics_30(
                'economic_processed.csv', df, 'Processed economic data saved to '
            )
        except Exception as e:
            logger.error(f"Error processing economic data: {str(e)}")
            return None
            
    def merge_datasets(self):
        """Merge all processed datasets."""
        try:
            logger.info("Merging processed datasets...")

            # Load processed datasets
            census_df = pd.read_csv(self.processed_dir / 'census_processed.csv')
            permits_df = pd.read_csv(self.processed_dir / 'permits_processed.csv')
            economic_df = pd.read_csv(self.processed_dir / 'economic_processed.csv')

            # Ensure all required columns exist
            required_cols = {
                'census': ['year', 'zip_code', 'total_population', 'median_household_income'],
                'permits': ['year', 'zip_code', 'total_permits', 'residential_permits', 'commercial_permits', 'housing_units'],
                'economic': ['year', 'unemployment_rate', 'gdp', 'per_capita_income', 'personal_income']
            }

            for df, name, cols in [
                (census_df, 'census', required_cols['census']),
                (permits_df, 'permits', required_cols['permits']),
                (economic_df, 'economic', required_cols['economic'])
            ]:
                if missing := [col for col in cols if col not in df.columns]:
                    logger.error(f"Missing required columns in {name} data: {missing}")
                    return None

            # Get the year range from census data
            min_year = census_df['year'].min()
            max_year = census_df['year'].max()

            # Filter permits data to match census years
            permits_df = permits_df[
                (permits_df['year'] >= min_year) & 
                (permits_df['year'] <= max_year)
            ]

            # Merge on year and ZIP code
            merged_df = census_df.merge(
                permits_df, 
                on=['year', 'zip_code'], 
                how='left',
                validate='1:1'
            )

            # Merge economic data (which is year-level only)
            merged_df = merged_df.merge(
                economic_df,
                on='year',
                how='left',
                validate='m:1'
            )

            # Fill missing values with appropriate defaults
            merged_df = merged_df.fillna({
                'total_permits': 0,
                'residential_permits': 0,
                'commercial_permits': 0,
                'total_construction_cost': 0,
                'residential_construction_cost': 0,
                'commercial_construction_cost': 0,
                'housing_units': 0,
                'is_residential': 0,
                'is_commercial': 0
            })

            # Validate the merged dataset
            if merged_df.empty:
                logger.error("Merged dataset is empty")
                return None

            if merged_df['year'].nunique() < economic_df['year'].nunique():
                logger.warning("Some years were lost during merge")

            if merged_df['zip_code'].nunique() < census_df['zip_code'].nunique():
                logger.warning("Some ZIP codes were lost during merge")

            return self._extracted_from_generate_retail_metrics_30(
                'merged_dataset.csv', merged_df, 'Merged dataset saved to '
            )
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            return None

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

    # TODO Rename this here and in `process_census_data`, `process_permit_data`, `process_economic_data`, `merge_datasets` and `generate_retail_metrics`
    def _extracted_from_generate_retail_metrics_30(self, arg0, arg1, arg2):
        output_file = self.processed_dir / arg0
        arg1.to_csv(output_file, index=False)
        logger.info(f"{arg2}{output_file}")
        return arg1
            
    def process_all(self):
        """Process all datasets."""
        try:
            logger.info("Starting complete data processing pipeline...")
            
            # Process individual datasets
            census_data = self.process_census_data()
            permit_data = self.process_permit_data()
            economic_data = self.process_economic_data()
            
            # Generate retail metrics
            retail_metrics = self.generate_retail_metrics()
            if retail_metrics is None:
                logger.error("Failed to generate retail metrics")
                return False
            
            # Merge datasets
            merged_data = self.merge_datasets()
            if merged_data is None:
                logger.error("Failed to merge datasets")
                return False
                
            logger.info("Data processing pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            return False 