"""
Data processing module for Chicago population analysis.
Handles data cleaning, transformation, and feature engineering.
"""

import logging
from typing import Dict, Optional, Tuple, List, Union
import json
import os
import traceback
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import geopandas as gpd

from src.config import settings
from .zoning import ZoningProcessor
from src.utils.helpers import resolve_column_name, ensure_directory
from src.config.column_alias_map import column_aliases, validate_column_value

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processor that coordinates all data processing pipelines."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.zoning_processor = ZoningProcessor()
        self.processed_data_dir = settings.PROCESSED_DATA_DIR
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def process_economic_data(self, df):
        """Process economic data."""
        try:
            logger.info(f"Processing economic data with shape: {df.shape}")

            # Ensure required columns exist
            required_cols = ['year', 'unemployment_rate', 'real_gdp', 'per_capita_income', 'personal_income']
            if missing_cols := [
                col for col in required_cols if col not in df.columns
            ]:
                logger.error(f"Required columns missing from economic data: {missing_cols}")
                return False

            # Convert year to numeric
            df['year'] = pd.to_numeric(df['year'], errors='coerce')

            # Drop rows with invalid years
            valid_years = df['year'].between(2010, 2030)
            if not valid_years.all():
                logger.warning(f"Found {(~valid_years).sum()} records with invalid years")
                df = df[valid_years]

            # Convert numeric columns
            numeric_cols = ['unemployment_rate', 'real_gdp', 'per_capita_income', 'personal_income']
            for col in numeric_cols:
                if col in df.columns and (missing_pct := (df[col].isna().sum() / len(df)) * 100) is not None:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"✅ {col} present, {missing_pct:.2f}% missing")

            # Fill missing values with forward fill then backward fill
            df = df.sort_values('year')
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

            # Calculate year-over-year changes
            for col in numeric_cols:
                pct_change_col = f"{col}_pct_change"
                df[pct_change_col] = df[col].pct_change() * 100
                df[pct_change_col] = df[pct_change_col].fillna(0)

            # Save processed data
            processed_path = self.processed_data_dir / 'economic_processed.csv'
            df.to_csv(processed_path, index=False)
            logger.info(f"Processed economic data saved to {processed_path}")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error processing economic data: ', e, False
            )
    
    def process_permits_data(self) -> bool:
        """Process building permit data."""
        try:
            logger.info("Loading raw permit records...")
            permit_file = settings.RAW_DATA_DIR / 'building_permits.csv'
            if not permit_file.exists():
                logger.error("Permit file not found")
                return False

            df = pd.read_csv(permit_file)
            logger.info(f"Loaded {len(df):,} raw permit records")

            # Ensure ZIP code is string type
            if 'contact_1_zipcode' in df.columns:
                df['zip_code'] = df['contact_1_zipcode'].astype(str).str.strip().str.zfill(5)
            else:
                logger.warning("No zip_code column found, attempting to extract from address")

            # Categorize permits
            df['residential_permits'] = df['work_description'].str.contains(
                'residential|house|apartment|condo|dwelling|home|townhouse|multi-family|single-family',
                case=False, regex=True
            ).astype(int)

            df['commercial_permits'] = df['work_description'].str.contains(
                'office|industrial|warehouse|factory|manufacturing|corporate|wholesale|distribution',
                case=False, regex=True
            ).astype(int)

            df['retail_permits'] = df['work_description'].str.contains(
                'retail|store|shop|restaurant|commercial|business|mall|market|sales',
                case=False, regex=True
            ).astype(int)

            # Calculate costs by type
            for permit_type in ['residential', 'commercial', 'retail']:
                cost_col = f'{permit_type}_construction_cost'
                df[cost_col] = df['reported_cost'].where(df[f'{permit_type}_permits'] == 1, 0)

            # Group by ZIP code and year
            df['year'] = pd.to_datetime(df['issue_date']).dt.year

            agg_dict = {
                'residential_permits': 'sum',
                'commercial_permits': 'sum', 
                'retail_permits': 'sum',
                'residential_construction_cost': 'sum',
                'commercial_construction_cost': 'sum',
                'retail_construction_cost': 'sum',
                'reported_cost': 'sum'
            }

            df_grouped = df.groupby(['zip_code', 'year']).agg(agg_dict).reset_index()

            # Ensure ZIP codes are strings
            df_grouped['zip_code'] = df_grouped['zip_code'].astype(str).str.strip().str.zfill(5)

            # Save processed data
            output_path = settings.PROCESSED_DATA_DIR / 'permits_processed.csv'
            df_grouped.to_csv(output_path, index=False)
            logger.info(f"Processed permit data saved to {output_path}")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error processing permit data: ', e, False
            )
    
    def _process_core_sources(self) -> Dict[str, bool]:
        """Process core data sources."""
        return {
            'census': self.process_census_data(),
            'permits': self.process_permits_data(),
            'business_licenses': self.process_business_licenses()
        }

    def _process_optional_sources(self) -> Dict[str, bool]:
        """Process optional data sources."""
        results = {'property': False, 'zoning': False}
        if os.path.exists(settings.PROPERTY_DATA_PATH):
            results['property'] = self.process_property_data()
        else:
            logger.warning("Property data file not found - skipping")

        if os.path.exists(settings.ZONING_DATA_PATH):
            results['zoning'] = self.process_zoning_data()
        else:
            logger.warning("Zoning data file not found - skipping")
        return results

    def _merge_processed_files(self, results: Dict[str, bool]) -> Optional[pd.DataFrame]:
        """Merge all processed data files into a single dataset."""
        try:
            logger.info("Merging processed datasets...")
            processed_files = {}
            
            # Load all successfully processed files
            for name, success in results.items():
                if not success:
                    continue
                    
                path = self.processed_data_dir / f"{name}_processed.csv"
                if not path.exists():
                    logger.warning(f"Processed file not found: {path}")
                    continue
                
                # Load file and log columns    
                processed_files[name] = pd.read_csv(path, low_memory=False)
                logger.info(f"{name} columns: {processed_files[name].columns.tolist()}")

            # Start with census data as base
            if 'census' not in processed_files:
                logger.error("Census data missing - required for base dataset")
                return None
                
            logger.info(f"Base census columns: {processed_files['census'].columns.tolist()}")
            merged_df = processed_files['census'].copy()
            
            # Ensure consistent data types for merge keys
            merged_df['zip_code'] = merged_df['zip_code'].astype(str).str.zfill(5)
            merged_df['year'] = pd.to_numeric(merged_df['year'], errors='coerce')

            # Merge permits data if available
            if 'permits' in processed_files:
                logger.info("Merging permits on ['zip_code', 'year']")
                logger.info(f"Pre-merge columns: {merged_df.columns.tolist()}")
                try:
                    permits_df = processed_files['permits'].copy()
                    permits_df['zip_code'] = permits_df['zip_code'].astype(str).str.zfill(5)
                    permits_df['year'] = pd.to_numeric(permits_df['year'], errors='coerce')
                    
                    merged_df = pd.merge(
                        merged_df,
                        permits_df,
                        on=['zip_code', 'year'],
                        how='left'
                    )
                    
                    # Fill missing permit values with 0
                    permit_cols = [
                        'residential_permits', 'commercial_permits', 'retail_permits',
                        'residential_construction_cost', 'commercial_construction_cost',
                        'retail_construction_cost', 'total_permits', 'total_construction_cost'
                    ]
                    for col in permit_cols:
                        if col in merged_df.columns:
                            merged_df[col] = merged_df[col].fillna(0)
                            logger.info(f"Filled {col} NaN values with 0")
                        else:
                            merged_df[col] = 0
                            logger.info(f"Added missing column {col} with zeros")
                except Exception as e:
                    logger.error(f"Error merging permits: {str(e)}")
                    # Add missing permit columns with zeros
                    permit_cols = [
                        'residential_permits', 'commercial_permits', 'retail_permits',
                        'residential_construction_cost', 'commercial_construction_cost',
                        'retail_construction_cost', 'total_permits', 'total_construction_cost'
                    ]
                    for col in permit_cols:
                        if col not in merged_df.columns:
                            merged_df[col] = 0
                            logger.info(f"Added missing column {col} with zeros")

            # Merge business license data if available
            if 'business_licenses' in processed_files:
                logger.info("Merging business_licenses on ['zip_code', 'year']")
                logger.info(f"Pre-merge columns: {merged_df.columns.tolist()}")
                try:
                    licenses_df = processed_files['business_licenses'].copy()
                    licenses_df['zip_code'] = licenses_df['zip_code'].astype(str).str.zfill(5)
                    licenses_df['year'] = pd.to_numeric(licenses_df['year'], errors='coerce')
                    
                    merged_df = pd.merge(
                        merged_df,
                        licenses_df,
                        on=['zip_code', 'year'],
                        how='left'
                    )
                except Exception as e:
                    logger.error(f"Error merging business_licenses: {str(e)}")

            # Merge economic data if available - note this is city-wide data
            if 'economic' in processed_files:
                logger.info("Merging economic data on ['year'] only - applying city-wide values")
                try:
                    economic_df = processed_files['economic'].copy()
                    economic_df['year'] = pd.to_numeric(economic_df['year'], errors='coerce')
                    merged_df['year'] = pd.to_numeric(merged_df['year'], errors='coerce')
                    
                    # Merge economic data on year only
                    merged_df = pd.merge(
                        merged_df,
                        economic_df,
                        on='year',
                        how='left'
                    )
                    logger.info(f"Economic columns added: {economic_df.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error merging economic data: {str(e)}")

            # Verify data types after merges
            logger.info("\nVerifying data types:")
            for col in merged_df.columns:
                logger.info(f"{col}: {merged_df[col].dtype}")

            # Save merged dataset
            merged_df.to_csv(settings.MERGED_DATA_PATH, index=False)
            logger.info(f"Merged dataset saved to {str(settings.MERGED_DATA_PATH)}")
            logger.info(f"Final merged columns: {merged_df.columns.tolist()}")
            
            # Log summary statistics
            logger.info("\nMerged dataset summary:")
            logger.info(f"Total rows: {len(merged_df)}")
            logger.info(f"Unique ZIP codes: {merged_df['zip_code'].nunique()}")
            logger.info(f"Years covered: {sorted(merged_df['year'].unique())}")
            
            return merged_df

        except Exception as e:
            logger.error(f"Error in merge process: {str(e)}")
            return None

    def process_all(self) -> bool:
        """Run all processing steps, including retail deficit."""
        try:
            logger.info("Starting data processing pipeline...")
            
            # Process core data sources
            core_results = self._process_core_sources()
            if not all(core_results.values()):
                failed = [k for k, v in core_results.items() if not v]
                logger.error(f"Core processing failed for: {', '.join(failed)}")
                return False
            
            # Process optional sources
            optional_results = self._process_optional_sources()
            
            # Process economic data
            economic_success = self.process_economic_data()
            if not economic_success:
                logger.warning("Economic data processing failed - some features may be missing")
            
            # Merge processed files
            merged_data = self._merge_processed_files({**core_results, **optional_results})
            if merged_data is None:
                logger.error("Failed to merge processed files")
                return False
                
            # Process retail deficit
            if not (processed_deficit := self.process_retail_deficit()):
                logger.warning("Retail deficit processing failed")
            
            logger.info("Data processing pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False
    
    def process_census_data(self, df):
        """Process census data."""
        try:
            logger.info(f"🔍 Raw census data columns: {df.columns.tolist()}")

            # Deduplicate 'zip_code' columns if present
            cols = df.columns.tolist()
            if cols.count('zip_code') > 1:
                df = df.loc[:, ~df.columns.duplicated()]
                logger.warning("Duplicate 'zip_code' columns found and removed in census data.")

            # Rename columns to match our schema
            column_map = {
                'zip code tabulation area': 'zip_code',
                'B01003_001E': 'total_population',
                'B19013_001E': 'median_household_income',
                'B25077_001E': 'median_home_value',
                'B23025_002E': 'labor_force'
            }

            # Apply column mapping
            df = df.rename(columns=column_map)
            logger.info(f"Census columns after renaming: {df.columns.tolist()}")

            # Ensure required columns exist
            required_cols = ['total_population', 'median_household_income', 'median_home_value', 'labor_force', 'zip_code', 'year']

            if missing_cols := [
                col for col in required_cols if col not in df.columns
            ]:
                logger.error(f"Required columns missing from census data: {missing_cols}")
                return None

            # Validate ZIP code column
            if 'zip_code' not in df.columns:
                # Try alternative column names
                if zcta_cols := [col for col in df.columns if any(x in col.lower() for x in ['zcta', 'zip', 'tabulation'])]:
                    logger.info(f"Found ZCTA column: {zcta_cols[0]}")
                    df = df.rename(columns={zcta_cols[0]: 'zip_code'})
                else:
                    logger.error("No ZCTA/ZIP column found")
                    logger.error(f"Available columns: {df.columns.tolist()}")
                    return None

            # Format ZIP codes
            df['zip_code'] = df['zip_code'].astype(str).str.strip().str.zfill(5)

            # Convert numeric columns
            numeric_cols = ['total_population', 'median_household_income', 'median_home_value', 'labor_force']
            for col in numeric_cols:
                if col in df.columns and (missing_pct := (df[col].isna().sum() / len(df)) * 100) is not None:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"✅ {col} present after mapping, {missing_pct:.2f}% missing")

            # Drop rows with missing values
            df = df.dropna(subset=numeric_cols)
            logger.info(f"Shape after dropping missing values: {df.shape}")

            # Save processed data
            processed_path = self.processed_data_dir / 'census_processed.csv'
            df.to_csv(processed_path, index=False)
            logger.info(f"Processed census data saved to {processed_path}")

            return df

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error processing census data: ', e, None
            )
    
    def process_retail_deficit(self) -> bool:
        """Process retail deficit data."""
        try:
            # Load retail metrics
            retail_metrics_path = self.processed_data_dir / 'retail_metrics.csv'
            if not retail_metrics_path.exists():
                logger.warning("Retail metrics not found, generating from scratch")
                retail_metrics = self.generate_retail_metrics()
                if retail_metrics is None:
                    logger.error("Failed to generate retail metrics")
                    return False
            else:
                retail_metrics = pd.read_csv(retail_metrics_path)

            # Calculate expected retail demand based on population and income
            retail_metrics['expected_retail_demand'] = (
                retail_metrics['total_population'] * 
                retail_metrics['median_household_income'] * 
                0.3  # Assume 30% of income goes to retail
            )

            # Calculate actual retail supply based on permits and construction
            retail_metrics['actual_retail_supply'] = retail_metrics['retail_construction_cost'].fillna(0)

            # Calculate retail deficit
            retail_metrics['retail_deficit'] = (
                retail_metrics['expected_retail_demand'] - 
                retail_metrics['actual_retail_supply']
            )

            # Calculate retail deficit per capita
            retail_metrics['retail_deficit_per_capita'] = (
                retail_metrics['retail_deficit'] / 
                retail_metrics['total_population']
            )

            # Save retail deficit metrics
            retail_deficit_path = self.processed_data_dir / 'retail_deficit.csv'
            retail_metrics.to_csv(retail_deficit_path, index=False)
            logger.info(f"Saved retail deficit metrics to {retail_deficit_path}")

            # Log summary statistics
            logger.info("Retail deficit summary:")
            logger.info(f"- Total expected demand: ${retail_metrics['expected_retail_demand'].sum():,.2f}")
            logger.info(f"- Total actual supply: ${retail_metrics['actual_retail_supply'].sum():,.2f}")
            logger.info(f"- Total deficit: ${retail_metrics['retail_deficit'].sum():,.2f}")
            logger.info(f"- Average deficit per capita: ${retail_metrics['retail_deficit_per_capita'].mean():,.2f}")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error processing retail deficit: ', e, False
            )

    def process_business_licenses(self) -> bool:
        """Process business license data and add active_licenses as total_licenses."""
        try:
            df = pd.read_csv(settings.BUSINESS_LICENSES_PATH)
            date_columns = ['license_start_date', 'expiration_date', 'application_created_date']
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            df['year'] = df['license_start_date'].dt.year
            processed_df = df.groupby(['zip_code', 'year']).agg({
                'license_id': 'count',
                'account_number': 'nunique'
            }).reset_index()
            processed_df = processed_df.rename(columns={
                'license_id': 'total_licenses',
                'account_number': 'unique_businesses'
            })
            # If you want to count only currently active licenses, add logic here.
            # For now, active_licenses is set to total_licenses (all licenses in that year).
            processed_df['active_licenses'] = processed_df['total_licenses']
            processed_df.to_csv(settings.BUSINESS_LICENSES_PROCESSED_PATH, index=False)
            logger.info("Successfully processed business license data")
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
            df = pd.read_csv(settings.MERGED_DATA_PATH, low_memory=False)
            logger.info(f"Loaded merged dataset with {len(df)} rows")

            # Verify required columns exist
            required_cols = [
                'residential_permits', 'commercial_permits', 'retail_permits',
                'residential_construction_cost', 'commercial_construction_cost', 'retail_construction_cost'
            ]

            if missing_cols := [
                col for col in required_cols if col not in df.columns
            ]:
                logger.error(f"Required columns missing from merged dataset: {missing_cols}")
                return None

            # Calculate total permits and costs
            permit_cols = ['residential_permits', 'commercial_permits', 'retail_permits']
            cost_cols = ['residential_construction_cost', 'commercial_construction_cost', 'retail_construction_cost']

            # Calculate total_permits if not present
            if 'total_permits' not in df.columns:
                logger.info("Calculating total_permits from individual permit types")
                df['total_permits'] = df[permit_cols].sum(axis=1)

            # Calculate total_construction_cost if not present
            if 'total_construction_cost' not in df.columns:
                logger.info("Calculating total_construction_cost from individual costs")
                df['total_construction_cost'] = df[cost_cols].sum(axis=1)

            # Calculate permit ratios safely
            for col in permit_cols:
                ratio_col = f"{col.replace('permits', 'permit')}_ratio"
                df[ratio_col] = np.where(
                    df['total_permits'] > 0,
                    df[col] / df['total_permits'],
                    0
                )

            # Calculate cost ratios safely
            for col in cost_cols:
                ratio_col = f"{col.replace('construction_cost', 'cost')}_ratio"
                df[ratio_col] = np.where(
                    df['total_construction_cost'] > 0,
                    df[col] / df['total_construction_cost'],
                    0
                )

            # Log summary statistics
            logger.info("Generated retail metrics:")
            logger.info(f"- Total permits: {df['total_permits'].sum():,}")
            logger.info(f"- Total construction cost: ${df['total_construction_cost'].sum():,.2f}")
            logger.info("Permit distribution:")
            for col in permit_cols:
                total = df[col].sum()
                pct = (total / df['total_permits'].sum() * 100) if df['total_permits'].sum() > 0 else 0
                logger.info(f"- {col}: {total:,} ({pct:.1f}%)")

            # Save retail metrics
            df.to_csv(settings.RETAIL_METRICS_PATH, index=False)
            logger.info(f"Saved retail metrics to {settings.RETAIL_METRICS_PATH}")

            return df

        except Exception as e:
            logger.error(f"Error generating retail metrics: {str(e)}")
            return None

    def process_property_data(self) -> bool:
        """Process property transaction data."""
        try:
            # Load property data
            df = pd.read_csv(settings.PROPERTY_DATA_PATH)
            
            # Convert date column
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            df['year'] = df['sale_date'].dt.year
            
            # Convert numeric columns
            numeric_cols = ['sale_price', 'property_sq_ft', 'year_built']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate metrics by ZIP code and year
            processed_df = df.groupby(['zip_code', 'year']).agg({
                'sale_price': ['mean', 'median', 'count'],
                'property_sq_ft': 'mean',
                'property_type': lambda x: x.value_counts().index[0]  # Most common property type
            }).reset_index()
            
            # Flatten column names
            processed_df.columns = ['_'.join(col).strip('_') for col in processed_df.columns.values]
            
            # Save processed data
            processed_df.to_csv(settings.PROPERTY_PROCESSED_PATH, index=False)
            logger.info("Successfully processed property data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing property data: {str(e)}")
            return False

    def process_zoning_data(self, df) -> bool:
        """Process zoning data."""
        try:
            if df is None:
                logger.warning("No zoning data provided for processing")
                return False

            logger.info(f"Processing zoning data with shape: {df.shape}")

            # Ensure required columns exist
            required_cols = ['zip_code', 'zoning_classification', 'zone_category', 'total_parcels', 'avg_lot_size']
            if missing_cols := [
                col for col in required_cols if col not in df.columns
            ]:
                logger.error(f"Required columns missing from zoning data: {missing_cols}")
                return False

            # Convert numeric columns
            numeric_cols = ['total_parcels', 'avg_lot_size', 'total_area']
            for col in numeric_cols:
                if col in df.columns and (missing_pct := (df[col].isna().sum() / len(df)) * 100) is not None:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"✅ {col} present, {missing_pct:.2f}% missing")

            # Fill missing values with forward fill then backward fill
            df = df.sort_values(['zip_code', 'zoning_classification'])
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

            # Calculate zoning metrics
            metrics = df.groupby('zip_code').agg({
                'total_parcels': 'sum',
                'avg_lot_size': 'mean',
                'total_area': 'sum'
            }).reset_index()

            # Add zoning diversity metrics
            zoning_counts = df.groupby('zip_code')['zoning_classification'].nunique()
            metrics['zoning_diversity'] = metrics['zip_code'].map(zoning_counts)

            # Save processed data
            processed_path = self.processed_data_dir / 'zoning_processed.csv'
            metrics.to_csv(processed_path, index=False)
            logger.info(f"Processed zoning data saved to {processed_path}")

            # Log summary statistics
            logger.info("Zoning metrics summary:")
            logger.info(f"- Total parcels: {metrics['total_parcels'].sum():,}")
            logger.info(f"- Average lot size: {metrics['avg_lot_size'].mean():,.0f} sq ft")
            logger.info(f"- Total area: {metrics['total_area'].sum():,.0f} sq ft")
            logger.info(f"- Average zoning diversity: {metrics['zoning_diversity'].mean():.1f} classifications per ZIP")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error processing zoning data: ', e, False
            )

    def run_pipeline(self) -> bool:
        """Run the full data processing pipeline."""
        try:
            logger.info("Starting data processing pipeline...")

            # Process each data source
            results = {
                'census': self.process_census_data(),
                'permits': self.process_permits_data(),
                'business_licenses': self.process_business_licenses(),
                'economic': self.process_economic_data()
            }

            # Log processing results
            for name, success in results.items():
                logger.info(f"{name} processing: {'✅ Success' if success else '❌ Failed'}")

            # Verify processed files exist and log their columns
            for name in results:
                path = self.processed_data_dir / f"{name}_processed.csv"
                if path.exists():
                    df = pd.read_csv(path, low_memory=False)
                    logger.info(f"\n{name} processed data:")
                    logger.info(f"- Shape: {df.shape}")
                    logger.info(f"- Columns: {df.columns.tolist()}")
                    if 'year' in df.columns:
                        logger.info(f"- Years: {df['year'].unique().tolist()}")
                    if 'zip_code' in df.columns:
                        logger.info(f"- ZIP codes: {len(df['zip_code'].unique())} unique")

            # Merge processed files
            merged_df = self._merge_processed_files(results)
            if merged_df is None:
                logger.error("Failed to merge processed files")
                return False

            # Verify merged dataset
            logger.info("\nMerged dataset verification:")
            logger.info(f"- Shape: {merged_df.shape}")
            logger.info(f"- Columns: {merged_df.columns.tolist()}")
            logger.info(f"- Missing values:\n{merged_df.isnull().sum()}")

            # Verify key columns
            key_columns = [
                'total_population', 'median_household_income',
                'residential_permits', 'commercial_permits', 'retail_permits',
                'residential_construction_cost', 'commercial_construction_cost', 'retail_construction_cost',
                'gdp', 'unemployment_rate'
            ]

            logger.info("\nKey column statistics:")
            for col in key_columns:
                if col in merged_df.columns:
                    stats = merged_df[col].describe()
                    logger.info(f"\n{col}:")
                    logger.info("- Present: ✅")
                    logger.info(f"- Non-null: {merged_df[col].count()}")
                    logger.info(f"- Mean: {stats['mean']:.2f}")
                    logger.info(f"- Min: {stats['min']:.2f}")
                    logger.info(f"- Max: {stats['max']:.2f}")
                else:
                    logger.warning(f"{col}: ❌ Missing")

            logger.info("Data processing pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            return False

    def process_all_data(self, census_data=None, permit_data=None, economic_data=None, zoning_data=None):
        """Process all data sources."""
        try:
            logger.info("Starting data processing pipeline...")

            # Process census data
            if census_data is not None:
                processed_census = self.process_census_data(census_data)
                if processed_census is None:
                    logger.error("Core processing failed for: census")
                    return False
                logger.info("Successfully processed census data")
            else:
                logger.warning("No census data provided for processing")
                return False

            # Process permit data
            if permit_data is not None:
                processed_permits = self.process_permit_data(permit_data)
                if not processed_permits:
                    logger.error("Core processing failed for: permits")
                    return False
                logger.info("Successfully processed permit data")
            else:
                logger.warning("No permit data provided for processing")
                return False

            # Process economic data
            if economic_data is not None:
                processed_economic = self.process_economic_data(economic_data)
                if not processed_economic:
                    logger.error("Core processing failed for: economic")
                    return False
                logger.info("Successfully processed economic data")
            else:
                logger.warning("No economic data provided for processing")
                return False

            # Process zoning data
            if zoning_data is not None:
                processed_zoning = self.process_zoning_data(zoning_data)
                if not processed_zoning:
                    logger.error("Core processing failed for: zoning")
                    return False
                logger.info("Successfully processed zoning data")
            else:
                logger.warning("No zoning data provided for processing")

            if processed_licenses := self.process_business_licenses():
                logger.info("Successfully processed business license data")

            else:
                logger.warning("Business license processing failed")
            # Process retail deficit
            if not (processed_deficit := self.process_retail_deficit()):
                logger.warning("Retail deficit processing failed")
            else:
                logger.info("Successfully processed retail deficit data")

            # Save processing summary
            self._save_processing_summary()

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error in data processing pipeline: ', e, False
            ) 

    def process_permit_data(self, df):
        """Process building permit data."""
        try:
            logger.info(f"Loaded {len(df)} raw permit records")

            # Check for permit ID column
            if 'permit_id' not in df.columns:
                logger.warning("No permit ID column found, using index")
                df['permit_id'] = df.index

            # Ensure required columns exist
            required_cols = ['permit_id', 'permit_type', 'total_fee', 'reported_cost']
            if missing_cols := [
                col for col in required_cols if col not in df.columns
            ]:
                logger.warning(f"Missing columns: {missing_cols}")
                for col in missing_cols:
                    df[col] = None
                logger.warning(f"Added missing column {col} with default value")

            # Extract ZIP code from address if needed
            if 'zip_code' not in df.columns:
                logger.warning("No zip_code column found, attempting to extract from address")

                # Try to extract from contact_1_zipcode first
                if 'contact_1_zipcode' in df.columns:
                    df['zip_code'] = df['contact_1_zipcode'].astype(str).str.strip().str.zfill(5)
                    logger.info("Extracted ZIP codes from contact_1_zipcode")

                # If still no ZIP codes, try to extract from address
                elif all(col in df.columns for col in ['street_number', 'street_direction', 'street_name']):
                    df['address'] = df['street_number'].astype(str) + ' ' + \
                                          df['street_direction'].astype(str) + ' ' + \
                                          df['street_name'].astype(str)

                    # Use the improved ZIP extraction logic
                    df['zip_code'] = df['address'].apply(self.extract_zip)
                else:
                    logger.error("No address columns found to extract ZIP code")
                    return False

            # Clean permit types
            df['permit_type'] = df['permit_type'].fillna('Other')
            df['permit_type'] = df['permit_type'].str.strip().str.title()

            # Map permit types to categories
            permit_type_map = {
                'New Construction': 'Residential',
                'Renovation/Alteration': 'Residential',
                'Addition': 'Residential',
                'Porch': 'Residential',
                'Garage': 'Residential',
                'Commercial': 'Commercial',
                'Business': 'Commercial',
                'Office': 'Commercial',
                'Industrial': 'Commercial',
                'Retail': 'Retail',
                'Restaurant': 'Retail',
                'Store': 'Retail',
                'Shop': 'Retail'
            }

            # Apply mapping with fallback to 'Other'
            df['permit_category'] = df['permit_type'].map(permit_type_map).fillna('Other')

            # Log permit type distribution
            type_counts = df['permit_category'].value_counts()
            logger.info("Permit type distribution:")
            for category, count in type_counts.items():
                logger.info(f"- {category}: {count:,} permits")

            # Convert costs to numeric
            df['reported_cost'] = pd.to_numeric(df['reported_cost'], errors='coerce')
            df['total_fee'] = pd.to_numeric(df['total_fee'], errors='coerce')

            # Add year column if missing
            if 'year' not in df.columns and 'issue_date' in df.columns:
                df['year'] = pd.to_datetime(df['issue_date']).dt.year
            elif 'year' not in df.columns:
                df['year'] = datetime.now().year
                logger.warning(f"No year column found, using current year: {df['year'].iloc[0]}")

            # Aggregate by type and year
            logger.info("Aggregating permits by type and year...")
            agg_df = df.groupby(['zip_code', 'year', 'permit_category']).agg({
                'permit_id': 'count',
                'reported_cost': 'sum'
            }).reset_index()

            # Pivot to get permit counts and costs by type
            permits_pivot = agg_df.pivot_table(
                index=['zip_code', 'year'],
                columns='permit_category',
                values=['permit_id', 'reported_cost'],
                fill_value=0
            ).reset_index()

            # Flatten column names
            permits_pivot.columns = [
                f"{col[1].lower()}_{col[0]}" if col[1] != "" 
                else col[0] for col in permits_pivot.columns
            ]

            # Add total columns
            permits_pivot['total_permits'] = permits_pivot[[col for col in permits_pivot.columns 
                                                          if col.endswith('permit_id')]].sum(axis=1)
            permits_pivot['total_construction_cost'] = permits_pivot[[col for col in permits_pivot.columns 
                                                                    if col.endswith('reported_cost')]].sum(axis=1)

            # Log summary statistics
            logger.info(f"Processed {len(df):,} permits:")
            for category in ['Residential', 'Commercial', 'Retail']:
                col = f"{category.lower()}_permit_id"
                if col in permits_pivot.columns:
                    logger.info(f"- {category}: {int(permits_pivot[col].sum()):,} permits")

            logger.info(f"Total construction cost: ${permits_pivot['total_construction_cost'].sum():,.2f}")

            # Save processed data
            processed_path = self.processed_data_dir / 'permits_processed.csv'
            permits_pivot.to_csv(processed_path, index=False)
            logger.info(f"Processed permit data saved to {processed_path}")

            # Log column names for debugging
            logger.info(f"Processed permits columns: {permits_pivot.columns.tolist()}")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error processing permit data: ', e, False
            ) 

    def _save_processing_summary(self):
        """Save processing summary to JSON."""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'processed_files': {
                    'census': str(self.processed_data_dir / 'census_processed.csv'),
                    'permits': str(self.processed_data_dir / 'permits_processed.csv'),
                    'economic': str(self.processed_data_dir / 'economic_processed.csv'),
                    'retail_metrics': str(self.processed_data_dir / 'retail_metrics.csv')
                },
                'metrics': {
                    'census_records': 0,
                    'permit_records': 0,
                    'economic_indicators': 0,
                    'retail_metrics': 0
                }
            }

            # Count records in each file
            for file_type, file_path in summary['processed_files'].items():
                path = Path(file_path)
                if path.exists():
                    try:
                        df = pd.read_csv(path)
                        summary['metrics'][f'{file_type}_records'] = len(df)
                    except Exception as e:
                        logger.warning(f"Could not read {file_type} file: {str(e)}")

            # Save summary
            summary_path = self.processed_data_dir / 'processing_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved processing summary to {summary_path}")

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error saving processing summary: ', e, False
            ) 

    def process_retail_data(self, data: pd.DataFrame) -> bool:
        """Process retail data.
        
        Args:
            data (pd.DataFrame): Raw retail data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get census data for population and income
            census_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'census_processed.csv')
            current_year = census_data['year'].max()
            census_current = census_data[census_data['year'] == current_year]

            # Calculate retail space from permits
            retail_space = data.groupby(['zip_code', 'year']).agg({
                'retail_permits': 'sum',
                'retail_construction_cost': 'sum'
            }).reset_index()

            # Merge with census data
            retail_metrics = retail_space.merge(
                census_current[['zip_code', 'total_population', 'median_household_income']],
                on='zip_code',
                how='outer'
            )

            # Fill missing values
            retail_metrics = retail_metrics.fillna({
                'retail_permits': 0,
                'retail_construction_cost': 0,
                'total_population': retail_metrics['total_population'].mean(),
                'median_household_income': retail_metrics['median_household_income'].mean(),
                'year': current_year
            })

            # Estimate retail space from construction cost
            # Assuming average cost of $200 per square foot for retail construction
            retail_metrics['retail_space'] = retail_metrics['retail_construction_cost'] / 200

            # Estimate annual retail spending per capita (30% of income)
            retail_metrics['retail_demand'] = retail_metrics['total_population'] * retail_metrics['median_household_income'] * 0.3

            # Calculate retail supply (annual sales per square foot)
            retail_metrics['retail_supply'] = retail_metrics['retail_space'] * 300  # Assume $300 annual sales per sq ft

            # Calculate retail gap (demand - supply)
            retail_metrics['retail_gap'] = retail_metrics['retail_demand'] - retail_metrics['retail_supply']

            # Calculate retail leakage (gap / demand)
            retail_metrics['retail_leakage'] = retail_metrics['retail_gap'] / retail_metrics['retail_demand']

            # Calculate vacancy rate (assume 10% base + gap factor)
            retail_metrics['vacancy_rate'] = 0.10 + (retail_metrics['retail_gap'] / retail_metrics['retail_demand']).clip(0, 0.2)

            # Calculate retail opportunity score (normalized gap)
            retail_metrics['opportunity_score'] = (retail_metrics['retail_gap'] - retail_metrics['retail_gap'].mean()) / retail_metrics['retail_gap'].std()

            # Identify high opportunity areas
            retail_metrics['high_opportunity'] = retail_metrics['opportunity_score'] > 1.0

            # Save retail metrics
            retail_metrics.to_csv(settings.PROCESSED_DATA_DIR / 'retail_metrics.csv', index=False)

            # Calculate and save retail deficit metrics
            retail_deficit = retail_metrics.copy()
            retail_deficit['retail_deficit'] = retail_deficit['retail_gap'].clip(lower=0)
            retail_deficit['retail_surplus'] = retail_deficit['retail_gap'].clip(upper=0).abs()
            retail_deficit.to_csv(settings.PROCESSED_DATA_DIR / 'retail_deficit.csv', index=False)

            # Log summary statistics
            total_demand = retail_metrics['retail_demand'].sum()
            total_supply = retail_metrics['retail_supply'].sum()
            total_deficit = retail_metrics['retail_gap'].sum()
            avg_deficit_per_capita = retail_metrics['retail_gap'].mean() / retail_metrics['total_population'].mean()

            logger.info("Retail deficit summary:")
            logger.info(f"- Total expected demand: ${total_demand:,.2f}")
            logger.info(f"- Total actual supply: ${total_supply:,.2f}")
            logger.info(f"- Total deficit: ${total_deficit:,.2f}")
            logger.info(f"- Average deficit per capita: ${avg_deficit_per_capita:,.2f}")

            # Save summary metrics
            summary_metrics = pd.DataFrame({
                'metric': ['total_demand', 'total_supply', 'total_deficit', 'avg_deficit_per_capita'],
                'value': [total_demand, total_supply, total_deficit, avg_deficit_per_capita]
            })
            summary_metrics.to_csv(settings.PROCESSED_DATA_DIR / 'retail_summary_metrics.csv', index=False)

            # Save retail metrics to a separate file for each ZIP code
            for zip_code in retail_metrics['zip_code'].unique():
                zip_metrics = retail_metrics[retail_metrics['zip_code'] == zip_code]
                zip_metrics.to_csv(settings.PROCESSED_DATA_DIR / f'retail_metrics_{zip_code}.csv', index=False)

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                'Error processing retail data: ', e, False
            ) 

    # TODO Rename this here and in `process_economic_data`, `process_permits_data`, `process_census_data`, `process_retail_deficit`, `process_zoning_data`, `process_all_data`, `process_permit_data`, `_save_processing_summary` and `process_retail_data`
    def _extracted_from_process_retail_data_48(self, arg0, e, arg2):
        logger.error(f"{arg0}{str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return arg2 

    def process_retail_metrics(self, df):
        required_cols = ['retail_space', 'retail_demand', 'retail_gap', 'vacancy_rate', 'retail_supply']
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            elif col == 'retail_gap' and 'retail_demand' in df.columns and 'retail_supply' in df.columns:
                df['retail_gap'] = df['retail_demand'] - df['retail_supply']
            else:
                df[col] = 0
                logger.warning(f"Added missing column {col} to retail data with default value 0")
        return df

    def extract_zip(self, address):
        # Extract 5-digit ZIP codes, prefer those starting with 606 (Chicago)
        if not isinstance(address, str):
            return None
        if zips := re.findall(r'\b60\d{3}\b', address):
            return zips[0]
        if all_zips := re.findall(r'\b\d{5}\b', address):
            logger.warning(f"Non-Chicago ZIP found in address: {address}")
            return all_zips[0]
        logger.warning(f"No valid ZIP found in address: {address}")
        return None 