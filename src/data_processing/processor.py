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
    
    def process_economic_data(self) -> bool:
        """Process economic data and add employment if possible."""
        try:
            df = pd.read_csv(settings.ECONOMIC_DATA_PATH)
            print('ECONOMIC DATA COLUMNS:', df.columns.tolist())  # Debug print
            
            # Clean and process economic data
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Economic indicators are city-wide, so we'll replicate them for each ZIP code
            chicago_zips = settings.CHICAGO_ZIP_CODES
            years = df['year'].unique()
            
            # Create a cross product of ZIP codes and years
            zip_years = pd.DataFrame([(zip_code, year) 
                                    for zip_code in chicago_zips 
                                    for year in years],
                                   columns=['zip_code', 'year'])
            
            # Merge with economic data
            df = zip_years.merge(df, on='year', how='left')
            
            # Save processed data
            df.to_csv(settings.ECONOMIC_PROCESSED_PATH, index=False)
            logger.info("Successfully processed economic data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing economic data: {str(e)}")
            return False
    
    def process_permits_data(self) -> bool:
        """Process building permit data."""
        try:
            # Load permit data
            df = pd.read_csv(settings.PERMITS_RAW_PATH, low_memory=False)
            logger.info(f"Loaded {len(df)} raw permit records")

            # Standardize column names - handle multiple possible names
            cost_columns = ['total_cost', 'estimated_cost', 'reported_cost', 'construction_cost']
            if cost_col := next(
                (col for col in cost_columns if col in df.columns), None
            ):
                # Standardize the column name
                df = df.rename(columns={cost_col: 'total_cost'})
                # Clean cost values
                df['total_cost'] = pd.to_numeric(df['total_cost'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
            else:
                # If no cost column found, create one with zeros
                df['total_cost'] = 0
                logger.warning("No cost column found in permits data, using zeros")

            # Handle permit ID
            id_columns = ['permit_id', 'id', 'permit_number', 'application_number']
            if id_col := next(
                (col for col in id_columns if col in df.columns), None
            ):
                df = df.rename(columns={id_col: 'permit_id'})
            else:
                df['permit_id'] = df.index
                logger.warning("No permit ID column found, using index")

            # Ensure required columns exist
            required_columns = {
                'permit_id': df.index,
                'total_cost': 0,
                'issue_date': pd.Timestamp.now(),
                'permit_type': 'UNKNOWN',
                'zip_code': '00000'
            }

            for col, default in required_columns.items():
                if col not in df.columns:
                    df[col] = default
                    logger.warning(f"Added missing column {col} with default value")

            # Convert issue_date to datetime and extract year
            df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
            df['year'] = df['issue_date'].dt.year

            # Add permit type classification
            df['permit_type'] = 'Other'
            
            # Classify permits based on work description
            retail_keywords = ['retail', 'store', 'shop', 'restaurant', 'commercial', 'business', 'mall', 'market', 'sales']
            residential_keywords = ['residential', 'house', 'apartment', 'condo', 'dwelling', 'home', 'townhouse', 'multi-family', 'single-family']
            commercial_keywords = ['office', 'industrial', 'warehouse', 'factory', 'manufacturing', 'corporate', 'wholesale', 'distribution']
            
            # Convert work description to lowercase for case-insensitive matching
            work_desc = df['work_description'].str.lower()
            
            # Classify permits
            df.loc[work_desc.str.contains('|'.join(retail_keywords), na=False), 'permit_type'] = 'Retail'
            df.loc[work_desc.str.contains('|'.join(residential_keywords), na=False), 'permit_type'] = 'Residential'
            df.loc[work_desc.str.contains('|'.join(commercial_keywords), na=False), 'permit_type'] = 'Commercial'

            # Log permit type distribution
            type_counts = df['permit_type'].value_counts()
            logger.info("Permit type distribution:")
            for permit_type, count in type_counts.items():
                logger.info(f"- {permit_type}: {count:,} permits")

            # Aggregate permits by type and year
            logger.info("Aggregating permits by type and year...")
            
            # Create aggregation dictionary
            agg_dict = {
                'permit_id': 'count',  # Count of permits
                'total_cost': 'sum'    # Sum of costs
            }
            
            # Group by ZIP code, year, and permit type
            grouped = df.groupby(['zip_code', 'year', 'permit_type']).agg(agg_dict)
            
            # Unstack permit types to get separate columns
            permit_counts = grouped['permit_id'].unstack(fill_value=0)
            construction_costs = grouped['total_cost'].unstack(fill_value=0)
            
            # Ensure all permit type columns exist
            for col in ['Residential', 'Commercial', 'Retail']:
                if col not in permit_counts.columns:
                    permit_counts[col] = 0
                if col not in construction_costs.columns:
                    construction_costs[col] = 0
            
            # Rename columns
            permit_counts = permit_counts.rename(columns={
                'Residential': 'residential_permits',
                'Commercial': 'commercial_permits',
                'Retail': 'retail_permits'
            })
            
            construction_costs = construction_costs.rename(columns={
                'Residential': 'residential_construction_cost',
                'Commercial': 'commercial_construction_cost',
                'Retail': 'retail_construction_cost'
            })
            
            # Reset index
            permit_counts = permit_counts.reset_index()
            construction_costs = construction_costs.reset_index()
            
            # Merge permit counts and construction costs
            processed_df = pd.merge(
                permit_counts,
                construction_costs,
                on=['zip_code', 'year'],
                how='outer'
            )
            
            # Fill any missing values
            processed_df = processed_df.fillna(0)
            
            # Calculate totals
            processed_df['total_permits'] = processed_df[['residential_permits', 'commercial_permits', 'retail_permits']].sum(axis=1)
            processed_df['total_construction_cost'] = processed_df[['residential_construction_cost', 'commercial_construction_cost', 'retail_construction_cost']].sum(axis=1)

            # Log summary statistics
            logger.info(f"Processed {len(df)} permits:")
            logger.info(f"- Residential: {processed_df['residential_permits'].sum():,} permits")
            logger.info(f"- Commercial: {processed_df['commercial_permits'].sum():,} permits")
            logger.info(f"- Retail: {processed_df['retail_permits'].sum():,} permits")
            logger.info(f"Total construction cost: ${processed_df['total_construction_cost'].sum():,.2f}")

            # Save processed data
            processed_df.to_csv(settings.PERMITS_PROCESSED_PATH, index=False)
            logger.info(f"Processed permit data saved to {str(settings.PERMITS_PROCESSED_PATH)}")
            
            # Log columns for debugging
            logger.info(f"Processed permits columns: {processed_df.columns.tolist()}")
            
            return True

        except Exception as e:
            logger.error(f"Error processing permit data: {str(e)}")
            return False
    
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

            # Merge permits data if available
            if 'permits' in processed_files:
                logger.info("Merging permits on ['zip_code', 'year']")
                logger.info(f"Pre-merge columns: {merged_df.columns.tolist()}")
                try:
                    merged_df = pd.merge(
                        merged_df,
                        processed_files['permits'],
                        on=['zip_code', 'year'],
                        how='left'
                    )
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
                    merged_df = pd.merge(
                        merged_df,
                        processed_files['business_licenses'],
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
                    # Ensure year is properly formatted
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
                    # Add missing economic columns with NaN
                    economic_cols = ['gdp', 'unemployment_rate', 'per_capita_income', 'personal_income']
                    for col in economic_cols:
                        if col not in merged_df.columns:
                            merged_df[col] = np.nan
                            logger.info(f"Added missing column {col} with NaN")

            # Check for missing permit columns
            permit_cols = ['residential_permits', 'commercial_permits', 'retail_permits']
            missing_permit_cols = [col for col in permit_cols if col not in merged_df.columns]
            if missing_permit_cols:
                logger.warning(f"Missing permit columns: {missing_permit_cols}")
                for col in missing_permit_cols:
                    logger.info(f"Added missing column {col} with zeros")
                    merged_df[col] = 0

            # Calculate total_permits if not present
            if 'total_permits' not in merged_df.columns:
                logger.info("Calculating total_permits from individual permit types")
                merged_df['total_permits'] = merged_df[permit_cols].sum(axis=1)

            # Save merged dataset
            merged_df.to_csv(settings.MERGED_DATA_PATH, index=False)
            logger.info(f"Merged dataset saved to {str(settings.MERGED_DATA_PATH)}")
            logger.info(f"Final merged columns: {merged_df.columns.tolist()}")
            
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
            retail_deficit_success = self.process_retail_deficit()
            if not retail_deficit_success:
                logger.warning("Retail deficit processing failed - some analyses may be incomplete")
            
            logger.info("Data processing pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False
    
    def process_census_data(self) -> bool:
        """Process census data."""
        try:
            # Load and validate raw census data
            df = pd.read_csv(settings.CENSUS_RAW_PATH, low_memory=False)
            
            # Define ACS column mapping
            ACS_COLUMN_MAP = {
                'B01003_001E': 'total_population',
                'B19013_001E': 'median_household_income',
                'B25077_001E': 'median_home_value',
                'B23025_002E': 'labor_force'
            }
            
            # Diagnostic check for raw columns
            print("\n🔍 Raw census data columns:", df.columns.tolist())
            
            # Apply column mapping
            df = df.rename(columns=ACS_COLUMN_MAP)
            logger.info(f"Census columns after renaming: {df.columns.tolist()}")
            
            # Validate total_population exists after mapping
            if 'total_population' in df.columns:
                missing_pct = df['total_population'].isna().mean() * 100
                print(f"✅ total_population present after mapping, {missing_pct:.2f}% missing")
            else:
                print("❌ total_population still missing after mapping")
                logger.error("total_population column missing after mapping")
                return False

            # Continue with normal processing
            required_cols = [
                'total_population',
                'median_household_income',
                'median_home_value',
                'labor_force',
                'zip_code',
                'year'
            ]
            
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Required column {col} missing from census data")
                    return False

            # Process and clean data
            df = df[required_cols].copy()
            df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
            
            # Ensure year is numeric and within valid range
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            valid_years = df['year'].between(2010, 2030)
            if not valid_years.all():
                logger.warning(f"Found {(~valid_years).sum()} records with invalid years")
                df = df[valid_years]
            
            # Convert numeric columns
            numeric_cols = ['total_population', 'median_household_income', 'median_home_value', 'labor_force']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill missing values with median by ZIP code
            for col in numeric_cols:
                df[col] = df.groupby('zip_code')[col].transform(lambda x: x.fillna(x.median()))
            
            # Save processed data
            df.to_csv(settings.CENSUS_PROCESSED_PATH, index=False)
            logger.info("Successfully processed census data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing census data: {str(e)}")
            return False
    
    def process_retail_deficit(self) -> bool:
        """Process retail deficit data and save results."""
        try:
            # Load required data
            try:
                retail_metrics = pd.read_csv(settings.RETAIL_DEFICIT_PATH)
                logger.info("Loaded retail metrics for deficit calculation")
            except FileNotFoundError:
                logger.warning("Retail metrics not found, generating from scratch")
                retail_metrics = self.generate_retail_metrics()
                if retail_metrics is None:
                    logger.error("Failed to generate retail metrics")
                    return False

            # Calculate retail deficit
            retail_metrics['retail_deficit'] = retail_metrics['expected_retail_demand'] - retail_metrics['actual_retail_supply']
            retail_metrics['deficit_ratio'] = retail_metrics['retail_deficit'] / retail_metrics['expected_retail_demand']

            # Categorize areas
            retail_metrics['deficit_category'] = pd.cut(
                retail_metrics['deficit_ratio'],
                bins=[-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf],
                labels=['High Surplus', 'Moderate Surplus', 'Balanced', 'Moderate Deficit', 'High Deficit']
            )

            # Save processed data
            retail_metrics.to_csv(settings.RETAIL_DEFICIT_PATH, index=False)
            logger.info(f"Saved retail deficit data to {settings.RETAIL_DEFICIT_PATH}")

            # Save analysis results
            self._save_retail_analysis_results(retail_metrics)

            return True

        except Exception as e:
            logger.error(f"Error processing retail deficit: {str(e)}")
            return False

    def _save_retail_analysis_results(self, df: pd.DataFrame) -> None:
        """Save retail analysis results to separate files."""
        try:
            # Create analysis results directory if it doesn't exist
            analysis_dir = settings.OUTPUT_DIR / 'analysis_results'
            analysis_dir.mkdir(parents=True, exist_ok=True)

            # Save deficit areas
            df[df['deficit_ratio'] > 0.05].to_csv(
                analysis_dir / 'retail_deficit_areas.csv',
                index=False
            )

            # Save surplus areas
            df[df['deficit_ratio'] < -0.05].to_csv(
                analysis_dir / 'retail_surplus_areas.csv',
                index=False
            )

            # Save retail leakage analysis
            df[['zip_code', 'retail_deficit', 'deficit_ratio', 'deficit_category']].to_csv(
                analysis_dir / 'retail_leakage.csv',
                index=False
            )

            logger.info("Saved retail analysis results to separate files")

        except Exception as e:
            logger.error(f"Error saving retail analysis results: {str(e)}")

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
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
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

    def process_zoning_data(self) -> bool:
        """Process zoning data."""
        try:
            # Load zoning data
            df = pd.read_csv(settings.ZONING_DATA_PATH)
            
            # Convert numeric columns
            df['total_parcels'] = pd.to_numeric(df['total_parcels'], errors='coerce')
            df['avg_lot_size'] = pd.to_numeric(df['avg_lot_size'], errors='coerce')
            
            # Calculate zoning distribution
            total_parcels = df.groupby('zip_code')['total_parcels'].transform('sum')
            df['zoning_percentage'] = (df['total_parcels'] / total_parcels) * 100
            
            # Calculate density metrics
            df['density_score'] = df['total_parcels'] / df['avg_lot_size']
            
            # Save processed data
            df.to_csv(settings.ZONING_PROCESSED_PATH, index=False)
            logger.info("Successfully processed zoning data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing zoning data: {str(e)}")
            return False 