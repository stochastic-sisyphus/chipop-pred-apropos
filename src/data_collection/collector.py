"""
Data collection module for Chicago population analysis.
"""
import os
import logging
import pandas as pd
from pathlib import Path
from census import Census
from fredapi import Fred
from sodapy import Socrata
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Optional, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import traceback

from src.config import settings

logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection from various sources including FRED, Census, and Chicago Data Portal."""

    def __init__(self):
        """Initialize data collector with API clients."""
        self.census = Census(settings.CENSUS_API_KEY)
        self.fred = Fred(api_key=settings.FRED_API_KEY)
        self.socrata = Socrata("data.cityofchicago.org", settings.CHICAGO_DATA_TOKEN)
        self.raw_dir = settings.RAW_DATA_DIR
        self.raw_dir.mkdir(exist_ok=True)
        
        # Cache for available Census years
        self._available_census_years = None

    def _get_available_census_years(self) -> Set[int]:
        """
        Get available years from Census API with caching.
        
        Returns:
            Set[int]: Set of available years
        """
        if self._available_census_years is not None:
            return self._available_census_years
            
        available_years = set()
        test_zip = str(settings.CHICAGO_ZIP_CODES[0])
        
        for year in range(2015, datetime.now().year):
            try:
                test_result = self.census.acs5.state_zipcode(
                    ['B01003_001E'],  # Total population
                    state_fips='17',   # Illinois
                    zcta=test_zip,
                    year=year
                )
                if test_result:
                    available_years.add(year)
                    logger.info(f"Census year {year} is available")
            except Exception as e:
                logger.debug(f"Census year {year} not available: {str(e)}")
                continue
                
        self._available_census_years = available_years
        return available_years

    def _fetch_census_zip_year(self, zip_code: str, year: int, variables: List[str], max_retries: int = 3) -> Optional[Dict]:
        """
        Fetch Census data for a specific ZIP code and year with retries.
        
        Args:
            zip_code (str): ZIP code to fetch data for
            year (int): Year to fetch data for
            variables (List[str]): Census variables to fetch
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Optional[Dict]: Census data row or None if failed
        """
        zip_code = str(zip_code).zfill(5)
        
        for attempt in range(max_retries):
            try:
                if result := self.census.acs5.state_zipcode(
                    variables,
                    state_fips='17',
                    zcta=zip_code,
                    year=year
                ):
                    row = result[0]
                    row['year'] = year
                    row['zip_code'] = zip_code  # Use consistent column name
                    logger.debug(f"Fetched data for ZIP {zip_code}, year {year}")
                    return row
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Back off before retry
                else:
                    logger.error(f"Failed to fetch ZIP {zip_code}, year {year}: {str(e)}")
                    
        return None

    def collect_census_data(self) -> Optional[pd.DataFrame]:
        """
        Collect Census data for all available years.
        
        Returns:
            Optional[pd.DataFrame]: Census data or None if failed
        """
        try:
            logger.info("Starting Census data collection...")
            
            # Get available years
            available_years = self._get_available_census_years()
            if not available_years:
                logger.error("No Census years available")
                return None
                
            logger.info(f"Collecting Census data for years: {sorted(available_years)}")
            
            # Census variables to collect
            variables = [
                'B01003_001E',  # Total population
                'B19013_001E',  # Median household income
                'B25077_001E',  # Median home value
                'B23025_002E',  # Labor force
                'B25001_001E',  # Housing units
                'B25003_001E',  # Occupied housing units
                'B25003_003E'   # Vacant housing units
            ]
            
            # Create tasks for parallel execution
            tasks = [
                (str(zip_code), year)
                for year in available_years
                for zip_code in settings.CHICAGO_ZIP_CODES
            ]
            
            # Collect data in parallel
            results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self._fetch_census_zip_year, z, y, variables): (z, y)
                    for z, y in tasks
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Census data"):
                    if result := future.result():
                        results.append(result)
                        
            if not results:
                logger.error("No Census data collected")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(results)
            logger.info(f"Raw Census data shape: {df.shape}")
            
            # Rename columns to be more descriptive
            column_map = {
                'B01003_001E': 'total_population',
                'B19013_001E': 'median_household_income',
                'B25077_001E': 'median_home_value',
                'B23025_002E': 'labor_force',
                'B25001_001E': 'total_housing_units',
                'B25003_001E': 'occupied_housing_units',
                'B25003_003E': 'vacant_housing_units',
                'zip code tabulation area': 'zip_code'  # Map ZCTA to zip_code
            }
            df = df.rename(columns=column_map)
            
            # Debug log current columns
            logger.info(f"Current columns: {df.columns.tolist()}")
            
            # Check for and handle duplicate columns
            dup_cols = df.columns[df.columns.duplicated()].tolist()
            if dup_cols:
                logger.warning(f"Found duplicate columns: {dup_cols}")
                # Keep first occurrence of each column
                df = df.loc[:, ~df.columns.duplicated()]
                logger.info(f"Columns after deduplication: {df.columns.tolist()}")
            
            # Validate zip_code column exists and is a Series
            if 'zip_code' not in df.columns:
                logger.error("No 'zip_code' column found after renaming")
                return None
                
            # Debug log ZIP code column info
            logger.info(f"ZIP code column type: {type(df['zip_code'])}")
            logger.info(f"ZIP code sample before processing:\n{df['zip_code'].head()}")
            
            # Handle ZIP code formatting
            try:
                # Convert to string and pad with zeros
                df['zip_code'] = df['zip_code'].astype(str).str.strip()
                df['zip_code'] = df['zip_code'].str.zfill(5)
                
                # Validate ZIP codes are 5 digits
                invalid_zips = df[~df['zip_code'].str.match(r'^\d{5}$')]
                if not invalid_zips.empty:
                    logger.warning(f"Found {len(invalid_zips)} invalid ZIP codes")
                    logger.warning(f"Invalid ZIP codes: {invalid_zips['zip_code'].unique().tolist()}")
                
                # Debug log after processing
                logger.info(f"ZIP code sample after processing:\n{df['zip_code'].head()}")
                
            except Exception as e:
                logger.error(f"Error formatting ZIP codes: {str(e)}")
                logger.error(f"ZIP code column type: {type(df['zip_code'])}")
                logger.error(f"ZIP code sample:\n{df['zip_code'].head()}")
                return None
            
            # Create template dataframe with all ZIP code and year combinations
            all_combinations = pd.DataFrame([(zip_code, year) 
                                          for zip_code in settings.CHICAGO_ZIP_CODES 
                                          for year in available_years],
                                         columns=['zip_code', 'year'])
            
            # Prepare columns for merge
            required_columns = ['zip_code', 'year'] + [col for col in column_map.values() if col != 'zip_code']
            df_merge = df[required_columns].copy()
            
            # Ensure no duplicate columns before merge
            df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]
            all_combinations = all_combinations.loc[:, ~all_combinations.columns.duplicated()]
            
            # Debug log merge info
            logger.info(f"Merge left columns: {all_combinations.columns.tolist()}")
            logger.info(f"Merge right columns: {df_merge.columns.tolist()}")
            
            try:
                # Merge with template
                df = pd.merge(all_combinations, df_merge, on=['zip_code', 'year'], how='left')
                logger.info(f"Merged data shape: {df.shape}")
                
                # Debug log final columns
                logger.info(f"Final columns before filling NAs: {df.columns.tolist()}")
                
                # Fill missing values
                numeric_cols = [col for col in df.columns if col not in ['zip_code', 'year', 'state']]
                
                # Convert all numeric columns at once
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill missing values using groupby means
                df_filled = df.copy()
                for col in numeric_cols:
                    try:
                        # Calculate means by ZIP code
                        zip_means = df.groupby('zip_code')[col].transform('mean')
                        
                        # Fill NAs without using inplace
                        df_filled[col] = df[col].fillna(zip_means)
                        
                        # If any NAs remain, fill with overall mean
                        if df_filled[col].isna().any():
                            overall_mean = df[col].mean()
                            df_filled[col] = df_filled[col].fillna(overall_mean)
                            
                        logger.info(f"Filled NAs in {col} using ZIP code means and overall mean")
                    except Exception as e:
                        logger.error(f"Error filling NAs for column {col}: {str(e)}")
                        logger.error(f"Column info - dtype: {df[col].dtype}, unique values: {df[col].nunique()}")
                        return None
                
                # Replace original with filled version
                df = df_filled
                
                # Final validation
                if df.isnull().any().any():
                    null_cols = df.columns[df.isnull().any()].tolist()
                    logger.warning(f"Found null values in columns: {null_cols}")
                    logger.warning(f"Null counts:\n{df[null_cols].isnull().sum()}")
                    
                    # Drop rows with any remaining nulls as a last resort
                    df = df.dropna()
                    logger.warning(f"Dropped rows with null values. New shape: {df.shape}")
                
                # Ensure ZIP codes are properly formatted
                df['zip_code'] = df['zip_code'].astype(str).str.strip().str.zfill(5)
                
                # Save processed data
                processed_path = settings.PROCESSED_DATA_DIR / 'census_processed.csv'
                df.to_csv(processed_path, index=False)
                logger.info(f"Processed Census data saved to {processed_path}")
                
                return df
                
            except Exception as e:
                logger.error(f"Error during merge: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
            
        except Exception as e:
            logger.error(f"Error in collect_census_data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def collect_permit_data(self):
        """Collect building permit data from Chicago Data Portal."""
        try:
            logger.info("Collecting building permit data...")
            
            # Query building permits with additional fields
            results = self.socrata.get(
                "ydr8-5enu",
                limit=100000,
                where="issue_date IS NOT NULL",
                select="""
                    issue_date, permit_type, street_number, street_direction, 
                    street_name, work_description, total_fee, reported_cost,
                    contact_1_type, contact_1_name, contact_1_city, 
                    contact_1_state, contact_1_zipcode, community_area,
                    census_tract, ward, pin_list, xcoordinate,
                    ycoordinate, latitude, longitude
                """.replace('\n', '').replace(' ', '')
            )
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)
            
            # Convert numeric columns
            numeric_cols = ['total_fee', 'reported_cost', 'xcoordinate', 'ycoordinate', 'latitude', 'longitude']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add retail classification
            retail_keywords = ['retail', 'store', 'shop', 'restaurant', 'commercial', 
                             'business', 'mall', 'market', 'sales']
            df['is_retail'] = df['work_description'].str.lower().str.contains(
                '|'.join(retail_keywords), na=False
            )
            
            # Add residential classification
            residential_keywords = ['residential', 'house', 'apartment', 'condo', 'dwelling',
                                  'home', 'townhouse', 'multi-family', 'single-family']
            df['is_residential'] = df['work_description'].str.lower().str.contains(
                '|'.join(residential_keywords), na=False
            )
            
            # Add commercial classification (non-retail commercial)
            commercial_keywords = ['office', 'industrial', 'warehouse', 'factory', 'manufacturing',
                                 'corporate', 'wholesale', 'distribution']
            df['is_commercial'] = df['work_description'].str.lower().str.contains(
                '|'.join(commercial_keywords), na=False
            )
            
            # Categorize permit types
            df['permit_category'] = 'other'
            df.loc[df['permit_type'].str.contains('PERMIT - NEW CONSTRUCTION', na=False), 'permit_category'] = 'new_construction'
            df.loc[df['permit_type'].str.contains('PERMIT - RENOVATION/ALTERATION', na=False), 'permit_category'] = 'renovation'
            df.loc[df['permit_type'].str.contains('PERMIT - ADDITION', na=False), 'permit_category'] = 'addition'
            
            # Calculate permit counts by type
            df['residential_permits'] = df['is_residential'].astype(int)
            df['commercial_permits'] = df['is_commercial'].astype(int)
            df['retail_permits'] = df['is_retail'].astype(int)
            
            # Calculate construction costs by type
            df['residential_construction_cost'] = df['reported_cost'].where(df['is_residential'], 0)
            df['commercial_construction_cost'] = df['reported_cost'].where(df['is_commercial'], 0)
            df['retail_construction_cost'] = df['reported_cost'].where(df['is_retail'], 0)
            
            # Add year column
            df['issue_date'] = pd.to_datetime(df['issue_date'])
            df['year'] = df['issue_date'].dt.year
            
            # Ensure contact_1_zipcode is string
            df['contact_1_zipcode'] = df['contact_1_zipcode'].astype(str).str.zfill(5)
            
            # Group by year for validation
            yearly_counts = df.groupby('year').agg({
                'permit_type': 'count',
                'residential_permits': 'sum',
                'commercial_permits': 'sum',
                'retail_permits': 'sum',
                'reported_cost': 'sum',
                'residential_construction_cost': 'sum',
                'commercial_construction_cost': 'sum',
                'retail_construction_cost': 'sum'
            }).reset_index()
            
            # Log summary statistics
            logger.info(f"Collected {len(df)} permits:")
            logger.info(f"- Residential: {df['residential_permits'].sum():,} permits")
            logger.info(f"- Commercial: {df['commercial_permits'].sum():,} permits")
            logger.info(f"- Retail: {df['retail_permits'].sum():,} permits")
            logger.info(f"Total construction cost: ${df['reported_cost'].sum():,.2f}")
            logger.info(f"- Residential: ${df['residential_construction_cost'].sum():,.2f}")
            logger.info(f"- Commercial: ${df['commercial_construction_cost'].sum():,.2f}")
            logger.info(f"- Retail: ${df['retail_construction_cost'].sum():,.2f}")
            
            return self._save_data_to_csv(
                'building_permits.csv', df, 'Building permit data'
            )
            
        except Exception as e:
            logger.error(f"Error collecting permit data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def _save_data_to_csv(self, filename: str, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Save DataFrame to CSV and log the operation.
        
        Args:
            filename: Name of the output file
            df: DataFrame to save
            data_type: Type of data being saved (for logging)
        
        Returns:
            The input DataFrame
        """
        output_file = self.raw_dir / filename
        df.to_csv(output_file, index=False)
        logger.info(f"{data_type} saved to {output_file}")
        return df
    
    def collect_economic_data(self) -> Optional[pd.DataFrame]:
        """
        Collect economic indicators from FRED for the Chicago metropolitan area.

        The following indicators are collected based on settings.FRED_SERIES:
        - CHIC917URN: Unemployment Rate
        - NGMP16980: Real GDP
        - PCPI17031: Per Capita Income
        - CHIC917PCPI: Personal Income

        Returns:
            pd.DataFrame with economic indicators by year, or None on failure.
        """
        if not isinstance(settings.FRED_SERIES, dict):
            logger.error(
                f"Expected dict for FRED_SERIES but got {type(settings.FRED_SERIES)}: {settings.FRED_SERIES}"
            )
            return None

        economic_data = {}
        rate_based_metrics = ['unemployment_rate', 'homeownership_rate']
        failed_series = []

        # First collect all available data
        for series_id, indicator_name in settings.FRED_SERIES.items():
            try:
                logger.debug(f"Requesting FRED series {series_id} ({indicator_name})")
                series = self.fred.get_series(series_id)

                if series is None or series.empty:
                    logger.warning(f"No data found for FRED series {series_id} ({indicator_name})")
                    failed_series.append(series_id)
                    continue

                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)

                if indicator_name in rate_based_metrics:
                    series = series.resample('YE').mean()
                else:
                    series = series.resample('YE').last()

                series.index = series.index.year
                economic_data[indicator_name] = series

                logger.info(f"Successfully collected {indicator_name} from {series_id}")

            except Exception as e:
                logger.error(f"Error collecting FRED series {series_id} ({indicator_name}): {str(e)}")
                failed_series.append(series_id)

        if failed_series:
            logger.warning(f"Failed FRED series: {', '.join(failed_series)}")

        if not economic_data:
            logger.error("No economic indicators collected")
            return None

        # Create DataFrame with all years
        df = pd.DataFrame(index=settings.DEFAULT_TRAIN_YEARS)
        
        # Add each indicator and handle missing years
        for indicator, series in economic_data.items():
            # Reindex series to match desired years
            series = series.reindex(df.index)
            # Forward fill, then backward fill any remaining NAs
            series = series.ffill().bfill()
            df[indicator] = series
            
        df.index.name = 'year'
        df = df.reset_index()
        
        output_path = self.raw_dir / 'economic_indicators.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(economic_data)} economic indicators to {output_path}")

        return df
            
    def collect_zoning_data(self):
        """Collect zoning data from Chicago Data Portal."""
        try:
            logger.info("Starting zoning data collection...")
            
            # First get a sample record to check available columns
            try:
                sample = self.socrata.get("dj47-wfun", limit=1)
                if not sample:
                    logger.error("Could not get sample zoning record")
                    return None
                    
                # Log available columns
                columns = list(sample[0].keys())
                logger.info(f"Available zoning columns: {columns}")
                
                # Build query based on available columns
                query = """
                    SELECT 
                        zone_class AS zoning_classification,
                        zone_type AS zone_category,
                        COUNT(*) AS total_parcels,
                        AVG(shape_area) AS avg_lot_size,
                        SUM(shape_area) AS total_area
                    GROUP BY zoning_classification, zone_category
                    LIMIT 1000000
                """
                zoning_data = self.socrata.get("dj47-wfun", query=query)
                df = pd.DataFrame.from_records(zoning_data)
                
                if len(df) > 0:
                    logger.info(f"Successfully collected zoning data: {len(df)} records")
                    
                    # Add ZIP code mapping based on location
                    df['zip_code'] = '60601'  # Default to Loop ZIP code
                    logger.warning("Using default ZIP code 60601 for all zoning records")
                    
                    # Convert area to square feet
                    df['avg_lot_size'] = pd.to_numeric(df['avg_lot_size'], errors='coerce')
                    df['total_area'] = pd.to_numeric(df['total_area'], errors='coerce')
                    df['total_parcels'] = pd.to_numeric(df['total_parcels'], errors='coerce')
                    
                    # Log summary statistics
                    logger.info(f"Zoning classifications: {df['zoning_classification'].nunique()}")
                    logger.info(f"Zone categories: {df['zone_category'].nunique()}")
                    logger.info(f"Total parcels: {int(df['total_parcels'].sum())}")
                    logger.info(f"Average lot size: {df['avg_lot_size'].mean():.0f} sq ft")
                    
                    # Save raw data
                    raw_path = self.raw_dir / 'zoning_data.csv'
                    df.to_csv(raw_path, index=False)
                    logger.info(f"Saved raw zoning data to {raw_path}")
                    
                    return df
                else:
                    logger.warning("No zoning data records returned")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to get zoning data: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
            
        except Exception as e:
            logger.error(f"Error collecting zoning data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def get_property_transactions(self) -> Optional[pd.DataFrame]:
        """Get property transaction data from Chicago Data Portal."""
        try:
            # Verify dataset ID exists
            if not settings.PROPERTY_TRANSACTIONS_DATASET:
                logger.warning("Property transactions dataset ID not configured")
                return None
                
            results = self.socrata.get(
                settings.PROPERTY_TRANSACTIONS_DATASET,
                select="""
                    zip_code,
                    sale_price,
                    property_type,
                    year_built,
                    total_value
                """,
                limit=1000000
            )
            
            if not results:
                logger.warning("No property transaction data returned")
                return None
                
            df = pd.DataFrame.from_records(results)
            
            # Save raw data
            df.to_csv(settings.PROPERTY_DATA_PATH, index=False)
            logger.info(f"Property transaction data saved to {settings.PROPERTY_DATA_PATH}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting property transaction data: {str(e)}")
            return None

    def get_business_licenses(self, limit: int = 1000000) -> Optional[pd.DataFrame]:
        """
        Collect business license data from Chicago Data Portal.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with business license data, or None if collection fails
        """
        for attempt in range(2):  # Try twice before failing
            try:
                logger.info("Collecting business license data...")
                
                results = self.socrata.get(
                    settings.BUSINESS_LICENSES_DATASET,
                    limit=limit,
                    select="""
                        license_id, account_number, legal_name, 
                        doing_business_as_name, license_code, 
                        license_description, business_activity,
                        application_created_date, license_start_date,
                        expiration_date, zip_code
                    """.replace('\n', '').replace(' ', ''),
                    where="expiration_date > '2025-04-25T00:00:00.000' AND UPPER(license_status) = 'AAI'",
                    order="license_start_date DESC"
                )
                
                df = pd.DataFrame.from_records(results)
                df['license_start_date'] = pd.to_datetime(df['license_start_date'], errors='coerce')
                df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
                df['application_created_date'] = pd.to_datetime(df['application_created_date'], errors='coerce')
                
                return self._save_data_to_csv(
                    'business_licenses.csv', df, 'Business license data'
                )
                
            except Exception as e:
                if attempt == 0:  # First attempt failed
                    logger.warning(f"First attempt to collect business license data failed: {str(e)}. Retrying...")
                    continue
                else:  # Second attempt failed
                    logger.error(f"Error collecting business license data after retry: {str(e)}")
                    return None

    def collect_all_data(self) -> bool:
        """Collect all required datasets."""
        try:
            # Collect Census data
            logger.info("Starting Census data collection...")
            census_data = self.collect_census_data()
            if census_data is None:
                logger.error("Census data collection failed")
                return False
            logger.info("Census data collection completed")
            
            # Collect permit data
            logger.info("Starting permit data collection...")
            permit_data = self.collect_permit_data()
            if permit_data is None:
                logger.error("Permit data collection failed")
                return False
            logger.info("Permit data collection completed")
            
            # Collect economic data
            logger.info("Starting economic data collection...")
            economic_data = self.collect_economic_data()
            if economic_data is None:
                logger.error("Economic data collection failed")
                return False
            logger.info("Economic data collection completed")
            
            # Try to get zoning data but don't fail if unsuccessful
            try:
                logger.info("Starting zoning data collection...")
                zoning_data = self.collect_zoning_data()
                if zoning_data is not None:
                    logger.info("Successfully collected zoning data")
                else:
                    logger.warning("Zoning data collection returned None")
            except Exception as e:
                logger.warning(f"Failed to collect zoning data: {str(e)}")
                
            # Try to get property data but don't fail if unsuccessful
            try:
                logger.info("Starting property transaction data collection...")
                property_data = self.get_property_transactions()
                if property_data is not None:
                    logger.info("Successfully collected property transaction data")
                else:
                    logger.warning("Property transaction data collection returned None")
            except Exception as e:
                logger.warning(f"Failed to collect property transaction data: {str(e)}")
                
            # Check core datasets were collected
            if all([census_data is not None,
                   permit_data is not None,
                   economic_data is not None]):
                logger.info("Successfully collected all core datasets")
                return True
                
            logger.error("Failed to collect one or more core datasets")
            return False
            
        except Exception as e:
            logger.error(f"Error in collect_all_data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False 