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
from typing import Dict, Optional, List

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
        
    def collect_census_data(self):
        """Collect demographic data from Census API."""
        try:
            logger.info("Collecting Census data...")

            # Define variables to collect
            variables = [
                'B01003_001E',  # Total population
                'B19013_001E',  # Median household income
                'B25077_001E',  # Median home value
                'B23025_002E'   # Labor force
            ]

            data = []
            max_retries = 3
            retry_delay = 5  # seconds
            
            for year in settings.DEFAULT_TRAIN_YEARS:
                # Get data for each ZIP code in Illinois (state FIPS: 17)
                for zip_code in settings.CHICAGO_ZIP_CODES:
                    for attempt in range(max_retries):
                        try:
                            if result := self.census.acs5.state_zipcode(
                                variables,
                                state_fips='17',  # Illinois
                                zcta=zip_code,
                                year=year,
                            ):
                                row = result[0]
                                row['year'] = year
                                row['zip_code'] = zip_code
                                data.append(row)
                                break  # Success, break retry loop
                        except Exception as e:
                            if attempt == max_retries - 1:  # Last attempt
                                logger.error(f"Failed to collect Census data for ZIP {zip_code} year {year} after {max_retries} attempts: {str(e)}")
                                continue  # Skip this ZIP code
                            else:
                                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for ZIP {zip_code} year {year}. Retrying...")
                                import time
                                time.sleep(retry_delay)

            if not data:
                logger.error("No Census data collected after all retries")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            return self._save_data_to_csv(
                'census_data.csv', df, 'Census data'
            )

        except Exception as e:
            logger.error(f"Error collecting Census data: {str(e)}")
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
            df['year'] = pd.to_datetime(df['issue_date']).dt.year
            
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
        - CHIC917URN: Chicago Metro Area Unemployment Rate (%)
        - NGMP16980: Chicago Metro Area Real GDP (Millions of chained 2012 dollars)
        - PCPI17031: Chicago Per Capita Personal Income (Dollars)
        - CHIC917PCPI: Chicago Personal Income (Thousands of dollars)
        - HORAMM17031: Chicago Homeownership Rate (%)
        
        Data is filtered to settings.DEFAULT_TRAIN_YEARS and aggregated annually:
        - Rate-based metrics (unemployment, homeownership): Annual mean
        - Absolute metrics (GDP, income): Last value of year
        
        Returns:
            pd.DataFrame: DataFrame containing economic indicators by date, or None if collection fails
        """
        economic_data = {}
        rate_based_metrics = ['unemployment_rate', 'homeownership_rate']
        
        # Track any failed series
        failed_series = []
        
        # Collect each economic indicator
        for series_id, indicator_name in settings.FRED_SERIES.items():
            try:
                series = self.fred.get_series(series_id)
                
                if series is None or series.empty:
                    logger.warning(f"No data found for FRED series {series_id} ({indicator_name})")
                    failed_series.append(series_id)
                    continue
                    
                # Convert index to datetime if not already
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                
                # Resample to annual frequency using appropriate method
                if indicator_name in rate_based_metrics:
                    # Use mean for rate-based metrics
                    series = series.resample('YE').mean()
                else:
                    # Use last value for absolute metrics
                    series = series.resample('YE').last()
                
                # Convert index to year
                series.index = series.index.year
                
                # Filter to relevant years
                series = series[series.index.isin(settings.DEFAULT_TRAIN_YEARS)]
                
                if series.empty:
                    logger.warning(f"No data in training years for series {series_id} ({indicator_name})")
                    failed_series.append(series_id)
                    continue
                
                economic_data[indicator_name] = series
                logger.info(f"Successfully collected {indicator_name} data from {series_id}")
                
            except Exception as e:
                logger.error(f"Error collecting {series_id} ({indicator_name}): {str(e)}")
                failed_series.append(series_id)
                continue
        
        if failed_series:
            logger.error(f"Failed to collect data for series: {', '.join(failed_series)}")
        
        if not economic_data:
            logger.error("No economic data was successfully collected")
            return None
            
        # Combine all indicators into a single DataFrame
        df = pd.DataFrame(economic_data)
        df.index.name = 'year'
        
        # Save to CSV
        output_path = self.raw_dir / 'economic_indicators.csv'
        df.to_csv(output_path)
        logger.info(f"Saved {len(economic_data)} economic indicators to {output_path}")
        
        return df
            
    def get_zoning_data(self) -> Optional[pd.DataFrame]:
        """
        Collect property and zoning data from Chicago Data Portal.
        
        Returns:
            DataFrame with property and zoning information, or None if collection fails
        """
        try:
            logger.info("Collecting zoning data...")
            
            results = self.socrata.get(
                settings.ZONING_DATASET,
                limit=1000000,
                select="""
                    property_use, township, 
                    year_constructed, total_building_area, total_land_area,
                    sale_date, sale_price, sale_type,
                    zip_code
                """.replace('\n', '').replace(' ', '')
            )
            
            df = pd.DataFrame.from_records(results)
            
            # Rename columns to match expected names
            df = df.rename(columns={
                'property_use': 'property_class',
                'year_constructed': 'year_built',
                'total_building_area': 'building_area',
                'total_land_area': 'land_area'
            })
            
            # Convert numeric columns
            numeric_cols = ['year_built', 'building_area', 'land_area', 'sale_price']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert dates
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            
            # Add year column
            df['year'] = df['sale_date'].dt.year
            
            return self._save_data_to_csv(
                'property_data.csv', df, 'Property and zoning data'
            )
            
        except Exception as e:
            logger.error(f"Error collecting zoning data: {str(e)}")
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

    def get_property_transactions(self, limit: int = 1000000) -> Optional[pd.DataFrame]:
        """
        Collect property transaction data from Chicago Data Portal.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with property transaction data, or None if collection fails
        """
        try:
            logger.info("Collecting property transaction data...")
            
            results = self.socrata.get(
                settings.PROPERTY_TRANSACTIONS_DATASET,  # Use correct dataset ID
                limit=limit,
                select="""
                    pin, sale_date, sale_price, sale_type,
                    property_class, township, zip_code,
                    year_built, building_area, land_area
                """.replace('\n', '').replace(' ', ''),
                where="sale_price > 0 AND sale_type IS NOT NULL",
                order="sale_date DESC"
            )
            
            if not results:
                logger.warning("No property transaction data available")
                return None
                
            df = pd.DataFrame.from_records(results)
            
            # Convert numeric columns
            numeric_cols = ['sale_price', 'year_built', 'building_area', 'land_area']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert dates
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            
            # Add year column
            df['year'] = df['sale_date'].dt.year
            
            # Filter out invalid transactions
            df = df[
                (df['sale_price'].notna()) & 
                (df['sale_type'].notna()) &
                (df['sale_price'] > 0)
            ]
            
            # Filter to relevant years
            df = df[df['year'].isin(settings.DEFAULT_TRAIN_YEARS)]
            
            if df.empty:
                logger.warning("No valid property transactions found in the training period")
                return None
            
            return self._save_data_to_csv(
                'property_transactions.csv', df, 'Property transaction data'
            )
            
        except Exception as e:
            logger.error(f"Error collecting property transaction data: {str(e)}")
            return None

    def collect_all_data(self):
        """Collect all required data."""
        # Critical data sources
        critical_data = {
            'census': self.collect_census_data(),
            'permits': self.collect_permit_data(),
            'economic': self.collect_economic_data()
        }

        # Check for critical failures
        if failed_critical := [name for name, data in critical_data.items() if data is None]:
            logger.error(f"Failed to collect critical data: {', '.join(failed_critical)}")
            return False

        # Optional data sources
        optional_data = {
            'property': self.get_property_transactions(),
            'zoning': self.get_zoning_data(),
            'licenses': self.get_business_licenses()
        }

        # Log warning for optional failures
        if failed_optional := [name for name, data in optional_data.items() if data is None]:
            logger.warning(f"Failed to collect optional data: {', '.join(failed_optional)}")

        logger.info("Successfully collected all critical data")
        return True 