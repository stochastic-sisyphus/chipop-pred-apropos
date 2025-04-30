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
                series = series[series.index.isin(settings.DEFAULT_TRAIN_YEARS)]

                if series.empty:
                    logger.warning(f"No valid data in training years for {series_id} ({indicator_name})")
                    failed_series.append(series_id)
                    continue

                economic_data[indicator_name] = series.values
                logger.info(f"Successfully collected {indicator_name} from {series_id}")

            except Exception as e:
                logger.error(f"Error collecting FRED series {series_id} ({indicator_name}): {str(e)}")
                failed_series.append(series_id)

        if failed_series:
            logger.warning(f"Failed FRED series: {', '.join(failed_series)}")

        if not economic_data:
            logger.error("No economic indicators collected")
            return None

        df = pd.DataFrame(economic_data, index=pd.Index(settings.DEFAULT_TRAIN_YEARS, name='year'))
        output_path = self.raw_dir / 'economic_indicators.csv'
        df.to_csv(output_path)
        logger.info(f"Saved {len(economic_data)} economic indicators to {output_path}")

        return df
            
    def get_zoning_data(self) -> Optional[pd.DataFrame]:
        """Get zoning data from Chicago Data Portal."""
        try:
            # Query zoning data without property_use column
            results = self.socrata.get(
                settings.ZONING_DATASET,
                select="""
                    zip_code,
                    zoning_classification,
                    COUNT(*) as total_parcels,
                    AVG(lot_area_sqft) as avg_lot_size
                """,
                group="zip_code, zoning_classification",
                where="zip_code IS NOT NULL",
                limit=1000000
            )

            if not results:
                logger.warning("No zoning data returned from API")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)

            # Clean numeric columns
            numeric_cols = ['total_parcels', 'avg_lot_size']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return self._extracted_from_get_property_transactions_31(
                df, "zoning_data.csv", "Saved zoning data"
            )
        except Exception as e:
            logger.error(f"Error getting zoning data: {str(e)}")
            return None
            
    def get_property_transactions(self, limit: int = 1000000) -> Optional[pd.DataFrame]:
        """Get property transaction data from Chicago Data Portal."""
        try:
            # Query property transactions
            results = self.socrata.get(
                settings.PROPERTY_TRANSACTIONS_DATASET,
                select="""
                    zip_code,
                    property_type,
                    sale_price,
                    sale_date,
                    year_built,
                    building_sqft,
                    land_sqft
                """,
                where="zip_code IS NOT NULL AND sale_price > 0",
                limit=limit
            )

            if not results:
                logger.warning("No property transaction data returned from API")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)

            # Clean numeric columns
            numeric_cols = ['sale_price', 'year_built', 'building_sqft', 'land_sqft']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert dates
            df['sale_date'] = pd.to_datetime(df['sale_date'])

            return self._extracted_from_get_property_transactions_31(
                df, "property_transactions.csv", "Saved property transaction data"
            )
        except Exception as e:
            logger.error(f"Error getting property transaction data: {str(e)}")
            return None

    # TODO Rename this here and in `get_zoning_data` and `get_property_transactions`
    def _extracted_from_get_property_transactions_31(self, df, arg1, arg2):
        df.to_csv(settings.DATA_RAW_DIR / arg1, index=False)
        logger.info(arg2)
        return df

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
            # Collect each dataset
            census_data = self.collect_census_data()
            permit_data = self.collect_permit_data()
            economic_data = self.collect_economic_data()
            
            # Try to get zoning data but don't fail if unsuccessful
            try:
                zoning_data = self.get_zoning_data()
                if zoning_data is not None:
                    logger.info("Successfully collected zoning data")
            except Exception as e:
                logger.warning(f"Failed to collect zoning data: {str(e)}")
                
            # Try to get property data but don't fail if unsuccessful
            try:
                property_data = self.get_property_transactions()
                if property_data is not None:
                    logger.info("Successfully collected property transaction data")
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
            return False 