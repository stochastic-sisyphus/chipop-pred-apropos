"""
Data collection module for the Chicago Housing Pipeline & Population Shift Project.
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from fredapi import Fred
from census import Census
from sodapy import Socrata
from ..config import settings
import traceback
import time

logger = logging.getLogger(__name__)

class DataCollector:
    """
    Class for collecting data from various sources.
    """
    
    def __init__(self, raw_dir=None):
        """
        Initialize the DataCollector.
        
        Args:
            raw_dir (Path, optional): Directory to save raw data. Defaults to settings.RAW_DATA_DIR.
        """
        self.raw_dir = raw_dir if raw_dir else settings.RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API clients
        self._init_api_clients()
    
    def _init_api_clients(self):
        """Initialize API clients for data collection."""
        # Census API
        self.census_api_key = settings.CENSUS_API_KEY
        if self.census_api_key and self.census_api_key != 'your_census_api_key':
            self.census = Census(self.census_api_key)
        else:
            self.census = None
            logger.warning("Census API key not set. Census data collection will be limited.")
        
        # FRED API
        self.fred_api_key = settings.FRED_API_KEY
        if self.fred_api_key and self.fred_api_key != 'your_fred_api_key':
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            self.fred = None
            logger.warning("FRED API key not set. Economic data collection will be limited.")
        
        # Chicago Data Portal
        self.chicago_token = settings.CHICAGO_DATA_TOKEN
        if self.chicago_token and self.chicago_token != 'your_chicago_data_token':
            self.chicago_client = Socrata("data.cityofchicago.org", self.chicago_token)
        else:
            self.chicago_client = None
            logger.warning("Chicago Data Portal token not set. Chicago data collection will be limited.")
        
        # HUD API
        self.hud_api_key = settings.HUD_API_KEY
        if not self.hud_api_key or self.hud_api_key == 'your_hud_api_key':
            logger.warning("HUD API key not set. Vacancy data collection will be limited.")
    
    def collect_data(self, use_sample=False):
        """
        Collect data from various sources or load sample data.
        
        Args:
            use_sample (bool): Whether to use sample data instead of collecting from APIs
            
        Returns:
            pd.DataFrame: Collected data
        """
        if use_sample:
            return self.load_sample_data()
        else:
            return self.collect_all_data()
    
    def load_sample_data(self):
        """
        Load sample data from files.
        
        Returns:
            pd.DataFrame: Sample data
        """
        logger.info("Loading sample data...")
        
        # Define sample data directory
        sample_dir = Path(settings.SAMPLE_DATA_DIR)
        
        # Check if sample data directory exists
        if not sample_dir.exists():
            logger.error(f"Sample data directory {sample_dir} does not exist")
            return None
        
        # Load sample data files
        data_frames = []
        
        # Census data
        census_path = sample_dir / "census_data.csv"
        if census_path.exists():
            try:
                census_df = pd.read_csv(census_path)
                data_frames.append(census_df)
                logger.info(f"Loaded sample census data: {len(census_df)} records")
            except Exception as e:
                logger.error(f"Error loading sample census data: {str(e)}")
        
        # Economic data
        economic_path = sample_dir / "economic_data.csv"
        if economic_path.exists():
            try:
                economic_df = pd.read_csv(economic_path)
                data_frames.append(economic_df)
                logger.info(f"Loaded sample economic data: {len(economic_df)} records")
            except Exception as e:
                logger.error(f"Error loading sample economic data: {str(e)}")
        
        # Building permits
        permits_path = sample_dir / "building_permits.csv"
        if permits_path.exists():
            try:
                permits_df = pd.read_csv(permits_path)
                data_frames.append(permits_df)
                logger.info(f"Loaded sample building permits data: {len(permits_df)} records")
            except Exception as e:
                logger.error(f"Error loading sample building permits data: {str(e)}")
        
        # Business licenses
        licenses_path = sample_dir / "business_licenses.csv"
        if licenses_path.exists():
            try:
                licenses_df = pd.read_csv(licenses_path)
                data_frames.append(licenses_df)
                logger.info(f"Loaded sample business licenses data: {len(licenses_df)} records")
            except Exception as e:
                logger.error(f"Error loading sample business licenses data: {str(e)}")
        
        # Check if we have any data
        if not data_frames:
            logger.error("No sample data files found")
            return None
        
        # Combine all data frames
        # Note: This is a simplified approach; in a real scenario, we would need to
        # properly merge these data frames based on common keys
        combined_data = pd.concat(data_frames, ignore_index=True)
        
        logger.info(f"Combined sample data: {len(combined_data)} records")
        
        return combined_data
    
    def collect_all_data(self):
        """
        Collect all data from various sources.
        
        Returns:
            pd.DataFrame: Combined data from all sources
        """
        logger.info("Collecting data...")
        
        # Collect Census data
        census_success = self.collect_census_data()
        
        # Collect economic data
        economic_success = self.collect_economic_data()
        
        # Collect building permits data
        permits_success = self.collect_building_permits()
        
        # Collect business licenses data
        licenses_success = self.collect_business_licenses()
        
        # Collect vacancy data
        vacancy_success = self.collect_vacancy_data()
        
        # Collect migration data
        migration_success = self.collect_migration_data()
        
        # Collect retail GDP data
        retail_gdp_success = self.collect_retail_gdp_data()
        
        # Check if all data collection was successful
        success_statuses = [
            census_success, 
            economic_success, 
            permits_success, 
            licenses_success,
            vacancy_success,
            migration_success,
            retail_gdp_success
        ]
        
        # Count successful collections - handle both boolean and DataFrame returns
        success_count = sum(1 for status in success_statuses if (
            isinstance(status, bool) and status or
            isinstance(status, pd.DataFrame) and not status.empty
        ))
        total_count = len(success_statuses)
        
        if success_count == total_count:
            logger.info("All data collection completed successfully")
        else:
            logger.warning(f"Some data collection tasks failed. {success_count}/{total_count} successful.")
        
        # Load and combine all collected data
        data_frames = []
        
        # Census data
        census_path = self.raw_dir / "census_data.csv"
        if census_path.exists():
            try:
                census_df = pd.read_csv(census_path)
                data_frames.append(census_df)
                logger.info(f"Loaded census data: {len(census_df)} records")
            except Exception as e:
                logger.error(f"Error loading census data: {str(e)}")
        
        # Economic data
        economic_path = self.raw_dir / "economic_data.csv"
        if economic_path.exists():
            try:
                economic_df = pd.read_csv(economic_path)
                data_frames.append(economic_df)
                logger.info(f"Loaded economic data: {len(economic_df)} records")
            except Exception as e:
                logger.error(f"Error loading economic data: {str(e)}")
        
        # Building permits
        permits_path = self.raw_dir / "building_permits.csv"
        if permits_path.exists():
            try:
                permits_df = pd.read_csv(permits_path)
                data_frames.append(permits_df)
                logger.info(f"Loaded building permits data: {len(permits_df)} records")
            except Exception as e:
                logger.error(f"Error loading building permits data: {str(e)}")
        
        # Business licenses
        licenses_path = self.raw_dir / "business_licenses.csv"
        if licenses_path.exists():
            try:
                licenses_df = pd.read_csv(licenses_path)
                data_frames.append(licenses_df)
                logger.info(f"Loaded business licenses data: {len(licenses_df)} records")
            except Exception as e:
                logger.error(f"Error loading business licenses data: {str(e)}")
        
        # Check if we have any data
        if not data_frames:
            logger.error("No data files found")
            return None
        
        # Combine all data frames
        # Note: This is a simplified approach; in a real scenario, we would need to
        # properly merge these data frames based on common keys
        combined_data = pd.concat(data_frames, ignore_index=True)
        
        logger.info(f"Combined data: {len(combined_data)} records")
        
        return combined_data
    
    def collect_census_data(self):
        """
        Collect Census demographic data for ZIP codes.
        
        Args:
            zip_codes (list): List of ZIP codes to collect data for
            
        Returns:
            pd.DataFrame: Census data
        """
        try:
            logger.info("Collecting Census demographic data.")
            
            # Illinois state FIPS code
            state_fips = '17'
            
            # Define Census variables to collect
            variables = {
                'B01003_001E': 'population',
                'B19013_001E': 'median_income',
                'B25001_001E': 'housing_units',
                'B25003_001E': 'occupied_units',
                'B25003_002E': 'owner_occupied',
                'B25003_003E': 'vacant_units',
                'B25035_001E': 'median_year_built',
                'B25064_001E': 'median_rent',
                'B25077_001E': 'median_home_value',
                'B25091_001E': 'with_mortgage'
            }
            
            census_data = []
            for zip_code in settings.CHICAGO_ZIP_CODES:
                try:
                    # Get Census data for ZIP code using state_zipcode method
                    data = self.census.acs5.state_zipcode(
                        list(variables.keys()),
                        state_fips,
                        zip_code
                    )
                    
                    if data and len(data) > 0:
                        census_data.append({
                            'zip_code': zip_code,
                            'population': data[0].get('B01003_001E', 0),
                            'median_income': data[0].get('B19013_001E', 0),
                            'housing_units': data[0].get('B25001_001E', 0),
                            'occupied_units': data[0].get('B25003_001E', 0),
                            'owner_occupied': data[0].get('B25003_002E', 0),
                            'vacant_units': data[0].get('B25003_003E', 0),
                            'median_year_built': data[0].get('B25035_001E', 0),
                            'median_rent': data[0].get('B25064_001E', 0),
                            'median_home_value': data[0].get('B25077_001E', 0),
                            'with_mortgage': data[0].get('B25091_001E', 0)
                        })
                    else:
                        logger.warning(f"No Census data found for ZIP code {zip_code}")
                        
                except Exception as e:
                    logger.warning(f"Failed to collect Census data for ZIP code {zip_code}: {str(e)}")
                    continue
            
            if not census_data:
                logger.error("No Census data collected for any ZIP code")
                return None
                
            df = pd.DataFrame(census_data)
            logger.info(f"Collected Census data for {len(df)} ZIP codes")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting Census data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def collect_economic_data(self):
        """
        Collect economic data from FRED API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Collecting economic data from FRED API.")
            
            # Check if FRED API key is set
            if not self.fred:
                logger.error("FRED API key not set. Cannot collect economic data.")
                return False
            
            # Define economic indicators to collect
            indicators = {
                'CHIC917URN': 'Chicago Unemployment Rate',
                'CHIC917PCPI': 'Chicago Per Capita Personal Income',
                'CHIC917HOUS': 'Chicago Housing Price Index',
                'CHIC917RETAIL': 'Chicago Retail Sales',
                'MORTGAGE30US': 'US 30-Year Fixed Mortgage Rate',
                'CPIAUCSL': 'Consumer Price Index',
                'GDPC1': 'Real GDP',
                'CSUSHPISA': 'Case-Shiller Home Price Index'
            }
            
            # Collect data for each indicator
            economic_data = {}
            success_count = 0
            failure_count = 0
            
            for series_id, description in indicators.items():
                try:
                    # Get data from FRED with retry logic
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            series = self.fred.get_series(series_id)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Retry {attempt+1}/{max_retries} for {series_id}: {str(e)}")
                                continue
                            else:
                                raise
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(series)
                    df.columns = [series_id]
                    df.index.name = 'date'
                    df.reset_index(inplace=True)
                    
                    # Add to economic data
                    economic_data[series_id] = {
                        'description': description,
                        'data': df.to_dict(orient='records')
                    }
                    
                    logger.info(f"Collected {description} data: {len(df)} records")
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to collect {description} data: {str(e)}")
                    failure_count += 1
            
            # Check if we have data
            if not economic_data:
                logger.error("No economic data collected from FRED API")
                return False
            
            # Save to file
            output_path = self.raw_dir / "economic_data.csv"
            
            # Create a flat table for CSV output
            flat_data = []
            for series_id, info in economic_data.items():
                for record in info['data']:
                    flat_data.append({
                        'indicator': series_id,
                        'description': info['description'],
                        'date': record['date'],
                        'value': record[series_id]
                    })
            
            # Convert to DataFrame and save
            df_flat = pd.DataFrame(flat_data)
            df_flat.to_csv(output_path, index=False)
            
            # Also save the full structured data as JSON
            json_path = self.raw_dir / "economic_indicators.json"
            with open(json_path, 'w') as f:
                json.dump(economic_data, f, indent=2, default=str)
            
            logger.info(f"Economic data saved to {output_path} and {json_path}")
            logger.info(f"Successfully collected {success_count} economic indicators")
            if failure_count > 0:
                logger.warning(f"Failed to collect {failure_count} economic indicators")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting economic data: {str(e)}")
            return False
    
    def collect_building_permits(self):
        """
        Collect building permits data from Chicago Data Portal.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Collecting building permits data.")
            
            # Check if Chicago Data Portal token is set
            if not self.chicago_client:
                logger.error("Chicago Data Portal token not set. Cannot collect building permits data.")
                return False
            
            # Define the dataset ID for building permits
            dataset_id = "ydr8-5enu"  # Building Permits dataset
            
            # Define query parameters
            query_params = {
                "$limit": 10000,
                "$where": "issue_date > '2010-01-01'",
                "$order": "issue_date DESC"
            }
            
            # Get data from Chicago Data Portal
            permits = self.chicago_client.get(dataset_id, **query_params)
            
            # Check if we have data
            if not permits:
                logger.error("No building permits data collected from Chicago Data Portal")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(permits)
            
            # Clean and transform data
            
            # Extract ZIP code from address if not present
            if 'zip_code' not in df.columns and 'work_location' in df.columns:
                # Try to extract ZIP code from work_location
                df['zip_code'] = df['work_location'].str.extract(r'IL\s+(\d{5})')
            
            # Filter for Chicago ZIP codes
            if 'zip_code' in df.columns:
                df = df[df['zip_code'].isin(settings.CHICAGO_ZIP_CODES)]
            
            # Extract permit type and residential info
            if 'permit_type' not in df.columns and 'permit_type_description' in df.columns:
                df['permit_type'] = df['permit_type_description']
            
            # Extract unit count if available
            if 'units' not in df.columns:
                # Try to find a column with unit information
                unit_columns = [col for col in df.columns if 'unit' in col.lower()]
                if unit_columns:
                    df['units'] = df[unit_columns[0]]
                else:
                    # Default to 1 unit for residential permits
                    df['units'] = 1
            
            # Save to file
            output_path = self.raw_dir / "building_permits.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(df)} building permits to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting building permits data: {str(e)}")
            return False
    
    def collect_business_licenses(self):
        """
        Collect business licenses data from Chicago Data Portal.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Collecting business licenses data.")
            
            # Check if Chicago Data Portal token is set
            if not self.chicago_client:
                logger.error("Chicago Data Portal token not set. Cannot collect business licenses data.")
                return False
            
            # Define the dataset ID for business licenses
            dataset_id = "r5kz-chrr"  # Business Licenses dataset
            
            # Define query parameters
            query_params = {
                "$limit": 10000,
                "$where": "license_start_date > '2010-01-01'",
                "$order": "license_start_date DESC"
            }
            
            # Get data from Chicago Data Portal
            licenses = self.chicago_client.get(dataset_id, **query_params)
            
            # Check if we have data
            if not licenses:
                logger.error("No business licenses data collected from Chicago Data Portal")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(licenses)
            
            # Clean and transform data
            
            # Extract ZIP code from address if not present
            if 'zip_code' not in df.columns and 'address' in df.columns:
                # Try to extract ZIP code from address
                df['zip_code'] = df['address'].str.extract(r'IL\s+(\d{5})')
            
            # Filter for Chicago ZIP codes
            if 'zip_code' in df.columns:
                df = df[df['zip_code'].isin(settings.CHICAGO_ZIP_CODES)]
            
            # Extract business type
            if 'business_type' not in df.columns and 'business_activity' in df.columns:
                df['business_type'] = df['business_activity']
            
            # Save to file
            output_path = self.raw_dir / "business_licenses.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(df)} business licenses to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting business licenses data: {str(e)}")
            return False
    
    def collect_vacancy_data(self):
        """
        Collect vacancy data from HUD API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Collecting vacancy data.")
            
            # Check if HUD API key is set
            if not self.hud_api_key or self.hud_api_key == 'your_hud_api_key':
                logger.error("HUD API key not set. Cannot collect vacancy data.")
                return False
            
            # Define API endpoint - using the correct HUD API endpoint
            api_url = "https://www.huduser.gov/portal/api/vacancy/survey"
            
            # Collect data for each ZIP code
            data = []
            success_count = 0
            failure_count = 0
            
            for zip_code in settings.CHICAGO_ZIP_CODES:
                try:
                    # Define query parameters
                    params = {
                        'type': 'zip',
                        'query': zip_code,
                        'year': datetime.now().year - 1,  # Use previous year
                        'format': 'json',
                        'state': 'IL'  # Add state parameter
                    }
                    
                    # Define headers
                    headers = {
                        'Authorization': f"Bearer {self.hud_api_key}",
                        'Accept': 'application/json',
                        'User-Agent': 'ChicagoHousingPipeline/1.0'  # Add user agent
                    }
                    
                    # Make API request with improved retry logic
                    max_retries = 5  # Increased retries
                    retry_delay = 10  # Increased delay
                    last_error = None
                    
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(
                                api_url, 
                                params=params, 
                                headers=headers, 
                                timeout=30
                            )
                            
                            # Check if request was successful
                            if response.status_code == 200:
                                # Parse response
                                vacancy_data = response.json()
                                
                                # Extract data with validation
                                if vacancy_data and 'data' in vacancy_data and len(vacancy_data['data']) > 0:
                                    for item in vacancy_data['data']:
                                        # Validate required fields
                                        if all(k in item for k in ['vacancy_rate', 'total_units']):
                                            record = {
                                                'zip_code': zip_code,
                                                'year': params['year'],
                                                'vacancy_rate': float(item.get('vacancy_rate', 0)),
                                                'rental_vacancy_rate': float(item.get('rental_vacancy_rate', 0)),
                                                'homeowner_vacancy_rate': float(item.get('homeowner_vacancy_rate', 0)),
                                                'total_units': int(item.get('total_units', 0)),
                                                'occupied_units': int(item.get('occupied_units', 0)),
                                                'vacant_units': int(item.get('vacant_units', 0))
                                            }
                                            data.append(record)
                                    
                                    success_count += 1
                                    break  # Success, exit retry loop
                                else:
                                    logger.warning(f"No valid vacancy data found for ZIP code {zip_code}")
                                    failure_count += 1
                                    break  # No data, exit retry loop
                            elif response.status_code == 429:  # Rate limit
                                retry_after = int(response.headers.get('Retry-After', retry_delay))
                                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                                time.sleep(retry_after)
                                continue
                            else:
                                logger.warning(f"Request failed with status {response.status_code}")
                                if attempt < max_retries - 1:
                                    time.sleep(retry_delay * (attempt + 1))
                                else:
                                    failure_count += 1
                                    break
                                
                        except requests.exceptions.RequestException as e:
                            last_error = e
                            if attempt < max_retries - 1:
                                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                                time.sleep(retry_delay * (attempt + 1))
                            else:
                                logger.error(f"All retry attempts failed for ZIP {zip_code}: {str(e)}")
                                failure_count += 1
                            break
                
                except Exception as e:
                    logger.error(f"Error processing ZIP code {zip_code}: {str(e)}")
                    failure_count += 1
                    continue
            
            # Check if we have data
            if not data:
                logger.error("No vacancy data collected for any ZIP code")
                # Create a minimal valid DataFrame with default values
                default_data = {
                    'zip_code': settings.CHICAGO_ZIP_CODES,
                    'year': [datetime.now().year - 1] * len(settings.CHICAGO_ZIP_CODES),
                    'vacancy_rate': [0.0] * len(settings.CHICAGO_ZIP_CODES),
                    'rental_vacancy_rate': [0.0] * len(settings.CHICAGO_ZIP_CODES),
                    'homeowner_vacancy_rate': [0.0] * len(settings.CHICAGO_ZIP_CODES),
                    'total_units': [0] * len(settings.CHICAGO_ZIP_CODES),
                    'occupied_units': [0] * len(settings.CHICAGO_ZIP_CODES),
                    'vacant_units': [0] * len(settings.CHICAGO_ZIP_CODES)
                }
                df = pd.DataFrame(default_data)
            else:
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Validate DataFrame
                required_columns = ['zip_code', 'year', 'vacancy_rate', 'total_units']
                if not all(col in df.columns for col in required_columns):
                    logger.error(f"Missing required columns in vacancy data: {required_columns}")
                    return False
                
                # Clean data
                df = df.fillna(0)  # Fill missing values with 0
                df = df.drop_duplicates()  # Remove duplicates
            
            # Save to file
            output_path = self.raw_dir / "vacancy_data.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved vacancy data for {success_count} ZIP codes to {output_path}")
            if failure_count > 0:
                logger.warning(f"Failed to collect vacancy data for {failure_count} ZIP codes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting vacancy data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_migration_data(self):
        """
        Collect migration data from Census API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Collecting migration data.")
            
            # Check if Census API key is set
            if not self.census:
                logger.error("Census API key not set. Cannot collect migration data.")
                return False
            
            # Define migration variables
            variables = [
                'B07001_001E',  # Total population
                'B07001_017E',  # Same house 1 year ago
                'B07001_033E',  # Moved within same county
                'B07001_049E',  # Moved from different county within same state
                'B07001_065E',  # Moved from different state
                'B07001_081E'   # Moved from abroad
            ]
            
            # Illinois state FIPS code
            state_fips = '17'
            
            migration_data = []
            for zip_code in settings.CHICAGO_ZIP_CODES:
                try:
                    # Get migration data for ZIP code
                    data = self.census.acs5.state_zipcode(
                        variables,
                        state_fips,
                        zip_code
                    )
                    
                    if data and len(data) > 0:
                        migration_data.append({
                            'zip_code': zip_code,
                            'total_population': data[0].get('B07001_001E', 0),
                            'same_house': data[0].get('B07001_017E', 0),
                            'same_county': data[0].get('B07001_033E', 0),
                            'same_state': data[0].get('B07001_049E', 0),
                            'different_state': data[0].get('B07001_065E', 0),
                            'abroad': data[0].get('B07001_081E', 0)
                        })
                    else:
                        logger.warning(f"No migration data found for ZIP code {zip_code}")
                        
                except Exception as e:
                    logger.warning(f"Failed to collect migration data for ZIP code {zip_code}: {str(e)}")
                    continue
            
            if not migration_data:
                logger.error("No migration data collected for any ZIP code")
                return False
                
            # Convert to DataFrame
            df = pd.DataFrame(migration_data)
            
            # Calculate migration rates
            df['migration_rate'] = (df['same_county'] + df['same_state'] + df['different_state'] + df['abroad']) / df['total_population']
            df['in_migration_rate'] = (df['same_state'] + df['different_state'] + df['abroad']) / df['total_population']
            df['out_migration_rate'] = df['different_state'] / df['total_population']
            
            # Save to file
            output_path = self.raw_dir / "migration_data.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved migration data for {len(df)} ZIP codes to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting migration data: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def collect_retail_gdp_data(self):
        """
        Collect retail GDP data from BEA API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Collecting retail GDP data.")
            
            # Check if BEA API key is set
            bea_api_key = getattr(settings, 'BEA_API_KEY', None)
            if not bea_api_key or bea_api_key == 'your_bea_api_key':
                logger.error("BEA API key not set. Cannot collect retail GDP data.")
                return False
            
            # Define API endpoint
            api_url = "https://apps.bea.gov/api/data"
            
            # Define query parameters
            params = {
                'UserID': bea_api_key,
                'method': 'GetData',
                'datasetname': 'Regional',
                'TableName': 'CAGDP2',
                'GeoFips': 'COUNTY',
                'LineCode': 29,  # Retail trade
                'Year': ','.join([str(year) for year in range(datetime.now().year - 5, datetime.now().year)]),
                'ResultFormat': 'JSON'
            }
            
            # Make API request
            response = requests.get(api_url, params=params)
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"Failed to collect retail GDP data: {response.status_code}")
                return False
            
            # Parse response
            try:
                retail_data = response.json()
            except Exception as e:
                logger.error(f"Failed to parse retail GDP data: {str(e)}")
                return False
            
            # Extract data
            data = []
            
            if retail_data and 'BEAAPI' in retail_data and 'Results' in retail_data['BEAAPI']:
                results = retail_data['BEAAPI']['Results']
                
                if 'Data' in results:
                    for item in results['Data']:
                        # Check if this is for Cook County (Chicago)
                        if item.get('GeoName', '').startswith('Cook'):
                            record = {
                                'year': item.get('TimePeriod'),
                                'retail_gdp': item.get('DataValue'),
                                'unit_of_measure': item.get('CL_UNIT'),
                                'region': item.get('GeoName')
                            }
                            data.append(record)
            
            # Check if we have data
            if not data:
                logger.error("No retail GDP data collected")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save to file
            output_path = self.raw_dir / "retail_gdp_data.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(df)} retail GDP records to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting retail GDP data: {str(e)}")
            return False
