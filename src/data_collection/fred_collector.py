"""
FRED API collector for Chicago Housing Pipeline.

This module handles data collection from the Federal Reserve Economic Data (FRED) API.
"""

import os
import logging
import traceback
import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import requests

from src.config import settings

logger = logging.getLogger(__name__)

class FREDCollector:
    """
    Collector for FRED API data.
    
    Collects economic indicators and market trends from the FRED API.
    """
    
    def __init__(self, api_key=None, cache_dir=None):
        """
        Initialize the FRED collector.
        
        Args:
            api_key (str, optional): FRED API key
            cache_dir (Path, optional): Directory to cache data
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY') or settings.FRED_API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.DATA_DIR) / "cache" / "fred"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load retry settings from configuration
        self.retry_settings = settings.API_RETRY_SETTINGS
        
        if not self.api_key:
            logger.warning("FRED API key not set. Set FRED_API_KEY environment variable or update settings.py")
    
    def collect_data(self, series_ids=None, start_date=None, end_date=None, use_sample=False):
        """
        Collect economic data from FRED API.
        
        This is an alias for the collect method to maintain compatibility with the pipeline.
        
        Args:
            series_ids (list, optional): FRED series IDs to collect. Defaults to standard set.
            start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to 10 years ago.
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            use_sample (bool, optional): Whether to use sample data instead of API. Defaults to False.
            
        Returns:
            pd.DataFrame: FRED economic data
        """
        return self.collect(series_ids, start_date, end_date, use_sample)
    
    def collect(self, series_ids=None, start_date=None, end_date=None, use_sample=False):
        """
        Collect economic data from FRED API.
        
        Args:
            series_ids (list, optional): FRED series IDs to collect. Defaults to standard set.
            start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to 10 years ago.
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            use_sample (bool, optional): Whether to use sample data instead of API. Defaults to False.
            
        Returns:
            pd.DataFrame: FRED economic data
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')  # ~10 years
            
            # Check if we should use sample data
            if use_sample:
                logger.info("Using sample data as requested")
                return self._generate_sample_data(series_ids, start_date, end_date)
            
            # Check cache first
            cache_name = f"fred_{start_date}_{end_date}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Set default series IDs if not provided
            if series_ids is None:
                # Use series IDs from settings if available
                if hasattr(settings, 'FRED_SERIES_IDS'):
                    # Use only the values (series IDs), not the keys
                    series_ids = []
                    for key, value in settings.FRED_SERIES_IDS.items():
                        # Skip alternative series (those with _ALT in the key)
                        if '_ALT' not in key:
                            series_ids.append(value)
                else:
                    series_ids = [
                        'MORTGAGE30US',  # 30-Year Fixed Rate Mortgage Average
                        'CPIAUCSL',      # Consumer Price Index for All Urban Consumers
                        'HOUST',         # Housing Starts: Total: New Privately Owned Housing Units Started
                        'RRVRUSQ156N',   # Rental Vacancy Rate in the United States
                        'MSPUS',         # Median Sales Price of Houses Sold for the United States
                        'UMCSENT',       # University of Michigan: Consumer Sentiment
                        'UNRATE',        # Unemployment Rate
                        'GDPC1',         # Real Gross Domestic Product
                        'RSAFS',         # Advance Retail Sales: Retail and Food Services, Total
                        'PERMIT',        # New Private Housing Units Authorized by Building Permits
                    ]
            
            # Check if API key is available
            if not self.api_key:
                logger.error("FRED API key not set. Using sample data.")
                return self._generate_sample_data(series_ids, start_date, end_date)
            
            # Initialize FRED API client
            try:
                import fredapi
                fred = fredapi.Fred(api_key=self.api_key)
            except ImportError:
                logger.error("fredapi package not installed. Install with: pip install fredapi")
                return self._generate_sample_data(series_ids, start_date, end_date)
            
            # Collect data for each series
            all_series = []
            failed_series = []
            
            for series_id in series_ids:
                logger.info(f"Collecting FRED data for series {series_id}")
                series_data = self._get_series_with_retry(fred, series_id, start_date, end_date)
                
                if series_data is not None:
                    # Convert to DataFrame
                    series_df = series_data.reset_index()
                    series_df.columns = ['date', 'value']
                    series_df['series_id'] = series_id
                    series_df['data_source'] = 'FRED'
                    all_series.append(series_df)
                else:
                    failed_series.append(series_id)
            
            # Log failed series
            if failed_series:
                logger.warning(f"Failed to collect data for {len(failed_series)} series: {failed_series}")
                
                # Try alternative series for failed ones
                for failed_id in failed_series:
                    alternative_data = self._get_alternative_series(fred, failed_id, start_date, end_date)
                    if alternative_data is not None:
                        all_series.append(alternative_data)
            
            if not all_series:
                logger.error("Failed to collect any FRED data")
                return self._generate_sample_data(series_ids, start_date, end_date)
            
            # Combine all series
            df = pd.concat(all_series, ignore_index=True)
            
            # Add year column
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            logger.info(f"Collected {len(df)} records from FRED API")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting FRED data: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_sample_data(series_ids, start_date, end_date)
    
    def _get_series_with_retry(self, fred, series_id, start_date, end_date):
        """
        Get series data with retry logic.
        
        Args:
            fred: FRED API client
            series_id (str): Series ID
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.Series or None: Series data or None if failed
        """
        max_retries = self.retry_settings['max_retries']
        initial_delay = self.retry_settings['initial_delay']
        backoff_factor = self.retry_settings['backoff_factor']
        max_delay = self.retry_settings['max_delay']
        jitter = self.retry_settings['jitter']
        
        for attempt in range(max_retries):
            try:
                series = fred.get_series(series_id, start_date, end_date)
                if series is not None and not series.empty:
                    return series
                else:
                    logger.warning(f"No data returned for series {series_id}")
                    if attempt < max_retries - 1:
                        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                        # Add jitter to prevent thundering herd
                        delay = delay * (1 + random.uniform(-jitter, jitter))
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for series {series_id}")
                        return None
            except Exception as e:
                if "Bad Request" in str(e) and "does not exist" in str(e):
                    logger.error(f"Series {series_id} does not exist. Skipping.")
                    return None
                elif attempt < max_retries - 1:
                    delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                    # Add jitter to prevent thundering herd
                    delay = delay * (1 + random.uniform(-jitter, jitter))
                    logger.warning(f"Attempt {attempt+1} failed for series {series_id}: {str(e)}")
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for series {series_id}: {str(e)}")
                    return None
        
        return None
    
    def _get_alternative_series(self, fred, failed_id, start_date, end_date):
        """
        Get alternative series for a failed series.
        
        Args:
            fred: FRED API client
            failed_id (str): Failed series ID
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame or None: Alternative series data or None if failed
        """
        # Find the original key for this value in settings
        original_keys = []
        for k, v in settings.FRED_SERIES_IDS.items():
            if v == failed_id and '_ALT' not in k:
                original_keys.append(k)
        
        if not original_keys:
            logger.warning(f"No original key found for failed series {failed_id}")
            return None
        
        original_key = original_keys[0]
        logger.info(f"Looking for alternative series for {original_key} ({failed_id})")
        
        # Check if we have alternatives in the data sources documentation
        alternatives = []
        if hasattr(settings, 'DATA_SOURCES'):
            for category, data in settings.DATA_SOURCES.items():
                if isinstance(data, dict) and 'primary' in data and isinstance(data['primary'], dict):
                    if data['primary'].get('series_id') == failed_id:
                        for alt in data.get('alternatives', []):
                            if isinstance(alt, dict) and 'series_id' in alt and alt.get('source') == 'FRED':
                                alternatives.append(alt['series_id'])
        
        # If no alternatives found in documentation, check for ALT keys in FRED_SERIES_IDS
        if not alternatives:
            for k, v in settings.FRED_SERIES_IDS.items():
                if original_key in k and '_ALT' in k:
                    alternatives.append(v)
        
        # If still no alternatives, use default fallbacks
        if not alternatives:
            if 'HOUSING_PRICE' in original_key:
                alternatives = ['MSPUS', 'CSUSHPISA']  # National housing price indices
            elif 'RETAIL_SALES' in original_key:
                alternatives = ['RSAFS', 'RETSCHUS']   # National retail sales
            elif 'VACANCY' in original_key:
                alternatives = ['RRVRUSQ156N']         # National rental vacancy rate
        
        # Try each alternative
        for alt_id in alternatives:
            logger.info(f"Trying alternative series {alt_id} for {original_key}")
            alt_data = self._get_series_with_retry(fred, alt_id, start_date, end_date)
            
            if alt_data is not None:
                # Convert to DataFrame
                alt_df = alt_data.reset_index()
                alt_df.columns = ['date', 'value']
                alt_df['series_id'] = original_key  # Use original key as series_id for consistency
                alt_df['original_series_id'] = alt_id  # Store the actual series ID used
                alt_df['data_source'] = 'FRED'
                logger.info(f"Successfully collected alternative data for {original_key} using {alt_id}")
                return alt_df
        
        logger.error(f"Failed to find any working alternative for {original_key}")
        return None
    
    def collect_local_indicators(self, metro_area='CHICAGO', indicators=None, use_sample=False):
        """
        Collect local economic indicators for a specific metro area.
        
        Args:
            metro_area (str): Metro area code
            indicators (list, optional): Indicator series IDs
            use_sample (bool, optional): Whether to use sample data instead of API. Defaults to False.
            
        Returns:
            pd.DataFrame: Local economic indicators
        """
        try:
            # Check if we should use sample data
            if use_sample:
                logger.info("Using sample local data as requested")
                return self._generate_sample_local_data(metro_area, indicators)
            
            # Check cache first
            cache_name = f"fred_local_{metro_area}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Set default indicators if not provided
            if indicators is None:
                # Use Chicago-specific series IDs from settings if available
                if hasattr(settings, 'FRED_SERIES_IDS'):
                    chicago_series = {}
                    for k, v in settings.FRED_SERIES_IDS.items():
                        if ('CHIC' in k or 'CHICAGO' in k) and '_ALT' not in k:
                            chicago_series[k] = v
                    indicators = list(chicago_series.values())
                else:
                    # Default Chicago indicators
                    indicators = [
                        'CHIC917URN',    # Unemployment Rate in Chicago-Naperville-Elgin, IL-IN-WI (MSA)
                        'NGMP16980',     # Total GDP for Chicago-Naperville-Elgin, IL-IN-WI (MSA)
                        'CUURA207SA0',   # CPI for Chicago-Naperville-Elgin, IL-IN-WI
                        'ATNHPIUS16980Q', # House Price Index for Chicago-Naperville-Elgin
                        'CHXRSA',        # S&P CoreLogic Case-Shiller IL-Chicago Home Price Index
                    ]
            
            # Collect data
            df = self.collect(series_ids=indicators, use_sample=use_sample)
            
            # Add metro area column
            if df is not None:
                df['metro_area'] = metro_area
            
            # Cache the data
            if df is not None:
                self._cache_data(df, cache_name)
                logger.info(f"Collected local indicators for {metro_area}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting local indicators: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_sample_local_data(metro_area, indicators)
    
    def collect_vacancy_data(self, year=None, use_sample=False):
        """
        Collect vacancy data from Census API.
        
        Args:
            year (int, optional): Year to collect data for. Defaults to latest ACS year in settings.
            use_sample (bool, optional): Whether to use sample data instead of API. Defaults to False.
            
        Returns:
            pd.DataFrame: Vacancy data by ZIP code
        """
        try:
            # Set default year if not provided
            if year is None:
                year = settings.CENSUS_ACS_YEAR
            
            # Check if we should use sample data
            if use_sample:
                logger.info("Using sample vacancy data as requested")
                return self._generate_sample_vacancy_data()
            
            # Check cache first
            cache_name = f"vacancy_{year}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Check if Census API key is available
            if not settings.CENSUS_API_KEY:
                logger.error("Census API key not set. Using sample data.")
                return self._generate_sample_vacancy_data()
            
            # Get vacancy data from Census API
            vacancy_data = self._get_vacancy_data_with_retry(year)
            
            if vacancy_data is None:
                logger.error("Failed to collect vacancy data from Census API")
                return self._generate_sample_vacancy_data()
            
            # Cache the data
            self._cache_data(vacancy_data, cache_name)
            
            logger.info(f"Collected vacancy data for {year}")
            return vacancy_data
            
        except Exception as e:
            logger.error(f"Error collecting vacancy data: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_sample_vacancy_data()
    
    def _get_vacancy_data_with_retry(self, year):
        """
        Get vacancy data from Census API with retry logic.
        
        Args:
            year (int): Year to collect data for
            
        Returns:
            pd.DataFrame or None: Vacancy data or None if failed
        """
        max_retries = self.retry_settings['max_retries']
        initial_delay = self.retry_settings['initial_delay']
        backoff_factor = self.retry_settings['backoff_factor']
        max_delay = self.retry_settings['max_delay']
        jitter = self.retry_settings['jitter']
        
        # Get endpoint configuration
        endpoint_config = settings.CENSUS_API_ENDPOINTS['vacancy']
        endpoint_url = endpoint_config['acs5'].format(year=year)
        variables = ','.join(endpoint_config['variables'])
        
        for attempt in range(max_retries):
            try:
                # Construct API URL
                api_url = f"{endpoint_url}?get={variables}&for={endpoint_config['for']}&in={endpoint_config['in']}&key={settings.CENSUS_API_KEY}"
                
                # Make request
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    # Parse response
                    data = response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data[1:], columns=data[0])
                    
                    # Rename columns
                    df = df.rename(columns={
                        endpoint_config['variables'][0]: 'total_housing_units',
                        endpoint_config['variables'][1]: 'occupied_units',
                        endpoint_config['variables'][2]: 'vacant_units',
                        'zip code tabulation area': 'zip_code'
                    })
                    
                    # Convert numeric columns
                    for col in ['total_housing_units', 'occupied_units', 'vacant_units']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Calculate vacancy rate
                    df['vacancy_rate'] = (df['vacant_units'] / df['total_housing_units']) * 100
                    
                    # Add year column
                    df['year'] = year
                    
                    return df
                else:
                    logger.warning(f"Census API request failed with status code {response.status_code}")
                    if attempt < max_retries - 1:
                        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                        # Add jitter to prevent thundering herd
                        delay = delay * (1 + random.uniform(-jitter, jitter))
                        logger.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for Census API request")
                        return None
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                    # Add jitter to prevent thundering herd
                    delay = delay * (1 + random.uniform(-jitter, jitter))
                    logger.warning(f"Attempt {attempt+1} failed for Census API request: {str(e)}")
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed for Census API request: {str(e)}")
                    return None
        
        return None
    
    def _cache_data(self, data, cache_name):
        """Cache data to file."""
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        data.to_pickle(cache_path)
        logger.info(f"Cached data to {cache_path}")
    
    def _load_cached_data(self, cache_name):
        """Load data from cache if available."""
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_pickle(cache_path)
        return None
    
    def _generate_sample_data(self, series_ids, start_date, end_date):
        """
        Generate sample FRED data when API fails.
        
        Args:
            series_ids (list): FRED series IDs
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame: Sample FRED data
        """
        logger.warning(f"Generating sample FRED data")
        
        # Check if sample data exists
        sample_path = Path(settings.DATA_DIR) / "sample" / "economic_data.csv"
        if sample_path.exists():
            logger.info(f"Loading sample data from {sample_path}")
            sample_df = pd.read_csv(sample_path)
            
            # Ensure the sample data has all required columns
            if 'date' in sample_df.columns and 'value' in sample_df.columns and 'series_id' in sample_df.columns:
                # Convert date to datetime
                sample_df['date'] = pd.to_datetime(sample_df['date'])
                
                # Add year column if missing
                if 'year' not in sample_df.columns:
                    sample_df['year'] = sample_df['date'].dt.year
                
                # Add data_source column if missing
                if 'data_source' not in sample_df.columns:
                    sample_df['data_source'] = 'FRED'
                
                # Ensure all requested series are present
                if series_ids is not None:
                    missing_series = [s for s in series_ids if s not in sample_df['series_id'].unique()]
                    if missing_series:
                        logger.warning(f"Sample data missing series: {missing_series}")
                        # Generate missing series
                        missing_df = self._generate_synthetic_series(missing_series, start_date, end_date)
                        if missing_df is not None:
                            sample_df = pd.concat([sample_df, missing_df], ignore_index=True)
                
                return sample_df
        
        # Generate synthetic data for all series
        return self._generate_synthetic_series(series_ids, start_date, end_date)
    
    def _generate_synthetic_series(self, series_ids, start_date, end_date):
        """
        Generate synthetic FRED data for missing series.
        
        Args:
            series_ids (list): FRED series IDs to generate
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            pd.DataFrame: Synthetic FRED data
        """
        logger.warning(f"Generating synthetic data for missing FRED series: {series_ids}")
        
        synthetic_records = []
        
        # Create date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_range = pd.date_range(start, end, freq='MS')  # Monthly start
        
        for series_id in series_ids:
            # Generate realistic values based on series type
            base_value = 100
            trend = 0.02  # 2% annual growth
            
            if 'HOUSING' in series_id or 'HOME' in series_id:
                base_value = 300000  # Home prices
                trend = 0.03
            elif 'RETAIL' in series_id or 'SALES' in series_id:
                base_value = 50000  # Retail sales
                trend = 0.015
            elif 'RATE' in series_id or 'PERCENT' in series_id:
                base_value = 3.5  # Interest rates, percentages
                trend = 0.001
            elif 'GDP' in series_id or 'INCOME' in series_id:
                base_value = 25000  # GDP, income
                trend = 0.025
            
            # Generate values with trend and noise
            values = []
            for i, date in enumerate(date_range):
                # Calculate trend value
                years_elapsed = i / 12.0  # Convert months to years
                trend_value = base_value * (1 + trend) ** years_elapsed
                
                # Add seasonal variation (5% amplitude)
                seasonal = 0.05 * np.sin(2 * np.pi * i / 12)
                
                # Add random noise (2% amplitude)
                noise = 0.02 * (np.random.random() - 0.5)
                
                # Combine components
                final_value = trend_value * (1 + seasonal + noise)
                values.append(max(0, final_value))  # Ensure non-negative
            
            # Create DataFrame
            series_df = pd.DataFrame({
                'date': date_range,
                'value': values,
                'series_id': series_id,
                'data_source': 'FRED'
            })
            
            # Add year column
            series_df['year'] = series_df['date'].dt.year
            
            synthetic_records.append(series_df)
        
        if synthetic_records:
            return pd.concat(synthetic_records, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _generate_sample_local_data(self, metro_area, indicators=None):
        """Generate sample local data for a specific metro area."""
        logger.warning(f"Generating sample local data for {metro_area}")
        
        # Generate sample data
        df = self._generate_sample_data(indicators, None, None)
        
        # Add metro area column
        if df is not None:
            df['metro_area'] = metro_area
        
        return df
    
    def _generate_sample_vacancy_data(self):
        """Generate sample vacancy data."""
        logger.warning("Generating sample vacancy data")
        
        # Create sample data for Chicago ZIP codes
        zip_codes = settings.CHICAGO_ZIP_CODES
        
        # Generate random data
        data = []
        for zip_code in zip_codes:
            total_units = np.random.randint(5000, 20000)
            vacancy_rate = np.random.uniform(3, 15)
            vacant_units = int(total_units * (vacancy_rate / 100))
            occupied_units = total_units - vacant_units
            
            data.append({
                'zip_code': zip_code,
                'total_housing_units': total_units,
                'occupied_units': occupied_units,
                'vacant_units': vacant_units,
                'vacancy_rate': vacancy_rate,
                'year': settings.CENSUS_ACS_YEAR
            })
        
        return pd.DataFrame(data)
