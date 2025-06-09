"""
BEA API collector for Chicago Housing Pipeline.

This module handles data collection from the Bureau of Economic Analysis (BEA) API.
"""

import os
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import json
from datetime import datetime

from src.config import settings

logger = logging.getLogger(__name__)

class BEACollector:
    """
    Collector for BEA API data.
    
    Collects economic data from the Bureau of Economic Analysis (BEA) API.
    """
    
    def __init__(self, api_key=None, cache_dir=None):
        """
        Initialize the BEA collector.
        
        Args:
            api_key (str, optional): BEA API key
            cache_dir (Path, optional): Directory to cache data
        """
        self.api_key = api_key or os.environ.get('BEA_API_KEY') or settings.BEA_API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.DATA_DIR) / "cache" / "bea"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://apps.bea.gov/api/data"
        
        if not self.api_key or self.api_key == 'your_bea_api_key':
            logger.warning("BEA API key not set. Set BEA_API_KEY environment variable or update settings.py")
    
    def collect_data(self, dataset='Regional', table_name='CAGDP2', geo_fips='COUNTY', year=None, use_sample=False):
        """
        Collect economic data from BEA API.
        
        Args:
            dataset (str): BEA dataset name (e.g., 'Regional', 'NIPA')
            table_name (str): BEA table name (e.g., 'CAGDP2' for GDP by county)
            geo_fips (str): Geographic level (e.g., 'COUNTY', 'MSA')
            year (int or str, optional): Year or year range (e.g., '2010,2020')
            use_sample (bool, optional): Whether to use sample data instead of API. Defaults to False.
            
        Returns:
            pd.DataFrame: Economic data
        """
        try:
            # If use_sample is True, directly use sample data
            if use_sample:
                logger.info("Using sample data as requested")
                return self._generate_sample_data(dataset, table_name, geo_fips, year)
                
            # Check cache first
            cache_name = f"bea_{dataset}_{table_name}_{geo_fips}_{year}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Set default year if not provided
            if year is None:
                year = datetime.now().year - 1  # Previous year
            
            # Prepare API parameters
            params = {
                'UserID': self.api_key,
                'method': 'GetData',
                'ResultFormat': 'JSON',
                'DatasetName': dataset,
                'TableName': table_name,
                'GeoFips': geo_fips,
                'Year': str(year)
            }
            
            # Make API request
            logger.info(f"Collecting BEA data for {dataset}/{table_name}, year {year}")
            
            if self.api_key == 'your_bea_api_key':
                logger.warning("Using sample data because BEA API key is not set")
                return self._generate_sample_data(dataset, table_name, geo_fips, year)
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Error fetching BEA data: {str(e)}")
                return self._generate_sample_data(dataset, table_name, geo_fips, year)
            
            # Parse response
            try:
                results = data['BEAAPI']['Results']['Data']
                df = pd.DataFrame(results)
            except (KeyError, TypeError) as e:
                logger.error(f"Error parsing BEA data: {str(e)}")
                return self._generate_sample_data(dataset, table_name, geo_fips, year)
            
            # Process data based on dataset and table
            if dataset == 'Regional' and table_name == 'CAGDP2':
                # County GDP data
                df = self._process_county_gdp(df)
            elif dataset == 'Regional' and table_name == 'CAINC1':
                # County income data
                df = self._process_county_income(df)
            else:
                # Generic processing
                df = self._process_generic(df)
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            logger.info(f"Collected {len(df)} records from BEA API")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting BEA data: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_sample_data(dataset, table_name, geo_fips, year)
    
    def collect_gdp_by_county(self, year=None, use_sample=False):
        """
        Collect GDP data by county.
        
        Args:
            year (int or str, optional): Year or year range
            use_sample (bool, optional): Whether to use sample data instead of API. Defaults to False.
            
        Returns:
            pd.DataFrame: County GDP data
        """
        return self.collect_data(dataset='Regional', table_name='CAGDP2', geo_fips='COUNTY', year=year, use_sample=use_sample)
    
    def collect_income_by_county(self, year=None, use_sample=False):
        """
        Collect income data by county.
        
        Args:
            year (int or str, optional): Year or year range
            use_sample (bool, optional): Whether to use sample data instead of API. Defaults to False.
            
        Returns:
            pd.DataFrame: County income data
        """
        return self.collect_data(dataset='Regional', table_name='CAINC1', geo_fips='COUNTY', year=year, use_sample=use_sample)
    
    def _process_county_gdp(self, df):
        """Process county GDP data."""
        # Rename columns
        if 'GeoFips' in df.columns:
            df = df.rename(columns={'GeoFips': 'county_fips'})
        if 'GeoName' in df.columns:
            df = df.rename(columns={'GeoName': 'county_name'})
        if 'DataValue' in df.columns:
            df = df.rename(columns={'DataValue': 'gdp_value'})
        
        # Convert numeric columns
        if 'gdp_value' in df.columns:
            df['gdp_value'] = pd.to_numeric(df['gdp_value'].str.replace(',', ''), errors='coerce')
        
        # Add year column if not present
        if 'TimePeriod' in df.columns:
            df = df.rename(columns={'TimePeriod': 'year'})
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Extract county and state
        if 'county_name' in df.columns:
            df['state'] = df['county_name'].str.extract(r', ([A-Z]{2})$')
            df['county'] = df['county_name'].str.replace(r', [A-Z]{2}$', '', regex=True)
        
        # Filter for Illinois counties
        if 'state' in df.columns:
            df = df[df['state'] == 'IL']
        
        return df
    
    def _process_county_income(self, df):
        """Process county income data."""
        # Rename columns
        if 'GeoFips' in df.columns:
            df = df.rename(columns={'GeoFips': 'county_fips'})
        if 'GeoName' in df.columns:
            df = df.rename(columns={'GeoName': 'county_name'})
        if 'DataValue' in df.columns:
            df = df.rename(columns={'DataValue': 'income_value'})
        
        # Convert numeric columns
        if 'income_value' in df.columns:
            df['income_value'] = pd.to_numeric(df['income_value'].str.replace(',', ''), errors='coerce')
        
        # Add year column if not present
        if 'TimePeriod' in df.columns:
            df = df.rename(columns={'TimePeriod': 'year'})
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Extract county and state
        if 'county_name' in df.columns:
            df['state'] = df['county_name'].str.extract(r', ([A-Z]{2})$')
            df['county'] = df['county_name'].str.replace(r', [A-Z]{2}$', '', regex=True)
        
        # Filter for Illinois counties
        if 'state' in df.columns:
            df = df[df['state'] == 'IL']
        
        return df
    
    def _process_generic(self, df):
        """Generic data processing."""
        # Convert numeric columns
        for col in df.columns:
            if col.lower() in ['datavalue', 'value', 'amount']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Add year column if not present
        if 'TimePeriod' in df.columns:
            df = df.rename(columns={'TimePeriod': 'year'})
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        return df
    
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
    
    def _generate_sample_data(self, dataset, table_name, geo_fips, year):
        """
        Generate sample BEA data when API fails.
        
        Args:
            dataset (str): BEA dataset name
            table_name (str): BEA table name
            geo_fips (str): Geographic level
            year (int or str): Year or year range
            
        Returns:
            pd.DataFrame: Sample BEA data
        """
        logger.warning(f"Generating sample BEA data for {dataset}/{table_name}, year {year}")
        
        # Check if sample data exists
        sample_path = Path(settings.DATA_DIR) / "sample" / "economic_data.csv"
        if sample_path.exists():
            logger.info(f"Loading sample data from {sample_path}")
            sample_df = pd.read_csv(sample_path)
            
            # Filter by year if year column exists and year is specified
            if 'year' in sample_df.columns and year is not None:
                if isinstance(year, str) and ',' in year:
                    # Year range
                    years = [int(y) for y in year.split(',')]
                    min_year, max_year = min(years), max(years)
                    sample_df = sample_df[(sample_df['year'] >= min_year) & (sample_df['year'] <= max_year)]
                else:
                    # Single year
                    try:
                        year_val = int(year)
                        sample_df = sample_df[sample_df['year'] == year_val]
                    except (ValueError, TypeError):
                        pass
            
            return sample_df
        
        # Generate synthetic data
        if dataset == 'Regional' and table_name == 'CAGDP2':
            # County GDP data
            return self._generate_sample_county_gdp(year)
        elif dataset == 'Regional' and table_name == 'CAINC1':
            # County income data
            return self._generate_sample_county_income(year)
        else:
            # Generic data
            return self._generate_sample_generic(dataset, table_name, year)
    
    def _generate_sample_county_gdp(self, year):
        """Generate sample county GDP data."""
        # Cook County FIPS: 17031 (Chicago)
        # DuPage County FIPS: 17043
        # Lake County FIPS: 17097
        # Will County FIPS: 17197
        # Kane County FIPS: 17089
        
        counties = [
            {'fips': '17031', 'name': 'Cook County, IL', 'base_gdp': 500000},
            {'fips': '17043', 'name': 'DuPage County, IL', 'base_gdp': 150000},
            {'fips': '17097', 'name': 'Lake County, IL', 'base_gdp': 120000},
            {'fips': '17197', 'name': 'Will County, IL', 'base_gdp': 100000},
            {'fips': '17089', 'name': 'Kane County, IL', 'base_gdp': 80000}
        ]
        
        data = []
        
        # Process year input
        years = []
        if isinstance(year, str) and ',' in year:
            # Year range
            years = [int(y) for y in year.split(',')]
        else:
            # Single year
            try:
                years = [int(year)]
            except (ValueError, TypeError):
                years = [datetime.now().year - 1]  # Default to previous year
        
        for year_val in years:
            for county in counties:
                # Add some variation by year
                growth_factor = 1 + (year_val - 2010) * 0.03  # 3% annual growth
                gdp_value = county['base_gdp'] * growth_factor
                
                data.append({
                    'county_fips': county['fips'],
                    'county_name': county['name'],
                    'county': county['name'].split(' County')[0],
                    'state': 'IL',
                    'year': year_val,
                    'gdp_value': gdp_value,
                    'industry': 'All industry total',
                    'unit': 'Millions of current dollars'
                })
        
        df = pd.DataFrame(data)
        
        # Save as sample data for future use
        sample_path = Path(settings.DATA_DIR) / "sample" / "economic_data.csv"
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        df.to_csv(sample_path, index=False)
        
        return df
    
    def _generate_sample_county_income(self, year):
        """Generate sample county income data."""
        # Cook County FIPS: 17031 (Chicago)
        # DuPage County FIPS: 17043
        # Lake County FIPS: 17097
        # Will County FIPS: 17197
        # Kane County FIPS: 17089
        
        counties = [
            {'fips': '17031', 'name': 'Cook County, IL', 'base_income': 65000},
            {'fips': '17043', 'name': 'DuPage County, IL', 'base_income': 85000},
            {'fips': '17097', 'name': 'Lake County, IL', 'base_income': 80000},
            {'fips': '17197', 'name': 'Will County, IL', 'base_income': 75000},
            {'fips': '17089', 'name': 'Kane County, IL', 'base_income': 70000}
        ]
        
        data = []
        
        # Process year input
        years = []
        if isinstance(year, str) and ',' in year:
            # Year range
            years = [int(y) for y in year.split(',')]
        else:
            # Single year
            try:
                years = [int(year)]
            except (ValueError, TypeError):
                years = [datetime.now().year - 1]  # Default to previous year
        
        for year_val in years:
            for county in counties:
                # Add some variation by year
                growth_factor = 1 + (year_val - 2010) * 0.02  # 2% annual growth
                income_value = county['base_income'] * growth_factor
                
                data.append({
                    'county_fips': county['fips'],
                    'county_name': county['name'],
                    'county': county['name'].split(' County')[0],
                    'state': 'IL',
                    'year': year_val,
                    'income_value': income_value,
                    'measure': 'Per capita personal income',
                    'unit': 'Dollars'
                })
        
        df = pd.DataFrame(data)
        
        # Save as sample data for future use
        sample_path = Path(settings.DATA_DIR) / "sample" / "economic_data.csv"
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        df.to_csv(sample_path, index=False)
        
        return df
    
    def _generate_sample_generic(self, dataset, table_name, year):
        """Generate generic sample data."""
        # Create a simple DataFrame with the requested parameters
        data = []
        
        # Process year input
        years = []
        if isinstance(year, str) and ',' in year:
            # Year range
            years = [int(y) for y in year.split(',')]
        else:
            # Single year
            try:
                years = [int(year)]
            except (ValueError, TypeError):
                years = [datetime.now().year - 1]  # Default to previous year
        
        for year_val in years:
            data.append({
                'dataset': dataset,
                'table_name': table_name,
                'year': year_val,
                'value': 100 + year_val - 2010,  # Simple value based on year
                'unit': 'Index'
            })
        
        return pd.DataFrame(data)
