"""
Census API collector for Chicago Housing Pipeline - FIXED VERSION.

This module handles data collection from the Census API without automatic fallbacks to sample data.
"""

import os
import logging
import pandas as pd
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

class CensusCollector:
    """
    Collector for Census API data - REAL DATA ONLY VERSION.
    
    Collects demographic and housing data from the Census API without automatic fallbacks.
    """
    
    def __init__(self, api_key=None, cache_dir=None):
        """Initialize the Census collector."""
        self.api_key = api_key or os.environ.get('CENSUS_API_KEY') or settings.CENSUS_API_KEY
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.DATA_DIR) / "cache" / "census"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_fips = '17'  # Illinois
        
        if not self.api_key or self.api_key == 'your_census_api_key':
            logger.warning("Census API key not set")
            self.api_key = None
    
    def collect_data(self, year=None, variables=None, geo_unit='zip code tabulation area', use_sample=False):
        """Collect demographic data from Census API."""
        return self.collect(year, variables, geo_unit, use_sample)
    
    def collect(self, year=None, variables=None, geo_unit='zip code tabulation area', use_sample=False):
        """
        Collect demographic data from Census API.
        
        Args:
            year (int, optional): Census year. Defaults to 2020.
            variables (list, optional): Census variables to collect
            geo_unit (str, optional): Geographic unit
            use_sample (bool, optional): Whether to use sample data. Defaults to False.
            
        Returns:
            pd.DataFrame: Census data
        """
        # Only use sample data if explicitly requested
        if use_sample:
            logger.info("✅ Using sample data as explicitly requested")
            return self._generate_sample_data(year or 2020, geo_unit)
        
        # Check for API key - REQUIRED for production data
        if not self.api_key:
            error_msg = "❌ Census API key not configured. Cannot collect real data."
            logger.error(error_msg)
            logger.error("Set CENSUS_API_KEY environment variable or run: python setup_api_keys.py")
            logger.error("💡 Use --use-sample-data flag to bypass this error")
            raise Exception(error_msg)
        
        # Set defaults
        if year is None:
            year = 2020
        if variables is None:
            variables = [
                'B01001_001E',  # Total population
                'B19013_001E',  # Median household income
                'B25001_001E',  # Total housing units
                'B25003_001E',  # Occupied housing units
                'B25003_003E',  # Renter-occupied housing units
            ]
        
        # Check cache first
        cache_name = f"census_real_{year}_{geo_unit.replace(' ', '_')}"
        cached_data = self._load_cached_data(cache_name)
        if cached_data is not None:
            logger.info(f"✅ Using cached real Census data: {len(cached_data)} records")
            return cached_data
        
        try:
            # Initialize Census API client
            from census import Census
            c = Census(self.api_key)
            
            logger.info(f"🚀 Collecting REAL Census data for year {year}")
            
            # Collect data for each Chicago ZIP code individually
            data = []
            success_count = 0
            
            for i, zip_code in enumerate(settings.CHICAGO_ZIP_CODES):
                try:
                    logger.debug(f"Collecting data for ZIP {zip_code}...")
                    
                    # Use ACS 5-year estimates for individual ZIP codes
                    zip_data = c.acs5.state_zipcode(
                        variables,
                        self.state_fips,
                        zip_code
                    )
                    
                    if zip_data and len(zip_data) > 0:
                        data.extend(zip_data)
                        success_count += 1
                        population = zip_data[0].get('B01001_001E', 'N/A')
                        logger.info(f"✅ REAL DATA: ZIP {zip_code} - Population: {population}")
                        
                        # **FIXED: Collect ALL Chicago ZIP codes, not just 10**
                        # Only show progress every 10 ZIP codes to reduce log spam
                        if success_count % 10 == 0:
                            logger.info(f"Progress: Collected real data for {success_count} ZIP codes...")
                    else:
                        logger.warning(f"No data returned for ZIP {zip_code}")
                        
                except Exception as zip_error:
                    logger.warning(f"Failed to collect data for ZIP {zip_code}: {str(zip_error)}")
                    continue
            
            # Check if we got any data
            if not data:
                error_msg = f"Failed to collect Census data for any ZIP codes. Attempted {len(settings.CHICAGO_ZIP_CODES)} ZIP codes."
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            logger.info(f"🎉 SUCCESS: Collected REAL Census data for {success_count} ZIP codes")
            
            # Convert to DataFrame and process
            df = pd.DataFrame(data)
            
            # Rename columns
            column_map = {
                'B01001_001E': 'population',
                'B19013_001E': 'median_income',
                'B25001_001E': 'housing_units',
                'B25003_001E': 'occupied_housing_units',
                'B25003_003E': 'renter_occupied_units',
                'zip code tabulation area': 'zip_code'
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            
            # Add year column
            df['year'] = year
            
            # Convert numeric columns
            numeric_columns = ['population', 'median_income', 'housing_units', 'occupied_housing_units', 'renter_occupied_units']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle ZIP code formatting
            if 'zip_code' in df.columns:
                df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
            elif 'zip code tabulation area' in df.columns:
                df['zip_code'] = df['zip code tabulation area'].astype(str).str.zfill(5)
                df = df.drop(columns=['zip code tabulation area'])
            
            # Filter for Chicago ZIP codes only
            if 'zip_code' in df.columns:
                chicago_zips = [str(z).zfill(5) for z in settings.CHICAGO_ZIP_CODES]
                df = df[df['zip_code'].isin(chicago_zips)]
                logger.info(f"✅ Filtered to {len(df)} Chicago ZIP codes with real data")
            
            # Validate critical columns
            required_cols = ['zip_code', 'population', 'housing_units', 'median_income']
            missing_cols = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
            if missing_cols:
                logger.error(f"❌ Missing critical Census columns: {missing_cols}")
                raise Exception(f"Census data missing required columns: {missing_cols}")
            
            logger.info(f"✅ Successfully collected all required Census columns: {required_cols}")
            
            # Cache the real data
            self._cache_data(df, cache_name)
            
            logger.info(f"🎉 FINAL SUCCESS: {len(df)} records of REAL Census data collected!")
            return df
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Real Census data collection failed!")
            logger.error(f"Error details: {str(e)}")
            logger.error("Solutions:")
            logger.error("1. Check Census API key: python main.py --check-api-keys")
            logger.error("2. Set up API keys: python setup_api_keys.py")
            logger.error("3. Use sample data flag: python main.py --use-sample-data")
            
            # NO AUTOMATIC FALLBACK - raise the error
            raise Exception(f"Census API collection failed: {str(e)}. Use --use-sample-data flag if you want to use sample data.")
    
    def collect_historical(self, start_year=2010, end_year=2020, variables=None, use_sample=False):
        """Collect historical Census data for multiple years."""
        all_data = []
        for year in range(start_year, end_year + 1):
            try:
                df = self.collect(year=year, variables=variables, use_sample=use_sample)
                if df is not None:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to collect data for year {year}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("Failed to collect any historical Census data")
            if use_sample:
                return self._generate_sample_historical_data(start_year, end_year)
            else:
                raise Exception("Failed to collect any historical Census data")
        
        # Combine all years
        historical_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Collected historical Census data from {start_year} to {end_year}: {len(historical_df)} records")
        return historical_df
    
    def _cache_data(self, data, cache_name):
        """Cache data to file."""
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        data.to_pickle(cache_path)
        logger.info(f"✅ Cached real data to {cache_path}")
    
    def _load_cached_data(self, cache_name):
        """Load data from cache if available."""
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_pickle(cache_path)
        return None
    
    def _generate_sample_data(self, year, geo_unit):
        """Generate sample Census data when explicitly requested."""
        logger.warning(f"Generating sample Census data for year {year}")
        
        # Check if sample data file exists
        sample_path = Path(settings.DATA_DIR) / "sample" / "census_data.csv"
        if sample_path.exists():
            logger.info(f"Loading sample data from {sample_path}")
            sample_df = pd.read_csv(sample_path)
            
            # Filter by year if year column exists
            if 'year' in sample_df.columns:
                sample_df = sample_df[sample_df['year'] == year]
            
            # Add year column if it doesn't exist
            if 'year' not in sample_df.columns:
                sample_df['year'] = year
            
            return sample_df
        
        # Generate synthetic data for Chicago ZIP codes if no sample file exists
        chicago_zips = settings.CHICAGO_ZIP_CODES[:10]  # Limit to 10 for consistency
        
        data = []
        for zip_code in chicago_zips:
            # Generate deterministic but varied values
            population = int(10000 + 30000 * (hash(f"{zip_code}_{year}") % 100) / 100)
            median_income = int(30000 + 100000 * (hash(f"{zip_code}_{year}_income") % 100) / 100)
            housing_units = int(population / 2.5)
            occupied_housing_units = int(housing_units * 0.9)
            renter_occupied_units = int(occupied_housing_units * 0.6)
            
            data.append({
                'zip_code': zip_code,
                'population': population,
                'median_income': median_income,
                'housing_units': housing_units,
                'occupied_housing_units': occupied_housing_units,
                'renter_occupied_units': renter_occupied_units,
                'year': year
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample records")
        return df
    
    def _generate_sample_historical_data(self, start_year, end_year):
        """Generate sample historical Census data when explicitly requested."""
        logger.warning(f"Generating sample historical Census data from {start_year} to {end_year}")
        
        all_data = []
        for year in range(start_year, end_year + 1):
            df = self._generate_sample_data(year, 'zip code tabulation area')
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            logger.error("Failed to generate sample historical Census data")
            return None
        
        historical_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Generated {len(historical_df)} sample historical records")
        return historical_df
