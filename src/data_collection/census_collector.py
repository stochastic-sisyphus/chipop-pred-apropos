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
        """Generate sample historical Census data with realistic Chicago neighborhood trends."""
        logger.warning(f"Generating sample historical Census data from {start_year} to {end_year}")

        # Chicago neighborhood characteristics for realistic trend generation
        neighborhood_profiles = {
            '60601': {'name': 'Loop', 'base_pop': 16000, 'growth': 0.025, 'income': 95000, 'income_growth': 0.03},
            '60602': {'name': 'Loop', 'base_pop': 1100, 'growth': 0.02, 'income': 92000, 'income_growth': 0.025},
            '60603': {'name': 'Loop', 'base_pop': 1100, 'growth': 0.015, 'income': 88000, 'income_growth': 0.02},
            '60604': {'name': 'South Loop', 'base_pop': 650, 'growth': 0.03, 'income': 72000, 'income_growth': 0.035},
            '60605': {'name': 'South Loop', 'base_pop': 34000, 'growth': 0.028, 'income': 85000, 'income_growth': 0.03},
            '60606': {'name': 'West Loop', 'base_pop': 3500, 'growth': 0.04, 'income': 98000, 'income_growth': 0.04},
            '60607': {'name': 'West Loop', 'base_pop': 30000, 'growth': 0.045, 'income': 105000, 'income_growth': 0.045},
            '60608': {'name': 'Pilsen', 'base_pop': 85000, 'growth': 0.01, 'income': 42000, 'income_growth': 0.025},
            '60609': {'name': 'Back of Yards', 'base_pop': 55000, 'growth': -0.005, 'income': 35000, 'income_growth': 0.015},
            '60610': {'name': 'Old Town', 'base_pop': 31000, 'growth': 0.02, 'income': 95000, 'income_growth': 0.025},
            '60612': {'name': 'Near West', 'base_pop': 36000, 'growth': 0.015, 'income': 52000, 'income_growth': 0.03},
            '60613': {'name': 'Lakeview', 'base_pop': 70000, 'growth': 0.008, 'income': 78000, 'income_growth': 0.02},
            '60614': {'name': 'Lincoln Park', 'base_pop': 65000, 'growth': 0.005, 'income': 115000, 'income_growth': 0.018},
            '60615': {'name': 'Bronzeville', 'base_pop': 42000, 'growth': 0.012, 'income': 38000, 'income_growth': 0.028},
            '60616': {'name': 'South Loop', 'base_pop': 45000, 'growth': 0.025, 'income': 65000, 'income_growth': 0.035},
            '60617': {'name': 'South Chicago', 'base_pop': 75000, 'growth': -0.008, 'income': 32000, 'income_growth': 0.01},
            '60618': {'name': 'Avondale', 'base_pop': 85000, 'growth': 0.01, 'income': 58000, 'income_growth': 0.022},
            '60619': {'name': 'Chatham', 'base_pop': 50000, 'growth': -0.01, 'income': 36000, 'income_growth': 0.012},
            '60620': {'name': 'Auburn Gresham', 'base_pop': 48000, 'growth': -0.012, 'income': 32000, 'income_growth': 0.01},
            '60621': {'name': 'Englewood', 'base_pop': 30000, 'growth': -0.02, 'income': 24000, 'income_growth': 0.008},
            '60622': {'name': 'Wicker Park', 'base_pop': 52000, 'growth': 0.018, 'income': 82000, 'income_growth': 0.035},
            '60623': {'name': 'Lawndale', 'base_pop': 75000, 'growth': -0.008, 'income': 28000, 'income_growth': 0.012},
            '60624': {'name': 'West Garfield', 'base_pop': 25000, 'growth': -0.015, 'income': 25000, 'income_growth': 0.01},
            '60625': {'name': 'Lincoln Square', 'base_pop': 55000, 'growth': 0.012, 'income': 62000, 'income_growth': 0.025},
            '60626': {'name': 'Rogers Park', 'base_pop': 55000, 'growth': 0.008, 'income': 45000, 'income_growth': 0.02},
            '60628': {'name': 'Roseland', 'base_pop': 52000, 'growth': -0.012, 'income': 34000, 'income_growth': 0.01},
            '60629': {'name': 'Chicago Lawn', 'base_pop': 80000, 'growth': 0.002, 'income': 38000, 'income_growth': 0.015},
            '60630': {'name': 'Jefferson Park', 'base_pop': 48000, 'growth': 0.005, 'income': 58000, 'income_growth': 0.018},
            '60631': {'name': 'Edgebrook', 'base_pop': 20000, 'growth': 0.003, 'income': 85000, 'income_growth': 0.015},
            '60632': {'name': 'Brighton Park', 'base_pop': 65000, 'growth': 0.005, 'income': 40000, 'income_growth': 0.018},
            '60633': {'name': 'Hegewisch', 'base_pop': 10000, 'growth': -0.005, 'income': 52000, 'income_growth': 0.012},
            '60634': {'name': 'Portage Park', 'base_pop': 65000, 'growth': 0.003, 'income': 55000, 'income_growth': 0.018},
            '60636': {'name': 'West Englewood', 'base_pop': 35000, 'growth': -0.018, 'income': 26000, 'income_growth': 0.008},
            '60637': {'name': 'Woodlawn', 'base_pop': 52000, 'growth': 0.008, 'income': 32000, 'income_growth': 0.022},
            '60638': {'name': 'Garfield Ridge', 'base_pop': 35000, 'growth': 0.002, 'income': 62000, 'income_growth': 0.015},
            '60639': {'name': 'Belmont Cragin', 'base_pop': 78000, 'growth': 0.005, 'income': 42000, 'income_growth': 0.018},
            '60640': {'name': 'Uptown', 'base_pop': 58000, 'growth': 0.015, 'income': 48000, 'income_growth': 0.028},
            '60641': {'name': 'Kilbourn Park', 'base_pop': 48000, 'growth': 0.008, 'income': 55000, 'income_growth': 0.02},
            '60642': {'name': 'Noble Square', 'base_pop': 8000, 'growth': 0.035, 'income': 92000, 'income_growth': 0.04},
            '60643': {'name': 'Morgan Park', 'base_pop': 28000, 'growth': -0.003, 'income': 58000, 'income_growth': 0.015},
            '60644': {'name': 'Austin', 'base_pop': 55000, 'growth': -0.01, 'income': 32000, 'income_growth': 0.01},
            '60645': {'name': 'West Ridge', 'base_pop': 55000, 'growth': 0.005, 'income': 48000, 'income_growth': 0.018},
            '60646': {'name': 'Sauganash', 'base_pop': 22000, 'growth': 0.002, 'income': 95000, 'income_growth': 0.015},
            '60647': {'name': 'Logan Square', 'base_pop': 72000, 'growth': 0.015, 'income': 65000, 'income_growth': 0.032},
            '60649': {'name': 'South Shore', 'base_pop': 52000, 'growth': -0.008, 'income': 28000, 'income_growth': 0.012},
            '60651': {'name': 'Humboldt Park', 'base_pop': 56000, 'growth': 0.005, 'income': 35000, 'income_growth': 0.02},
            '60652': {'name': 'Ashburn', 'base_pop': 42000, 'growth': 0.002, 'income': 58000, 'income_growth': 0.015},
            '60653': {'name': 'Bronzeville', 'base_pop': 18000, 'growth': 0.02, 'income': 42000, 'income_growth': 0.03},
            '60654': {'name': 'River North', 'base_pop': 24000, 'growth': 0.035, 'income': 125000, 'income_growth': 0.035},
            '60655': {'name': 'Mt Greenwood', 'base_pop': 20000, 'growth': 0.001, 'income': 82000, 'income_growth': 0.012},
            '60656': {'name': 'Norwood Park', 'base_pop': 38000, 'growth': 0.002, 'income': 72000, 'income_growth': 0.015},
            '60657': {'name': 'Lakeview', 'base_pop': 95000, 'growth': 0.008, 'income': 88000, 'income_growth': 0.022},
            '60659': {'name': 'North Park', 'base_pop': 45000, 'growth': 0.005, 'income': 52000, 'income_growth': 0.018},
            '60660': {'name': 'Edgewater', 'base_pop': 58000, 'growth': 0.01, 'income': 55000, 'income_growth': 0.022},
            '60661': {'name': 'West Loop', 'base_pop': 12000, 'growth': 0.05, 'income': 115000, 'income_growth': 0.045},
        }

        all_data = []
        base_year = 2010  # Reference year for calculations

        for year in range(start_year, end_year + 1):
            years_from_base = year - base_year

            for zip_code in settings.CHICAGO_ZIP_CODES:
                zip_str = str(zip_code)
                profile = neighborhood_profiles.get(zip_str, {
                    'name': 'Other',
                    'base_pop': 40000,
                    'growth': 0.005,
                    'income': 55000,
                    'income_growth': 0.02
                })

                # Calculate values with compound growth and some noise
                import random
                random.seed(hash(f"{zip_code}_{year}"))
                noise = random.uniform(0.97, 1.03)

                population = int(profile['base_pop'] * ((1 + profile['growth']) ** years_from_base) * noise)
                median_income = int(profile['income'] * ((1 + profile['income_growth']) ** years_from_base) * noise)
                housing_units = int(population / 2.3 * random.uniform(0.95, 1.05))
                occupied_units = int(housing_units * random.uniform(0.88, 0.94))
                renter_occupied = int(occupied_units * random.uniform(0.45, 0.65))

                all_data.append({
                    'zip_code': zip_str,
                    'population': population,
                    'median_income': median_income,
                    'housing_units': housing_units,
                    'occupied_housing_units': occupied_units,
                    'renter_occupied_units': renter_occupied,
                    'year': year
                })

        historical_df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(historical_df)} sample historical records with realistic Chicago trends")
        return historical_df
