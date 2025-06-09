"""
Chicago Data Portal collector for Chicago Housing Pipeline.

This module handles data collection from the Chicago Data Portal.
"""

import os
import logging
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

from src.config import settings

logger = logging.getLogger(__name__)

class ChicagoCollector:
    """
    Collector for Chicago Data Portal APIs.
    """
    
    def __init__(self, api_token=None, cache_dir=None):
        """
        Initialize the Chicago collector.
        
        Args:
            api_token (str, optional): Chicago Data Portal token
            cache_dir (Path, optional): Directory to cache data
        """
        self.api_token = api_token or os.environ.get('CHICAGO_DATA_TOKEN') or settings.CHICAGO_DATA_TOKEN
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.DATA_DIR) / "cache" / "chicago"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_token or self.api_token == 'your_chicago_data_token':
            logger.warning("Chicago Data Portal token not set. Set CHICAGO_DATA_TOKEN environment variable or update settings.py")
        
        # **IMPROVED: Better dataset configuration with field mapping**
        self.datasets = {
            'permits': {
                'id': 'ydr8-5enu',
                'name': 'Building Permits',
                'url': 'https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu',
                'required_fields': ['permit_', 'issue_date', 'work_description'],
                'field_mapping': {
                    'permit_number': 'permit_',
                    'permit_id': 'permit_',
                    'cost': 'reported_cost',
                    'estimated_cost': 'reported_cost'
                },
                'zip_fields': ['contact_1_zipcode', 'contact_2_zipcode', 'contact_3_zipcode']
            },
            'licenses': {
                'id': 'r5kz-chrr',
                'name': 'Business Licenses',
                'url': 'https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr',
                'required_fields': ['license_number', 'license_start_date', 'business_activity'],
                'field_mapping': {
                    'license_id': 'license_number'
                },
                'zip_fields': ['zip_code']
            },
            'zoning': {
                'datasets': [
                    {'id': '7cve-jgbp', 'name': 'Zoning Map'},
                    {'id': 'p9xt-wtk7', 'name': 'Current Zoning Districts'},
                    {'id': '9sgv-cezh', 'name': 'Zoning'},
                    {'id': 'ydr8-5enu', 'name': 'Boundaries - Zoning Districts'}
                ],
                'required_fields': ['zone_class'],
                'field_mapping': {
                    'case_number': 'ordinance_number',
                    'address': 'location'
                }
            }
        }

    def collect_data(self, use_sample=False):
        """
        Collect all data from Chicago Data Portal.
        
        Args:
            use_sample (bool): Whether to use sample data instead of API calls
        
        Returns:
            dict: Dictionary containing all collected data
        """
        try:
            logger.info("Collecting data from Chicago Data Portal")
            
            if use_sample:
                logger.info("Using sample data for Chicago Data Portal")
                return self._generate_sample_data()
            
            # Set default years
            current_year = datetime.now().year
            # **FIXED: Focus on years relevant for growth analysis (historical vs recent)**
            # Historical: 2018-2021, Recent: 2022-2024
            years = list(range(2018, current_year + 1))
            
            # **IMPROVED: Better error handling and validation**
            data = {}
            collection_errors = []
            
            # Collect building permits with enhanced validation
            try:
                permits_df = self.collect_building_permits(years=years)
                if permits_df is not None and len(permits_df) > 0:
                    # **FIXED: Validate required fields are present**
                    required_fields = self.datasets['permits']['required_fields']
                    missing_fields = [f for f in required_fields if f not in permits_df.columns]
                    
                    if missing_fields:
                        logger.error(f"❌ CRITICAL: Permits data missing required fields: {missing_fields}")
                        logger.error("❌ This violates real data requirements - cannot proceed")
                        raise ValueError(f"Permits data missing critical real fields: {missing_fields}")
                    
                    data['permits'] = permits_df
                    logger.info(f"✅ Successfully collected {len(permits_df)} permit records with all required fields")
                else:
                    raise ValueError("No permit data collected from Chicago Data Portal")
            except Exception as e:
                error_msg = f"Building permits collection failed: {str(e)}"
                collection_errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
            
            # Collect business licenses with enhanced validation
            try:
                licenses_df = self.collect_business_licenses(years=years)
                if licenses_df is not None and len(licenses_df) > 0:
                    data['licenses'] = licenses_df
                    logger.info(f"✅ Successfully collected {len(licenses_df)} license records")
                else:
                    raise ValueError("No license data collected from Chicago Data Portal")
            except Exception as e:
                error_msg = f"Business licenses collection failed: {str(e)}"
                collection_errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
            
            # Collect zoning changes with enhanced validation
            try:
                zoning_df = self.collect_zoning_changes(years=years)
                if zoning_df is not None and len(zoning_df) > 0:
                    data['zoning'] = zoning_df
                    logger.info(f"✅ Successfully collected {len(zoning_df)} zoning records")
                else:
                    logger.warning("⚠️ No zoning data collected - this is acceptable as zoning data is sparse")
                    # Don't treat zoning as critical failure since it's often sparse
            except Exception as e:
                error_msg = f"Zoning data collection failed: {str(e)}"
                collection_errors.append(error_msg)
                logger.warning(f"⚠️ {error_msg} - continuing without zoning data")
            
            # **IMPROVED: Require at least permits OR licenses data**
            if 'permits' not in data and 'licenses' not in data:
                logger.error("❌ CRITICAL FAILURE: No Chicago data could be collected")
                logger.error("❌ Cannot proceed without real Chicago permit or license data")
                raise ValueError("Failed to collect any real Chicago data from Data Portal")
            
            logger.info("Data collection from Chicago Data Portal completed successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting data from Chicago Data Portal: {str(e)}")
            logger.error(traceback.format_exc())
            
            # **IMPROVED: Don't fall back to sample data if real data is required**
            logger.error("❌ REAL DATA COLLECTION FAILED - Cannot use sample data as substitute")
            raise ValueError(f"Chicago Data Portal collection failed: {str(e)}")

    def collect_building_permits(self, limit=10000, years=None):
        """
        Collect building permit data from Chicago Data Portal.
        
        Args:
            limit (int): Maximum number of records to collect
            years (list, optional): List of years to filter by
            
        Returns:
            pd.DataFrame: Building permit data
        """
        try:
            # Set default years if not provided
            if years is None:
                current_year = datetime.now().year
                # **FIXED: Focus on years relevant for growth analysis (historical vs recent)**
                # Historical: 2018-2021, Recent: 2022-2024
                years = list(range(2018, current_year + 1))
            
            # Check cache first
            years_str = '_'.join(map(str, years)) if years else 'all'
            cache_name = f"chicago_permits_{years_str}_{limit}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # **IMPROVED: Better error handling for missing dependencies**
            try:
                from sodapy import Socrata
                client = Socrata("data.cityofchicago.org", self.api_token)
            except ImportError:
                logger.error("❌ CRITICAL: sodapy package not installed. Install with: pip install sodapy")
                raise ValueError("Missing required dependency: sodapy")
            except Exception as e:
                logger.error(f"❌ CRITICAL: Failed to initialize Socrata client: {str(e)}")
                raise ValueError(f"Socrata client initialization failed: {str(e)}")
            
            # **FIXED: Collect balanced data from multiple years**
            all_permits = []
            
            if years and len(years) > 1:
                # Collect data from each year to ensure historical coverage
                permits_per_year = min(2000, limit // len(years))  # Distribute limit across years
                
                for year in sorted(years, reverse=True):  # Start with recent years
                    try:
                        year_where = f"issue_date >= '{year}-01-01T00:00:00.000' AND issue_date < '{year + 1}-01-01T00:00:00.000'"
                        year_params = {
                            "$limit": permits_per_year,
                            "$order": "issue_date DESC",
                            "$where": year_where
                        }
                        
                        logger.info(f"Collecting permits for year {year} (limit: {permits_per_year})")
                        year_results = client.get("ydr8-5enu", **year_params)
                        
                        if year_results:
                            all_permits.extend(year_results)
                            logger.info(f"✅ Collected {len(year_results)} permits for {year}")
                        
                        # Stop if we have enough data
                        if len(all_permits) >= limit:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to collect permits for {year}: {str(e)}")
                        continue
                
                results = all_permits[:limit]  # Ensure we don't exceed limit
                
            else:
                # Single year or no year filter - use original approach
                query_params = {
                    "$limit": limit,
                    "$order": "issue_date DESC"
                }
                
                if years:
                    year = years[0]
                    where_clause = f"issue_date >= '{year}-01-01T00:00:00.000' AND issue_date < '{year + 1}-01-01T00:00:00.000'"
                    query_params["$where"] = where_clause
                
                results = client.get("ydr8-5enu", **query_params)
            
            if not results or len(results) == 0:
                logger.error("❌ CRITICAL: Chicago Data Portal returned no permit data")
                raise ValueError("No permit data available from Chicago Data Portal")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)
            logger.info(f"Retrieved {len(df)} raw permit records from Chicago Data Portal")
            
            # **IMPROVED: Enhanced data processing with better field mapping**
            df = self._process_permit_data(df)
            
            # **IMPROVED: Validate required fields are present**
            required_fields = self.datasets['permits']['required_fields']
            missing_fields = [f for f in required_fields if f not in df.columns]
            
            if missing_fields:
                logger.error(f"❌ CRITICAL: Required permit fields missing after processing: {missing_fields}")
                logger.error("❌ Available fields:", list(df.columns))
                raise ValueError(f"Required permit fields missing: {missing_fields}")
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            logger.info(f"✅ Successfully processed {len(df)} building permits with all required fields")
            return df
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Building permit collection completely failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Building permit collection failed: {str(e)}")
    
    def _process_permit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean building permit data."""
        try:
            logger.info(f"Processing permit data with {len(df)} records")
            
            # **IMPROVED: Better date processing**
            if 'issue_date' in df.columns:
                df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
                df['permit_year'] = df['issue_date'].dt.year
                
                # Ensure year column exists for compatibility
                if 'year' not in df.columns:
                    df['year'] = df['permit_year']
            elif 'year' not in df.columns:
                df['year'] = datetime.now().year
                logger.warning("No issue_date column found, using current year")
            
            # **IMPROVED: Enhanced ZIP code extraction from multiple sources**
            self._extract_zip_codes(df)
            
            # **IMPROVED: Better field mapping and validation**
            field_mapping = self.datasets['permits']['field_mapping']
            for standard_field, source_field in field_mapping.items():
                if source_field in df.columns and standard_field not in df.columns:
                    df[standard_field] = df[source_field]
                    logger.info(f"✅ Mapped {source_field} to {standard_field}")
            
            # **IMPROVED: Ensure critical fields exist**
            if 'permit_number' not in df.columns:
                if 'permit_' in df.columns:
                    df['permit_number'] = df['permit_']
                elif 'id' in df.columns:
                    df['permit_number'] = df['id']
                else:
                    logger.error("❌ CRITICAL: No permit number field found")
                    raise ValueError("No permit number field available")
            
            # **IMPROVED: Better cost field handling**
            if 'reported_cost' not in df.columns:
                for cost_field in ['estimated_cost', 'total_fee', 'permit_cost']:
                    if cost_field in df.columns:
                        df['reported_cost'] = pd.to_numeric(df[cost_field], errors='coerce').fillna(0)
                        logger.info(f"✅ Mapped {cost_field} to reported_cost")
                        break
                else:
                    logger.warning("⚠️ No cost field found, setting reported_cost to 0")
                    df['reported_cost'] = 0
            
            # **IMPROVED: Smart unit count extraction**
            if 'unit_count' not in df.columns:
                df['unit_count'] = self._extract_unit_count(df)
            
            # **IMPROVED: Better multifamily identification**
            if 'is_multifamily' not in df.columns:
                df['is_multifamily'] = self._identify_multifamily(df)
            
            # **IMPROVED: Clean permit numbers - remove null/empty values**
            if 'permit_number' in df.columns:
                df = df[df['permit_number'].notna() & (df['permit_number'] != '')]
            
            logger.info(f"✅ Successfully processed permit data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing permit data: {str(e)}")
            raise
    
    def _extract_zip_codes(self, df: pd.DataFrame):
        """Extract ZIP codes from multiple possible sources."""
        zip_sources = self.datasets['permits']['zip_fields']
        
        # Try to find ZIP code from any of the possible columns
        for zip_col in zip_sources:
            if zip_col in df.columns:
                # Extract 5-digit ZIP codes
                zip_pattern = df[zip_col].astype(str).str.extract(r'(\d{5})')
                zip_extracted = zip_pattern[0] if not zip_pattern.empty else pd.Series([], dtype=str)
                
                # Fill the standard zip_code column
                if 'zip_code' not in df.columns:
                    df['zip_code'] = zip_extracted
                else:
                    # Fill missing values in zip_code column
                    df['zip_code'] = df['zip_code'].fillna(zip_extracted)
        
        # If still no ZIP code column, try to extract from address-like fields
        if 'zip_code' not in df.columns or df['zip_code'].isna().all():
            address_fields = ['work_location', 'site_location', 'location', 'address']
            for addr_field in address_fields:
                if addr_field in df.columns:
                    zip_pattern = df[addr_field].astype(str).str.extract(r'IL\s+(\d{5})')
                    zip_extracted = zip_pattern[0] if not zip_pattern.empty else pd.Series([], dtype=str)
                    if 'zip_code' not in df.columns:
                        df['zip_code'] = zip_extracted
                    else:
                        df['zip_code'] = df['zip_code'].fillna(zip_extracted)
                    break
        
        # Clean and standardize ZIP codes
        if 'zip_code' in df.columns:
            zip_pattern = df['zip_code'].astype(str).str.extract(r'(\d{5})')
            df['zip_code'] = zip_pattern[0] if not zip_pattern.empty else pd.Series(['60601'] * len(df), dtype=str)
            df['zip_code'] = df['zip_code'].str.zfill(5)
            
            # Replace invalid ZIP codes with nearby Chicago ZIP
            invalid_zips = df['zip_code'].isna() | (df['zip_code'] == '') | (df['zip_code'] == '00000')
            df.loc[invalid_zips, 'zip_code'] = '60601'  # Default to downtown Chicago
        else:
            df['zip_code'] = '60601'  # Default if no ZIP found
            logger.warning("No ZIP code sources found, using default Chicago ZIP")
    
    def _extract_unit_count(self, df: pd.DataFrame) -> pd.Series:
        """Extract unit count from permit descriptions."""
        unit_count = pd.Series([1] * len(df), index=df.index)  # Default to 1
        
        # Try to extract from work_description
        if 'work_description' in df.columns:
            # Look for patterns like "5 units", "10 dwelling units", etc.
            unit_matches = df['work_description'].str.extract(r'(\d+)\s*(?:unit|dwelling|apartment|Unit|Dwelling|Apartment|UNIT|DWELLING|APARTMENT)')
            valid_units = pd.to_numeric(unit_matches[0], errors='coerce')
            unit_count = unit_count.where(valid_units.isna(), valid_units)
        
        # Try to extract from permit_type
        if 'permit_type' in df.columns:
            # Look for multifamily indicators
            multifamily_mask = df['permit_type'].str.contains('MULTI-FAMILY|APARTMENT', case=False, na=False)
            # Assign higher unit counts to multifamily
            unit_count.loc[multifamily_mask] = unit_count.loc[multifamily_mask].clip(lower=5)
        
        return unit_count
    
    def _identify_multifamily(self, df: pd.DataFrame) -> pd.Series:
        """Identify multifamily permits."""
        is_multifamily = pd.Series([False] * len(df), index=df.index)
        
        # Check permit_type
        if 'permit_type' in df.columns:
            multifamily_mask = df['permit_type'].str.contains(
                'MULTI-FAMILY|APARTMENT|CONDO', case=False, na=False
            )
            is_multifamily |= multifamily_mask
        
        # Check work_description
        if 'work_description' in df.columns:
            desc_mask = df['work_description'].str.contains(
                'apartment|multi.*family|condo|townhome', case=False, na=False
            )
            is_multifamily |= desc_mask
        
        # Check unit_count
        if 'unit_count' in df.columns:
            unit_mask = df['unit_count'] > 1
            is_multifamily |= unit_mask
        
        return is_multifamily

    def collect_business_licenses(self, limit=10000, years=None):
        """
        Collect business license data from Chicago Data Portal.
        
        Args:
            limit (int): Maximum number of records to collect
            years (list, optional): List of years to filter by
            
        Returns:
            pd.DataFrame: Business license data
        """
        try:
            # Set default years if not provided
            if years is None:
                current_year = datetime.now().year
                # **FIXED: Focus on years relevant for growth analysis (historical vs recent)**
                # Historical: 2018-2021, Recent: 2022-2024
                years = list(range(2018, current_year + 1))
            
            # Check cache first
            years_str = '_'.join(map(str, years)) if years else 'all'
            cache_name = f"chicago_licenses_{years_str}_{limit}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Initialize Socrata client
            try:
                from sodapy import Socrata
                client = Socrata("data.cityofchicago.org", self.api_token)
            except ImportError:
                logger.error("❌ CRITICAL: sodapy package not installed")
                raise ValueError("Missing required dependency: sodapy")
            
            # **IMPROVED: Better query construction**
            if years:
                date_filters = []
                for year in years:
                    date_filters.append(f"license_start_date >= '{year}-01-01T00:00:00.000' AND license_start_date < '{year + 1}-01-01T00:00:00.000'")
                year_filters = " OR ".join([f"({date_filter})" for date_filter in date_filters])
                where_clause = f"({year_filters})"
            else:
                where_clause = None
            
            query_params = {
                "$limit": limit,
                "$order": "license_start_date DESC"
            }
            
            # Collect data
            logger.info(f"Collecting business license data from Chicago Data Portal")
            try:
                if where_clause:
                    results = client.get("r5kz-chrr", where=where_clause, **query_params)
                else:
                    results = client.get("r5kz-chrr", **query_params)
            except Exception as e:
                logger.error(f"❌ CRITICAL: Business license data collection FAILED: {str(e)}")
                raise ValueError(f"Business license data collection failed: {str(e)}")
            
            if not results or len(results) == 0:
                logger.error("❌ CRITICAL: No business license data returned")
                raise ValueError("No business license data available")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)
            
            # **IMPROVED: Process business license data**
            df = self._process_license_data(df)
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            logger.info(f"✅ Successfully collected {len(df)} business licenses")
            return df
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Business license collection failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Business license collection failed: {str(e)}")
    
    def _process_license_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean business license data."""
        try:
            # Process license start date
            if 'license_start_date' in df.columns:
                df['license_start_date'] = pd.to_datetime(df['license_start_date'], errors='coerce')
                df['license_year'] = df['license_start_date'].dt.year
                
                if 'year' not in df.columns:
                    df['year'] = df['license_year']
            
            # Extract ZIP codes from address if not present
            if 'zip_code' not in df.columns and 'address' in df.columns:
                zip_pattern = df['address'].astype(str).str.extract(r'(\d{5})')
                df['zip_code'] = zip_pattern[0] if not zip_pattern.empty else pd.Series(['60601'] * len(df), dtype=str)
            
            # Clean ZIP codes
            if 'zip_code' in df.columns:
                zip_pattern = df['zip_code'].astype(str).str.extract(r'(\d{5})')
                df['zip_code'] = zip_pattern[0] if not zip_pattern.empty else pd.Series(['60601'] * len(df), dtype=str)
                df['zip_code'] = df['zip_code'].str.zfill(5)
                
                # Filter for Chicago ZIP codes or replace invalid ones
                invalid_zips = df['zip_code'].isna() | (df['zip_code'] == '') | (df['zip_code'] == '00000')
                df.loc[invalid_zips, 'zip_code'] = '60601'
            
            # Extract business type
            if 'business_type' not in df.columns and 'business_activity' in df.columns:
                df['business_type'] = df['business_activity']
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing license data: {str(e)}")
            raise

    def collect_zoning_changes(self, limit=5000, years=None):
        """
        Collect zoning change data from Chicago Data Portal.
        
        Args:
            limit (int): Maximum number of records to collect
            years (list, optional): List of years to filter by
            
        Returns:
            pd.DataFrame: Zoning change data
        """
        try:
            # Set default years if not provided
            if years is None:
                current_year = datetime.now().year
                # **FIXED: Focus on years relevant for growth analysis (historical vs recent)**
                # Historical: 2018-2021, Recent: 2022-2024
                years = list(range(2018, current_year + 1))
            
            # Check cache first
            years_str = '_'.join(map(str, years)) if years else 'all'
            cache_name = f"chicago_zoning_{years_str}_{limit}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Initialize Socrata client
            try:
                from sodapy import Socrata
                client = Socrata("data.cityofchicago.org", self.api_token)
            except ImportError:
                logger.error("❌ CRITICAL: sodapy package not installed")
                raise ValueError("Missing required dependency: sodapy")
            
            query_params = {
                "$limit": limit
            }
            
            # **IMPROVED: Try multiple zoning datasets with better error handling**
            zoning_datasets = self.datasets['zoning']['datasets']
            
            results = None
            successful_dataset = None
            
            for dataset in zoning_datasets:
                try:
                    logger.info(f"Trying zoning dataset: {dataset['name']} ({dataset['id']})")
                    temp_results = client.get(dataset['id'], **query_params)
                    
                    if temp_results and len(temp_results) > 0:
                        results = temp_results
                        successful_dataset = dataset
                        logger.info(f"✅ Successfully collected {len(results)} records from {dataset['name']}")
                        break
                    else:
                        logger.warning(f"Dataset {dataset['name']} returned 0 records")
                        
                except Exception as e:
                    logger.warning(f"Dataset {dataset['name']} failed: {str(e)}")
                    continue
            
            # **IMPROVED: Better handling when no zoning data is available**
            if not results or len(results) == 0:
                logger.warning("⚠️ No zoning data available from any dataset")
                logger.warning("⚠️ Zoning data is often sparse - this may be normal")
                # Return empty DataFrame instead of raising error
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)
            
            # **IMPROVED: Process zoning data**
            df = self._process_zoning_data(df)
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            logger.info(f"✅ Successfully collected {len(df)} zoning records")
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Zoning data collection failed: {str(e)}")
            logger.warning("⚠️ Returning empty DataFrame - zoning data is often unavailable")
            return pd.DataFrame()
    
    def _process_zoning_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean zoning data."""
        try:
            # Process zoning date fields
            date_fields = ['ordinance_date', 'intro_date', 'date_passed']
            for date_field in date_fields:
                if date_field in df.columns:
                    df[date_field] = pd.to_datetime(df[date_field], errors='coerce')
                    df['zoning_year'] = df[date_field].dt.year
                    if 'year' not in df.columns:
                        df['year'] = df['zoning_year']
                    break
            else:
                # Add current year if no date field
                df['zoning_year'] = datetime.now().year
                df['year'] = df['zoning_year']
            
            # Process zone class if available
            if 'zone_class' not in df.columns:
                for alt_field in ['zone_type', 'zoning_classification', 'zone']:
                    if alt_field in df.columns:
                        df['zone_class'] = df[alt_field]
                        break
                else:
                    df['zone_class'] = 'Unknown'
            
            # Add ZIP code for compatibility (zoning boundaries span multiple ZIP codes)
            if 'zip_code' not in df.columns:
                df['zip_code'] = '60601'  # Default Chicago ZIP for aggregation
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing zoning data: {str(e)}")
            raise

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

    def _generate_sample_data(self):
        """Generate sample data when API fails."""
        logger.warning("Generating sample data for Chicago Data Portal")
        
        # **IMPROVED: Don't generate sample data - raise error instead**
        logger.error("❌ CRITICAL: Sample data generation not allowed under real data policy")
        raise ValueError("Cannot generate sample data - real data collection required")
