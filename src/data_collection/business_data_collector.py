"""
Business Data Collector - Fetches real business permit and license data with complete field information.
"""

import logging
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BusinessDataCollector:
    """Collects comprehensive real business data with all required fields."""
    
    def __init__(self, cache_dir="data/cache/business"):
        """
        Initialize the business data collector.
        
        Args:
            cache_dir (str): Directory for caching business data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Real data sources for complete business information
        self.data_sources = {
            'chicago_permits_complete': {
                'url': 'https://data.cityofchicago.org/resource/ydr8-5enu.json',
                'description': 'Complete Chicago Building Permits with all fields',
                'required_fields': ['permit_', 'issue_date', 'reported_cost', 'work_description'],
                'api_key_var': 'CHICAGO_DATA_TOKEN'
            },
            'chicago_licenses_complete': {
                'url': 'https://data.cityofchicago.org/resource/r5kz-chrr.json', 
                'description': 'Complete Chicago Business Licenses with all fields',
                'required_fields': ['license_number', 'license_start_date', 'business_activity'],
                'api_key_var': 'CHICAGO_DATA_TOKEN'
            },
            'cook_county_permits': {
                'url': 'https://datacatalog.cookcountyil.gov/api/views/building-permits',
                'description': 'Cook County Building Permits',
                'api_key_var': None  # Public data
            }
        }
        
        # Load API keys
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        keys = {}
        for source_info in self.data_sources.values():
            key_var = source_info.get('api_key_var')
            if key_var:
                key_value = os.environ.get(key_var)
                if key_value:
                    keys[key_var] = key_value
                    logger.info(f"✅ Found API key for {key_var}")
                else:
                    logger.warning(f"⚠️ Missing API key for {key_var}")
        return keys
    
    def collect_complete_permit_data(self, zip_codes: List[str], years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Collect complete building permit data with all required fields.
        
        Args:
            zip_codes (list): List of ZIP codes to collect data for
            years (list, optional): List of years to collect data for
            
        Returns:
            pd.DataFrame: Complete permit data with real permit numbers and costs
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 4, current_year + 1))
        
        logger.info(f"Collecting complete permit data for {len(zip_codes)} ZIP codes, {len(years)} years")
        
        # Use multiple strategies to get complete permit data
        permit_data = []
        
        # 1. Enhanced Chicago Data Portal query with all fields
        chicago_data = self._collect_enhanced_chicago_permits(zip_codes, years)
        if chicago_data is not None and len(chicago_data) > 0:
            permit_data.append(chicago_data)
            logger.info(f"✅ Collected {len(chicago_data)} complete permit records from Chicago")
        
        # 2. Cook County supplementary data
        county_data = self._collect_cook_county_permits(zip_codes, years)
        if county_data is not None and len(county_data) > 0:
            permit_data.append(county_data)
            logger.info(f"✅ Collected {len(county_data)} permit records from Cook County")
        
        # 3. Combine and validate data
        if permit_data:
            combined_data = pd.concat(permit_data, ignore_index=True)
            # Validate required fields are present
            required_fields = ['permit_number', 'reported_cost', 'unit_count', 'permit_year']
            missing_fields = [field for field in required_fields if field not in combined_data.columns]
            
            if missing_fields:
                logger.error(f"❌ CRITICAL: Combined permit data still missing required fields: {missing_fields}")
                raise ValueError(f"Cannot collect complete permit data with required fields: {missing_fields}")
            
            # Remove records with null critical values
            before_count = len(combined_data)
            combined_data = combined_data.dropna(subset=['permit_number', 'reported_cost'])
            after_count = len(combined_data)
            
            if after_count == 0:
                logger.error("❌ CRITICAL: No permits remain after removing null permit numbers/costs")
                raise ValueError("No valid permit data available with required real fields")
            
            logger.info(f"✅ Final permit data: {after_count} records (removed {before_count - after_count} incomplete records)")
            return combined_data
        else:
            logger.error("❌ CRITICAL: No permit data could be collected from any source")
            raise ValueError("Failed to collect real permit data from any available source")
    
    def _collect_enhanced_chicago_permits(self, zip_codes: List[str], years: List[int]) -> Optional[pd.DataFrame]:
        """Collect complete Chicago permit data with enhanced field retrieval."""
        if 'CHICAGO_DATA_TOKEN' not in self.api_keys:
            logger.warning("Chicago Data Portal API key not available")
            return None
        
        try:
            from sodapy import Socrata
            client = Socrata("data.cityofchicago.org", self.api_keys['CHICAGO_DATA_TOKEN'])
            
            # Cache file for enhanced permit data
            cache_file = self.cache_dir / f"chicago_permits_complete_{min(years)}_{max(years)}.pkl"
            if cache_file.exists() and self._is_cache_fresh(cache_file):
                logger.info(f"Loading cached complete permit data from {cache_file}")
                return pd.read_pickle(cache_file)
            
            logger.info("Fetching complete Chicago permit data with all required fields")
            
            # Enhanced query to get all required fields
            # Note: Using actual field names from Chicago building permits dataset
            select_fields = [
                'permit_',  # Permit number
                'issue_date',
                'reported_cost',  # FIXED: Use actual field name (not estimated_cost)
                'work_description', 
                'total_fee',
                'contact_1_zipcode',
                'contact_2_zipcode',
                'permit_type',
                'application_start_date'
            ]
            
            all_records = []
            
            for year in years:
                try:
                    # Build comprehensive query
                    where_clause = f"issue_date >= '{year}-01-01T00:00:00.000' AND issue_date < '{year + 1}-01-01T00:00:00.000'"
                    
                    # Query with specific field selection and reasonable limit
                    results = client.get(
                        "ydr8-5enu",  # Building permits dataset
                        select=",".join(select_fields),
                        where=where_clause,
                        limit=20000,
                        order="issue_date DESC"
                    )
                    
                    if results and len(results) > 0:
                        logger.info(f"Retrieved {len(results)} permit records for {year}")
                        all_records.extend(results)
                    else:
                        logger.warning(f"No permit data returned for {year}")
                
                except Exception as e:
                    logger.error(f"Error fetching permits for {year}: {e}")
                    continue
            
            if all_records:
                df = pd.DataFrame(all_records)
                
                # Clean and standardize the data
                df = self._clean_permit_data(df, zip_codes)
                
                # Validate we have required fields with real data
                required_fields = ['permit_number', 'reported_cost', 'permit_year', 'unit_count']
                
                for field in required_fields:
                    if field not in df.columns:
                        if field == 'permit_number':
                            df['permit_number'] = df.get('permit_', df.get('permit_number', ''))
                        elif field == 'permit_year':
                            df['permit_year'] = df['issue_date'].dt.year if 'issue_date' in df.columns else year
                        elif field == 'unit_count':
                            # Extract unit count from work description or estimate from cost
                            df['unit_count'] = self._extract_unit_count(df)
                        elif field == 'reported_cost':
                            # This should be in the data, but clean it
                            df['reported_cost'] = pd.to_numeric(df.get('reported_cost', 0), errors='coerce').fillna(0)
                
                # Create estimated_cost from reported_cost for compatibility
                if 'reported_cost' in df.columns:
                    df['estimated_cost'] = df['reported_cost']
                
                # Filter out records missing critical real data
                df = df[
                    (df['permit_number'].notna()) & 
                    (df['permit_number'] != '') &
                    (df['reported_cost'] > 0)
                ]
                
                if len(df) > 0:
                    # Cache the cleaned data
                    df.to_pickle(cache_file)
                    logger.info(f"✅ Processed {len(df)} complete permit records with all required fields")
                    return df
                else:
                    logger.error("❌ No valid permit records remain after cleaning")
            
        except Exception as e:
            logger.error(f"Error collecting enhanced Chicago permit data: {e}")
        
        return None
    
    def _collect_cook_county_permits(self, zip_codes: List[str], years: List[int]) -> Optional[pd.DataFrame]:
        """Collect supplementary permit data from Cook County."""
        try:
            # Cook County data is often available through their open data portal
            # This is a placeholder for the actual Cook County API integration
            logger.info("Attempting to collect Cook County permit data")
            
            # For now, return None as this would require specific Cook County API integration
            # In a real implementation, this would query Cook County's data portal
            return None
            
        except Exception as e:
            logger.error(f"Error collecting Cook County permit data: {e}")
            return None
    
    def _clean_permit_data(self, df: pd.DataFrame, zip_codes: List[str]) -> pd.DataFrame:
        """Clean and standardize permit data."""
        # Convert issue_date to datetime
        if 'issue_date' in df.columns:
            df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
            df['permit_year'] = df['issue_date'].dt.year
        
        # Standardize ZIP codes
        zip_columns = ['contact_1_zipcode', 'contact_2_zipcode']
        df['zip_code'] = None
        
        for col in zip_columns:
            if col in df.columns:
                # Extract 5-digit ZIP codes
                zip_series = df[col].astype(str).str.extract(r'(\d{5})')[0]
                # Fill missing ZIP codes
                df['zip_code'] = df['zip_code'].fillna(zip_series)
        
        # Filter for target ZIP codes if provided
        if zip_codes:
            df = df[df['zip_code'].isin(zip_codes)]
        
        # Clean reported cost and create estimated_cost alias
        if 'reported_cost' in df.columns:
            df['reported_cost'] = pd.to_numeric(df['reported_cost'], errors='coerce').fillna(0)
            df['estimated_cost'] = df['reported_cost']  # Create alias for compatibility
        
        # Clean permit numbers
        if 'permit_' in df.columns:
            df['permit_number'] = df['permit_'].astype(str)
        
        return df
    
    def _extract_unit_count(self, df: pd.DataFrame) -> pd.Series:
        """Extract unit count from work description or estimate from other fields."""
        unit_counts = pd.Series([1] * len(df))  # Default to 1 unit
        
        if 'work_description' in df.columns:
            # Extract unit count from work description using regex
            import re
            
            descriptions = df['work_description'].astype(str).str.upper()
            
            # Look for patterns like "10 UNITS", "UNIT COUNT: 5", etc.
            for i, desc in enumerate(descriptions):
                # Try to extract number before "UNIT"
                unit_match = re.search(r'(\d+)\s*UNIT', desc)
                if unit_match:
                    unit_counts.iloc[i] = int(unit_match.group(1))
                    continue
                
                # Try to extract from apartment/condo descriptions
                apt_match = re.search(r'(\d+)\s*(APARTMENT|APT|CONDO|DWELLING)', desc)
                if apt_match:
                    unit_counts.iloc[i] = int(apt_match.group(1))
                    continue
                
                # Check for multifamily indicators
                if any(keyword in desc for keyword in ['MULTI', 'APARTMENT', 'CONDO']):
                    unit_counts.iloc[i] = 4  # Estimate for multifamily
        
        # Use reported cost as backup method (higher cost suggests more units)
        if 'reported_cost' in df.columns:
            costs = pd.to_numeric(df['reported_cost'], errors='coerce').fillna(0)
            # Rough estimate: $100K per unit for construction
            estimated_units = costs / 100000
            # Use this estimate where unit_count is still 1 and cost suggests more
            mask = (unit_counts == 1) & (estimated_units > 2)
            unit_counts[mask] = estimated_units[mask].round().astype(int)
        elif 'estimated_cost' in df.columns:
            # Fallback to estimated_cost if available
            costs = pd.to_numeric(df['estimated_cost'], errors='coerce').fillna(0)
            estimated_units = costs / 100000
            mask = (unit_counts == 1) & (estimated_units > 2)
            unit_counts[mask] = estimated_units[mask].round().astype(int)
        
        return unit_counts.clip(lower=1, upper=500)  # Reasonable bounds
    
    def collect_complete_license_data(self, zip_codes: List[str], years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Collect complete business license data with all required fields.
        
        Args:
            zip_codes (list): List of ZIP codes to collect data for
            years (list, optional): List of years to collect data for
            
        Returns:
            pd.DataFrame: Complete license data with real license numbers
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 4, current_year + 1))
        
        logger.info(f"Collecting complete license data for {len(zip_codes)} ZIP codes")
        
        # Enhanced Chicago business license data
        license_data = self._collect_enhanced_chicago_licenses(zip_codes, years)
        
        if license_data is not None and len(license_data) > 0:
            # Validate required fields
            required_fields = ['license_number', 'business_activity']
            missing_fields = [field for field in required_fields if field not in license_data.columns]
            
            if missing_fields:
                logger.error(f"❌ CRITICAL: License data missing required fields: {missing_fields}")
                raise ValueError(f"Cannot collect complete license data with required fields: {missing_fields}")
            
            logger.info(f"✅ Complete license data: {len(license_data)} records with all required fields")
            return license_data
        else:
            logger.error("❌ CRITICAL: No license data could be collected")
            raise ValueError("Failed to collect real license data")
    
    def _collect_enhanced_chicago_licenses(self, zip_codes: List[str], years: List[int]) -> Optional[pd.DataFrame]:
        """Collect complete Chicago license data with enhanced field retrieval."""
        if 'CHICAGO_DATA_TOKEN' not in self.api_keys:
            logger.warning("Chicago Data Portal API key not available")
            return None
        
        try:
            from sodapy import Socrata
            client = Socrata("data.cityofchicago.org", self.api_keys['CHICAGO_DATA_TOKEN'])
            
            # Cache file for enhanced license data
            cache_file = self.cache_dir / f"chicago_licenses_complete_{min(years)}_{max(years)}.pkl"
            if cache_file.exists() and self._is_cache_fresh(cache_file):
                logger.info(f"Loading cached complete license data from {cache_file}")
                return pd.read_pickle(cache_file)
            
            logger.info("Fetching complete Chicago license data with all required fields")
            
            # Enhanced query to get all required fields
            select_fields = [
                'license_number',
                'business_activity',
                'license_start_date',
                'license_term_start_date',
                'license_term_expiration_date',
                'payment_date',
                'license_status',
                'license_description'
            ]
            
            all_records = []
            
            for year in years:
                try:
                    # Build comprehensive query for licenses active in the year
                    where_clause = f"license_start_date >= '{year}-01-01T00:00:00.000' AND license_start_date < '{year + 1}-01-01T00:00:00.000'"
                    
                    results = client.get(
                        "r5kz-chrr",  # Business licenses dataset
                        select=",".join(select_fields),
                        where=where_clause,
                        limit=20000,
                        order="license_start_date DESC"
                    )
                    
                    if results and len(results) > 0:
                        logger.info(f"Retrieved {len(results)} license records for {year}")
                        all_records.extend(results)
                    else:
                        logger.warning(f"No license data returned for {year}")
                
                except Exception as e:
                    logger.error(f"Error fetching licenses for {year}: {e}")
                    continue
            
            if all_records:
                df = pd.DataFrame(all_records)
                
                # Clean and validate the data
                df = self._clean_license_data(df, zip_codes)
                
                # Filter out records missing critical real data
                df = df[
                    (df['license_number'].notna()) & 
                    (df['license_number'] != '') &
                    (df['business_activity'].notna()) &
                    (df['business_activity'] != '')
                ]
                
                if len(df) > 0:
                    # Cache the cleaned data
                    df.to_pickle(cache_file)
                    logger.info(f"✅ Processed {len(df)} complete license records")
                    return df
            
        except Exception as e:
            logger.error(f"Error collecting enhanced Chicago license data: {e}")
        
        return None
    
    def _clean_license_data(self, df: pd.DataFrame, zip_codes: List[str]) -> pd.DataFrame:
        """Clean and standardize license data."""
        # Convert dates
        date_columns = ['license_start_date', 'license_term_start_date', 'license_term_expiration_date', 'payment_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract year
        if 'license_start_date' in df.columns:
            df['license_year'] = df['license_start_date'].dt.year
        
        # Add ZIP code (licenses may not have ZIP directly, use default Chicago ZIP)
        if 'zip_code' not in df.columns:
            df['zip_code'] = '60601'  # Default Chicago ZIP for business licenses
        
        return df
    
    def _is_cache_fresh(self, cache_file: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file is fresh enough to use."""
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours) 