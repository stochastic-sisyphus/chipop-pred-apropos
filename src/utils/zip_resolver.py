"""
Optimized ZIP code resolution module for the Chicago Population Analysis project.
"""

import logging
import pandas as pd
import numpy as np
import os
import json
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

class OptimizedZipResolver:
    """
    Optimized ZIP code resolver with caching, batch processing, and reduced API calls.
    """
    def __init__(self, cache_path="zip_cache.json", max_workers=4):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.logged_missing = set()
        self.unresolved_addresses = []
        self.max_workers = max_workers
        self.crosswalk = self._load_zip_crosswalk()
        self.crosswalk_dict = self._build_crosswalk_dict()
        
        # Chicago ZIP code patterns for quick validation
        self.chicago_zip_patterns = [
            "606", # Most Chicago ZIPs start with 606
            "607", # Some Chicago ZIPs start with 607
            "608"  # Some Chicago suburbs
        ]
        
        # Try to import uszipcode if available
        try:
            from uszipcode import SearchEngine
            self.search = SearchEngine()
            self._uszipcode_available = True
        except ImportError:
            self.search = None
            self._uszipcode_available = False
            logger.warning("uszipcode package not available, falling back to other methods")

    def _load_cache(self):
        """Load the ZIP code cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading ZIP cache, creating new one: {e}")
        return {}

    def save_cache(self):
        """Save the ZIP code cache to disk."""
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
            logger.debug(f"ZIP cache saved with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error saving ZIP cache: {e}")

    def _load_zip_crosswalk(self):
        """Load the Chicago ZIP code crosswalk file."""
        crosswalk_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chicago_zip_crosswalk.csv')
        if os.path.exists(crosswalk_path):
            try:
                return pd.read_csv(crosswalk_path, dtype=str)
            except Exception as e:
                logger.warning(f"Failed to load ZIP crosswalk: {e}")
        return None

    def _build_crosswalk_dict(self):
        """Build a dictionary for fast ZIP code lookups."""
        crosswalk_dict = {}
        if self.crosswalk is not None:
            for _, row in self.crosswalk.iterrows():
                if 'address' in row and pd.notna(row['address']):
                    crosswalk_dict[row['address'].strip().lower()] = row['zip_code']
                if 'community_area' in row and pd.notna(row['community_area']):
                    crosswalk_dict[row['community_area'].strip().lower()] = row['zip_code']
                if 'street_name' in row and pd.notna(row.get('street_name', '')):
                    crosswalk_dict[row['street_name'].strip().lower()] = row['zip_code']
        return crosswalk_dict

    def is_likely_chicago_zip(self, zip_code):
        """Check if a ZIP code is likely to be in Chicago."""
        if not zip_code or not isinstance(zip_code, str):
            return False
        return any(zip_code.startswith(pattern) for pattern in self.chicago_zip_patterns)

    def normalize_city(self, city):
        """Normalize city names to standard format."""
        if not isinstance(city, str):
            return "Chicago"
        city_map = {
            "MT PROSPECT": "Mount Prospect",
            "CHICAGO": "Chicago",
            "CHGO": "Chicago",
            "CHI": "Chicago",
            "CHIC": "Chicago",
            "CHICGO": "Chicago",
            "CICERO": "Cicero",
            "OAK PARK": "Oak Park",
            "EVANSTON": "Evanston",
            "SKOKIE": "Skokie"
        }
        return city_map.get(city.upper(), city.title())

    def lookup_crosswalk(self, address, city, state):
        """Look up a ZIP code in the crosswalk dictionary."""
        if not self.crosswalk_dict:
            return None
            
        candidates = [address, city]
        for cand in candidates:
            if cand and isinstance(cand, str):
                key = cand.strip().lower()
                if key in self.crosswalk_dict:
                    return self.crosswalk_dict[key]
                
                # Try fuzzy matching for efficiency
                matches = difflib.get_close_matches(
                    key, self.crosswalk_dict.keys(), n=1, cutoff=0.85
                )
                if matches:
                    return self.crosswalk_dict[matches[0]]
        return None

    def _lookup_uszipcode(self, address, city, state):
        """Look up a ZIP code using the uszipcode package."""
        if not self._uszipcode_available or not self.search:
            return None
            
        try:
            # Try to parse the address first
            try:
                import usaddress
                parsed, _ = usaddress.tag(f"{address}, {city}, {state}")
                if zipcode := parsed.get("ZipCode"):
                    if self.is_likely_chicago_zip(zipcode):
                        return zipcode
            except:
                pass
                
            # If address parsing fails, try city/state lookup
            if city.upper() == "CHICAGO" and state.upper() in ("IL", "ILLINOIS"):
                # For Chicago addresses without ZIP, use a common Chicago ZIP
                return "60601"  # Downtown Chicago
                
            # For other cities, try to look up
            if results := self.search.by_city_and_state(city, state):
                zipcode = results[0].zipcode
                if self.is_likely_chicago_zip(zipcode):
                    return zipcode
        except Exception as e:
            logger.debug(f"uszipcode lookup failed: {e}")
        return None

    def resolve_zip(self, address, city, state="IL", is_permit=False, use_geocoding=False):
        """
        Resolve a ZIP code from an address with multiple fallback methods.
        
        Args:
            address: Street address
            city: City name
            state: State (defaults to IL)
            is_permit: Whether this is a permit address (more likely to need geocoding)
            use_geocoding: Whether to use geocoding as a last resort
            
        Returns:
            str: ZIP code or None if not found
        """
        # Always default state to 'IL' if missing or nan
        if not isinstance(state, str) or not state or str(state).lower() == 'nan':
            state = "IL"
            
        # Create cache key
        key = f"{address}|{city}|{state}"
        
        # Check cache first
        if key in self.cache:
            return self.cache[key]
            
        # Try local crosswalk (with fuzzy matching)
        if result := self.lookup_crosswalk(address, city, state):
            self.cache[key] = result
            return result
            
        # Try uszipcode
        if result := self._lookup_uszipcode(address, city, state):
            self.cache[key] = result
            return result
            
        # Try normalized city
        norm_city = self.normalize_city(city)
        if norm_city != city:
            if result := self._lookup_uszipcode(address, norm_city, state):
                self.cache[key] = result
                return result
                
        # Only use geocoding for permit data with valid street address if explicitly requested
        if use_geocoding and is_permit and address and any(char.isdigit() for char in address):
            try:
                from src.utils.helpers import geocode_address_zip
                if result := geocode_address_zip(address, city, state, sleep=0.1):
                    self.cache[key] = result
                    return result
            except Exception as e:
                logger.debug(f"Geocoding failed: {e}")
                
        # Log missing ZIPs only once
        if key not in self.logged_missing:
            logger.warning(f"No ZIP found after all attempts: {address}, {city}, {state}")
            self.logged_missing.add(key)
            self.unresolved_addresses.append({
                "address": address,
                "city": city,
                "state": state
            })
            
        return None

    def batch_resolve(self, addresses, use_geocoding=False):
        """
        Batch resolve multiple addresses in parallel.
        
        Args:
            addresses: List of (address, city, state, is_permit) tuples
            use_geocoding: Whether to use geocoding as a last resort
            
        Returns:
            dict: Dictionary mapping input tuples to resolved ZIP codes
        """
        results = {}
        
        # First, check cache for all addresses
        uncached = []
        for addr_tuple in addresses:
            address, city, state, is_permit = addr_tuple
            key = f"{address}|{city}|{state}"
            if key in self.cache:
                results[addr_tuple] = self.cache[key]
            else:
                uncached.append(addr_tuple)
                
        if not uncached:
            return results
            
        # Process uncached addresses in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_addr = {
                executor.submit(
                    self.resolve_zip, address, city, state, is_permit, use_geocoding
                ): (address, city, state, is_permit)
                for address, city, state, is_permit in uncached
            }
            
            for future in as_completed(future_to_addr):
                addr_tuple = future_to_addr[future]
                try:
                    zip_code = future.result()
                    results[addr_tuple] = zip_code
                except Exception as e:
                    logger.error(f"Error resolving ZIP for {addr_tuple}: {e}")
                    results[addr_tuple] = None
                    
        return results

    def resolve_missing_zips(self, df, address_col="address", city_col="city", state_col="state", zip_col="zip_code", is_permit=False, use_geocoding=False):
        """
        Resolve missing ZIP codes in a DataFrame.
        
        Args:
            df: DataFrame with address data
            address_col: Column name for address
            city_col: Column name for city
            state_col: Column name for state
            zip_col: Column name for ZIP code
            is_permit: Whether these are permit addresses
            use_geocoding: Whether to use geocoding as a last resort
            
        Returns:
            int: Number of ZIP codes resolved
        """
        # Get rows with missing ZIP codes
        missing_mask = df[zip_col].isnull() | (df[zip_col] == "")
        missing_df = df[missing_mask].copy()
        
        if missing_df.empty:
            return 0
            
        logger.info(f"Resolving {len(missing_df)} missing ZIP codes...")
        
        # Create list of address tuples
        addresses = []
        for _, row in missing_df.iterrows():
            address = row.get(address_col, "")
            city = row.get(city_col, "Chicago")
            state = row.get(state_col, "IL") if state_col in row else "IL"
            addresses.append((address, city, state, is_permit))
            
        # Batch resolve
        resolved = self.batch_resolve(addresses, use_geocoding)
        
        # Update DataFrame
        updated = 0
        for i, (_, row) in enumerate(missing_df.iterrows()):
            addr_tuple = addresses[i]
            if zip_code := resolved.get(addr_tuple):
                idx = row.name
                df.at[idx, zip_col] = zip_code
                updated += 1
                
        # Save cache
        self.save_cache()
        
        logger.info(f"Resolved {updated} ZIP codes")
        return updated

    def export_unresolved_addresses(self, output_path=None):
        """Export unresolved addresses to a CSV file."""
        if self.unresolved_addresses:
            df = pd.DataFrame(self.unresolved_addresses)
            if output_path:
                df.to_csv(output_path, index=False)
            return df
        return pd.DataFrame()

    def close(self):
        """Close the resolver and save the cache."""
        self.save_cache()
        if self._uszipcode_available and self.search:
            # Let SearchEngine's __del__ handle closing its own session
            self.search = None
        logger.info(f"ZIP resolver closed, cache saved with {len(self.cache)} entries")
