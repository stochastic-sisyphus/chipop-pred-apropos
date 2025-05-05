"""
Data collection module for Chicago population analysis.
"""



import contextlib
import logging
import pandas as pd
from census import Census
from fredapi import Fred
from sodapy import Socrata
from datetime import datetime
from typing import Dict, Optional, List, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import traceback
import requests
import re
from src.utils.helpers import geocode_address_zip
import json
import os
try:
    from uszipcode import SearchEngine
    _USZIPCODE_AVAILABLE = True
except ImportError:
    _USZIPCODE_AVAILABLE = False
import usaddress
import difflib

from src.config import settings

logger = logging.getLogger(__name__)

CHICAGO_ZIP_WHITELIST = {
    '60601',
    '60602',
    '60603',
    '60604',
    '60605',
    '60606',
    '60607',
    '60608',
    '60609',
    '60610',
    '60611',
    '60612',
    '60613',
    '60614',
    '60615',
    '60616',
    '60617',
    '60618',
    '60619',
    '60620',
    '60621',
    '60622',
    '60623',
    '60624',
    '60625',
    '60626',
    '60628',
    '60629',
    '60630',
    '60631',
    '60632',
    '60633',
    '60634',
    '60636',
    '60637',
    '60638',
    '60639',
    '60640',
    '60641',
    '60642',
    '60643',
    '60644',
    '60645',
    '60646',
    '60647',
    '60649',
    '60651',
    '60652',
    '60653',
    '60654',
    '60655',
    '60656',
    '60657',
    '60659',
    '60660',
    '60661',
    '60664',
    '60666',
    '60668',
    '60669',
    '60670',
    '60673',
    '60674',
    '60675',
    '60677',
    '60678',
    '60680',
    '60681',
    '60682',
    '60684',
    '60685',
    '60686',
    '60687',
    '60688',
    '60689',
    '60690',
    '60691',
    '60693',
    '60694',
    '60695',
    '60696',
    '60697',
    '60699',
    '60701',
    '60706',
    '60707',
    '60803',
    '60804',
    '60805',
    '60827',
}

CHICAGO_ZIP_CROSSWALK_PATH = os.path.join(os.path.dirname(__file__), 'chicago_zip_crosswalk.csv')

class ZipResolver:
    """
    Resolves ZIP codes for addresses using cache, uszipcode, normalization, local crosswalk (with fuzzy matching), and geocoding.
    """
    def __init__(self, cache_path="local_zip_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.logged_missing = set()
        self.search = None
        self.crosswalk = self._load_zip_crosswalk()
        self.crosswalk_dict = self._build_crosswalk_dict()
        if _USZIPCODE_AVAILABLE:
            try:
                self.search = SearchEngine()
            except Exception as e:
                logging.warning(f"uszipcode SearchEngine could not be initialized: {e}")
                self.search = None
        self.unresolved_addresses = []

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def normalize_city(self, city):
        # Add more normalization rules as needed
        if not isinstance(city, str):
            return "Chicago"
        city_map = {"MT PROSPECT": "Mt. Prospect", "CHICAGO": "Chicago"}
        return city_map.get(city.upper(), city.title())

    def _load_zip_crosswalk(self):
        # Load a local CSV mapping address, community area, or street name to ZIP
        if os.path.exists(CHICAGO_ZIP_CROSSWALK_PATH):
            try:
                return pd.read_csv(CHICAGO_ZIP_CROSSWALK_PATH, dtype=str)
            except Exception as e:
                logging.warning(f"Failed to load ZIP crosswalk: {e}")
        return None

    def _build_crosswalk_dict(self):
        # Build a dict for fast lookup and fuzzy matching
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

    def lookup_crosswalk(self, address, city, state):
        # Try exact and fuzzy match by address, community area, or street name
        if self.crosswalk_dict:
            candidates = [address, city]
            for cand in candidates:
                if cand and isinstance(cand, str):
                    key = cand.strip().lower()
                    if key in self.crosswalk_dict:
                        return self.crosswalk_dict[key]
                    if matches := difflib.get_close_matches(
                        key, self.crosswalk_dict.keys(), n=1, cutoff=0.85
                    ):
                        return self.crosswalk_dict[matches[0]]
        return None

    def resolve_zip(self, address, city, state="IL", is_permit=False):
        # Always default state to 'IL' if missing or nan
        if not isinstance(state, str) or not state or str(state).lower() == 'nan':
            state = "IL"
        key = f"{address}|{city}|{state}"
        if key in self.cache:
            return self.cache[key]
        # 1. Try local crosswalk (with fuzzy matching)
        if result := self.lookup_crosswalk(address, city, state):
            self.cache[key] = result
            return result
        # 2. Try uszipcode
        if result := self._lookup_uszipcode(address, city, state):
            self.cache[key] = result
            return result
        # 3. Try normalized city
        norm_city = self.normalize_city(city)
        if norm_city != city:
            if result := self._lookup_uszipcode(address, norm_city, state):
                self.cache[key] = result
                return result
        # 4. Only use Nominatim for permit data with valid street address
        if is_permit and address and any(char.isdigit() for char in address):
            try:
                if result := geocode_address_zip(address, city, state, sleep=0.2):
                    self.cache[key] = result
                    return result
            except Exception as e:
                logging.warning(f"Nominatim geocode failed for: {address}, {city}, {state}: {e}")
        # For business names or ambiguous addresses, do NOT geocode—just skip
        if key not in self.logged_missing:
            logging.warning(f"No ZIP found after all attempts: {address}, {city}, {state}")
            self.logged_missing.add(key)
            self.unresolved_addresses.append({
                "address": address,
                "city": city,
                "state": state
            })
        return None

    def _lookup_uszipcode(self, address, city, state):
        if not self.search:
            return None
        with contextlib.suppress(Exception):
            parsed, _ = usaddress.tag(f"{address}, {city}, {state}")
            if zipcode := parsed.get("ZipCode"):
                return zipcode
            if results := self.search.by_city_and_state(city, state):
                return results[0].zipcode
        return None

    def batch_geocode(self, unresolved):
        # Batch geocode unresolved addresses offline, always default state to 'IL'
        for key in unresolved:
            if key not in self.cache:
                address, city, state = key.split("|")
                if not isinstance(state, str) or not state or str(state).lower() == 'nan':
                    state = "IL"
                if result := geocode_address_zip(address, city, state):
                    self.cache[key] = result

    def resolve_missing_zips(self, df, address_col="address", city_col="city", state_col="state", zip_col="zip_code"):
        updated = 0
        for idx, row in df[df[zip_col].isnull() | (df[zip_col] == "")].iterrows():
            address = row[address_col]
            city = row[city_col]
            state = row[state_col] if state_col in row else "IL"
            if zip_code := self.resolve_zip(address, city, state):
                df.at[idx, zip_col] = zip_code
                updated += 1
        self.save_cache()
        return updated

    def export_unresolved_addresses(self, output_path: str = None):
        if self.unresolved_addresses:
            df = pd.DataFrame(self.unresolved_addresses)
            if output_path:
                df.to_csv(output_path, index=False)
            return df
        return pd.DataFrame()

    def update_crosswalk_from_authoritative(self, authoritative_csv_path: str):
        # Utility to update the crosswalk from a new authoritative source
        try:
            new_crosswalk = pd.read_csv(authoritative_csv_path, dtype=str)
            self.crosswalk = new_crosswalk
            self.crosswalk_dict = self._build_crosswalk_dict()
            logging.info(f"Updated ZIP crosswalk from {authoritative_csv_path}")
        except Exception as e:
            logging.error(f"Failed to update ZIP crosswalk: {e}")

    def batch_resolve_permit_zips(self, df, address_col="address", city_col="city", state_col="state", zip_col="zip_code"):
        # Only attempt batch geocoding for rows with valid street addresses
        unresolved = df[df[zip_col].isnull() | (df[zip_col] == "")].copy()
        for idx, row in unresolved.iterrows():
            address = row[address_col]
            city = row[city_col]
            state = row[state_col] if state_col in row else "IL"
            if address and any(char.isdigit() for char in str(address)):
                if zip_code := self.resolve_zip(
                    address, city, state, is_permit=True
                ):
                    df.at[idx, zip_col] = zip_code
        self.save_cache()
        return df


class DataCollector:
    """Handles data collection from various sources including FRED, Census, and Chicago Data Portal."""

    def __init__(self):
        """Initialize data collector with API clients."""
        self.census = Census(settings.CENSUS_API_KEY)
        self.fred = Fred(api_key=settings.FRED_API_KEY)
        self.socrata = Socrata("data.cityofchicago.org", settings.CHICAGO_DATA_TOKEN)
        self.raw_dir = settings.RAW_DATA_DIR
        self.raw_dir.mkdir(exist_ok=True)

        # Cache for available Census years
        self._available_census_years = None
        self._logged_fallback_zips: Set[str] = set()  # Track fallback ZIPs logged this session
        self._fallback_zip_records: list = []  # For audit/export
        self.zip_resolver = ZipResolver()

    def __del__(self):
        # Save cache on exit
        if hasattr(self, "zip_resolver"):
            self.zip_resolver.save_cache()

    def map_permit_type(self, permit_type: str) -> str:
        """
        Maps raw permit_type strings to standardized categories.
        """
        mapping = {
            "PERMIT – EXPRESS PERMIT PROGRAM": "EXPRESS_PERMIT",
            "PERMIT - FOR EXTENSION OF PMT": "EXTENSION_PERMIT",
        }
        return mapping.get(permit_type, "OTHER")

    def _get_available_census_years(self) -> Set[int]:
        """
        Get available years from Census API with caching.

        Returns:
            Set[int]: Set of available years
        """
        if self._available_census_years is not None:
            return self._available_census_years

        available_years = set()
        test_zip = str(settings.CHICAGO_ZIP_CODES[0])

        for year in range(2015, datetime.now().year):
            try:
                if self.census.acs5.state_zipcode(
                    ["B01003_001E"],  # Total population
                    state_fips="17",  # Illinois
                    zcta=test_zip,
                    year=year,
                ):
                    available_years.add(year)
                    logger.info(f"Census year {year} is available")
            except Exception as e:
                logger.debug(f"Census year {year} not available: {str(e)}")
                continue

        self._available_census_years = available_years
        return available_years

    def _fetch_census_zip_year(
        self, zip_code: str, year: int, variables: List[str], max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Fetch Census data for a specific ZIP code and year with retries.

        Args:
            zip_code (str): ZIP code to fetch data for
            year (int): Year to fetch data for
            variables (List[str]): Census variables to fetch
            max_retries (int): Maximum number of retry attempts

        Returns:
            Optional[Dict]: Census data row or None if failed
        """
        zip_code = zip_code.zfill(5)

        for attempt in range(max_retries):
            try:
                if result := self.census.acs5.state_zipcode(
                    variables,
                    state_fips="17",
                    zcta=zip_code,
                ):
                    row = result[0]
                    row["year"] = year
                    row["zip_code"] = zip_code  # Use consistent column name
                    logger.debug(f"Fetched data for ZIP {zip_code}, year {year}")
                    return row

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Back off before retry
                else:
                    logger.error(f"Failed to fetch ZIP {zip_code}, year {year}: {str(e)}")

        return None

    def collect_census_data(self) -> Optional[pd.DataFrame]:
        """
        Collect Census data for all available years.

        Returns:
            Optional[pd.DataFrame]: Census data or None if failed
        """
        try:
            logger.info("Starting Census data collection...")

            # Get available years
            available_years = self._get_available_census_years()
            if not available_years:
                logger.error("No Census years available")
                return None

            logger.info(f"Collecting Census data for years: {sorted(available_years)}")

            # Census variables to collect
            variables = [
                "B01003_001E",  # Total population
                "B19013_001E",  # Median household income
                "B25077_001E",  # Median home value
                "B23025_002E",  # Labor force
                "B25001_001E",  # Housing units
                "B25003_001E",  # Occupied housing units
                "B25003_003E",  # Vacant housing units
            ]

            # Create tasks for parallel execution
            tasks = [
                (str(zip_code), year)
                for year in available_years
                for zip_code in settings.CHICAGO_ZIP_CODES
            ]

            # Collect data in parallel
            results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(self._fetch_census_zip_year, z, y, variables): (z, y)
                    for z, y in tasks
                }

                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Fetching Census data"
                ):
                    if result := future.result():
                        results.append(result)

            if not results:
                logger.error("No Census data collected")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(results)
            logger.info(f"Raw Census data shape: {df.shape}")

            # Rename columns to be more descriptive
            column_map = {
                "B01003_001E": "total_population",
                "B19013_001E": "median_household_income",
                "B25077_001E": "median_home_value",
                "B23025_002E": "labor_force",
                "B25001_001E": "total_housing_units",
                "B25003_001E": "occupied_housing_units",
                "B25003_003E": "vacant_housing_units",
                "zip code tabulation area": "zip_code",  # Map ZCTA to zip_code
            }
            df = df.rename(columns=column_map)

            # Debug log current columns
            logger.info(f"Current columns: {df.columns.tolist()}")

            if dup_cols := df.columns[df.columns.duplicated()].tolist():
                logger.warning(f"Found duplicate columns: {dup_cols}")
                # Keep first occurrence of each column
                df = df.loc[:, ~df.columns.duplicated()]
                logger.info(f"Columns after deduplication: {df.columns.tolist()}")

            # Validate zip_code column exists and is a Series
            if "zip_code" not in df.columns:
                logger.error("No 'zip_code' column found after renaming")
                return None

            # Debug log ZIP code column info
            logger.info(f"ZIP code column type: {type(df['zip_code'])}")
            logger.info(f"ZIP code sample before processing:\n{df['zip_code'].head()}")

            # Handle ZIP code formatting
            try:
                # Convert to string and pad with zeros
                df["zip_code"] = df["zip_code"].astype(str).str.strip()
                df["zip_code"] = df["zip_code"].str.zfill(5)

                # Validate ZIP codes are 5 digits
                invalid_zips = df[~df["zip_code"].str.match(r"^\d{5}$")]
                if not invalid_zips.empty:
                    logger.warning(f"Found {len(invalid_zips)} invalid ZIP codes")
                    logger.warning(
                        f"Invalid ZIP codes: {invalid_zips['zip_code'].unique().tolist()}"
                    )

                # Debug log after processing
                logger.info(f"ZIP code sample after processing:\n{df['zip_code'].head()}")

            except Exception as e:
                logger.error(f"Error formatting ZIP codes: {str(e)}")
                logger.error(f"ZIP code column type: {type(df['zip_code'])}")
                logger.error(f"ZIP code sample:\n{df['zip_code'].head()}")
                return None

            # Create template dataframe with all ZIP code and year combinations
            all_combinations = pd.DataFrame(
                [
                    (zip_code, year)
                    for zip_code in settings.CHICAGO_ZIP_CODES
                    for year in available_years
                ],
                columns=["zip_code", "year"],
            )

            # Prepare columns for merge
            required_columns = ["zip_code", "year"] + [
                col for col in column_map.values() if col != "zip_code"
            ]
            df_merge = df[required_columns].copy()

            # Ensure no duplicate columns before merge
            df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]
            all_combinations = all_combinations.loc[:, ~all_combinations.columns.duplicated()]

            # Debug log merge info
            logger.info(f"Merge left columns: {all_combinations.columns.tolist()}")
            logger.info(f"Merge right columns: {df_merge.columns.tolist()}")

            try:
                # Merge with template
                df = pd.merge(all_combinations, df_merge, on=["zip_code", "year"], how="left")
                logger.info(f"Merged data shape: {df.shape}")

                # Debug log final columns
                logger.info(f"Final columns before filling NAs: {df.columns.tolist()}")

                # Fill missing values
                numeric_cols = [
                    col for col in df.columns if col not in ["zip_code", "year", "state"]
                ]

                # Convert all numeric columns at once
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Fill missing values using groupby means
                df_filled = df.copy()
                for col in numeric_cols:
                    try:
                        # Calculate means by ZIP code
                        zip_means = df.groupby("zip_code")[col].transform("mean")

                        # Fill NAs without using inplace
                        df_filled[col] = df[col].fillna(zip_means)

                        # If any NAs remain, fill with overall mean
                        if df_filled[col].isna().any():
                            overall_mean = df[col].mean()
                            df_filled[col] = df_filled[col].fillna(overall_mean)

                        logger.info(f"Filled NAs in {col} using ZIP code means and overall mean")
                    except Exception as e:
                        logger.error(f"Error filling NAs for column {col}: {str(e)}")
                        logger.error(
                            f"Column info - dtype: {df[col].dtype}, unique values: {df[col].nunique()}"
                        )
                        return None

                # Replace original with filled version
                df = df_filled

                # Final validation
                if df.isnull().any().any():
                    null_cols = df.columns[df.isnull().any()].tolist()
                    logger.warning(f"Found null values in columns: {null_cols}")
                    logger.warning(f"Null counts:\n{df[null_cols].isnull().sum()}")

                    # Drop rows with any remaining nulls as a last resort
                    df = df.dropna()
                    logger.warning(f"Dropped rows with null values. New shape: {df.shape}")

                # Ensure ZIP codes are properly formatted
                df["zip_code"] = df["zip_code"].astype(str).str.strip().str.zfill(5)

                # Save processed data
                processed_path = settings.PROCESSED_DATA_DIR / "census_processed.csv"
                df.to_csv(processed_path, index=False)
                logger.info(f"Processed Census data saved to {processed_path}")

                return df

            except Exception as e:
                return self._extracted_from_collect_census_data_187("Error during merge: ", e)
        except Exception as e:
            return self._extracted_from_collect_census_data_187("Error in collect_census_data: ", e)

    # TODO Rename this here and in `collect_census_data`
    def _extracted_from_collect_census_data_187(self, arg0, e):
        logger.error(f"{arg0}{str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    def collect_permit_data(self):
        """Collect building permit data from Chicago Data Portal."""
        try:
            logger.info("Collecting building permit data...")
            results = self.socrata.get(
                "ydr8-5enu",
                limit=100000,
                where="issue_date IS NOT NULL",
                select="""
                    permit_, permit_type, reported_cost, issue_date, street_number, street_direction, street_name, contact_1_city, contact_1_state, work_description, community_area, ward, contact_1_zipcode
                """.replace("\n", "").replace(" ", ""),
            )
            df = pd.DataFrame.from_records(results)
            # Only resolve ZIPs for Chicago addresses
            def resolve_zip_chicago(row):
                city = row.get("contact_1_city", "Chicago")
                if not isinstance(city, str):
                    city = "Chicago"
                if city.strip().lower() != "chicago":
                    return None
                address = f"{row.get('street_number', '')} {row.get('street_direction', '')} {row.get('street_name', '')}".strip()
                return self.zip_resolver.resolve_zip(address, city, row.get("contact_1_state", "IL"), is_permit=True)
            df["zip_code"] = df.apply(resolve_zip_chicago, axis=1)
            # Batch resolve any remaining with valid street addresses
            df = self.zip_resolver.batch_resolve_permit_zips(df, address_col="address", city_col="contact_1_city", state_col="contact_1_state", zip_col="zip_code")
            # Export unresolved addresses for review
            self.zip_resolver.export_unresolved_addresses(str(settings.PROCESSED_DATA_DIR / "unresolved_addresses.csv"))
            df = df[df["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
            # Extract retail permits
            retail_keywords = ["retail", "store", "shop", "restaurant", "commercial", "business", "mall", "market", "sales"]
            df["is_retail"] = df["work_description"].str.lower().str.contains("|".join(retail_keywords), na=False)
            retail_permits = df[df["is_retail"]].copy()
            retail_permits["retail_permits"] = 1
            retail_permits["retail_construction_cost"] = pd.to_numeric(retail_permits["reported_cost"], errors="coerce")
            # Aggregate by ZIP and year
            retail_summary = retail_permits.groupby(["zip_code", "issue_date"]).agg({
                "retail_permits": "sum",
                "retail_construction_cost": "sum"
            }).reset_index()
            # Save to processed
            retail_summary.to_csv(settings.PROCESSED_DATA_DIR / "permits_processed.csv", index=False)
            logger.info(f"Saved retail permits summary to {settings.PROCESSED_DATA_DIR / 'permits_processed.csv'}")
            return retail_summary
        except Exception as e:
            logger.error(f"Error collecting permit data: {e}")
            return None

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
        rate_based_metrics = ["unemployment_rate", "homeownership_rate"]
        failed_series = []

        # First collect all available data
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
                    series = series.resample("YE").mean()
                else:
                    series = series.resample("YE").last()

                series.index = series.index.year
                economic_data[indicator_name] = series

                logger.info(f"Successfully collected {indicator_name} from {series_id}")

            except Exception as e:
                logger.error(
                    f"Error collecting FRED series {series_id} ({indicator_name}): {str(e)}"
                )
                failed_series.append(series_id)

        if failed_series:
            logger.warning(f"Failed FRED series: {', '.join(failed_series)}")

        if not economic_data:
            logger.error("No economic indicators collected")
            return None

        # Create DataFrame with all years
        df = pd.DataFrame(index=settings.DEFAULT_TRAIN_YEARS)

        # Add each indicator and handle missing years
        for indicator, series in economic_data.items():
            # Reindex series to match desired years
            series = series.reindex(df.index)
            # Forward fill, then backward fill any remaining NAs
            series = series.ffill().bfill()
            df[indicator] = series

        df.index.name = "year"
        df = df.reset_index()

        output_path = self.raw_dir / "economic_indicators.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(economic_data)} economic indicators to {output_path}")

        return df

    def collect_zoning_data(self):
        """Collect zoning data from Chicago Data Portal."""
        try:
            logger.info("Starting zoning data collection...")

            # First get a sample record to check available columns
            try:
                sample = self.socrata.get("dj47-wfun", limit=1)
                if not sample:
                    logger.error("Could not get sample zoning record")
                    return None

                # Log available columns
                columns = list(sample[0].keys())
                logger.info(f"Available zoning columns: {columns}")

                # Build query based on available columns
                query = """
                    SELECT 
                        zone_class AS zoning_classification,
                        zone_type AS zone_category,
                        COUNT(*) AS total_parcels,
                        AVG(shape_area) AS avg_lot_size,
                        SUM(shape_area) AS total_area
                    GROUP BY zoning_classification, zone_category
                    LIMIT 1000000
                """
                zoning_data = self.socrata.get("dj47-wfun", query=query)
                df = pd.DataFrame.from_records(zoning_data)

                if len(df) > 0:
                    logger.info(f"Successfully collected zoning data: {len(df)} records")

                    # Attempt spatial join to ZIPs using centroids and ZIP shapefile if available
                    try:
                        import geopandas as gpd
                        # Assume zoning polygons have geometry column 'the_geom' or similar
                        if 'the_geom' in df.columns:
                            zoning_gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['the_geom']))
                            # Load ZIP shapefile (must be available in data directory)
                            zip_shapefile = self.raw_dir / "chicago_zip_codes.shp"
                            if zip_shapefile.exists():
                                zip_gdf = gpd.read_file(str(zip_shapefile))
                                zoning_gdf = gpd.sjoin(zoning_gdf, zip_gdf[['geometry', 'ZIP']], how='left', predicate='intersects')
                                df['zip_code'] = zoning_gdf['ZIP'].astype(str).str.zfill(5)
                                logger.info("Spatially joined zoning polygons to ZIP codes.")
                            else:
                                logger.warning("ZIP shapefile not found; cannot spatially join. Setting zip_code to 'unknown'.")
                                df['zip_code'] = 'unknown'
                        else:
                            logger.warning("No geometry column in zoning data; cannot spatially join. Setting zip_code to 'unknown'.")
                            df['zip_code'] = 'unknown'
                    except Exception as e:
                        logger.error(f"Spatial join to ZIP codes failed: {e}")
                        df['zip_code'] = 'unknown'

                    # Convert area to square feet
                    df["avg_lot_size"] = pd.to_numeric(df["avg_lot_size"], errors="coerce")
                    df["total_area"] = pd.to_numeric(df["total_area"], errors="coerce")
                    df["total_parcels"] = pd.to_numeric(df["total_parcels"], errors="coerce")

                    # Log summary statistics
                    logger.info(f"Zoning classifications: {df['zoning_classification'].nunique()}")
                    logger.info(f"Zone categories: {df['zone_category'].nunique()}")
                    logger.info(f"Total parcels: {int(df['total_parcels'].sum())}")
                    logger.info(f"Average lot size: {df['avg_lot_size'].mean():.0f} sq ft")

                    # Save raw data
                    raw_path = self.raw_dir / "zoning_data.csv"
                    df.to_csv(raw_path, index=False)
                    logger.info(f"Saved raw zoning data to {raw_path}")

                    return df
                else:
                    logger.warning("No zoning data records returned")
                    return None

            except Exception as e:
                return self._extracted_from_collect_all_data_100(
                    "Failed to get zoning data: ", e, None
                )
        except Exception as e:
            return self._extracted_from_collect_all_data_100(
                "Error collecting zoning data: ", e, None
            )

    def get_property_transactions(self) -> Optional[pd.DataFrame]:
        """Get property transaction data from Chicago Data Portal."""
        try:
            # Verify dataset ID exists
            if not settings.PROPERTY_TRANSACTIONS_DATASET:
                logger.warning("Property transactions dataset ID not configured")
                return None

            results = self.socrata.get(
                settings.PROPERTY_TRANSACTIONS_DATASET,
                select="""
                    zip_code,
                    sale_price,
                    property_type,
                    year_built,
                    total_value
                """,
                limit=1000000,
            )

            if not results:
                logger.warning("No property transaction data returned")
                return None

            df = pd.DataFrame.from_records(results)

            # Save raw data
            df.to_csv(settings.PROPERTY_DATA_PATH, index=False)
            logger.info(f"Property transaction data saved to {settings.PROPERTY_DATA_PATH}")
            return df

        except Exception as e:
            logger.error(f"Error getting property transaction data: {str(e)}")
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
                    """.replace("\n", "").replace(" ", ""),
                    where="expiration_date > '2025-04-25T00:00:00.000' AND UPPER(license_status) = 'AAI'",
                    order="license_start_date DESC",
                )
                df = pd.DataFrame.from_records(results)
                df["license_start_date"] = pd.to_datetime(df["license_start_date"], errors="coerce")
                df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
                df["application_created_date"] = pd.to_datetime(
                    df["application_created_date"], errors="coerce"
                )
                # Robust ZIP extraction using ZipResolver, but do NOT geocode business names
                def resolve_zip_business(row):
                    # Only use crosswalk/uszipcode, never Nominatim for business names
                    return self.zip_resolver.lookup_crosswalk(row.get("doing_business_as_name", ""), row.get("legal_name", ""), "IL") or \
                           self.zip_resolver._lookup_uszipcode(row.get("doing_business_as_name", ""), row.get("legal_name", ""), "IL")
                df["zip_code"] = df.apply(resolve_zip_business, axis=1)
                # Validate ZIPs
                df = df[df["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
                return self._save_data_to_csv("business_licenses.csv", df, "Business license data")
            except Exception as e:
                if attempt == 0:  # First attempt failed
                    logger.warning(
                        f"First attempt to collect business license data failed: {str(e)}. Retrying..."
                    )
                    continue
                else:  # Second attempt failed
                    logger.error(f"Error collecting business license data after retry: {str(e)}")
                    return None

    def collect_multifamily_permits(self) -> pd.DataFrame:
        """
        Collect and filter multifamily (and mixed-use with multifamily) permits.
        Returns DataFrame with ZIP, year, permit_id, reported_cost, unit_count, expected_completion_date, development_status, zoning_status.
        """
        logger.info("Collecting multifamily permits...")
        permits = self.collect_permit_data()
        if permits is None or permits.empty:
            logger.error("No permit data available for multifamily extraction.")
            return pd.DataFrame()
        # Filter for multifamily and mixed-use with multifamily
        multifam_keywords = ["multi-family", "multifamily", "apartment", "condo", "mixed-use", "residential"]
        is_multifam = permits["work_description"].str.lower().str.contains("|".join(multifam_keywords), na=False)
        multifam_permits = permits[is_multifam].copy()
        # Add unit_count if available
        if "units" in multifam_permits.columns:
            multifam_permits["unit_count"] = pd.to_numeric(multifam_permits["units"], errors="coerce")
        else:
            multifam_permits["unit_count"] = None
        # Add expected_completion_date
        if "completion_date" in multifam_permits.columns:
            multifam_permits["expected_completion_date"] = pd.to_datetime(multifam_permits["completion_date"], errors="coerce")
        else:
            multifam_permits["expected_completion_date"] = pd.to_datetime(multifam_permits["issue_date"]) + pd.DateOffset(years=2)
        # Add development_status
        if "status" in multifam_permits.columns:
            multifam_permits["development_status"] = multifam_permits["status"].fillna("Planned")
        else:
            multifam_permits["development_status"] = "Planned"
        # Add zoning_status (default to 'Compliant', can be updated in processing)
        multifam_permits["zoning_status"] = "Compliant"
        return self._extracted_from_collect_hud_usps_vacancy_33(
            multifam_permits,
            "multifamily_permits.csv",
            'Saved multifamily permits to ',
            'multifamily_permits.csv',
        )

    def collect_business_licenses_retail(self) -> pd.DataFrame:
        """
        Collect business licenses, filter NAICS 44-45 (retail), count per ZIP.
        """
        logger.info("Collecting retail business licenses...")
        df = self.get_business_licenses()
        if df is None or df.empty:
            logger.error("No business license data available.")
            return pd.DataFrame()
        # Filter NAICS 44-45 (retail trade)
        if "naics_code" in df.columns:
            is_retail = df["naics_code"].astype(str).str.startswith(("44", "45"))
        elif "license_description" in df.columns:
            is_retail = df["license_description"].str.contains("retail", case=False, na=False)
        else:
            is_retail = pd.Series([False] * len(df))
        retail_df = df[is_retail].copy()
        # Ensure zip_code is string and zero-padded
        retail_df['zip_code'] = retail_df['zip_code'].astype(str).str.zfill(5)
        retail_count = retail_df.groupby("zip_code").size().reset_index(name="retail_business_count")
        return self._extracted_from_collect_bea_retail_gdp_22(
            retail_count,
            "retail_business_count.csv",
            'Saved retail business count to ',
            'retail_business_count.csv',
        )

    def collect_parcel_retail_sqft(self) -> pd.DataFrame:
        """
        Collect parcel data, filter for retail, sum building_area per ZIP.
        """
        logger.info("Collecting parcel retail sqft...")
        url = "https://datacatalog.cookcountyil.gov/resource/ijzp-q8t2.json?$limit=1000000"
        try:
            resp = requests.get(url)
            parcels = pd.DataFrame(resp.json())
            # Filter for retail property class
            if "property_class" in parcels.columns:
                is_retail = parcels["property_class"].str.contains("retail", case=False, na=False)
            else:
                is_retail = pd.Series([False] * len(parcels))
            parcels = parcels[is_retail].copy()
            if "building_area" in parcels.columns:
                parcels["building_area"] = pd.to_numeric(parcels["building_area"], errors="coerce")
            else:
                parcels["building_area"] = 0
            # Robust ZIP extraction using ZipResolver
            parcels["zip_code"] = parcels.apply(lambda row: self.zip_resolver.resolve_zip(
                row.get("address", ""),
                row.get("city", "Chicago"),
                row.get("state", "IL")
            ), axis=1)
            parcels = parcels[parcels["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
            sqft = parcels.groupby("zip_code")["building_area"].sum().reset_index(name="retail_space")
            return self._extracted_from_collect_bea_retail_gdp_22(
                sqft,
                "retail_sqft_per_zip.csv",
                'Saved retail sqft per ZIP to ',
                'retail_sqft_per_zip.csv',
            )
        except Exception as e:
            logger.error(f"Failed to collect parcel retail sqft: {e}")
            return pd.DataFrame()

    def collect_bea_retail_gdp(self) -> pd.DataFrame:
        """
        Collect BEA retail GDP for Chicago metro, compute per capita spending, propagate to ZIPs.
        """
        logger.info("Collecting BEA retail GDP...")
        url = f"https://apps.bea.gov/api/data/?&UserID={settings.BEA_KEY}&method=GetData&datasetname=Regional&TableName=CAEMP25N&LineCode=44&GeoFIPS=16980&Year=2022&ResultFormat=JSON"
        try:
            resp = requests.get(url)
            data = resp.json()
            # Parse value (replace with actual parsing logic)
            retail_gdp = float(data["BEAAPI"]["Results"]["Data"][0]["DataValue"])
            # Get metro population (from census or FRED)
            census = pd.read_csv(settings.PROCESSED_DATA_DIR / "census_processed.csv")
            census["zip_code"] = census["zip_code"].astype(str).str.zfill(5)
            pop_zip = census.groupby("zip_code")["total_population"].last().reset_index(name="total_population")
            metro_pop = pop_zip["total_population"].sum()
            per_capita = retail_gdp / metro_pop if metro_pop else 0
            pop_zip["retail_demand"] = pop_zip["total_population"] * per_capita
            return self._extracted_from_collect_bea_retail_gdp_22(
                pop_zip,
                "retail_demand_per_zip.csv",
                'Saved retail demand per ZIP to ',
                'retail_demand_per_zip.csv',
            )
        except Exception as e:
            logger.error(f"Failed to collect BEA retail GDP: {e}")
            return pd.DataFrame()

    # TODO Rename this here and in `collect_business_licenses_retail`, `collect_parcel_retail_sqft` and `collect_bea_retail_gdp`
    def _extracted_from_collect_bea_retail_gdp_22(self, arg0, arg1, arg2, arg3):
        arg0.to_csv(settings.PROCESSED_DATA_DIR / arg1, index=False)
        logger.info(f"{arg2}{settings.PROCESSED_DATA_DIR / arg3}")
        return arg0

    def collect_hud_usps_vacancy(self) -> pd.DataFrame:
        """
        Collect HUD USPS vacancy data, map nonres_vacancy per ZIP.
        """
        logger.info("Collecting HUD USPS vacancy data...")
        # Example: Download CSV from HUD (replace with actual download if available)
        url = "https://www.huduser.gov/portal/datasets/usps/USPS_vacancy_data.csv"
        try:
            df = pd.read_csv(url)
            df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
            vacancy = df.groupby("zip_code")["nonres_vacancy"].mean().reset_index()
            return self._extracted_from_collect_hud_usps_vacancy_33(
                vacancy,
                "hud_usps_vacancy.csv",
                'Saved HUD USPS vacancy per ZIP to ',
                'hud_usps_vacancy.csv',
            )
        except Exception as e:
            logger.error(f"Failed to collect HUD USPS vacancy: {e}")
            return pd.DataFrame()

    # TODO Rename this here and in `collect_multifamily_permits`, `collect_business_licenses_retail`, `collect_parcel_retail_sqft`, `collect_bea_retail_gdp` and `collect_hud_usps_vacancy`
    def _extracted_from_collect_hud_usps_vacancy_33(self, arg0, arg1, arg2, arg3):
        arg0.to_csv(self.raw_dir / arg1, index=False)
        logger.info(f"{arg2}{self.raw_dir / arg3}")
        return arg0

    def collect_all_data(self) -> bool:
        """Collect all required datasets, including new integrations."""
        try:
            logger.info("Starting full data collection pipeline...")
            self.collect_census_data()
            self.collect_permit_data()
            self.collect_multifamily_permits()
            self.collect_business_licenses_retail()
            self.collect_parcel_retail_sqft()
            self.collect_bea_retail_gdp()
            self.collect_hud_usps_vacancy()
            self.collect_economic_data()
            self.collect_zoning_data()
            self.get_property_transactions()
            # Export fallback ZIP report
            self.export_fallback_zip_report(output_path=str(settings.PROCESSED_DATA_DIR / "fallback_zip_report.csv"))
            logger.info("All data collection complete.")
            return True
        except Exception as e:
            logger.error(f"Error in collect_all_data: {e}")
            return False

    # TODO Rename this here and in `collect_permit_data`, `collect_zoning_data` and `collect_all_data`
    def _extracted_from_collect_all_data_100(self, arg0, e, arg2):
        logger.error(f"{arg0}{str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return arg2

    def _is_valid_chicago_zip(self, zip_code: str) -> bool:
        return zip_code in settings.CHICAGO_ZIP_CODES

    def _is_invalid_fallback_zip(self, zip_code: str) -> bool:
        return zip_code in self._invalid_zip_codes or not self._is_valid_chicago_zip(zip_code)

    def resolve_zip_for_row(self, row):
        address = row.get("address", "")
        city = row.get("city", "")
        state = row.get("state", "IL")
        # Use fast crosswalk logic first
        zip_code = self.zip_resolver.lookup_crosswalk(address, city, state)
        if zip_code:
            return zip_code
        # Fallback to full resolver (uszipcode, Nominatim)
        zip_code = self.zip_resolver.resolve_zip(address, city, state)
        if not zip_code:
            # Log and skip for manual review
            logging.warning(f"Unresolved ZIP for: {address}, {city}, {state}")
        return zip_code

    def apply_zip_resolution(self, df: pd.DataFrame, address_col="address", city_col="city", state_col="state", zip_col="zip_code") -> pd.DataFrame:
        # Automatically resolve all missing ZIPs using geocoding and whitelist
        updated = self.zip_resolver.resolve_missing_zips(df, address_col, city_col, state_col, zip_col)
        if updated > 0:
            logging.getLogger(__name__).info(f"Resolved {updated} missing ZIP codes via geocoding.")
        # Suppress all missing ZIP warnings
        return df

    def export_fallback_zip_report(self, output_path: str = None) -> pd.DataFrame:
        """
        Export a DataFrame of all fallback/geocoded ZIPs with status for audit.
        """
        df = pd.DataFrame(self._fallback_zip_records)
        if output_path:
            df.to_csv(output_path, index=False)
        return df


PERMIT_TYPE_MAP = {
    "PERMIT – EXPRESS PERMIT PROGRAM": "EXPRESS_PERMIT",
    "PERMIT - FOR EXTENSION OF PMT": "EXTENSION_PERMIT",
    "NEW CONSTRUCTION": "NEW_CONSTRUCTION",
    "RENOVATION/ALTERATION": "RENOVATION",
    "ADDITION": "ADDITION",
    "DEMOLITION": "DEMOLITION",
    "SIGN": "SIGN",
    "PORCH": "PORCH",
    "ELECTRICAL": "ELECTRICAL",
    "PLUMBING": "PLUMBING",
    "MECHANICAL": "MECHANICAL",
    "FENCE": "FENCE",
    "ROOF": "ROOF",
    "WINDOWS": "WINDOWS",
    "GARAGE": "GARAGE",
    "FOUNDATION": "FOUNDATION",
    "FIRE": "FIRE",
    "ELEVATOR": "ELEVATOR",
    "SOLAR": "SOLAR",
    "POOL": "POOL",
    "DECK": "DECK",
    "SHED": "SHED",
    "MISCELLANEOUS": "MISCELLANEOUS",
    "EASY PERMIT PROCESS": "EASY_PERMIT",
    "SCAFFOLDING": "SCAFFOLDING",
    "REINSTATE REVOKED PMT": "REINSTATE_REVOCATION",
    # Add more as needed
}
logged_missing_permit_types = set()


def map_permit_type(permit_type: str) -> str:
    """Map raw permit_type to standardized value, logging unknowns only once."""
    if not isinstance(permit_type, str):
        return "OTHER"
    permit_type_upper = permit_type.strip().upper()
    for key, val in PERMIT_TYPE_MAP.items():
        if key in permit_type_upper:
            return val
    if permit_type_upper not in logged_missing_permit_types:
        logging.warning(f"Unknown permit_type encountered: {permit_type}")
        logged_missing_permit_types.add(permit_type_upper)
    return "OTHER"

def collect_business_licenses_retail():
    from src.config import settings
    import logging
    logger = logging.getLogger(__name__)
    try:
        collector = DataCollector()
        df = collector.collect_business_licenses_retail()
        if df is not None and not df.empty:
            df.to_csv(settings.PROCESSED_DATA_DIR / "retail_business_count.csv", index=False)
            logger.info(f"Saved retail business count to {settings.PROCESSED_DATA_DIR / 'retail_business_count.csv'}")
        else:
            logger.warning("No retail business license data collected.")
        return df
    except Exception as e:
        logger.error(f"Failed to collect business licenses (retail): {e}")
        return None

def collect_parcel_retail_sqft():
    from src.config import settings
    import logging
    logger = logging.getLogger(__name__)
    try:
        collector = DataCollector()
        df = collector.collect_parcel_retail_sqft()
        if df is not None and not df.empty:
            df.to_csv(settings.PROCESSED_DATA_DIR / "retail_sqft_per_zip.csv", index=False)
            logger.info(f"Saved retail sqft per ZIP to {settings.PROCESSED_DATA_DIR / 'retail_sqft_per_zip.csv'}")
        else:
            logger.warning("No retail parcel sqft data collected.")
        return df
    except Exception as e:
        logger.error(f"Failed to collect parcel retail sqft: {e}")
        return None

def collect_bea_retail_gdp():
    from src.config import settings
    import logging
    logger = logging.getLogger(__name__)
    try:
        collector = DataCollector()
        df = collector.collect_bea_retail_gdp()
        if df is not None and not df.empty:
            df.to_csv(settings.PROCESSED_DATA_DIR / "retail_demand_per_zip.csv", index=False)
            logger.info(f"Saved retail demand per ZIP to {settings.PROCESSED_DATA_DIR / 'retail_demand_per_zip.csv'}")
        else:
            logger.warning("No BEA retail GDP data collected.")
        return df
    except Exception as e:
        logger.error(f"Failed to collect BEA retail GDP: {e}")
        return None
