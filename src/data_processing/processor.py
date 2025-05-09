"""
Data processing module for Chicago population analysis.
Handles data cleaning, transformation, and feature engineering.
"""

import logging
from typing import Dict, Optional, List, Any, Tuple
import json
import os
import traceback
import re
from unicodedata import name

import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

from src.config import settings
from .zoning import ZoningProcessor
from src.utils.validate_data import flag_insufficient_data, check_required_columns
from src.utils.helpers import clean_zip, resolve_column_name
from src.config.column_alias_map import column_aliases

# Set up logging
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processor that coordinates all data processing pipelines."""

    def __init__(self):
        """Initialize the data processor."""
        self.zoning_processor = ZoningProcessor()
        self.processed_data_dir = settings.PROCESSED_DATA_DIR
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def process_population_data(self) -> bool:
        """Process population data.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_csv(settings.POPULATION_DATA_PATH)

            # Clean and process population data
            df.columns = df.columns.str.lower().str.replace(" ", "_")
            df = (
                df.groupby(["zip_code", "year"])
                .agg( # type: ignore
                    {"population": "sum", "households": "sum", "median_income": "mean"}
                )
                .reset_index()
            )

            # Save processed data
            df.to_csv(settings.POPULATION_PROCESSED_PATH, index=False)
            logger.info("Successfully processed population data")
            return True

        except Exception as e:
            logger.error(f"Error processing population data: {str(e)}")
            return False

    def process_economic_data(self, df: pd.DataFrame) -> bool:
        """Process economic data."""
        try:
            logger.info(f"Processing economic data with shape: {df.shape}")

            # Ensure required columns exist
            required_cols = [
                "year",
                "unemployment_rate",
                "real_gdp",
                "per_capita_income",
                "personal_income",
            ]
            if missing_cols := [col for col in required_cols if col not in df.columns]:
                logger.error(
                    f"Required columns missing from economic data: {missing_cols}"
                )
                return False

            # Convert year to numeric
            df["year"] = pd.to_numeric(df["year"], errors="coerce")

            # Drop rows with invalid years
            valid_years = df["year"].between(2010, 2030)
            if not valid_years.all():
                logger.warning(
                    f"Found {(~valid_years).sum()} records with invalid years"
                )
                df = df[valid_years]

            # Convert numeric columns
            numeric_cols = [
                "unemployment_rate",
                "real_gdp",
                "per_capita_income",
                "personal_income",
            ]
            for col in numeric_cols:
                if (
                    col in df.columns
                    and (missing_pct := (df[col].isna().sum() / len(df)) * 100)
                    is not None
                ):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    logger.info(f"✅ {col} present, {missing_pct:.2f}% missing")

            # Fill missing values with forward fill then backward fill
            df = df.sort_values("year")
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

            # Calculate year-over-year changes
            for col in numeric_cols:
                pct_change_col = f"{col}_pct_change"
                df[pct_change_col] = df[col].pct_change() * 100
                df[pct_change_col] = df[pct_change_col].fillna(0)

            # Save processed data
            processed_path = self.processed_data_dir / "economic_processed.csv"
            df.to_csv(processed_path, index=False)
            logger.info(f"Processed economic data saved to {processed_path}")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error processing economic data: ", e, False
            )

    def process_permits_data(self) -> bool:
        """Process building permit data."""
        try:
            logger.info("Loading raw permit records...")
            permit_file = settings.RAW_DATA_DIR / "building_permits.csv"
            if not permit_file.exists():
                logger.error("Permit file not found")
                return False

            df = pd.read_csv(permit_file)
            logger.info(f"Loaded {len(df):,} raw permit records")
            print("[DEBUG] Permits raw shape:", df.shape)
            print("[DEBUG] Permits raw columns:", list(df.columns))
            print("[DEBUG] Permits raw head:\n", df.head())
            print("[DEBUG] Permits raw nulls:\n", df.isnull().sum())

            # Ensure ZIP code is string type
            if "contact_1_zipcode" in df.columns:
                # clean_zip will return a 5-digit string or None
                df["zip_code"] = df["contact_1_zipcode"].apply(clean_zip)
            else:
                logger.warning(
                    "No zip_code column found, attempting to extract from address"
                )

            # Filter to valid Chicago ZIPs only
            df_valid, df_invalid = self.validate_zip_codes(df, "zip_code")
            logger.info(
                f"Filtered out {len(df_invalid)} records with invalid ZIPs from permits data."
            )
            print("[DEBUG] Permits valid shape:", df_valid.shape)
            print("[DEBUG] Permits valid ZIPs unique:", df_valid["zip_code"].nunique())
            print(
                "[DEBUG] Permits valid years unique:",
                df_valid["year"].nunique() if "year" in df_valid.columns else "N/A",
            )
            df = df_valid

            # Categorize permits
            df["residential_permits"] = (
                df["work_description"]
                .str.contains(
                    "residential|house|apartment|condo|dwelling|home|townhouse|multi-family|single-family",
                    case=False,
                    regex=True,
                )
                .fillna(0).astype(int)
            )

            df["commercial_permits"] = (
                df["work_description"]
                .str.contains(
                    "office|industrial|warehouse|factory|manufacturing|corporate|wholesale|distribution",
                    case=False,
                    regex=True,
                )
                .fillna(0).astype(int)
            )

            df["retail_permits"] = (
                df["work_description"]
                .str.contains(
                    "retail|store|shop|restaurant|commercial|business|mall|market|sales",
                    case=False,
                    regex=True,
                )
                .fillna(0).astype(int)
            )

            # Calculate costs by type
            for permit_type in ["residential", "commercial", "retail"]:
                cost_col = f"{permit_type}_construction_cost"
                df[cost_col] = df["reported_cost"].where(
                    df[f"{permit_type}_permits"] == 1, 0
                )

            # Group by ZIP code and year
            df["year"] = pd.to_datetime(df["issue_date"]).dt.year

            agg_dict = {
                "residential_permits": "sum",
                "commercial_permits": "sum",
                "retail_permits": "sum",
                "residential_construction_cost": "sum",
                "commercial_construction_cost": "sum",
                "retail_construction_cost": "sum",
                "reported_cost": "sum",
            }

            df_grouped = df.groupby(["zip_code", "year"]).agg(agg_dict).reset_index()

            # Ensure ZIP codes are strings
            # clean_zip will return a 5-digit string or None
            df_grouped["zip_code"] = df_grouped["zip_code"].apply(clean_zip)

            # Save processed data
            output_path = settings.PROCESSED_DATA_DIR / "permits_processed.csv"
            df_grouped.to_csv(output_path, index=False)
            logger.info(f"Processed permit data saved to {output_path}")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error processing permit data: ", e, False
            )

    def _process_core_sources(self) -> Dict[str, bool]:
        """Process core data sources."""
        census_df = self.process_census_data(pd.read_csv(settings.CENSUS_DATA_PATH))
        return {
            "census": census_df is not None and not census_df.empty,
            "permits": self.process_permits_data(),
            "business_licenses": self.process_business_licenses(),
        }

    def _process_optional_sources(self) -> Dict[str, bool]:
        """Process optional data sources."""
        results = {"property": False, "zoning": False}
        if os.path.exists(settings.PROPERTY_DATA_PATH):
            results["property"] = self.process_property_data()
        else:
            logger.warning("Property data file not found - skipping")

        if os.path.exists(settings.ZONING_DATA_PATH):
            results["zoning"] = self.process_zoning_data()
        else:
            logger.warning("Zoning data file not found - skipping")
        return results

    def _merge_processed_files(
        self, results: Dict[str, bool]
    ) -> Optional[pd.DataFrame]:
        processed_files = self._load_processed_files(results)
        # If any required file is missing, use empty DataFrame with required columns
        required_cols = [
            'zip_code', 'year', 'total_population', 'median_household_income', 'total_housing_units',
            'retail_space', 'retail_demand', 'retail_gap', 'retail_supply', 'retail_permits', # Base retail columns
            'retail_construction_cost', 'retail_business_count', 'retail_leakage',
            'residential_permits', 'commercial_permits', 'reported_cost', 'total_licenses',
            'unique_businesses', 'active_licenses', 'median_home_value', 'labor_force', 'state'
        ]
        for key in ['census', 'permits', 'business_licenses', 'retail_metrics', 'economic', 'zoning', 'property', 'multifamily_permits']:
            if key not in processed_files or processed_files[key] is None or processed_files[key].empty:
                logger.warning(f"Processed file {key} missing or empty. Using empty DataFrame.")
                processed_files[key] = pd.DataFrame(columns=required_cols)

        # Ensure retail_sqft_per_zip and retail_business_count are loaded if they exist
        for retail_raw_key, retail_raw_path_attr in [('retail_sqft_per_zip', 'PROCESSED_DATA_DIR / "retail_sqft_per_zip.csv"'), 
                                                     ('retail_business_count', 'PROCESSED_DATA_DIR / "retail_business_count.csv"')]:
            path = eval(f"settings.{retail_raw_path_attr}") # Use eval carefully or define paths directly
            if path.exists():
                processed_files[retail_raw_key] = pd.read_csv(path, dtype={'zip_code': str})
            elif retail_raw_key not in processed_files: # If not already loaded (e.g. as empty df)
                logger.warning(f"{path} not found. {retail_raw_key} will be missing.")
                processed_files[retail_raw_key] = pd.DataFrame(columns=['zip_code']) # Ensure key exists

        merged_df = processed_files['census'].copy()
        merged_df = self._align_keys(merged_df)
        logger.info(f"Initial census columns: {list(merged_df.columns)}")
        # Merge in order: permits+licenses, economic, zoning, property, multifamily, retail_metrics
        merged_df = self._safe_merge(merged_df, processed_files['permits'], ['zip_code', 'year'], 'left', 'permits')
        merged_df = self._safe_merge(merged_df, processed_files['business_licenses'], ['zip_code', 'year'], 'left', 'business_licenses')        
        merged_df = self._safe_merge(merged_df, processed_files['economic'], ['year'], 'left', 'economic')
        merged_df = self._safe_merge(merged_df, processed_files['zoning'], ['zip_code', 'year'], 'left', 'zoning')
        merged_df = self._safe_merge(merged_df, processed_files['property'], ['zip_code', 'year'], 'left', 'property')
        merged_df = self._safe_merge(merged_df, processed_files['multifamily_permits'], ['zip_code', 'year'], 'left', 'multifamily_permits')
        
        # Merge raw retail components if available
        if 'retail_sqft_per_zip' in processed_files and not processed_files['retail_sqft_per_zip'].empty:
            merged_df = self._safe_merge(merged_df, processed_files['retail_sqft_per_zip'], ['zip_code'], 'left', 'sqft')
        if 'retail_business_count' in processed_files and not processed_files['retail_business_count'].empty:
            merged_df = self._safe_merge(merged_df, processed_files['retail_business_count'], ['zip_code'], 'left', 'bizcount')

        # Note: retail_metrics.csv is not merged here anymore, it's calculated from merged_df later.
        # if processed_files['retail_metrics'] is not None and not processed_files['retail_metrics'].empty:
        #     merged_df = self._safe_merge(merged_df, processed_files['retail_metrics'], ['zip_code', 'year'], 'left', 'retail_metrics')
        # else:
        #     logger.warning("retail_metrics is missing or empty. Skipping retail_metrics merge.")
        # Ensure all required columns are present
        merged_df = self._ensure_required_columns(merged_df, required_cols)
        # Filter to valid Chicago ZIPs
        valid_zips = set(settings.CHICAGO_ZIP_CODES)
        pre_filter_shape = merged_df.shape
        merged_df = merged_df[merged_df['zip_code'].isin(valid_zips)]
        logger.info(f"Filtered to valid Chicago ZIPs: {merged_df.shape[0]} rows (from {pre_filter_shape[0]})")
        if merged_df.empty:
            debug_path = settings.INTERIM_DATA_DIR / "debug_merged_before_zip_filter.csv"
            merged_df.to_csv(debug_path, index=False)
            logger.error(f"Merged DataFrame is empty after ZIP filtering. Debug saved to {debug_path}")
            return None
        return merged_df

    def _align_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].apply(clean_zip) # clean_zip returns 5-digit string or None
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        return df

    def _safe_merge(self, left: pd.DataFrame, right: pd.DataFrame, on: list, how: str, label: str) -> pd.DataFrame:
        try:
            left = self._align_keys(left)
            right = self._align_keys(right)
            merged = left.merge(right, on=on, how=how, suffixes=("", f"_{label}"))
            # Deduplicate columns and resolve suffixes
            merged = self._deduplicate_and_resolve_columns(merged)
            logger.info(f"After merging {label}: columns={list(merged.columns)} shape={merged.shape}")
            logger.info(f"zip_code dtype: {merged['zip_code'].dtype}, unique: {merged['zip_code'].nunique()}")
            logger.info(f"year dtype: {merged['year'].dtype}, unique: {merged['year'].nunique()}")
            return merged
        except Exception as e:
            logger.error(f"Merge failed for {label}: {e}")
            # Attempt to coerce dtypes and retry
            left = self._align_keys(left)
            right = self._align_keys(right)
            try:
                merged = left.merge(right, on=on, how=how, suffixes=("", f"_{label}"))
                merged = self._deduplicate_and_resolve_columns(merged)
                logger.info(f"After retry merging {label}: columns={list(merged.columns)} shape={merged.shape}")
                return merged
            except Exception as e2:
                logger.error(f"Retry merge failed for {label}: {e2}")
                return left

    def _deduplicate_and_resolve_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove duplicate columns and resolve _x/_y suffixes
        cols = df.columns
        to_drop = [col for col in cols if col.endswith('_x') or col.endswith('_y')]
        for col in to_drop:
            base = col[:-2]
            if base in df.columns:
                df.drop(col, axis=1, inplace=True)
            else:
                df.rename(columns={col: base}, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _ensure_required_columns(self, df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
                logger.warning(f"Column {col} missing after merge. Filled with NaN.")
        return df

    def _load_processed_files(
        self, results: Dict[str, bool]
    ) -> Dict[str, pd.DataFrame]:
        """Load all successfully processed files."""
        processed_files = {}
        for name, success in results.items():
            if not success:
                logger.warning(f"Skipping load for {name} as processing was not successful.")
                processed_files[name] = pd.DataFrame() # Ensure key exists, even if empty
                continue
            path = self.processed_data_dir / f"{name}_processed.csv"
            if not path.exists():
                logger.warning(f"Processed file not found: {path}. Using empty DataFrame for {name}.")
                processed_files[name] = pd.DataFrame() # Ensure key exists
                continue
            try:
                # Explicitly load zip_code as string
                df_loaded = pd.read_csv(path, low_memory=False, dtype={'zip_code': str})
                if 'zip_code' in df_loaded.columns:
                    # Apply clean_zip after loading to ensure it's standardized
                    df_loaded['zip_code'] = df_loaded['zip_code'].apply(clean_zip)
                processed_files[name] = df_loaded
                logger.info(f"Loaded {name} from {path}, shape: {df_loaded.shape}. Columns: {df_loaded.columns.tolist()}")
                if 'zip_code' in df_loaded.columns and not df_loaded.empty:
                    logger.info(f"{name} unique ZIPs after load & clean: {df_loaded['zip_code'].nunique()}, sample: {df_loaded['zip_code'].dropna().unique()[:5]}")
            except Exception as e:
                logger.error(f"Error loading processed file {path}: {e}. Using empty DataFrame for {name}.")
                processed_files[name] = pd.DataFrame()
        return processed_files

    def _initialize_base_dataset(
        self, processed_files: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Initialize base dataset from census data."""
        if "census" not in processed_files:
            logger.error("Census data missing - required for base dataset")
            return None
        merged_df = processed_files["census"].copy()
        merged_df["zip_code"] = merged_df["zip_code"].apply(clean_zip) # Already applied in _load_processed_files, but safe
        merged_df["year"] = pd.to_numeric(merged_df["year"], errors="coerce")
        return merged_df

    def _prepare_permits_data(self, permits_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare permits data for merging."""
        permits_df = permits_df.copy()
        permits_df["zip_code"] = permits_df["zip_code"].apply(clean_zip) # Already applied in _load_processed_files, but safe
        permits_df["year"] = pd.to_numeric(permits_df["year"], errors="coerce")
        self._handle_missing_retail_data(permits_df)
        return permits_df

    def _handle_missing_retail_data(self, permits_df: pd.DataFrame) -> None:
        """Handle missing retail data with fallback calculations."""
        if (permits_df["retail_permits"] > 0).sum() == 0:
            logger.warning(
                "No retail_permits > 0 found; substituting retail_permits from commercial_permits * 0.2"
            )
            permits_df["retail_permits"] = permits_df["commercial_permits"] * 0.2
        if (permits_df["retail_construction_cost"] > 0).sum() == 0:
            logger.warning(
                "No retail_construction_cost > 0 found; estimating from commercial_construction_cost * 0.2"
            )
            permits_df["retail_construction_cost"] = (
                permits_df["commercial_construction_cost"] * 0.2
            )

    def _compute_retail_metrics(
        self, permits_df: pd.DataFrame, merged_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute retail-related metrics."""
        retail_space = permits_df["retail_construction_cost"] / 200
        retail_demand = (
            merged_df["total_population"] * merged_df["median_household_income"] * 0.3
        )
        retail_supply = retail_space * 300
        permits_df["retail_space"] = retail_space
        permits_df["retail_demand"] = retail_demand
        permits_df["retail_supply"] = retail_supply
        permits_df["retail_gap"] = retail_demand - retail_supply
        permits_df["vacancy_rate"] = 0.1
        return permits_df

    def _fill_missing_permit_values(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing permit values with zeros."""
        permit_cols = [
            "residential_permits",
            "commercial_permits",
            "retail_permits",
            "residential_construction_cost",
            "commercial_construction_cost",
            "retail_construction_cost",
            "total_permits",
            "total_construction_cost",
        ]
        for col in permit_cols:
            merged_df[col] = merged_df[col].fillna(0) if col in merged_df.columns else 0
        return merged_df

    def _merge_business_licenses(
        self, merged_df: pd.DataFrame, processed_files: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge business license data."""
        if "business_licenses" not in processed_files:
            return merged_df
        try:
            licenses_df = processed_files["business_licenses"].copy()
            licenses_df["zip_code"] = licenses_df["zip_code"].apply(clean_zip) # Already applied in _load_processed_files, but safe
            licenses_df["year"] = pd.to_numeric(licenses_df["year"], errors="coerce")
            merged_df = pd.merge(
                merged_df, licenses_df, on=["zip_code", "year"], how="left"
            )
        except Exception as e:
            logger.error(f"Error merging business_licenses: {str(e)}")
        return merged_df

    def _merge_economic_data(
        self, merged_df: pd.DataFrame, processed_files: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge economic data (city-wide)."""
        if "economic" not in processed_files:
            return merged_df
        try:
            economic_df = processed_files["economic"].copy()
            economic_df["year"] = pd.to_numeric(economic_df["year"], errors="coerce")
            merged_df["year"] = pd.to_numeric(merged_df["year"], errors="coerce")
            merged_df = pd.merge(merged_df, economic_df, on="year", how="left")
        except Exception as e:
            logger.error(f"Error merging economic data: {str(e)}")
        return merged_df

    def _verify_and_initialize_columns(self, merged_df: pd.DataFrame) -> None:
        """Verify data types and initialize missing retail columns."""
        retail_columns = {
            "retail_space": 0,
            "retail_demand": 0,
            "retail_gap": 0,
            "retail_supply": 0,
            "vacancy_rate": 0.1,
        }
        for col, default_val in retail_columns.items():
            if col not in merged_df.columns:
                merged_df[col] = default_val
            logger.warning(
                f"{col} not found - initialized with default value {default_val}"
            )

    def _save_and_log_results(self, merged_df: pd.DataFrame) -> None:
        """Save merged dataset and log results."""
        merged_df.to_csv(settings.MERGED_DATA_PATH, index=False)
        logger.info(f"Merged dataset saved to {str(settings.MERGED_DATA_PATH)}")
        logger.info(f"Final merged columns: {merged_df.columns.tolist()}")
        logger.info(f"Total rows: {len(merged_df)}")
        logger.info(f"Unique ZIP codes: {merged_df['zip_code'].nunique()}")
        logger.info(f"Years covered: {sorted(merged_df['year'].unique()) if 'year' in merged_df.columns else []}")

    def process_all(self) -> bool:
        """Run all processing steps, including retail deficit."""
        try:
            logger.info("Starting data processing pipeline...")

            # Process core data sources
            core_results = self._process_core_sources()
            if not core_results or not all(core_results.values()):
                failed = [k for k, v in core_results.items() if not v]
                logger.error(f"Core processing failed for: {', '.join(failed)}")
                return False

            # Process optional sources
            optional_results = self._process_optional_sources()
            logger.info(f"Optional sources processed: {optional_results}")

            # Process economic data if raw file exists
            if settings.ECONOMIC_RAW_PATH.exists():
                try:
                    economic_raw_df = pd.read_csv(settings.ECONOMIC_RAW_PATH, dtype={'year': str}) # ensure year is str for now
                    if not self.process_economic_data(economic_raw_df): # Pass the loaded df
                        logger.error("Economic data processing failed within process_all.")
                        # Potentially return False if this is critical
                except Exception as e:
                    logger.error(f"Failed to load or process raw economic data in process_all: {e}")
            else:
                logger.warning(f"Raw economic data file not found at {settings.ECONOMIC_RAW_PATH} - skipping its processing.")

            # Merge all base processed files (census, permits, business_licenses, economic, zoning, property, multifamily)
            # This now also includes retail_sqft_per_zip and retail_business_count if they were loaded.
            all_processed_results = {**core_results, **optional_results, 
                                     "economic": settings.ECONOMIC_RAW_PATH.exists(), # Flag if economic was attempted
                                     "retail_sqft_per_zip": (settings.PROCESSED_DATA_DIR / "retail_sqft_per_zip.csv").exists(),
                                     "retail_business_count": (settings.PROCESSED_DATA_DIR / "retail_business_count.csv").exists()}
            merged_df = self._merge_processed_files(all_processed_results)
            
            if merged_df is not None and not merged_df.empty:
                logger.info(f"Initial merged_df shape after _merge_processed_files: {merged_df.shape}")
                # Enrich with calculated retail features (demand, supply, gap, etc.)
                merged_df = self.enrich_retail_metrics(merged_df.copy()) # This should add columns
                logger.info(f"merged_df shape after enrich_with_retail_features: {merged_df.shape}")
                self._save_and_log_results(merged_df) # Save the fully enriched dataset
            else:
                logger.error("Merged DataFrame is empty or None after merging base sources.")
                return False

            # Save specific output views from the final merged_df
            self.save_outputs(merged_df)
            logger.info("Data processing pipeline completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False

    def process_census_data(self, df: pd.DataFrame) -> bool:
        """Process census data."""
        try:
            logger.info(f"🔍 Raw census data columns: {df.columns.tolist()}")

            # Deduplicate 'zip_code' columns if present
            cols = df.columns.tolist()
            if cols.count("zip_code") > 1:
                df = df.loc[:, ~df.columns.duplicated()]
                logger.warning(
                    "Duplicate 'zip_code' columns found and removed in census data."
                )

            # Rename columns to match our schema
            column_map = {
                "zip code tabulation area": "zip_code",
                "B01003_001E": "total_population",
                "B19013_001E": "median_household_income",
                "B25077_001E": "median_home_value",
                "B23025_002E": "labor_force",
                "B25001_001E": "total_housing_units",
            }

            # Apply column mapping
            df = df.rename(columns=column_map)
            logger.info(f"Census columns after renaming: {df.columns.tolist()}")

            # Ensure required columns exist
            required_cols = [
                "total_population",
                "median_household_income",
                "median_home_value",
                "labor_force",
                "zip_code",
                "year",
            ]

            if missing_cols := [col for col in required_cols if col not in df.columns]:
                logger.error(
                    f"Required columns missing from census data: {missing_cols}"
                )
                return False

            # Validate ZIP code column
            if "zip_code" not in df.columns:
                # Try alternative column names
                if zcta_cols := [
                    col
                    for col in df.columns
                    if any(x in col.lower() for x in ["zcta", "zip", "tabulation"])
                ]:
                    logger.info(f"Found ZCTA column: {zcta_cols[0]}")
                    df = df.rename(columns={zcta_cols[0]: "zip_code"})
                else:
                    logger.error("No ZCTA/ZIP column found")
                    logger.error(f"Available columns: {df.columns.tolist()}")
                    return False

            # Format ZIP codes
            df["zip_code"] = df["zip_code"].apply(clean_zip)
            
            # Convert numeric columns
            numeric_cols = [
                "total_population",
                "median_household_income",
                "median_home_value",
                "labor_force",
            ]
            for col in numeric_cols:
                if (
                    col in df.columns
                    and (missing_pct := (df[col].isna().sum() / len(df)) * 100)
                    is not None
                ):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    logger.info(
                        f"✅ {col} present after mapping, {missing_pct:.2f}% missing"
                    )

            # Drop rows with missing values
            df = df.dropna(subset=numeric_cols)
            logger.info(f"Shape after dropping missing values: {df.shape}")

            # Save processed data
            processed_path = self.processed_data_dir / "census_processed.csv"
            df.to_csv(processed_path, index=False)
            logger.info(f"Processed census data saved to {processed_path}")

            # After saving processed data, check for flatline ZIPs
            try:
                for col in ["total_population", "total_housing_units"]:
                    if col in df.columns:
                        changes = df.groupby("zip_code")[col].agg(["first", "last"])
                        stagnant = (changes["first"] == changes["last"]).sum()
                        logger.info(
                            f"{col}: {stagnant} ZIPs have no change over period"
                        )
            except Exception as e:
                logger.warning(f"Could not compute flatline ZIPs: {e}")

            return df

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error processing census data: ", e, None
            )

    def process_retail_deficit(self) -> Optional[pd.DataFrame]:
        """Process retail deficit data with robust fallback and logging/reporting."""
        try:
            retail_metrics_path = self.processed_data_dir / "retail_metrics.csv"
            if not retail_metrics_path.exists():
                logger.warning("Retail metrics not found, generating from scratch")
                retail_metrics = self.generate_retail_metrics()
                if retail_metrics is None:
                    logger.error("Failed to generate retail metrics")
                    return None
            else:
                retail_metrics = pd.read_csv(retail_metrics_path)

            # Enrich retail metrics before saving
            retail_metrics = self.enrich_retail_metrics(retail_metrics)

            # Drop Other_x and Other_y if present
            for col in ["Other_x", "Other_y"]:
                if col in retail_metrics.columns:
                    retail_metrics = retail_metrics.drop(columns=[col])

            # Save retail_metrics.csv
            retail_metrics.to_csv(
                settings.PROCESSED_DATA_DIR / "retail_metrics.csv", index=False
            )
            logger.info(
                f"Saved retail metrics to {settings.PROCESSED_DATA_DIR / 'retail_metrics.csv'}"
            )

            # Calculate and save retail deficit metrics
            retail_deficit = retail_metrics.copy()
            retail_deficit["retail_deficit"] = retail_deficit["retail_gap"].clip(lower=0)
            retail_deficit["retail_surplus"] = (
                retail_deficit["retail_gap"].clip(upper=0).abs()
            )

            # Drop Other_x and Other_y if present
            for col in ["Other_x", "Other_y"]:
                if col in retail_deficit.columns:
                    retail_deficit = retail_deficit.drop(columns=[col])

            # Save retail_deficit.csv
            retail_deficit.to_csv(
                settings.PROCESSED_DATA_DIR / "retail_deficit.csv", index=False
            )
            logger.info(
                f"Saved retail deficit metrics to {settings.PROCESSED_DATA_DIR / 'retail_deficit.csv'}"
            )

            return retail_deficit  # Return the DataFrame now

        except Exception as e:
            logger.error(f"Error processing retail deficit: {str(e)}")
            return None

    def process_business_licenses(self) -> bool:
        """Process business license data and add active_licenses as total_licenses."""
        try:
            df = pd.read_csv(settings.BUSINESS_LICENSES_PATH)
            date_columns = [
                "license_start_date",
                "expiration_date",
                "application_created_date",
                # "zip_code", # Removed: zip_code should not be parsed as a date
            ]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            df["year"] = df["license_start_date"].dt.year
            processed_df = (
                df.groupby(["zip_code", "year"])
                .agg({"license_id": "count", "account_number": "nunique"})
                .reset_index()
            )
            processed_df = processed_df.rename(
                columns={
                    "license_id": "total_licenses",
                    "account_number": "unique_businesses",
                }
            )
            # If you want to count only currently active licenses, add logic here.
            # For now, active_licenses is set to total_licenses (all licenses in that year).
            processed_df["active_licenses"] = processed_df["total_licenses"]
            # Ensure zip_code is cleaned before saving
            processed_df["zip_code"] = processed_df["zip_code"].apply(clean_zip)
            processed_df.to_csv(settings.BUSINESS_LICENSES_PROCESSED_PATH, index=False)
            logger.info("Successfully processed business license data")
            return True
        except Exception as e:
            logger.error(f"Error processing business licenses: {str(e)}")
            return False

    def clean_zoning_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform zoning data.

        Args:
            df: Raw zoning DataFrame

        Returns:
            Cleaned zoning DataFrame
        """
        try:
            logger.info("Cleaning zoning data...")

            # Extract zoning categories
            df["zone_class"] = df["zone_class"].str.extract("([A-Z]+)")

            # Create indicator columns for zone types
            df["is_residential"] = df["zone_class"].isin(["RS", "RT", "RM"])
            df["is_business"] = df["zone_class"].isin(["B", "C"])
            df["is_manufacturing"] = df["zone_class"].isin(["M"])
            df["is_planned_development"] = df["zone_class"].isin(["PD"])

            # Clean address data
            df["zip_code"] = df["zip_code"].apply(clean_zip)

            logger.info("Successfully cleaned zoning data")
            return df

        except Exception as e:
            logger.error(f"Error cleaning zoning data: {str(e)}")
            return None

    def clean_business_licenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform business license data.

        Args:
            df: Raw business licenses DataFrame

        Returns:
            Cleaned business licenses DataFrame
        """
        try:
            logger.info("Cleaning business license data...")

            # Extract year and month
            df["start_year"] = df["license_start_date"].dt.year
            df["start_month"] = df["license_start_date"].dt.month

            # Calculate license duration
            df["license_duration"] = (
                df["license_expiration_date"] - df["license_start_date"]
            ).dt.days

            # Clean address data
            df["zip_code"] = df["zip_code"].apply(clean_zip)

            # Group licenses by type
            df["is_retail"] = df["license_description"].str.contains(
                "RETAIL", case=False, na=False
            )
            df["is_restaurant"] = df["license_description"].str.contains(
                "FOOD|RESTAURANT", case=False, na=False
            )

            logger.info("Successfully cleaned business license data")
            return df

        except Exception as e:
            logger.error(f"Error cleaning business license data: {str(e)}")
            return None

    def clean_property_transactions(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Clean and transform property transaction data.

        Args:
            df: Raw property transactions DataFrame

        Returns:
            Cleaned property transactions DataFrame
        """
        try:
            logger.info("Cleaning property transaction data...")

            # Convert numeric columns
            df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")

            # Extract year and month
            df["sale_year"] = df["sale_date"].dt.year
            df["sale_month"] = df["sale_date"].dt.month

            # Remove outliers
            df = df[df["sale_price"] > 1000]  # Remove likely errors

            # Calculate price per square foot where possible
            if "square_feet" in df.columns:
                df["price_per_sqft"] = df["sale_price"] / pd.to_numeric(
                    df["square_feet"], errors="coerce"
                )

            # Clean address data
            df["zip_code"] = df["zip_code"].apply(clean_zip)

            logger.info("Successfully cleaned property transaction data")
            return df

        except Exception as e:
            logger.error(f"Error cleaning property transaction data: {str(e)}")
            return None

    def aggregate_by_zip(
        self, df: pd.DataFrame, value_cols: List[str], agg_funcs: Dict[str, str]
    ) -> Optional[pd.DataFrame]:
        """
        Aggregate data by ZIP code using specified aggregation functions.

        Args:
            df: DataFrame to aggregate
            value_cols: Columns to aggregate
            agg_funcs: Dictionary mapping columns to aggregation functions

        Returns:
            Aggregated DataFrame
        """
        try:
            logger.info("Aggregating data by ZIP code...")

            # Ensure ZIP code column exists
            if "zip_code" not in df.columns:
                raise ValueError("DataFrame must contain 'zip_code' column")

            # Perform aggregation
            agg_df = df.groupby("zip_code")[value_cols].agg(agg_funcs).reset_index()

            logger.info("Successfully aggregated data")
            return agg_df

        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            return None

    def process_property_data(self) -> bool:
        """Process property transaction data."""
        try:
            # Load property data
            df = pd.read_csv(settings.PROPERTY_DATA_PATH)

            # Convert date column
            df["sale_date"] = pd.to_datetime(df["sale_date"])
            df["year"] = df["sale_date"].dt.year

            # Convert numeric columns
            numeric_cols = ["sale_price", "property_sq_ft", "year_built"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Calculate metrics by ZIP code and year
            processed_df = (
                df.groupby(["zip_code", "year"])
                .agg(
                    {
                        "sale_price": ["mean", "median", "count"],
                        "property_sq_ft": "mean",
                        "property_type": lambda x: x.value_counts().index[
                            0
                        ],  # Most common property type
                    }
                )
                .reset_index()
            )

            processed_df["zip_code"] = processed_df["zip_code"].apply(clean_zip)

            # Flatten column names
            processed_df.columns = [
                "_".join(col).strip("_") for col in processed_df.columns.values
            ]

            # Save processed data
            processed_df.to_csv(settings.PROPERTY_PROCESSED_PATH, index=False)
            logger.info("Successfully processed property data")
            return True

        except Exception as e:
            logger.error(f"Error processing property data: {str(e)}")
            return False

    def process_zoning_data(self, df: pd.DataFrame) -> bool:
        """Process zoning data."""
        try:
            if df is None:  # Already explicit, but add comment for maintainers
                logger.warning("No zoning data provided for processing")
                return False

            logger.info(f"Processing zoning data with shape: {df.shape}")

            # Ensure required columns exist
            required_cols = [
                "zip_code",
                "zoning_classification",
                "zone_category",
                "total_parcels",
                "avg_lot_size",
            ]
            if missing_cols := [col for col in required_cols if col not in df.columns]:
                logger.error(
                    f"Required columns missing from zoning data: {missing_cols}"
                )
                return False

            # Convert numeric columns
            numeric_cols = ["total_parcels", "avg_lot_size", "total_area"]
            for col in numeric_cols:
                if (
                    col in df.columns
                    and (missing_pct := (df[col].isna().sum() / len(df)) * 100)
                    is not None
                ):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    logger.info(f"✅ {col} present, {missing_pct:.2f}% missing")

            # Fill missing values with forward fill then backward fill
            df = df.sort_values(["zip_code", "zoning_classification"])
            df[numeric_cols] = df[numeric_cols].ffill().bfill()

            # Calculate zoning metrics
            metrics = (
                df.groupby("zip_code")
                .agg(
                    {
                        "total_parcels": "sum",
                        "avg_lot_size": "mean",
                        "total_area": "sum",
                    }
                )
                .reset_index()
            )

            # Add zoning diversity metrics
            zoning_counts = df.groupby("zip_code")["zoning_classification"].nunique()
            metrics["zoning_diversity"] = metrics["zip_code"].map(zoning_counts)

            # Save processed data
            processed_path = self.processed_data_dir / "zoning_processed.csv"
            metrics.to_csv(processed_path, index=False)
            logger.info(f"Processed zoning data saved to {processed_path}")

            # Log summary statistics
            logger.info("Zoning metrics summary:")
            logger.info(f"- Total parcels: {metrics['total_parcels'].sum():,}")
            logger.info(
                f"- Average lot size: {metrics['avg_lot_size'].mean():,.0f} sq ft"
            )
            logger.info(f"- Total area: {metrics['total_area'].sum():,.0f} sq ft")
            logger.info(
                f"- Average zoning diversity: {metrics['zoning_diversity'].mean():.1f} classifications per ZIP"
            )

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error processing zoning data: ", e, False
            )

    def process_multifamily_permits(self) -> pd.DataFrame:
        """
        Process multifamily permits and propagate to processed data.
        """
        logger = logging.getLogger(__name__)
        multifam_path = settings.RAW_DATA_DIR / "multifamily_permits.csv"
        if not multifam_path.exists():
            logger.error("multifamily_permits.csv not found.")
            return pd.DataFrame()
        df = pd.read_csv(multifam_path, dtype={'zip_code': str})
        df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
        df["year"] = pd.to_datetime(df["issue_date"]).dt.year
        # Group by ZIP and year, sum unit_count
        mf_agg = (
            df.groupby(["zip_code", "year"]).agg({"unit_count": "sum"}).reset_index()
        )
        mf_agg["zip_code"] = mf_agg["zip_code"].apply(clean_zip)
        mf_agg.to_csv(
            settings.PROCESSED_DATA_DIR / "multifamily_permits_processed.csv",
            index=False,
        )
        logger.info(
            f"Saved processed multifamily permits to {settings.PROCESSED_DATA_DIR / 'multifamily_permits_processed.csv'}"
        )
        return mf_agg

    def process_retail_opportunity_areas(self):
        """
        Compute top 5 opportunity ZIPs for retail development based on multifamily growth and retail lag.
        """
        logger = logging.getLogger(__name__)
        # Load multifamily permits
        mf = pd.read_csv(
            settings.PROCESSED_DATA_DIR / "multifamily_permits_processed.csv", dtype={'zip_code': str}
        )
        mf["zip_code"] = mf["zip_code"].apply(clean_zip)
        # Load permits processed for retail lag
        permits = pd.read_csv(settings.PROCESSED_DATA_DIR / "permits_processed.csv", dtype={'zip_code': str})
        permits["zip_code"] = permits["zip_code"].apply(clean_zip)
        # Calculate historical (oldest 10 years) and recent (most recent 10 years)
        min_year, max_year = mf["year"].min(), mf["year"].max()
        mid_year = min_year + (max_year - min_year) // 2
        hist = (
            mf[mf["year"] <= mid_year]
            .groupby("zip_code")["unit_count"]
            .sum()
            .reset_index(name="hist_units")
        )
        recent = (
            mf[mf["year"] > mid_year]
            .groupby("zip_code")["unit_count"]
            .sum()
            .reset_index(name="recent_units")
        )
        growth = pd.merge(hist, recent, on="zip_code", how="outer").fillna(0)
        growth["growth_pct"] = np.where(
            growth["hist_units"] > 0,
            (growth["recent_units"] - growth["hist_units"]) / growth["hist_units"],
            np.where(growth["recent_units"] > 0, 1.0, 0),
        )
        # Merge with permits for retail lag
        # permits["zip_code"] is already cleaned
        permits["retail_lag"] = permits["retail_permits"] / (
            permits["residential_permits"] + 1e-6
        )
        # Merge with growth
        merged = pd.merge(growth, permits, on="zip_code", how="left")
        # Exclude downtown ZIPs
        downtown_zips = [
            "60601",
            "60602",
            "60603",
            "60604",
            "60605",
            "60606",
            "60607",
            "60610",
            "60611",
        ]
        merged = merged[~merged["zip_code"].isin(downtown_zips)]
        # Filter for >=20% growth and retail_lag < 0.5
        filtered = merged[(merged["growth_pct"] >= 0.2) & (merged["retail_lag"] < 0.5)]
        top5 = filtered.sort_values("growth_pct", ascending=False).head(5)
        top5["zip_code"] = top5["zip_code"].apply(clean_zip)
        top5.to_csv(
            settings.PROCESSED_DATA_DIR / "retail_opportunity_areas.csv", index=False
        )
        logger.info(
            f"Saved top 5 retail opportunity ZIPs to {settings.PROCESSED_DATA_DIR / 'retail_opportunity_areas.csv'}"
        )
        return top5

    def process_permit_data(self, df):
        """Process building permit data."""
        try:
            logger.info(f"Loaded {len(df)} raw permit records")

            # Check for permit ID column
            if "permit_id" not in df.columns:
                logger.warning("No permit ID column found, using index")
                df["permit_id"] = df.index

            # Ensure required columns exist
            required_cols = ["permit_id", "permit_type", "total_fee", "reported_cost"]
            if missing_cols := [col for col in required_cols if col not in df.columns]:
                logger.warning(f"Missing columns: {missing_cols}")
                for col in missing_cols:
                    df[col] = None
                logger.warning(f"Added missing column {col} with default value")

            # Extract ZIP code from address if needed
            if "zip_code" not in df.columns:
                logger.warning(
                    "No zip_code column found, attempting to extract from address"
                )

                # Try to extract from contact_1_zipcode first
                if "contact_1_zipcode" in df.columns:
                    # clean_zip returns 5-digit string or None
                    df["zip_code"] = df["contact_1_zipcode"].apply(clean_zip)
                    logger.info("Extracted ZIP codes from contact_1_zipcode")

                # If still no ZIP codes, try to extract from address
                elif all(
                    col in df.columns
                    for col in ["street_number", "street_direction", "street_name"]
                ):
                    df["address"] = (
                        df["street_number"].astype(str)
                        + " "
                        + df["street_direction"].astype(str)
                        + " "
                        + df["street_name"].astype(str)
                    )

                    # Use the improved ZIP extraction logic
                    df["zip_code"] = df["address"].apply(self.extract_zip)
                else:
                    logger.error("No address columns found to extract ZIP code")
                    return False

            # Clean permit types
            df["permit_type"] = df["permit_type"].fillna("Other")
            df["permit_type"] = df["permit_type"].str.strip().str.title()

            # Map permit types to categories
            permit_type_map = {
                "New Construction": "Residential",
                "Renovation/Alteration": "Residential",
                "Addition": "Residential",
                "Porch": "Residential",
                "Garage": "Residential",
                "Commercial": "Commercial",
                "Business": "Commercial",
                "Office": "Commercial",
                "Industrial": "Commercial",
                "Retail": "Retail",
                "Restaurant": "Retail",
                "Store": "Retail",
                "Shop": "Retail",
            }

            # Apply mapping with fallback to 'Other'
            df["permit_category"] = (
                df["permit_type"].map(permit_type_map).fillna("Other")
            )

            # Log permit type distribution
            type_counts = df["permit_category"].value_counts()
            logger.info("Permit type distribution:")
            for category, count in type_counts.items():
                logger.info(f"- {category}: {count:,} permits")

            # Convert costs to numeric
            df["reported_cost"] = pd.to_numeric(df["reported_cost"], errors="coerce")
            df["total_fee"] = pd.to_numeric(df["total_fee"], errors="coerce")

            # Add year column if missing
            if "year" not in df.columns and "issue_date" in df.columns:
                df["year"] = pd.to_datetime(df["issue_date"]).dt.year
            elif "year" not in df.columns:
                df["year"] = datetime.now().year
                logger.warning(
                    f"No year column found, using current year: {df['year'].iloc[0]}"
                )

            # Aggregate by type and year
            logger.info("Aggregating permits by type and year...")
            agg_df = (
                df.groupby(["zip_code", "year", "permit_category"])
                .agg({"permit_id": "count", "reported_cost": "sum"})
                .reset_index()
            )

            # Pivot to get permit counts and costs by type
            permits_pivot = agg_df.pivot_table(
                index=["zip_code", "year"],
                columns="permit_category",
                values=["permit_id", "reported_cost"],
                fill_value=0,
            ).reset_index()

            # Flatten column names
            permits_pivot.columns = [
                f"{col[1].lower()}_{col[0]}" if col[1] != "" else col[0]
                for col in permits_pivot.columns
            ]

            # Add total columns
            permits_pivot["total_permits"] = permits_pivot[
                [col for col in permits_pivot.columns if col.endswith("permit_id")]
            ].sum(axis=1)
            permits_pivot["total_construction_cost"] = permits_pivot[
                [col for col in permits_pivot.columns if col.endswith("reported_cost")]
            ].sum(axis=1)

            # Log summary statistics
            logger.info(f"Processed {len(df):,} permits:")
            for category in ["Residential", "Commercial", "Retail"]:
                col = f"{category.lower()}_permit_id"
                if col in permits_pivot.columns:
                    logger.info(
                        f"- {category}: {int(permits_pivot[col].sum()):,} permits"
                    )

            logger.info(
                f"Total construction cost: ${permits_pivot['total_construction_cost'].sum():,.2f}"
            )

            # Save processed data
            processed_path = self.processed_data_dir / "permits_processed.csv"
            permits_pivot["zip_code"] = permits_pivot["zip_code"].apply(clean_zip)
            permits_pivot.to_csv(processed_path, index=False)
            logger.info(f"Processed permit data saved to {processed_path}")

            # Log column names for debugging
            logger.info(f"Processed permits columns: {permits_pivot.columns.tolist()}")

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error processing permit data: ", e, False
            )

    def _save_processing_summary(self):
        """Save processing summary to JSON."""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "processed_files": {
                    "census": str(self.processed_data_dir / "census_processed.csv"),
                    "permits": str(self.processed_data_dir / "permits_processed.csv"),
                    "economic": str(self.processed_data_dir / "economic_processed.csv"),
                    "retail_metrics": str(
                        self.processed_data_dir / "retail_metrics.csv"
                    ),
                },
                "metrics": {
                    "census_records": 0,
                    "permit_records": 0,
                    "economic_indicators": 0,
                    "retail_metrics": 0,
                },
            }

            # Count records in each file
            for file_type, file_path in summary["processed_files"].items():
                path = Path(file_path)
                if path.exists():
                    try:
                        df = pd.read_csv(path)
                        summary["metrics"][f"{file_type}_records"] = len(df)
                    except Exception as e:
                        logger.warning(f"Could not read {file_type} file: {str(e)}")

            # Save summary
            summary_path = self.processed_data_dir / "processing_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved processing summary to {summary_path}")

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error saving processing summary: ", e, False
            )

    def process_retail_data(self, data: pd.DataFrame) -> bool:
        """Process retail data.

        Args:
            data (pd.DataFrame): Raw retail data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Filter to valid Chicago ZIPs only
            data_valid, data_invalid = self.validate_zip_codes(data, "zip_code")
            logger.info(
                f"Filtered out {len(data_invalid)} records with invalid ZIPs from retail data."
            )
            data = data_valid

            # Get census data for population and income
            census_data = pd.read_csv(
                settings.PROCESSED_DATA_DIR / "census_processed.csv", dtype={'zip_code': str}
            )
            current_year = census_data["year"].max()
            census_current = census_data[census_data["year"] == current_year]

            # Calculate retail space from permits
            retail_space = (
                data.groupby(["zip_code", "year"])
                .agg({"retail_permits": "sum", "retail_construction_cost": "sum"})
                .reset_index()
            )

            retail_space["zip_code"] = retail_space["zip_code"].apply(clean_zip)
            # Merge with census data
            retail_metrics = retail_space.merge(
                census_current[
                    ["zip_code", "total_population", "median_household_income"]
                ],
                on="zip_code",
                how="outer",
                suffixes=('', '_census') # Add suffixes to avoid column name clashes if any
            )

            # Fill missing values
            retail_metrics = retail_metrics.fillna(
                {
                    "retail_permits": 0,
                    "retail_construction_cost": 0,
                    "total_population": retail_metrics["total_population"].mean(),
                    "median_household_income": retail_metrics[
                        "median_household_income"
                    ].mean(),
                    "year": current_year,
                }
            )

            # Estimate retail space from construction cost
            # Assuming average cost of $200 per square foot for retail construction
            retail_metrics["retail_space"] = (
                retail_metrics["retail_construction_cost"] / 200
            )

            # Estimate annual retail spending per capita (30% of income)
            retail_metrics["retail_demand"] = (
                retail_metrics["total_population"]
                * retail_metrics["median_household_income"]
                * 0.3
            )

            # Calculate retail supply (annual sales per square foot)
            retail_metrics["retail_supply"] = (
                retail_metrics["retail_space"] * 300
            )  # Assume $300 annual sales per sq ft

            # Calculate retail gap (demand - supply)
            retail_metrics["retail_gap"] = (
                retail_metrics["retail_demand"] - retail_metrics["retail_supply"]
            )

            # Calculate retail leakage (gap / demand)
            retail_metrics["retail_leakage"] = (
                retail_metrics["retail_gap"] / retail_metrics["retail_demand"]
            )

            # Calculate vacancy rate (assume 10% base + gap factor)
            retail_metrics["vacancy_rate"] = 0.10 + (
                retail_metrics["retail_gap"] / retail_metrics["retail_demand"]
            ).clip(0, 0.2)

            # Calculate retail opportunity score (normalized gap)
            retail_metrics["opportunity_score"] = (
                retail_metrics["retail_gap"] - retail_metrics["retail_gap"].mean()
            ) / retail_metrics["retail_gap"].std()

            # Identify high opportunity areas
            retail_metrics["high_opportunity"] = (
                retail_metrics["opportunity_score"] > 1.0
            )

            # Save retail metrics
            retail_metrics["zip_code"] = retail_metrics["zip_code"].apply(clean_zip)
            retail_metrics.to_csv(
                settings.PROCESSED_DATA_DIR / "retail_metrics.csv", index=False
            )

            # Calculate and save retail deficit metrics
            retail_deficit = retail_metrics.copy()
            retail_deficit["retail_deficit"] = retail_deficit["retail_gap"].clip(
                lower=0
            )
            retail_deficit["retail_surplus"] = (
                retail_deficit["retail_gap"].clip(upper=0).abs()
            )
            # Drop Other_x and Other_y if present
            for col in ["Other_x", "Other_y"]:
                if col in retail_deficit.columns:
                    retail_deficit = retail_deficit.drop(columns=[col])
            # Save retail_deficit.csv
            retail_deficit["zip_code"] = retail_deficit["zip_code"].apply(clean_zip)
            retail_deficit.to_csv(
                settings.PROCESSED_DATA_DIR / "retail_deficit.csv", index=False
            )
            logger.info(
                f"Saved retail deficit metrics to {settings.PROCESSED_DATA_DIR / 'retail_deficit.csv'}"
            )
            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error processing retail data: ", e, False
            )

    def _extracted_from_process_retail_data_48(self, arg0, e, arg2):
        logger.error(f"{arg0}{str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return arg2

    def process_retail_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            "retail_space",
            "retail_demand",
            "retail_gap",
            "vacancy_rate",
            "retail_supply",
        ]
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            elif (
                col == "retail_gap"
                and "retail_demand" in df.columns
                and "retail_supply" in df.columns
            ):
                df["retail_gap"] = df["retail_demand"] - df["retail_supply"]
            else:
                df[col] = 0
                logger.warning(
                    f"Added missing column {col} to retail data with default value 0"
                )
        return df

    def extract_zip(self, address: Any, community_area: Optional[Any] = None, ward: Optional[Any] = None) -> Optional[str]:
        # Extract 5-digit ZIP codes, prefer those starting with 606 (Chicago)
        if not isinstance(address, str):
            return None
        if zips := re.findall(r"\b60\d{3}\b", address):
            return zips[0]
        if all_zips := re.findall(r"\b\d{5}\b", address):
            logger.warning(f"Non-Chicago ZIP found in address: {address}")
            return all_zips[0]
        # Fallback: use community_area or ward as proxy (placeholder logic)
        if community_area is not None:
            logger.warning(
                f"No ZIP found, using community_area {community_area} as proxy for ZIP"
            )
            return f"606{str(community_area).zfill(2)}"  # Example proxy
        if ward is not None:
            logger.warning(f"No ZIP found, using ward {ward} as proxy for ZIP")
            return f"606{str(ward).zfill(2)}"  # Example proxy
        logger.warning(f"No valid ZIP found in address: {address}")
        return None

    def is_valid_chicago_zip(self, zip_code: str) -> bool:
        return isinstance(zip_code, str) and re.match(r"^606\d{2}$", zip_code)

    def validate_zip_codes(self, df: pd.DataFrame, zip_col: str = "zip_code") -> Tuple[pd.DataFrame, pd.DataFrame]:
        valid = df[zip_col].astype(str).str.match(r"^606[0-9]{2}$")
        percent_valid = valid.mean() * 100
        logger.info(
            f"Valid Chicago ZIPs: {percent_valid:.2f}% ({valid.sum()} of {len(df)})"
        )
        # If <80% valid, attempt to recover ZIPs using extract_zip
        if percent_valid < 80:
            logger.warning(
                f"Low valid ZIP percentage ({percent_valid:.2f}%). Attempting ZIP recovery."
            )

            def recover_zip(row):
                if re.match(r"^606\d{2}$", str(row[zip_col])):
                    return row[zip_col]
                # Try extracting from address fields
                address_fields = [
                    row.get("street_number", ""),
                    str(row.get("street_direction", "")), # Ensure string for concatenation
                    row.get("street_name", ""),
                    row.get("contact_1_city", "Chicago"),
                ]
                address = " ".join([str(x) for x in address_fields if pd.notnull(x)])
                zips = re.findall(r"\b60\d{3}\b", address)
                if zips:
                    logger.warning(f"Recovered ZIP from address: {zips[0]}")
                    return clean_zip(zips[0]) # Clean it
                # Try community_area
                if pd.notnull(row.get("community_area")):
                    ca_zip = f"606{str(int(row['community_area'])).zfill(2)}"
                    logger.warning(f"Recovered ZIP from community_area: {ca_zip}")
                    return clean_zip(ca_zip) # Clean it
                # Try ward
                if pd.notnull(row.get("ward")):
                    ward_zip = f"606{str(int(row['ward'])).zfill(2)}"
                    logger.warning(f"Recovered ZIP from ward: {ward_zip}")
                    return clean_zip(ward_zip) # Clean it
                # Try contact_1_zipcode again if it exists and wasn't initially valid
                if pd.notnull(row.get('contact_1_zipcode')):
                    return clean_zip(row.get('contact_1_zipcode'))
                logger.warning(f"Could not recover ZIP for row: {row}")
                return None

            df[zip_col] = df.apply(recover_zip, axis=1)
            # Re-validate
            valid = df[zip_col].astype(str).str.match(r"^606[0-9]{2}$")
            percent_valid = valid.mean() * 100 if len(df) > 0 else 0
            logger.info(
                f"Post-recovery valid Chicago ZIPs: {percent_valid:.2f}% ({valid.sum()} of {len(df)})"
            )
        return df[valid].copy(), df[~valid].copy()

    def validate_processed_data(self) -> None:
        files = [
            "data/processed/permits_processed.csv",
            "data/processed/retail_metrics.csv",
            "data/processed/retail_deficit.csv",
        ]
        for file in files:
            df = pd.read_csv(file, dtype={'zip_code': str})
            print(f"\n{file}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if "zip_code" in df.columns:
                valid = df["zip_code"].astype(str).str.match(r"^606[0-9]{2}$")
                print(
                    f"  Valid ZIPs: {valid.sum()} / {len(df)} ({valid.mean()*100:.2f}%)"
                )
            print(f"  Nulls per column:\n{df.isnull().sum()}")

    @staticmethod
    def _get_retail_feature_specs(src_pop: Optional[str], src_income: Optional[str], src_rcc: Optional[str], src_rp: Optional[str]) -> List[Dict[str, Any]]:
        """Returns the specification for retail feature enrichment."""
        return [
            {
                "target": "retail_space",
                "rules": {
                    "retail_construction_cost": lambda d: d[src_rcc] / 200 if src_rcc in d and pd.notna(d[src_rcc]).any() else np.nan,
                    "retail_permits": lambda d: d[src_rp] * 10000 if src_rp in d and pd.notna(d[src_rp]).any() else np.nan,
                },
                "default": 0,
            },
            {   "target": "retail_business_count", "rules": {}, "default": 0, }, # Assumed from merge or defaults to 0
            {
                "target": "retail_demand",
                "rules": {
                    "total_population,median_household_income": lambda d: d[src_pop] * d[src_income] * 0.3 if src_pop in d and src_income in d and pd.notna(d[src_pop]).any() and pd.notna(d[src_income]).any() else np.nan,
                },
                "default": 0,
            },
            {
                "target": "retail_supply",
                "rules": { "retail_space": lambda d: d["retail_space"] * 300 if "retail_space" in d and pd.notna(d["retail_space"]).any() else np.nan, },
                "default": 0,
            },
            {
                "target": "retail_gap",
                "rules": { "retail_demand,retail_supply": lambda d: d["retail_demand"] - d["retail_supply"] if "retail_demand" in d and "retail_supply" in d and pd.notna(d["retail_demand"]).any() and pd.notna(d["retail_supply"]).any() else np.nan, },
                "default": 0,
            },
            {   "target": "vacancy_rate", "rules": {"default": lambda d: 0.1}, "default": 0.1, },
            {
                "target": "retail_leakage",
                "rules": { "retail_gap,retail_demand": lambda d: np.where(d["retail_demand"] != 0, d["retail_gap"] / d["retail_demand"], 0) if "retail_gap" in d and "retail_demand" in d and pd.notna(d["retail_gap"]).any() and pd.notna(d["retail_demand"]).any() else np.nan, },
                "default": 0,
            },
            {   "target": src_rcc or "retail_construction_cost", "rules": {}, "default": 0, }, # Ensure source col exists
            {   "target": src_rp or "retail_permits", "rules": {}, "default": 0, }, # Ensure source col exists
        ]

    def _perform_retail_feature_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure retail-specific columns are present and populated using available data or proxies.
        If source data is missing, fill with np.nan and log a warning. If all values are zero, log a warning.
        Returns the DataFrame with all required retail columns, and stores warnings in df.attrs.
        """
        df = df.copy()
        warnings = []

        # --- Define Canonical Names for Source Columns ---
        src_pop = resolve_column_name(df, "total_population", column_aliases)
        src_income = resolve_column_name(df, "median_household_income", column_aliases)
        src_rcc = resolve_column_name(df, "retail_construction_cost", column_aliases)
        src_rp = resolve_column_name(df, "retail_permits", column_aliases)

        canonical_to_resolved_map = {
            "total_population": src_pop,
            "median_household_income": src_income,
            "retail_construction_cost": src_rcc,
            "retail_permits": src_rp,
            "retail_space": "retail_space",
            "retail_demand": "retail_demand",
            "retail_supply": "retail_supply",
            "retail_gap": "retail_gap",
        }

        for canonical, resolved in canonical_to_resolved_map.items():
            if resolved not in df.columns and canonical in ["total_population", "median_household_income", "retail_construction_cost", "retail_permits"]:
                df[resolved] = np.nan
                warnings.append(f"Source column '{resolved}' (for {canonical}) not found. Added as NaN, may affect dependent calculations.")
                logger.warning(f"Source column '{resolved}' (for {canonical}) not found. Added as NaN.")

        feature_specs = self._get_retail_feature_specs(src_pop, src_income, src_rcc, src_rp)

        # --- Enrich Features Iteratively ---
        for spec in feature_specs:
            target_col = spec["target"]
            rules = spec["rules"]
            default_value = spec["default"]

            is_pure_default_rule = "default" in rules and len(rules) == 1
            if target_col in df.columns and \
               (not pd.to_numeric(df[target_col], errors='coerce').isnull().all() and \
                not (pd.to_numeric(df[target_col], errors='coerce').fillna(0) == 0).all()) and \
               not is_pure_default_rule: # If column exists, is meaningfully populated, and not a pure default rule, skip.
                logger.debug(f"Column '{target_col}' already exists and is populated. Skipping rule-based enrichment.")
                continue

            populated_by_rule, enrich_msg = self._enrich_column(df, target_col, rules, canonical_to_resolved_map)
            warnings.append(enrich_msg)

            if not populated_by_rule and (target_col not in df.columns or pd.to_numeric(df[target_col], errors='coerce').isnull().all()):
                df[target_col] = default_value
                msg = f"Applied specification default value '{default_value}' to '{target_col}' as no rule populated it meaningfully."
                logging.info(msg) # Changed from warning to info as this is expected fallback
                warnings.append(msg)
            
            if target_col in df.columns and (pd.to_numeric(df[target_col], errors='coerce').fillna(0) == 0).all():
                msg = f"Warning: All values in '{target_col}' are zero after enrichment (even after spec default if applied). This might affect dependent calculations."
                logging.warning(msg)
                warnings.append(msg)

        # --- Final Type Conversion and NaN Fill for all target columns ---
        for spec in feature_specs:
            col = spec["target"]
            default_val = spec["default"] # Default from spec
            if col in df.columns:
                # Ensure column is numeric, then fill NaNs that might have resulted from failed rule applications
                # or if the column was all NaN before default application in _enrich_column.
                df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure numeric
                if df[col].isnull().all(): # If *still* all NaN after everything
                    df[col] = default_val
                    warnings.append(f"Final Pass: Column '{col}' was all NaN, filled with spec default {default_val}.")
                elif df[col].isnull().any(): # If some NaNs remain
                    df[col] = df[col].fillna(default_val) 
                    warnings.append(f"Final Pass: Column '{col}' had some NaNs, filled with spec default {default_val}.")
            else: # Safeguard: Should be created by _enrich_column with default if not by rule
                df[col] = default_val
                warnings.append(f"Safeguard: Column '{col}' was entirely missing after enrichment loop, added with default {default_val}.")

        df.attrs["retail_warnings"] = list(set(warnings)) # Unique warnings
        return df

    def _enrich_column(self, df: pd.DataFrame, col_to_enrich: str, rules: dict, base_col_map: dict) -> Tuple[bool, str]:
        """
        Helper method to enrich a single column based on available data rules.
        Returns a tuple: (bool indicating if successfully populated with non-NaN, message string).
        """
        for required_cols_key, calculation in rules.items():
            if required_cols_key == "default":
                if col_to_enrich not in df.columns: # Ensure column exists
                    df[col_to_enrich] = np.nan
                df[col_to_enrich] = calculation(df) # Apply the default calculation rule
                logger.info(f"Applied default calculation rule for '{col_to_enrich}'.")
                if (
                    not pd.to_numeric(df[col_to_enrich], errors='coerce')
                    .isnull()
                    .all()
                ):
                    return True, f"Set '{col_to_enrich}' using its default calculation rule."

                logger.debug(f"Default rule for '{col_to_enrich}' resulted in all NaN.")
                return False, f"Default rule for '{col_to_enrich}' resulted in all NaN."
            canonical_source_names = required_cols_key.split(',')
            sources_present_and_valid, actual_source_cols_in_df = self._validate_rule_sources(df, col_to_enrich, required_cols_key, base_col_map)

            if sources_present_and_valid:
                try:
                    original_values = df[col_to_enrich].copy() if col_to_enrich in df.columns else None
                    df.loc[:, col_to_enrich] = calculation(df) # Use .loc for safer assignment
                    # If calculation results in non-NaN data, this rule effectively populated.
                    if not pd.to_numeric(df[col_to_enrich], errors='coerce').isnull().all():
                        logger.info(f"Enriched '{col_to_enrich}' using rule based on '{required_cols_key}' (resolved as {actual_source_cols_in_df}).")
                        return True, f"Enriched '{col_to_enrich}' using rule based on '{required_cols_key}'."
                    # Else (calculation resulted in all NaN values)
                    logger.warning(f"Rule for '{col_to_enrich}' based on '{required_cols_key}' resulted in all NaN values.")
                    if original_values is not None: # Revert if it made things worse or didn't help
                        df.loc[:, col_to_enrich] = original_values
                    # Continue to next rule
                except Exception as e:
                    logger.error(f"Error applying rule for '{col_to_enrich}' based on '{required_cols_key}': {e}")
                    if col_to_enrich not in df.columns: df[col_to_enrich] = np.nan # Ensure column exists for default fill
                    else: df.loc[:, col_to_enrich] = np.nan # Set to NaN to allow default fill
        # If loop completes, no rule successfully populated the column with non-NaN data.
        msg = f"No rule successfully populated '{col_to_enrich}' with non-NaN data."
        logger.info(msg)
        return False, msg

    def _validate_rule_sources(self, df: pd.DataFrame, col_to_enrich: str, required_cols_key: str, base_col_map: dict) -> Tuple[bool, List[str]]:
        """Validates if all source columns for a rule are present in the DataFrame."""
        canonical_source_names = required_cols_key.split(',')
        actual_source_cols_in_df = []
        
        for canonical_name in canonical_source_names:
            actual_col = base_col_map.get(canonical_name)
            if not actual_col or actual_col not in df.columns:
                logger.debug(f"Rule for '{col_to_enrich}' (dep: {required_cols_key}): Source column '{actual_col}' (from canonical '{canonical_name}') not found in DataFrame.")
                return False, [] # Source not found
            actual_source_cols_in_df.append(actual_col)
            
            # Note: Checking if source is all NaN can be done here, but calculation lambdas should ideally handle NaNs.
            # if pd.to_numeric(df[actual_col], errors='coerce').isnull().all():
            #     logger.debug(f"Rule for '{col_to_enrich}' (dep: {required_cols_key}): Source '{actual_col}' is all NaN.")
        return True, actual_source_cols_in_df

    def enrich_retail_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required retail columns are present and populated using available data or proxies.
        Logs warnings for defaulted/estimated metrics or all-zero columns.
        """
        # This method now primarily serves as a specific entry point for retail enrichment,
        # but the core logic is handled by the more generic _perform_retail_feature_enrichment.
        logger.info("Enriching retail metrics...")
        # Make a copy to avoid modifying the original DataFrame passed to this public method
        df_to_enrich = df.copy()
        df_enriched = self._perform_retail_feature_enrichment(df_to_enrich)
        
        # Log any warnings collected during the enrichment process
        if "retail_warnings" in df_enriched.attrs and df_enriched.attrs["retail_warnings"]:
            logger.warning("Retail enrichment process generated the following notes/warnings:")
            for warning_msg in df_enriched.attrs["retail_warnings"]:
                logger.warning(f"- {warning_msg}")
        
        return df_enriched

    def save_outputs(self, merged_df: pd.DataFrame) -> None:
        """
        Save all required outputs for the pipeline. Refactored for clarity and maintainability.
        """
        merged_df = self._filter_valid_zips(merged_df)
        merged_df = self._flag_insufficient_data(merged_df)
        self._save_retail_metrics(merged_df)
        self._save_retail_deficit(merged_df)
        self._save_population_shift_patterns(merged_df)
        self._save_zip_summary(merged_df)
        self._save_ten_year_growth_areas(merged_df)
        self._save_emerging_housing_areas(merged_df)
        self._save_retail_housing_opportunity(merged_df)
        self._save_downtown_comparison(merged_df)
        self._save_high_leakage_areas(merged_df)
        self._save_lowest_retail_provision(merged_df)
        self._save_top_impacted_areas(merged_df)
        self._save_model_metrics(merged_df)
        self._save_retail_deficit_feature_importance(merged_df)

    def _filter_valid_zips(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]

    def _flag_insufficient_data(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        key_cols = ["retail_space", "retail_demand", "retail_gap", "retail_supply", "total_housing_units", "total_population"]
        for col in key_cols:
            if col in merged_df.columns and (merged_df[col] == 0).all():
                logger.warning(f"All values in {col} are zero. Downstream metrics may be misleading.")
                merged_df[f"all_zero_{col}"] = True
        return flag_insufficient_data(merged_df, key_cols)

    def _save_retail_metrics(self, df: pd.DataFrame) -> None:
        retail_metrics_cols = ["zip_code", "year", "retail_space", "retail_demand", "retail_gap", "retail_supply"]
        for col in ["retail_space", "retail_demand", "retail_gap", "retail_supply"]:
            if col in df.columns:
                df[col] = df[col].replace(0, pd.NA)
                # Ensure the column is numeric before fillna with mean to avoid downcasting issues
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df.groupby("zip_code")[col].transform(lambda x: x.fillna(x.mean())) # Now fillna should be safer
                df[col] = df[col].fillna(df[col].mean())
                df[f"{col}_status"] = df[col].apply(lambda x: "insufficient data" if pd.isna(x) or x == 0 else "ok")
        if all(col in df.columns for col in retail_metrics_cols):
            df["zip_code"] = df["zip_code"].apply(clean_zip)
            df[retail_metrics_cols].to_csv(settings.PROCESSED_DATA_DIR / "retail_metrics.csv", index=False)
            logger.info(f"Saved retail metrics to {settings.PROCESSED_DATA_DIR / 'retail_metrics.csv'}")

    def _save_retail_deficit(self, df: pd.DataFrame) -> None:
        if "retail_gap" in df.columns:
            df["zip_code"] = df["zip_code"].apply(clean_zip)
            retail_deficit = df[["zip_code", "year", "retail_gap"]].copy()
            retail_deficit.to_csv(settings.PROCESSED_DATA_DIR / "retail_deficit.csv", index=False)
            logger.info(f"Saved retail deficit to {settings.PROCESSED_DATA_DIR / 'retail_deficit.csv'}")

    def _save_population_shift_patterns(self, df: pd.DataFrame) -> None:
        if "total_population" in df.columns:
            pop_shift = df[["zip_code", "year", "total_population"]].copy()
            pop_shift["zip_code"] = pop_shift["zip_code"].apply(clean_zip)
            pop_shift.to_csv(settings.PREDICTIONS_DIR / "population_shift_patterns.csv", index=False)
            logger.info(f"Saved population shift patterns to {settings.PREDICTIONS_DIR / 'population_shift_patterns.csv'}")

    def _save_zip_summary(self, df: pd.DataFrame) -> None:
        zip_summary_cols = ["zip_code", "total_population", "median_household_income", "total_housing_units", "retail_space", "retail_demand", "retail_gap", "retail_supply"]
        zip_summary = df.groupby("zip_code").last().reset_index()[zip_summary_cols]
        zip_summary["zip_code"] = zip_summary["zip_code"].apply(clean_zip)
        zip_summary.to_csv(settings.PREDICTIONS_DIR / "zip_summary.csv", index=False)
        logger.info(f"Saved zip summary to {settings.PREDICTIONS_DIR / 'zip_summary.csv'}")

    def _save_ten_year_growth_areas(self, merged_df: pd.DataFrame) -> None:
        # Only proceed if required columns exist
        required_cols = ["zip_code", "total_housing_units", "retail_space", "retail_supply", "retail_demand"]
        for col in required_cols:
            if col not in merged_df.columns:
                logger.warning(f"Column '{col}' missing from merged_df. Skipping ten_year_growth_areas output.")
                return
        if "total_population" in merged_df.columns and "year" in merged_df.columns:
            pop_growth = merged_df.groupby("zip_code").apply(lambda g: (g["total_population"].iloc[-1] - g["total_population"].iloc[0]) / g["total_population"].iloc[0] if len(g) > 1 and g["total_population"].iloc[0] else 0).reset_index(name="pop_growth")
            threshold = pop_growth["pop_growth"].quantile(0.9)
            ten_year_growth = pop_growth[pop_growth["pop_growth"] >= threshold].copy() # Use .copy() if it's a slice
            ten_year_growth.loc[:, "zip_code"] = ten_year_growth["zip_code"].apply(clean_zip)
            ten_year_growth.to_csv(settings.PREDICTIONS_DIR / "ten_year_growth_areas.csv", index=False)
            logger.info(f"Saved ten year growth areas to {settings.PREDICTIONS_DIR / 'ten_year_growth_areas.csv'}")

    def _save_emerging_housing_areas(self, df: pd.DataFrame) -> None:
        if "total_housing_units" in df.columns and "year" in df.columns:
            housing_growth = df.groupby("zip_code").apply(lambda g: (g["total_housing_units"].iloc[-1] - g["total_housing_units"].iloc[0]) / g["total_housing_units"].iloc[0] if len(g) > 1 and g["total_housing_units"].iloc[0] else 0).reset_index(name="housing_growth")
            threshold = housing_growth["housing_growth"].quantile(0.9)
            emerging_housing = housing_growth[housing_growth["housing_growth"] >= threshold]
            emerging_housing["zip_code"] = emerging_housing["zip_code"].apply(clean_zip)
            emerging_housing.to_csv(settings.PREDICTIONS_DIR / "emerging_housing_areas.csv", index=False)
            logger.info(f"Saved emerging housing areas to {settings.PREDICTIONS_DIR / 'emerging_housing_areas.csv'}")

    def _save_retail_housing_opportunity(self, df: pd.DataFrame) -> None:
        if "total_housing_units" in df.columns and "retail_space" in df.columns:
            housing_q = df["total_housing_units"].quantile(0.8)
            retail_q = df["retail_space"].quantile(0.2)
            opportunity = df[(df["total_housing_units"] >= housing_q) & (df["retail_space"] <= retail_q)]
            opportunity_to_save = opportunity[["zip_code", "total_housing_units", "retail_space"]].copy()
            opportunity_to_save["zip_code"] = opportunity_to_save["zip_code"].apply(clean_zip)
            opportunity_to_save.to_csv(settings.PREDICTIONS_DIR / "retail_housing_opportunity.csv", index=False)
            logger.info(f"Saved retail housing opportunity to {settings.PREDICTIONS_DIR / 'retail_housing_opportunity.csv'}")

    def _save_downtown_comparison(self, df: pd.DataFrame) -> None:
        downtown_zips = ["60601", "60602", "60603", "60604", "60605", "60606", "60607"]
        downtown = df[df["zip_code"].isin(downtown_zips)].copy() # Use .copy() if it's a slice
        if not downtown.empty:
            downtown.loc[:, "zip_code"] = downtown["zip_code"].apply(clean_zip)
            downtown.to_csv(settings.PREDICTIONS_DIR / "downtown_comparison.csv", index=False)
            logger.info(f"Saved downtown comparison to {settings.PREDICTIONS_DIR / 'downtown_comparison.csv'}")

    def _save_high_leakage_areas(self, df):
        if "retail_gap" in df.columns:
            leakage_q = df["retail_gap"].quantile(0.9)
            high_leakage = df[df["retail_gap"] >= leakage_q]
            high_leakage_to_save = high_leakage[["zip_code", "retail_gap"]].copy()
            high_leakage_to_save["zip_code"] = high_leakage_to_save["zip_code"].apply(clean_zip)
            high_leakage_to_save.to_csv(settings.PREDICTIONS_DIR / "high_leakage_areas.csv", index=False)
            logger.info(f"Saved high leakage areas to {settings.PREDICTIONS_DIR / 'high_leakage_areas.csv'}")

    def _save_lowest_retail_provision(self, df):
        if "retail_space" in df.columns:
            low_retail_q = df["retail_space"].quantile(0.1)
            lowest_retail = df[df["retail_space"] <= low_retail_q]
            lowest_retail_to_save = lowest_retail[["zip_code", "retail_space"]].copy()
            lowest_retail_to_save["zip_code"] = lowest_retail_to_save["zip_code"].apply(clean_zip)
            lowest_retail_to_save.to_csv(settings.PREDICTIONS_DIR / "lowest_retail_provision.csv", index=False)
            logger.info(f"Saved lowest retail provision to {settings.PREDICTIONS_DIR / 'lowest_retail_provision.csv'}")

    def _save_top_impacted_areas(self, df):
        if "retail_gap" in df.columns and "total_housing_units" in df.columns and "year" in df.columns:
            housing_growth = df.groupby("zip_code").apply(lambda g: (g["total_housing_units"].iloc[-1] - g["total_housing_units"].iloc[0]) / g["total_housing_units"].iloc[0] if len(g) > 1 and g["total_housing_units"].iloc[0] else 0).reset_index(name="housing_growth")
            merged_growth = pd.merge(df, housing_growth, on="zip_code", how="left")
            impacted = merged_growth[(merged_growth["retail_gap"] >= merged_growth["retail_gap"].quantile(0.9)) & (merged_growth["housing_growth"] >= merged_growth["housing_growth"].quantile(0.9))]
            impacted_to_save = impacted[["zip_code", "retail_gap", "housing_growth"]].copy()
            impacted_to_save["zip_code"] = impacted_to_save["zip_code"].apply(clean_zip)
            impacted_to_save.to_csv(settings.PREDICTIONS_DIR / "top_impacted_areas.csv", index=False)
            logger.info(f"Saved top impacted areas to {settings.PREDICTIONS_DIR / 'top_impacted_areas.csv'}")

    def _save_model_metrics(self, df: pd.DataFrame) -> None:
        if "model_metric" in df.columns:
            df[["zip_code", "model_metric"]].to_csv(settings.MODEL_METRICS_DIR / "model_metrics.csv", index=False)
            logger.info(f"Saved model metrics to {settings.MODEL_METRICS_DIR / 'model_metrics.csv'}")

    def _save_retail_deficit_feature_importance(self, df: pd.DataFrame) -> None:
        if "retail_deficit_feature_importance" in df.columns:
            feature_importance_to_save = df[["zip_code", "retail_deficit_feature_importance"]].copy()
            feature_importance_to_save["zip_code"] = feature_importance_to_save["zip_code"].apply(clean_zip)
            feature_importance_to_save.to_csv(settings.MODEL_METRICS_DIR / "retail_deficit_feature_importance.csv", index=False)
            logger.info(f"Saved retail deficit feature importance to {settings.MODEL_METRICS_DIR / 'retail_deficit_feature_importance.csv'}")

    def compute_retail_metrics(
    ) -> pd.DataFrame:
        """
        Compute retail metrics for each ZIP code.
        This method now expects that the necessary source data (permits, parcels, licenses, census, BEA)
        has already been processed and merged into a comprehensive DataFrame, or that individual
        processed files for these sources are available.

        It primarily orchestrates the loading of these pre-processed components if they haven't
        been passed in or merged already, and then calculates derived retail metrics.
        """
        logger.info("Computing or retrieving retail metrics for each ZIP code...")

        # Attempt to load a pre-merged dataset first, or individual components
        # This part assumes that individual processed files exist if a fully merged_df isn't available
        # For simplicity, this example will focus on using an existing merged_df or census_df
        # and expect other components to be merged into it by prior steps or enrich_with_retail_features.

        # Placeholder: In a full pipeline, you'd load or ensure census_df, permits_df, etc. are available.
        # For this refactoring, we assume enrich_with_retail_features will work on a df that has these.
        # The core logic of calculating retail_space, supply, demand, gap is now within enrich_with_retail_features.

        # This method could be used to trigger the enrichment if needed, or simply load
        # the result if enrich_with_retail_features has already run and saved its output.
        
        # For now, let's assume this method is more about ensuring the retail_metrics.csv exists
        # or can be generated by calling enrich_with_retail_features on a suitable base.
        
        retail_metrics_path = settings.PROCESSED_DATA_DIR / "retail_metrics.csv"
        if retail_metrics_path.exists():
            logger.info(f"Loading existing retail metrics from {retail_metrics_path}")
            return pd.read_csv(retail_metrics_path, dtype={'zip_code': str})
        else:
            logger.warning(f"{retail_metrics_path} not found. Retail metrics might be incomplete or need generation.")
            # Fallback: create an empty df with expected columns if no base data to enrich is available here.
            # In a real pipeline, you'd load the merged_dataset.csv and call enrich_with_retail_features.
            return pd.DataFrame(columns=["zip_code", "year", "retail_space", "retail_demand", "retail_supply", "retail_gap"])
