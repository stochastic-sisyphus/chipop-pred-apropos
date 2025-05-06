"""
Data processing module for Chicago population analysis.
Handles data cleaning, transformation, and feature engineering.
"""

import logging
from typing import Dict, Optional, List
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
                .agg(
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

    def process_economic_data(self, df):
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
                df["zip_code"] = (
                    df["contact_1_zipcode"].astype(str).str.strip().str.zfill(5)
                )
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
                .astype(int)
            )

            df["commercial_permits"] = (
                df["work_description"]
                .str.contains(
                    "office|industrial|warehouse|factory|manufacturing|corporate|wholesale|distribution",
                    case=False,
                    regex=True,
                )
                .astype(int)
            )

            df["retail_permits"] = (
                df["work_description"]
                .str.contains(
                    "retail|store|shop|restaurant|commercial|business|mall|market|sales",
                    case=False,
                    regex=True,
                )
                .astype(int)
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
            df_grouped["zip_code"] = (
                df_grouped["zip_code"].astype(str).str.strip().str.zfill(5)
            )

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
        return {
            "census": self.process_census_data(pd.read_csv(settings.CENSUS_DATA_PATH)),
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
        """
        Merge all processed files, enforcing ZIP validation and robust propagation.
        """
        processed_files = self._load_processed_files(results)
        # Only keep valid Chicago ZIPs in all merges
        for key, df in processed_files.items():
            if "zip_code" in df.columns:
                processed_files[key] = df[
                    df["zip_code"].isin(settings.CHICAGO_ZIP_CODES)
                ]
        # Merge permits_processed.csv (retail_permits, retail_construction_cost)
        permits = processed_files.get("permits_processed.csv")
        # Merge retail_business_count.csv
        retail_business = processed_files.get("retail_business_count.csv")
        # Merge retail_sqft_per_zip.csv (retail_space)
        retail_sqft = processed_files.get("retail_sqft_per_zip.csv")
        # Merge retail_demand_per_zip.csv (retail_demand)
        retail_demand = processed_files.get("retail_demand_per_zip.csv")
        # Merge all on zip_code
        merged = None
        try:
            merged = permits
            if retail_business is not None:
                merged = merged.merge(retail_business, on="zip_code", how="left")
            if retail_sqft is not None:
                merged = merged.merge(retail_sqft, on="zip_code", how="left")
            if retail_demand is not None:
                merged = merged.merge(retail_demand, on="zip_code", how="left")
            # Compute retail_supply (proxy: retail_business_count * retail_space per business)
            if (
                "retail_business_count" in merged.columns
                and "retail_space" in merged.columns
            ):
                merged["retail_supply"] = merged["retail_business_count"] * (
                    merged["retail_space"] / merged["retail_business_count"]
                ).fillna(0)
            # Compute retail_gap = retail_demand - retail_supply
            if "retail_demand" in merged.columns and "retail_supply" in merged.columns:
                merged["retail_gap"] = merged["retail_demand"] - merged["retail_supply"]
            # Compute retail_lag = retail_gap / retail_demand
            if "retail_gap" in merged.columns and "retail_demand" in merged.columns:
                merged["retail_lag"] = merged["retail_gap"] / merged[
                    "retail_demand"
                ].replace(0, np.nan)
            # Add robust error handling for missing/insufficient data
            for col in [
                "retail_permits",
                "retail_construction_cost",
                "retail_business_count",
                "retail_space",
                "retail_demand",
                "retail_supply",
                "retail_gap",
                "retail_lag",
            ]:
                if col not in merged.columns:
                    merged[col] = np.nan
                if merged[col].isnull().all():
                    logging.warning(
                        f"All values missing for {col} in merged retail data."
                    )
            return merged
        except Exception as e:
            logging.error(f"Failed to merge processed retail files: {e}")
            return None

    def _load_processed_files(
        self, results: Dict[str, bool]
    ) -> Dict[str, pd.DataFrame]:
        """Load all successfully processed files."""
        processed_files = {}
        for name, success in results.items():
            if not success:
                continue
            path = self.processed_data_dir / f"{name}_processed.csv"
            if not path.exists():
                logger.warning(f"Processed file not found: {path}")
                continue
            processed_files[name] = pd.read_csv(path, low_memory=False)
            logger.info(f"{name} columns: {processed_files[name].columns.tolist()}")
        return processed_files

    def _initialize_base_dataset(
        self, processed_files: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Initialize base dataset from census data."""
        if "census" not in processed_files:
            logger.error("Census data missing - required for base dataset")
            return None
        merged_df = processed_files["census"].copy()
        merged_df["zip_code"] = merged_df["zip_code"].astype(str).str.zfill(5)
        merged_df["year"] = pd.to_numeric(merged_df["year"], errors="coerce")
        return merged_df

    def _prepare_permits_data(self, permits_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare permits data for merging."""
        permits_df = permits_df.copy()
        permits_df["zip_code"] = permits_df["zip_code"].astype(str).str.zfill(5)
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
            licenses_df["zip_code"] = licenses_df["zip_code"].astype(str).str.zfill(5)
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
        logger.info(f"Years covered: {sorted(merged_df['year'].unique())}")

    def process_all(self) -> bool:
        """Run all processing steps, including retail deficit."""
        try:
            logger.info("Starting data processing pipeline...")

            # Process core data sources
            core_results = self._process_core_sources()
            if not all(core_results.values()):
                failed = [k for k, v in core_results.items() if not v]
                logger.error(f"Core processing failed for: {', '.join(failed)}")
                return False

            # Process optional sources
            optional_results = self._process_optional_sources()

            # Process economic data
            economic_success = self.process_economic_data(
                pd.read_csv(settings.RAW_DATA_DIR / "economic_indicators.csv")
            )
            if not economic_success:
                logger.warning(
                    "Economic data processing failed - some features may be missing"
                )

            # Process multifamily permits
            self.process_multifamily_permits()

            # Merge processed files
            merged_data = self._merge_processed_files(
                {**core_results, **optional_results}
            )
            if merged_data is None:
                logger.error("Failed to merge processed files")
                return False

            # Process retail deficit
            if not self.process_retail_deficit():
                logger.warning("Retail deficit processing failed")

            # Process retail opportunity areas
            self.process_retail_opportunity_areas()

            # After merging all processed files, ensure retail metrics are present and merged
            merged_df = self._merge_processed_files({**core_results, **optional_results})
            if merged_df is None:
                logger.error("Failed to merge processed files.")
                return False
            # Merge retail_space
            retail_space_path = settings.PROCESSED_DATA_DIR / "retail_sqft_per_zip.csv"
            if retail_space_path.exists():
                retail_space_df = pd.read_csv(retail_space_path)
                if "year" in merged_df.columns and "year" in retail_space_df.columns:
                    merged_df = pd.merge(merged_df, retail_space_df, on=["zip_code", "year"], how="left")
                else:
                    merged_df = pd.merge(merged_df, retail_space_df, on=["zip_code"], how="left")
                logger.info("Merged retail_space into main metrics DataFrame.")
            else:
                logger.warning("retail_sqft_per_zip.csv not found; retail_space will be missing.")
            # Merge retail_supply
            retail_supply_path = settings.PROCESSED_DATA_DIR / "retail_business_count.csv"
            if retail_supply_path.exists():
                retail_supply_df = pd.read_csv(retail_supply_path)
                merged_df = pd.merge(merged_df, retail_supply_df.rename(columns={"retail_business_count": "retail_supply"}), on=["zip_code"], how="left")
                logger.info("Merged retail_supply into main metrics DataFrame.")
            else:
                logger.warning("retail_business_count.csv not found; retail_supply will be missing.")
            # Merge retail_demand
            retail_demand_path = settings.PROCESSED_DATA_DIR / "retail_demand_per_zip.csv"
            if retail_demand_path.exists():
                retail_demand_df = pd.read_csv(retail_demand_path)
                if "year" in merged_df.columns and "year" in retail_demand_df.columns:
                    merged_df = pd.merge(merged_df, retail_demand_df, on=["zip_code", "year"], how="left")
                else:
                    merged_df = pd.merge(merged_df, retail_demand_df, on=["zip_code"], how="left")
                logger.info("Merged retail_demand into main metrics DataFrame.")
            else:
                logger.warning("retail_demand_per_zip.csv not found; retail_demand will be missing.")
            # Compute retail_gap
            if "retail_demand" in merged_df.columns and "retail_supply" in merged_df.columns:
                merged_df["retail_gap"] = merged_df["retail_demand"] - merged_df["retail_supply"]
                logger.info("Computed retail_gap = retail_demand - retail_supply.")
            else:
                logger.warning("Cannot compute retail_gap: missing retail_demand or retail_supply.")
            # Let NaN propagate, do not set to zero
            # Save and log results
            self._save_and_log_results(merged_df)
            logger.info("Data processing pipeline completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error in process_all: {e}")
            return False

    def process_census_data(self, df):
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
                return None

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
                    return None

            # Format ZIP codes
            df["zip_code"] = df["zip_code"].astype(str).str.strip().str.zfill(5)

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

    def process_retail_deficit(self) -> bool:
        """Process retail deficit data with robust fallback and logging/reporting."""
        try:
            retail_metrics_path = self.processed_data_dir / "retail_metrics.csv"
            if not retail_metrics_path.exists():
                logger.warning("Retail metrics not found, generating from scratch")
                retail_metrics = self.generate_retail_metrics()
                if retail_metrics is None:
                    logger.error("Failed to generate retail metrics")
                    return False
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
            retail_deficit.to_csv(
                settings.PROCESSED_DATA_DIR / "retail_deficit.csv", index=False
            )
            logger.info(
                f"Saved retail deficit metrics to {settings.PROCESSED_DATA_DIR / 'retail_deficit.csv'}"
            )
            return True
        except Exception as e:
            logger.error(f"Error processing retail deficit: {str(e)}")
            return False

    def process_business_licenses(self) -> bool:
        """Process business license data and add active_licenses as total_licenses."""
        try:
            df = pd.read_csv(settings.BUSINESS_LICENSES_PATH)
            date_columns = [
                "license_start_date",
                "expiration_date",
                "application_created_date",
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
            df["zip_code"] = df["zip_code"].astype(str).str.extract("(\d{5})")

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
            df["zip_code"] = df["zip_code"].astype(str).str.extract("(\d{5})")

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

    def clean_property_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
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
            df["zip_code"] = df["zip_code"].astype(str).str.extract("(\d{5})")

            logger.info("Successfully cleaned property transaction data")
            return df

        except Exception as e:
            logger.error(f"Error cleaning property transaction data: {str(e)}")
            return None

    def aggregate_by_zip(
        self, df: pd.DataFrame, value_cols: List[str], agg_funcs: Dict[str, str]
    ) -> pd.DataFrame:
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

    def generate_retail_metrics(self) -> pd.DataFrame:
        """Generate retail metrics from processed data, merging in retail_sqft_per_zip and retail_business_count."""
        try:
            import numpy as np

            df = pd.read_csv(settings.MERGED_DATA_PATH, low_memory=False)
            # Ensure zip_code is string and zero-padded
            df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
            # Load retail_sqft_per_zip and retail_business_count
            retail_sqft_path = settings.RAW_DATA_DIR / "retail_sqft_per_zip.csv"
            retail_business_count_path = (
                settings.RAW_DATA_DIR / "retail_business_count.csv"
            )
            retail_sqft_per_zip_df = None
            retail_business_count_df = None
            if retail_sqft_path.exists():
                retail_sqft_per_zip_df = pd.read_csv(retail_sqft_path)
                retail_sqft_per_zip_df["zip_code"] = (
                    retail_sqft_per_zip_df["zip_code"].astype(str).str.zfill(5)
                )
            else:
                logger.warning(
                    "retail_sqft_per_zip.csv not found; retail_space will fallback to zero."
                )
            if retail_business_count_path.exists():
                retail_business_count_df = pd.read_csv(retail_business_count_path)
                retail_business_count_df["zip_code"] = (
                    retail_business_count_df["zip_code"].astype(str).str.zfill(5)
                )
            else:
                logger.warning(
                    "retail_business_count.csv not found; retail_supply will fallback to zero."
                )
            # Merge into df by zip_code (left join)
            if retail_sqft_per_zip_df is not None:
                df = df.merge(retail_sqft_per_zip_df, on="zip_code", how="left")
                df["retail_space"] = df["retail_sqft_per_zip"].fillna(0)
            elif "retail_space" not in df.columns:
                df["retail_space"] = 0
            if retail_business_count_df is not None:
                df = df.merge(retail_business_count_df, on="zip_code", how="left")
                df["retail_supply"] = df["retail_business_count"].fillna(0)
            elif "retail_supply" not in df.columns:
                df["retail_supply"] = 0
            # If all values in retail_space or retail_supply are zero, try to enrich using fallback formulas
            if (df["retail_space"] == 0).all():
                logger.warning(
                    "All values in retail_space are zero after merge. Attempting enrichment using fallback formulas."
                )
                if "retail_construction_cost" in df.columns:
                    df["retail_space"] = df["retail_construction_cost"] / 200
                elif "retail_permits" in df.columns:
                    df["retail_space"] = df["retail_permits"] * 10000
                else:
                    logger.warning("Could not enrich retail_space; leaving as zero.")
            if (df["retail_supply"] == 0).all():
                logger.warning(
                    "All values in retail_supply are zero after merge. Attempting enrichment using fallback formulas."
                )
                if "retail_space" in df.columns:
                    df["retail_supply"] = df["retail_space"] * 300
                else:
                    logger.warning("Could not enrich retail_supply; leaving as zero.")
            # Log sample of merged columns
            logger.info(df[["zip_code", "retail_space", "retail_supply"]].head())
            # Calculate retail_gap if both demand and supply present
            if "retail_demand" in df.columns and "retail_supply" in df.columns:
                df["retail_gap"] = df["retail_demand"] - df["retail_supply"]
            # Enrich with other retail features before saving
            df = self.enrich_with_retail_features(df)
            # Save and log
            df.to_csv(settings.RETAIL_METRICS_PATH, index=False)
            logger.info(f"Saved retail metrics to {settings.RETAIL_METRICS_PATH}")
            return df
        except Exception as e:
            logger.error(f"Error generating retail metrics: {str(e)}")
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

    def process_zoning_data(self, df) -> bool:
        """Process zoning data."""
        try:
            if df is None:
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
        df = pd.read_csv(multifam_path)
        df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
        df["year"] = pd.to_datetime(df["issue_date"]).dt.year
        # Group by ZIP and year, sum unit_count
        mf_agg = (
            df.groupby(["zip_code", "year"]).agg({"unit_count": "sum"}).reset_index()
        )
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
            settings.PROCESSED_DATA_DIR / "multifamily_permits_processed.csv"
        )
        # Load permits processed for retail lag
        permits = pd.read_csv(settings.PROCESSED_DATA_DIR / "permits_processed.csv")
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
        permits["zip_code"] = permits["zip_code"].astype(str).str.zfill(5)
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
        top5.to_csv(
            settings.PROCESSED_DATA_DIR / "retail_opportunity_areas.csv", index=False
        )
        logger.info(
            f"Saved top 5 retail opportunity ZIPs to {settings.PROCESSED_DATA_DIR / 'retail_opportunity_areas.csv'}"
        )
        return top5

    def run_pipeline(self) -> bool:
        """Run the full data processing pipeline."""
        try:
            logger.info("Starting data processing pipeline...")

            # Process each data source
            results = {
                "census": self.process_census_data(
                    pd.read_csv(settings.CENSUS_DATA_PATH)
                ),
                "permits": self.process_permits_data(),
                "business_licenses": self.process_business_licenses(),
                "economic": self.process_economic_data(
                    pd.read_csv(settings.RAW_DATA_DIR / "economic_indicators.csv")
                ),
            }

            # Log processing results
            for name, success in results.items():
                logger.info(
                    f"{name} processing: {'✅ Success' if success else '❌ Failed'}"
                )

            # Verify processed files exist and log their columns
            for name in results:
                path = self.processed_data_dir / f"{name}_processed.csv"
                if path.exists():
                    df = pd.read_csv(path, low_memory=False)
                    logger.info(f"\n{name} processed data:")
                    logger.info(f"- Shape: {df.shape}")
                    logger.info(f"- Columns: {df.columns.tolist()}")
                    if "year" in df.columns:
                        logger.info(f"- Years: {df['year'].unique().tolist()}")
                    if "zip_code" in df.columns:
                        logger.info(
                            f"- ZIP codes: {len(df['zip_code'].unique())} unique"
                        )

            # Merge processed files
            merged_data = self._merge_processed_files(results)
            if merged_data is None:
                logger.error("Failed to merge processed files")
                return False

            # Verify merged dataset
            logger.info("\nMerged dataset verification:")
            logger.info(f"- Shape: {merged_data.shape}")
            logger.info(f"- Columns: {merged_data.columns.tolist()}")
            logger.info(f"- Missing values:\n{merged_data.isnull().sum()}")

            # Verify key columns
            key_columns = [
                "total_population",
                "median_household_income",
                "residential_permits",
                "commercial_permits",
                "retail_permits",
                "residential_construction_cost",
                "commercial_construction_cost",
                "retail_construction_cost",
                "gdp",
                "unemployment_rate",
            ]

            logger.info("\nKey column statistics:")
            for col in key_columns:
                if col in merged_data.columns:
                    stats = merged_data[col].describe()
                    logger.info(f"\n{col}:")
                    logger.info("- Present: ✅")
                    logger.info(f"- Non-null: {merged_data[col].count()}")
                    logger.info(f"- Mean: {stats['mean']:.2f}")
                    logger.info(f"- Min: {stats['min']:.2f}")
                    logger.info(f"- Max: {stats['max']:.2f}")
                else:
                    logger.warning(f"{col}: ❌ Missing")

            logger.info("Data processing pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            return False

    def process_all_data(
        self, census_data=None, permit_data=None, economic_data=None, zoning_data=None
    ):
        """Process all data sources."""
        try:
            logger.info("Starting data processing pipeline...")

            # Process census data
            if census_data is not None:
                processed_census = self.process_census_data(census_data)
                if processed_census is None:
                    logger.error("Core processing failed for: census")
                    return False
                logger.info("Successfully processed census data")
            else:
                logger.warning("No census data provided for processing")
                return False

            # Process permit data
            if permit_data is not None:
                processed_permits = self.process_permit_data(permit_data)
                if not processed_permits:
                    logger.error("Core processing failed for: permits")
                    return False
                logger.info("Successfully processed permit data")
            else:
                logger.warning("No permit data provided for processing")
                return False

            # Process economic data
            if economic_data is not None:
                processed_economic = self.process_economic_data(economic_data)
                if not processed_economic:
                    logger.error("Core processing failed for: economic")
                    return False
                logger.info("Successfully processed economic data")
            else:
                logger.warning("No economic data provided for processing")
                return False

            # Process zoning data
            if zoning_data is not None:
                processed_zoning = self.process_zoning_data(zoning_data)
                if not processed_zoning:
                    logger.error("Core processing failed for: zoning")
                    return False
                logger.info("Successfully processed zoning data")
            else:
                logger.warning("No zoning data provided for processing")

            if self.process_business_licenses():
                logger.info("Successfully processed business license data")

            # Process retail deficit
            if not self.process_retail_deficit():
                logger.warning("Retail deficit processing failed")
            else:
                logger.info("Successfully processed retail deficit data")

            # Save processing summary
            self._save_processing_summary()

            # Validate processed data integrity
            self.validate_processed_data()

            return True

        except Exception as e:
            return self._extracted_from_process_retail_data_48(
                "Error in data processing pipeline: ", e, False
            )

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
                    df["zip_code"] = (
                        df["contact_1_zipcode"].astype(str).str.strip().str.zfill(5)
                    )
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
                settings.PROCESSED_DATA_DIR / "census_processed.csv"
            )
            current_year = census_data["year"].max()
            census_current = census_data[census_data["year"] == current_year]

            # Calculate retail space from permits
            retail_space = (
                data.groupby(["zip_code", "year"])
                .agg({"retail_permits": "sum", "retail_construction_cost": "sum"})
                .reset_index()
            )

            # Merge with census data
            retail_metrics = retail_space.merge(
                census_current[
                    ["zip_code", "total_population", "median_household_income"]
                ],
                on="zip_code",
                how="outer",
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

    def process_retail_metrics(self, df):
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

    def extract_zip(self, address, community_area=None, ward=None):
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

    def is_valid_chicago_zip(self, zip_code):
        return isinstance(zip_code, str) and re.match(r"^606\d{2}$", zip_code)

    def validate_zip_codes(self, df, zip_col="zip_code"):
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
                    row.get("street_direction", ""),
                    row.get("street_name", ""),
                    row.get("contact_1_city", "Chicago"),
                ]
                address = " ".join([str(x) for x in address_fields if pd.notnull(x)])
                zips = re.findall(r"\b60\d{3}\b", address)
                if zips:
                    logger.warning(f"Recovered ZIP from address: {zips[0]}")
                    return zips[0]
                # Try community_area
                if pd.notnull(row.get("community_area")):
                    ca_zip = f"606{str(int(row['community_area'])).zfill(2)}"
                    logger.warning(f"Recovered ZIP from community_area: {ca_zip}")
                    return ca_zip
                # Try ward
                if pd.notnull(row.get("ward")):
                    ward_zip = f"606{str(int(row['ward'])).zfill(2)}"
                    logger.warning(f"Recovered ZIP from ward: {ward_zip}")
                    return ward_zip
                logger.warning(f"Could not recover ZIP for row: {row}")
                return None

            df[zip_col] = df.apply(recover_zip, axis=1)
            # Re-validate
            valid = df[zip_col].astype(str).str.match(r"^606[0-9]{2}$")
            percent_valid = valid.mean() * 100
            logger.info(
                f"Post-recovery valid Chicago ZIPs: {percent_valid:.2f}% ({valid.sum()} of {len(df)})"
            )
        return df[valid].copy(), df[~valid].copy()

    def validate_processed_data(self):
        files = [
            "data/processed/permits_processed.csv",
            "data/processed/retail_metrics.csv",
            "data/processed/retail_deficit.csv",
        ]
        for file in files:
            df = pd.read_csv(file)
            print(f"\n{file}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if "zip_code" in df.columns:
                valid = df["zip_code"].astype(str).str.match(r"^606[0-9]{2}$")
                print(
                    f"  Valid ZIPs: {valid.sum()} / {len(df)} ({valid.mean()*100:.2f}%)"
                )
            print(f"  Nulls per column:\n{df.isnull().sum()}")

    def enrich_with_retail_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure retail-specific columns are present and populated using available data or proxies.
        If source data is missing, fill with np.nan and log a warning. If all values are zero, log a warning.
        Returns the DataFrame with all required retail columns, and a list of warnings for downstream reporting.
        """
        required = [
            "retail_space",
            "retail_demand",
            "retail_gap",
            "retail_supply",
            "vacancy_rate",
        ]
        warnings = []

        enrichment_rules = {
            "retail_space": {
                "retail_construction_cost": lambda x: x["retail_construction_cost"]
                / 200,
                "retail_permits": lambda x: x["retail_permits"] * 10000,
            },
            "retail_demand": {
                "total_population,median_household_income": lambda x: x[
                    "total_population"
                ]
                * x["median_household_income"]
                * 0.3
            },
            "retail_gap": {
                "retail_demand,retail_supply": lambda x: x["retail_demand"]
                - x["retail_supply"]
            },
            "retail_supply": {"retail_space": lambda x: x["retail_space"] * 300},
            "vacancy_rate": {"default": lambda _: 0.1},
        }

        for col in required:
            if col not in df.columns:
                if msg := self._enrich_column(df, col, enrichment_rules[col]):
                    logging.warning(msg)
                    warnings.append(msg)
            elif (df[col] == 0).all():
                msg = f"All values in {col} are zero. Downstream metrics may be misleading."
                logging.warning(msg)
                warnings.append(msg)

        df.attrs["retail_warnings"] = warnings
        return df

    def _enrich_column(self, df: pd.DataFrame, col: str, rules: dict) -> str:
        """Helper method to enrich a single column based on available data."""
        for required_cols, calculation in rules.items():
            if required_cols == "default":
                df[col] = calculation(df)
                return f"Set {col} to default value"

            required_cols = required_cols.split(",")
            if all(col in df.columns for col in required_cols):
                df[col] = calculation(df)
                return f"Filled missing {col} using {required_cols}"

        df[col] = 0
        return f"Could not enrich {col}, set to 0"

    def enrich_retail_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required retail columns are present and populated using available data or proxies.
        Log a WARNING for each metric that was defaulted or estimated. If all values are zero, log a WARNING.
        """
        required = [
            "retail_space",
            "retail_demand",
            "retail_gap",
            "retail_supply",
            "vacancy_rate",
        ]
        warnings = []

        enrichment_rules = {
            "retail_space": {
                "retail_construction_cost": lambda x: x["retail_construction_cost"]
                / 200,
                "retail_permits": lambda x: x["retail_permits"] * 10000,
                "default": lambda _: 0,
            },
            "retail_demand": {
                "total_population,median_household_income": lambda x: x[
                    "total_population"
                ]
                * x["median_household_income"]
                * 0.3,
                "default": lambda _: 0,
            },
            "retail_gap": {
                "retail_demand,retail_supply": lambda x: x["retail_demand"]
                - x["retail_supply"],
                "default": lambda _: 0,
            },
            "retail_supply": {
                "retail_space": lambda x: x["retail_space"] * 300,
                "default": lambda _: 0,
            },
            "vacancy_rate": {"default": lambda _: 0.1},
        }

        for col in required:
            if col not in df.columns:
                if msg := self._enrich_column(df, col, enrichment_rules[col]):
                    logger.warning(msg)
                    warnings.append(msg)
            elif (df[col] == 0).all():
                msg = f"All values in {col} are zero. Downstream metrics may be misleading."
                logger.warning(msg)
                warnings.append(msg)

        df.attrs["retail_warnings"] = warnings
        return df

    def save_outputs(self, merged_df: pd.DataFrame):
        # Only keep valid Chicago ZIPs
        merged_df = merged_df[merged_df["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
        # Add/ensure data_status column
        required_cols = ["retail_space", "retail_demand", "retail_gap", "retail_supply"]

        def status_row(row):
            for col in required_cols:
                if col in row and (pd.isna(row[col]) or row[col] == 0):
                    return "insufficient data"
            return "ok"

        merged_df["data_status"] = merged_df.apply(status_row, axis=1)
        # Write outputs
        merged_df.to_csv(
            settings.PROCESSED_DATA_DIR / "merged_dataset.csv", index=False
        )
        # ... repeat for all other outputs as needed ...
