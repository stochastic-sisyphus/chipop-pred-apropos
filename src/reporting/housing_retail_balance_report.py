"""
Module for generating housing-retail balance analysis reports.
"""

import pandas as pd
from typing import Dict, Any
import logging
import jinja2
from datetime import datetime
from jinja2 import Template
import traceback
import numpy as np

from src.config import settings
from src.models.housing_model import HousingModel
from src.models.retail_model import RetailModel
from src.visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "total_population",
    "total_housing_units",
    "retail_space",
    "retail_demand",
    "retail_gap",
    "retail_supply",
    "vacancy_rate",
]


class HousingRetailBalanceReport:
    """Generates housing-retail balance analysis reports."""

    def __init__(self):
        """Initialize the housing-retail balance report generator."""
        self.housing_model = HousingModel()
        self.retail_model = RetailModel()
        self.visualizer = Visualizer()

        # Set up Jinja2 environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(settings.REPORT_TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Define output path
        self.output_path = settings.REPORTS_DIR / "housing_retail_balance_report.md"
        self.census_data_path = settings.PROCESSED_DATA_DIR / "census_processed.csv"
        self.permits_data_path = settings.PROCESSED_DATA_DIR / "permits_processed.csv"
        self.retail_data_path = settings.PROCESSED_DATA_DIR / "retail_deficit.csv"

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        try:
            # Load data
            census_data = pd.read_csv(self.census_data_path)
            permits_data = pd.read_csv(self.permits_data_path)
            retail_data = pd.read_csv(self.retail_data_path)

            # Ensure ZIP codes are strings
            census_data["zip_code"] = census_data["zip_code"].astype(str).str.zfill(5)
            permits_data["zip_code"] = permits_data["zip_code"].astype(str).str.zfill(5)
            retail_data["zip_code"] = retail_data["zip_code"].astype(str).str.zfill(5)

            # Get most recent year
            current_year = census_data["year"].max()

            # Filter for current year
            census_current = census_data[census_data["year"] == current_year].copy()
            permits_current = permits_data[permits_data["year"] == current_year].copy()
            retail_current = retail_data[retail_data["year"] == current_year].copy()

            # Ensure required columns exist
            required_columns = {
                "census": ["total_population", "total_housing_units"],
                "retail": ["retail_space", "retail_demand", "retail_gap"],
            }

            for col in required_columns["census"]:
                if col not in census_current.columns:
                    logger.error(f"Missing required census column: {col}")
                    return None

            for col in required_columns["retail"]:
                if col not in retail_current.columns:
                    logger.warning(f"Missing required retail column: {col}")
                    retail_current[col] = 0

            # Merge data
            merged = census_current.merge(
                permits_current, on="zip_code", how="left", suffixes=("", "_permits")
            )
            merged = merged.merge(
                retail_current, on="zip_code", how="left", suffixes=("", "_retail")
            )

            # Fill missing values
            merged = merged.fillna(
                {
                    "total_population": merged["total_population"].mean(),
                    "total_housing_units": merged["total_housing_units"].mean(),
                    "retail_space": 0,
                    "retail_demand": 0,
                    "retail_gap": 0,
                }
            )

            # Calculate per capita metrics
            merged["retail_per_capita"] = merged["retail_space"] / merged["total_population"]
            merged["housing_per_capita"] = (
                merged["total_housing_units"] / merged["total_population"]
            )

            # Calculate balance score (0 = perfect balance, higher = more imbalanced)
            merged["balance_score"] = abs(
                merged["retail_per_capita"] - merged["housing_per_capita"]
            )

            # Determine balance categories
            merged["balance_category"] = pd.cut(
                merged["balance_score"],
                bins=[-float("inf"), 0.1, 0.3, 0.5, float("inf")],
                labels=[
                    "Balanced",
                    "Slightly Imbalanced",
                    "Moderately Imbalanced",
                    "Severely Imbalanced",
                ],
            )

            return merged

        except Exception as e:
            logger.error(f"Error loading and preparing data: {str(e)}")
            return None

    def analyze_balance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze housing-retail balance."""
        try:
            # Use and display all new retail metrics
            required_cols = [
                "retail_permits", "retail_construction_cost", "retail_business_count",
                "retail_space", "retail_demand", "retail_supply", "retail_gap", "retail_lag"
            ]
            for col in required_cols:
                if col not in data.columns:
                    data[col] = np.nan
            # Log missing data
            for col in required_cols:
                if data[col].isnull().all():
                    logger.warning(f"Retail metric column {col} is missing for all ZIPs.")
            # Only keep valid Chicago ZIPs
            data = data[data["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
            return data

        except Exception as e:
            logger.error(f"Failed to analyze housing-retail balance: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def generate_report(
        self, census_data, permit_data, economic_data, zoning_data, retail_metrics, retail_deficit
    ):
        try:
            # Ensure all outputs only include valid Chicago ZIPs and missing data is flagged, not zero
            retail_metrics = retail_metrics[retail_metrics["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
            for col in ["retail_gap", "retail_demand", "retail_supply", "retail_space", "vacancy_rate"]:
                if col in retail_metrics.columns and (retail_metrics[col] == 0).all():
                    retail_metrics[col] = np.nan
                    retail_metrics[f"{col}_status"] = "insufficient data"
                elif col in retail_metrics.columns:
                    retail_metrics[f"{col}_status"] = retail_metrics[col].apply(lambda x: "insufficient data" if pd.isna(x) or x == 0 else "ok")
            # Load report template
            template_path = settings.TEMPLATES_DIR / "reports/housing_retail_balance_report.md"
            with open(template_path, "r") as f:
                template = Template(f.read())
            # Merge and prepare data
            merged = census_data.merge(
                retail_metrics, on="zip_code", how="left", suffixes=("", "_retail")
            )
            # Log available columns
            logging.info(f"HousingRetailBalanceReport: merged columns: {list(merged.columns)}")
            # Check for missing/zeroed required columns
            missing = []
            all_zero = []
            for col in REQUIRED_COLS:
                if col not in merged.columns:
                    missing.append(col)
                    merged[col] = 0
                    logging.warning(f"HousingRetailBalanceReport: Missing column {col}, set to 0.")
                elif (merged[col] == 0).all():
                    all_zero.append(col)
                    logging.warning(f"HousingRetailBalanceReport: All values in {col} are zero.")
            notes = []
            if missing:
                notes.append(f"Missing columns: {', '.join(missing)}")
            if all_zero:
                notes.append(f"All zero columns: {', '.join(all_zero)}")
            # Calculate metrics
            metrics = {
                "total_population": int(merged["total_population"].sum()),
                "total_housing_units": int(merged["total_housing_units"].sum()),
                "total_retail_space": float(merged["retail_space"].sum()),
                "avg_retail_density": (
                    float(merged["retail_space"].sum() / merged["total_population"].sum())
                    if merged["total_population"].sum()
                    else 0
                ),
                "avg_housing_density": (
                    float(merged["total_housing_units"].sum() / merged["total_population"].sum())
                    if merged["total_population"].sum()
                    else 0
                ),
                "retail_vacancy_rate": float(merged["vacancy_rate"].mean()),
            }
            # Prepare context
            context = {
                "generation_date": datetime.now().strftime("%Y-%m-%d"),
                "summary": metrics,
                "notes": notes,
                "missing_or_defaulted": missing + all_zero,
            }
            # Render template with .get() and missing/defaulted block
            try:
                rendered = template.render(**{k: context.get(k, "N/A") for k in context})
            except Exception as e:
                logging.error(f"HousingRetailBalanceReport: Template rendering failed: {e}")
                rendered = (
                    f"Report generation failed. Error: {e}\nNotes: {context.get('notes', [])}"
                )
            return rendered
        except Exception as e:
            logging.error(f"HousingRetailBalanceReport: Failed to generate report: {e}")
            return f"Report generation failed. Error: {e}"


def generate_report(
    census_data, permit_data, economic_data, zoning_data, retail_metrics, retail_deficit
):
    """Generate housing retail balance report."""
    try:
        # Load report template
        template_path = settings.TEMPLATES_DIR / "reports/housing_retail_balance_report.md"
        with open(template_path, "r") as f:
            template = Template(f.read())

        # Calculate balance metrics
        metrics = {
            "total_housing_units": int(census_data["total_housing_units"].sum()),
            "occupied_units": int(census_data["occupied_housing_units"].sum()),
            "vacant_units": int(census_data["vacant_housing_units"].sum()),
            "total_population": int(census_data["total_population"].sum()),
            "total_retail_space": (
                retail_metrics["retail_space"].sum()
                if "retail_space" in retail_metrics.columns
                else 0
            ),
            "retail_per_capita": (
                retail_metrics["retail_space"].sum() / census_data["total_population"].sum()
                if "retail_space" in retail_metrics.columns
                else 0
            ),
            "retail_vacancy_rate": (
                retail_metrics["vacancy_rate"].mean()
                if "vacancy_rate" in retail_metrics.columns
                else 0
            ),
        }

        # Calculate balance scores
        if "retail_space" in retail_metrics.columns:
            retail_metrics["balance_score"] = retail_metrics.apply(
                lambda row: min(
                    row["retail_space"]
                    / (row["total_population"] * 20),  # 20 sq ft per person benchmark
                    1.0,
                ),
                axis=1,
            )
        else:
            retail_metrics["balance_score"] = 0

        # Identify imbalanced areas
        imbalanced_areas = []
        if "balance_score" in retail_metrics.columns:
            imbalanced_areas = (
                retail_metrics[retail_metrics["balance_score"] < 0.5]
                .sort_values("balance_score")
                .to_dict("records")
            )

        # Identify ZIPs with missing retail metrics
        missing_cols = [
            "retail_space", "retail_supply", "retail_demand"
        ]
        missing_report = {}
        for col in missing_cols:
            missing_zips = retail_metrics[retail_metrics[col].isnull()]["zip_code"].tolist()
            if missing_zips:
                logger.warning(f"Missing {col} for ZIPs: {missing_zips}")
                missing_report[col] = missing_zips

        return template.render(
            generation_date=datetime.now().strftime("%Y-%m-%d"),
            current_analysis={
                "housing": {
                    "total_units": metrics["total_housing_units"],
                    "density": metrics["total_housing_units"] / metrics["total_population"],
                    "pipeline_units": (
                        permit_data["residential_permits"].sum()
                        if "residential_permits" in permit_data.columns
                        else 0
                    ),
                },
                "retail": {
                    "total_space": metrics["total_retail_space"],
                    "per_capita": metrics["retail_per_capita"],
                    "vacancy_rate": metrics["retail_vacancy_rate"],
                },
            },
            analysis_results={
                "imbalance_areas": imbalanced_areas,
                "development_patterns": [],
                "mixed_use_opportunities": [],
            },
            recommendations={"development": [], "policy": []},
            missing_data=missing_report,
        )
    except Exception as e:
        logger.error(f"Failed to generate housing retail balance report: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
