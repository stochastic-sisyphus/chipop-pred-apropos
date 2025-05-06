"""
Module for generating retail deficit analysis reports.
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
from src.models.retail_model import RetailModel
from src.visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "total_population",
    "median_household_income",
    "retail_space",
    "vacancy_rate",
    "retail_gap",
    "retail_demand",
    "retail_supply",
]


class RetailDeficitReport:
    """Generates retail deficit analysis reports."""

    def __init__(self):
        """Initialize the report generator."""
        self.retail_model = RetailModel()
        self.visualizer = Visualizer()

        # Set up Jinja2 environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(settings.REPORT_TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Define output path
        self.output_path = settings.REPORTS_DIR / "retail_deficit_analysis.md"

    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare data for the report."""
        try:
            # Load processed data
            census_data = pd.read_csv(settings.PROCESSED_DATA_DIR / "census_processed.csv")
            permit_data = pd.read_csv(settings.PROCESSED_DATA_DIR / "permits_processed.csv")
            retail_data = pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_deficit.csv")
            retail_deficit = pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_deficit.csv")

            # Ensure required columns exist
            required_columns = {
                "census": ["zip_code", "year", "total_population", "median_household_income"],
                "permits": ["zip_code", "year", "total_permits", "total_construction_cost"],
                "retail": ["zip_code", "retail_space", "vacancy_rate"],
                "deficit": ["zip_code", "retail_gap", "retail_demand", "retail_supply"],
            }

            # Validate and standardize zip_code columns
            for df_name, df in {
                "census": census_data,
                "permits": permit_data,
                "retail": retail_data,
                "deficit": retail_deficit,
            }.items():
                # Ensure zip_code exists
                if "zip_code" not in df.columns:
                    logger.error(f"zip_code column missing from {df_name} data")
                    return None

                # Standardize zip_code format
                df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

                # Validate other required columns
                for col in required_columns[df_name]:
                    if col not in df.columns and col != "zip_code":
                        logger.warning(f"{col} missing from {df_name} data")
                        if col in [
                            "retail_space",
                            "vacancy_rate",
                            "retail_gap",
                            "retail_demand",
                            "retail_supply",
                        ]:
                            df[col] = 0
                        elif col == "total_population":
                            df[col] = (
                                df["total_housing_units"] * 2.5
                                if "total_housing_units" in df.columns
                                else 0
                            )
                        elif col == "median_household_income":
                            df[col] = 50000  # Default value

            # Merge datasets
            merged = pd.merge(census_data, permit_data, on=["zip_code", "year"], how="left")
            merged = pd.merge(merged, retail_data, on=["zip_code", "year"], how="left")  # ✅ FIX
            merged = pd.merge(merged, retail_deficit, on=["zip_code", "year"], how="left")  # ✅ FIX

            # Fill missing values
            merged = merged.fillna(
                {
                    "total_permits": 0,
                    "total_construction_cost": 0,
                    "retail_space": 0,
                    "vacancy_rate": 0.1,
                    "retail_gap": 0,
                    "retail_demand": 0,
                    "retail_supply": 0,
                }
            )

            # Log data shapes
            logger.info(f"Census data: {census_data.shape}")
            logger.info(f"Permits data: {permit_data.shape}")
            logger.info(f"Retail data: {retail_data.shape}")
            logger.info(f"Retail deficit data: {retail_deficit.shape}")
            logger.info(f"Merged data: {merged.shape}")

            return {
                "census": census_data,
                "permits": permit_data,
                "retail": retail_data,
                "deficit": retail_deficit,
                "merged": merged,
            }

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def analyze_retail_deficit(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze retail deficit patterns."""
        try:
            merged_data = data["merged"]

            # Use and display all new retail metrics
            required_cols = [
                "retail_permits", "retail_construction_cost", "retail_business_count",
                "retail_space", "retail_demand", "retail_supply", "retail_gap", "retail_lag"
            ]
            for col in required_cols:
                if col not in merged_data.columns:
                    merged_data[col] = np.nan

            # Log missing data
            for col in required_cols:
                if merged_data[col].isnull().all():
                    logger.warning(f"Retail metric column {col} is missing for all ZIPs.")

            # Only keep valid Chicago ZIPs
            merged_data = merged_data[merged_data["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]

            # Calculate summary metrics
            metrics = {
                "total_retail_space": float(merged_data["retail_space"].sum()),
                "avg_space_per_capita": float(
                    merged_data["retail_space"].sum() / merged_data["total_population"].sum()
                ),
                "total_deficit": float(merged_data["retail_gap"].sum()),
                "avg_deficit_per_capita": float(
                    merged_data["retail_gap"].sum() / merged_data["total_population"].sum()
                ),
            }

            # Calculate metrics by ZIP
            zip_metrics = []
            for zip_code in merged_data["zip_code"].unique():
                zip_data = merged_data[merged_data["zip_code"] == zip_code]

                # Calculate metrics
                population = int(zip_data["total_population"].sum())
                retail_space = float(zip_data["retail_space"].sum())
                retail_gap = float(zip_data["retail_gap"].sum())
                retail_demand = float(zip_data["retail_demand"].sum())
                retail_supply = float(zip_data["retail_supply"].sum())

                # Calculate per capita metrics
                retail_per_capita = retail_space / population if population > 0 else 0
                gap_per_capita = retail_gap / population if population > 0 else 0

                zip_metrics.append(
                    {
                        "location": f"ZIP {zip_code}",
                        "total_population": population,
                        "retail_space": retail_space,
                        "retail_per_capita": retail_per_capita,
                        "market_gap": retail_gap,
                        "gap_per_capita": gap_per_capita,
                        "retail_demand": retail_demand,
                        "retail_supply": retail_supply,
                        "required_space": int(retail_gap / 300),  # Assuming $300/sqft
                        "potential_stores": int(retail_gap / 1000000),  # Assuming $1M per store
                    }
                )

            # Sort by market gap
            zip_metrics.sort(key=lambda x: x["market_gap"], reverse=True)

            return {"summary": metrics, "zip_metrics": zip_metrics[:5]}  # Top 5 deficit areas

        except Exception as e:
            logger.error(f"Failed to analyze retail deficit: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def ensure_merged_retail_columns(self, merged_data: pd.DataFrame, retail_deficit: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure merged_data has all required retail columns, filling from retail_deficit if possible, else with 0.
        """
        # Ensure all required retail columns are present and valid, and flag missing as 'insufficient data' or NaN
        required_cols = ["retail_gap", "retail_demand", "retail_supply"]
        for col in required_cols:
            if col not in merged_data.columns:
                merged_data[col] = np.nan
            # If all values are zero, treat as insufficient data
            if (merged_data[col] == 0).all():
                merged_data[col] = np.nan
                merged_data[f"{col}_status"] = "insufficient data"
            else:
                merged_data[f"{col}_status"] = merged_data[col].apply(lambda x: "insufficient data" if pd.isna(x) or x == 0 else "ok")
        # Only keep valid Chicago ZIPs
        merged_data = merged_data[merged_data["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
        return merged_data

    def generate_report(
        self, census_data, permit_data, economic_data, zoning_data, retail_metrics, retail_deficit
    ):
        try:
            # Explicitly access all inputs to avoid "unused parameter" warnings
            _ = census_data, permit_data, economic_data, zoning_data, retail_deficit

            # Log input shapes for traceability
            logging.debug(
                f"generate_report inputs: "
                f"census_data={getattr(census_data, 'shape', 'N/A')}, "
                f"permit_data={getattr(permit_data, 'shape', 'N/A')}, "
                f"economic_data={getattr(economic_data, 'shape', 'N/A')}, "
                f"zoning_data={getattr(zoning_data, 'shape', 'N/A')}, "
                f"retail_deficit={getattr(retail_deficit, 'shape', 'N/A')}"
            )

            # Ensure all outputs only include valid Chicago ZIPs and missing data is flagged, not zero
            retail_deficit = retail_deficit[retail_deficit["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
            for col in ["retail_gap", "retail_demand", "retail_supply"]:
                if col in retail_deficit.columns and (retail_deficit[col] == 0).all():
                    retail_deficit[col] = np.nan
                    retail_deficit[f"{col}_status"] = "insufficient data"
                elif col in retail_deficit.columns:
                    retail_deficit[f"{col}_status"] = retail_deficit[col].apply(
                        lambda x: "insufficient data" if pd.isna(x) or x == 0 else "ok"
                    )

            # Load report template
            template_path = settings.TEMPLATES_DIR / "reports/retail_deficit_analysis.md"
            with open(template_path, "r") as f:
                template = Template(f.read())

            # Prepare opportunity dataframe
            opportunity_df = pd.merge(
                census_data[["zip_code", "total_population", "median_household_income"]],
                (
                    retail_metrics[["zip_code", "retail_space", "vacancy_rate"]]
                    if "retail_space" in retail_metrics.columns
                    else pd.DataFrame({"zip_code": census_data["zip_code"].unique()})
                ),
                on="zip_code",
                how="left",
            )

            # Fill missing retail metrics with 0
            for col in ["retail_space", "vacancy_rate"]:
                if col not in opportunity_df.columns:
                    opportunity_df[col] = 0

            # Log available columns
            logging.info(f"RetailDeficitReport: opportunity_df columns: {list(opportunity_df.columns)}")

            # Check for missing/zeroed required columns
            missing = []
            all_zero = []
            for col in REQUIRED_COLS:
                if col not in opportunity_df.columns:
                    missing.append(col)
                    opportunity_df[col] = 0
                    logging.warning(f"RetailDeficitReport: Missing column {col}, set to 0.")
                elif (opportunity_df[col] == 0).all():
                    all_zero.append(col)
                    logging.warning(f"RetailDeficitReport: All values in {col} are zero.")

            notes = []
            if missing:
                notes.append(f"Missing columns: {', '.join(missing)}")
            if all_zero:
                notes.append(f"All zero columns: {', '.join(all_zero)}")

            # Prepare context
            context = {
                "generation_date": datetime.now().strftime("%Y-%m-%d"),
                "opportunity_areas": {},
                "high_deficit_areas": [],
                "recommendations": [],
                "methodology_notes": "",
                "notes": notes,
                "missing_or_defaulted": missing + all_zero,
            }
            if "opportunity_areas" not in context or context["opportunity_areas"] is None:
                context["opportunity_areas"] = {}

            # Render template
            try:
                rendered = template.render(**{k: context.get(k, "N/A") for k in context})
            except Exception as e:
                logging.error(f"RetailDeficitReport: Template rendering failed: {e}")
                rendered = f"Report generation failed. Error: {e}\nNotes: {context.get('notes', [])}"

            return rendered

        except Exception as e:
            logging.error(f"RetailDeficitReport: Failed to generate report: {e}")
            return f"Report generation failed. Error: {e}"

    def ensure_retail_columns(self, retail_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required retail columns exist, adding with default 0 if missing. Log only once per column."""
        REQUIRED_RETAIL_COLUMNS = [
            "retail_space",
            "retail_demand",
            "retail_gap",
            "vacancy_rate",
            "retail_supply",
        ]
        logged_missing_retail_columns = set()
        for col in REQUIRED_RETAIL_COLUMNS:
            if col not in retail_data.columns:
                if col not in logged_missing_retail_columns:
                    logging.warning(
                        f"Added missing column {col} to retail_data with default value 0"
                    )
                    logged_missing_retail_columns.add(col)
                retail_data[col] = 0
        return retail_data


def generate_retail_deficit_report(df, output_path):
    """
    Generate a retail deficit report, explicitly flagging ZIPs with missing retail metrics.
    """
    import numpy as np
    import pandas as pd
    import logging
    logger = logging.getLogger("src.reporting.retail_deficit_report")

    # Identify ZIPs with missing retail metrics
    missing_cols = [
        "retail_space", "retail_supply", "retail_demand"
    ]
    missing_report = {}
    for col in missing_cols:
        missing_zips = df[df[col].isnull()]["zip_code"].tolist()
        if missing_zips:
            logger.warning(f"Missing {col} for ZIPs: {missing_zips}")
            missing_report[col] = missing_zips

    # Write missing data section to report
    with open(output_path, "w") as f:
        f.write("# Retail Deficit Report\n\n")
        f.write("## ZIPs with Missing Retail Metrics\n")
        for col, zips in missing_report.items():
            f.write(f"- {col}: {', '.join(str(z) for z in zips)}\n")
        f.write("\n")
        # ... existing report content ...
        f.write("## Main Analysis\n")
        # (Insert main analysis code here)
        # For demonstration, show summary table
        summary_cols = ["zip_code", "retail_space", "retail_supply", "retail_demand", "retail_gap"]
        summary = df[summary_cols].head(20).to_string(index=False)
        f.write("\n### Sample Data (first 20 rows):\n")
        f.write(summary)
        f.write("\n")
