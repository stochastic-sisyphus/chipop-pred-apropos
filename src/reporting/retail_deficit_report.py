"""
Module for generating retail deficit analysis reports.
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging
import jinja2
from datetime import datetime
from jinja2 import Template
import traceback
import numpy as np

from src.config import settings
from src.models.retail_model import RetailModel
from src.visualization.visualizer import Visualizer
from src.utils.helpers import clean_zip

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

    def load_and_prepare_data(self, processed_data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[pd.DataFrame]:
        """Load and prepare data for the report."""
        try:
            retail_deficit_df = None
            if processed_data_dict and 'retail_deficit' in processed_data_dict:
                retail_deficit_df = processed_data_dict['retail_deficit']
                if isinstance(retail_deficit_df, pd.DataFrame) and not retail_deficit_df.empty:
                    logger.info("Using 'retail_deficit' from processed_data_dict for RetailDeficitReport.")
                else:
                    logger.warning("'retail_deficit' in processed_data_dict is not a valid DataFrame or is empty.")
                    retail_deficit_df = None # Reset if invalid

            if retail_deficit_df is None: # Fallback to loading from file
                deficit_path = settings.PROCESSED_DATA_DIR / "retail_deficit.csv"
                if deficit_path.exists():
                    retail_deficit_df = pd.read_csv(deficit_path, dtype={'zip_code': str})
                    logger.info(f"Loaded retail_deficit.csv from {deficit_path} for RetailDeficitReport.")
                else:
                    logger.error(f"retail_deficit.csv not found at {deficit_path} and not in processed_data_dict.")
                    return None
            
            if retail_deficit_df.empty:
                logger.warning("Retail deficit data is empty.")
                return None

            if "zip_code" not in retail_deficit_df.columns:
                logger.error("Retail deficit data is missing 'zip_code' column.")
                return None
            retail_deficit_df["zip_code"] = retail_deficit_df["zip_code"].astype(str).apply(clean_zip)

            # Ensure required columns for analysis are present, fill with 0 or NaN if missing
            # This step might be less critical if DataProcessor ensures these columns
            for col in ["retail_gap", "retail_demand", "retail_supply", "total_population", "retail_space"]:
                if col not in retail_deficit_df.columns:
                    logger.warning(f"Column '{col}' missing in retail_deficit_df. Will be filled with NaN/0.")
                    retail_deficit_df[col] = np.nan # Or 0, depending on downstream expectation

            return retail_deficit_df

        except Exception as e:
            return self._extracted_from_analyze_retail_deficit_88(
                'Error loading data: ', e
            )

    def analyze_retail_deficit(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze retail deficit patterns."""
        # This method now expects a single DataFrame (retail_deficit_df)
        # The 'data' parameter from the original was a dict, now it should be the DataFrame itself.
        merged_data = data # Assuming 'data' is the retail_deficit_df
        try:
            # merged_data = data["merged"] # Old way

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

            # Ensure necessary columns for calculation are numeric and handle NaNs
            merged_data["retail_space"] = pd.to_numeric(merged_data["retail_space"], errors='coerce').fillna(0)
            merged_data["total_population"] = pd.to_numeric(merged_data["total_population"], errors='coerce').fillna(0)
            merged_data["retail_gap"] = pd.to_numeric(merged_data["retail_gap"], errors='coerce').fillna(0)

            # Calculate summary metrics
            metrics = {
                "total_retail_space": float(merged_data["retail_space"].sum()),
                "avg_space_per_capita": float(
                    merged_data["retail_space"].sum() / merged_data["total_population"].sum() if merged_data["total_population"].sum() != 0 else 0
                ),
                "total_deficit": float(merged_data["retail_gap"].sum()),
                "avg_deficit_per_capita": float(
                    merged_data["retail_gap"].sum() / merged_data["total_population"].sum() if merged_data["total_population"].sum() != 0 else 0
                ),
            }

            # Calculate metrics by ZIP
            zip_metrics = []
            # Ensure 'zip_code' is present before trying to access unique values
            for zip_code in merged_data["zip_code"].unique():
                zip_data = merged_data[merged_data["zip_code"] == zip_code]

                # Calculate metrics
                population = int(zip_data["total_population"].sum())
                retail_space = float(zip_data["retail_space"].sum())
                retail_gap = float(zip_data["retail_gap"].sum())
                retail_demand = float(zip_data["retail_demand"].sum()) if "retail_demand" in zip_data.columns else 0
                retail_supply = float(zip_data["retail_supply"].sum()) if "retail_supply" in zip_data.columns else 0

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
            return self._extracted_from_analyze_retail_deficit_88(
                'Failed to analyze retail deficit: ', e
            )

    # TODO Rename this here and in `load_and_prepare_data` and `analyze_retail_deficit`
    def _extracted_from_analyze_retail_deficit_88(self, arg0, e):
        logger.error(f"{arg0}{str(e)}")
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

    def _load_template_content(self) -> str:
        """Loads the report template content."""
        template_path = settings.TEMPLATES_DIR / "reports/retail_deficit_analysis.md"
        try:
            with open(template_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"RetailDeficitReport: Template file not found at {template_path}")
            raise
        except Exception as e:
            logger.error(f"RetailDeficitReport: Error loading template {template_path}: {e}")
            raise

    def _prepare_report_context(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the context dictionary for template rendering."""
        opportunity_df_for_check = pd.DataFrame(analysis_results.get("zip_metrics", []))
        logging.info(f"RetailDeficitReport: opportunity_df_for_check columns: {list(opportunity_df_for_check.columns)}")

        missing = []
        all_zero = []
        for col in REQUIRED_COLS: # Class level REQUIRED_COLS
            if not opportunity_df_for_check.empty and col not in opportunity_df_for_check.columns:
                missing.append(col)
                logging.warning(f"RetailDeficitReport: Missing column {col} in zip_metrics for context. Template might need to handle this.")
            elif not opportunity_df_for_check.empty and col in opportunity_df_for_check.columns and \
                     (pd.to_numeric(opportunity_df_for_check[col], errors='coerce').fillna(0) == 0).all():
                all_zero.append(col)
                logging.warning(f"RetailDeficitReport: All values in {col} (in zip_metrics) are zero or NaN.")

        notes = []
        if missing:
            notes.append(f"Missing columns in zip_metrics: {', '.join(missing)}")
        if all_zero:
            notes.append(f"All zero/NaN columns in zip_metrics: {', '.join(all_zero)}")

        return {
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "summary_metrics": analysis_results.get("summary", {}),
            "top_deficit_areas": analysis_results.get("zip_metrics", []),
            "recommendations": [],  # Placeholder, as in original
            "methodology_notes": "",  # Placeholder
            "notes": notes,
            "missing_or_defaulted": missing + all_zero,
        }

    def _render_and_save_report(self, template_content: str, context: Dict[str, Any]) -> str:
        """Renders the report using the template and context, then saves it."""
        try:
            template = Template(template_content)
            # Use .get(k, "N/A") for robustness during rendering
            rendered_report = template.render(**{k: context.get(k, "N/A") for k in context})
        except Exception as e:
            logging.error(f"RetailDeficitReport: Template rendering failed: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            error_notes = context.get('notes', [])
            rendered_report = f"Report generation failed. Error: {e}\nNotes: {error_notes}"
            self._extracted_from__render_and_save_report_13(
                rendered_report, 'Retail Deficit Report (failure state) saved at '
            )
            raise  # Re-raise to indicate failure to the caller

        self._extracted_from__render_and_save_report_13(
            rendered_report, 'Retail Deficit Report generated at '
        )
        return rendered_report

    # TODO Rename this here and in `_render_and_save_report`
    def _extracted_from__render_and_save_report_13(self, rendered_report, arg1):
        # Still try to save the failure report
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write(rendered_report)
        logger.info(f"{arg1}{self.output_path}")

    def generate_report(
        self, processed_data_dict: Dict[str, pd.DataFrame]
    ):
        try:
            retail_deficit_df = self.load_and_prepare_data(processed_data_dict)
            if retail_deficit_df is None or retail_deficit_df.empty:
                logger.error("RetailDeficitReport: Failed to load or prepare retail_deficit data.")
                return "Report generation failed: Missing or empty retail_deficit data."

            # Perform analysis to get data for the template
            analysis_results = self.analyze_retail_deficit(retail_deficit_df)
            if not analysis_results: # analyze_retail_deficit returns None on failure
                logger.error("RetailDeficitReport: Analysis of retail deficit data failed.")
                return "Report generation failed: Analysis returned no results."

            # Log input shapes for traceability
            logging.debug(
                f"RetailDeficitReport generate_report using retail_deficit_df with shape: {retail_deficit_df.shape}"
            )

            template_content = self._load_template_content()
            context = self._prepare_report_context(analysis_results)
            return self._render_and_save_report(template_content, context)
        except FileNotFoundError: # Caught from _load_template_content
             return "Report generation failed: Template file not found."
        except jinja2.exceptions.TemplateError as e: # Caught from _render_and_save_report
            logger.error(f"RetailDeficitReport: Jinja2 template error during report generation: {e}")
            return f"Report generation failed due to a template error: {e}"
        except Exception as e:
            logger.error(f"RetailDeficitReport: An unexpected error occurred during generate_report: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Attempt to write a failure report if not already done by _render_and_save_report
            try:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_path, "w") as f:
                    f.write(f"# Report Generation Failed\n\nError: {e}\n\nTraceback:\n{traceback.format_exc()}")
            except Exception as e_write:
                logger.error(f"Failed to write failure report: {e_write}")
            return f"Report generation failed due to an unexpected error: {e}"

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

    def _generate_opportunity_df(self, retail_deficit_df: pd.DataFrame) -> pd.DataFrame:
        # This method might be obsolete if analyze_retail_deficit prepares the data for the template.
        # If still needed, ensure it selects columns that exist.
        logger.debug(f"RetailDeficitReport._generate_opportunity_df input shape: {retail_deficit_df.shape if retail_deficit_df is not None else 'None'}")
        if retail_deficit_df is not None and not retail_deficit_df.empty:
            cols_to_select = [
                "zip_code", "total_population", "median_household_income",
                "retail_space", "vacancy_rate"
            ]
            if existing_cols := [
                col for col in cols_to_select if col in retail_deficit_df.columns
            ]:
                return retail_deficit_df[existing_cols].copy()
            else:
                return pd.DataFrame()
        return pd.DataFrame(columns=["zip_code"]) # Return empty DataFrame with at least zip_code if input is None/empty
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
        if missing_zips := df[df[col].isnull()]["zip_code"].tolist():
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

def generate_report(merged_df: pd.DataFrame, output_path: str):
    """
    Generate the retail deficit report, flagging ZIPs with insufficient retail data and all-zero columns.
    """
    logger.info("Generating retail deficit report...")
    # Identify ZIPs with insufficient retail data
    required_cols = ["retail_gap", "retail_demand", "retail_supply", "total_housing_units"]
    for col in required_cols:
        if col not in merged_df.columns:
            logger.warning(f"Column '{col}' missing from merged_df. Skipping related analysis.")
        elif (merged_df[col] == 0).all():
            logger.warning(f"All values in {col} are zero. Downstream metrics may be misleading.")
    insufficient_zips = []
    if all(col in merged_df.columns for col in required_cols):
        insufficient_zips = merged_df[
            (merged_df["retail_gap"].isna()) | (merged_df["retail_gap"] == 0) |
            (merged_df["retail_demand"].isna()) | (merged_df["retail_demand"] == 0) |
            (merged_df["retail_supply"].isna()) | (merged_df["retail_supply"] == 0)
        ]["zip_code"].tolist()
        if insufficient_zips:
            logger.warning(f"Insufficient retail data for ZIPs: {insufficient_zips}")
    # Add a column to flag insufficient data
    merged_df["retail_deficit_data_status"] = merged_df["zip_code"].apply(lambda z: "insufficient" if z in insufficient_zips else "ok")
    # Only include ZIPs with sufficient data in main analysis
    sufficient_df = merged_df[merged_df["retail_deficit_data_status"] == "ok"]
    # ... existing report generation logic ...
    # Save report
    sufficient_df.to_csv(output_path, index=False)
    logger.info(f"Saved retail deficit report to {output_path}")
