from pathlib import Path
import logging
import pandas as pd
from typing import Dict, Any, Optional # Added Optional, Dict, Any
import jinja2 # Added jinja2 import
from src.config import settings # Assuming settings has MERGED_DATA_PATH

logger = logging.getLogger(__name__) # Added logger definition

REQUIRED_COLS = [
    "total_population",
    "median_household_income",
    "median_home_value",
    "labor_force",
    "total_housing_units",
    "retail_space",
    "retail_demand",
    "retail_gap",
    "retail_supply",
    "vacancy_rate",
]


class ChicagoZipSummaryReport:
    def __init__(self):
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(settings.REPORT_TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.output_path = settings.REPORTS_DIR / "chicago_zip_summary.md"

    def load_and_prepare_data(self, processed_data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[pd.DataFrame]:
        """Loads and prepares the merged dataset for the summary report."""
        try:
            if processed_data_dict and 'merged_data' in processed_data_dict:
                df = processed_data_dict['merged_data']
                logger.info("Using merged_data from provided dictionary for ChicagoZipSummaryReport.")
            elif settings.MERGED_DATA_PATH.exists():
                df = pd.read_csv(settings.MERGED_DATA_PATH)
                logger.info(f"Loaded merged_dataset.csv for ChicagoZipSummaryReport from {settings.MERGED_DATA_PATH}")
            else:
                logger.error(f"merged_dataset.csv not found at {settings.MERGED_DATA_PATH}")
                return None
            
            if df.empty:
                logger.warning("Merged data is empty.")
                return None
            
            # Basic validation or preparation can go here
            # For example, ensuring 'zip_code' and 'year' exist
            if 'zip_code' not in df.columns or 'year' not in df.columns:
                logger.error("merged_dataset.csv is missing 'zip_code' or 'year' column.")
                return None

            return df
        except Exception as e:
            logger.error(f"Error loading/preparing data for ChicagoZipSummaryReport: {e}")
            return None

    def generate_report(self, processed_data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        try:
            # ... existing code to load and prepare data ...
            data_df = self.load_and_prepare_data(processed_data_dict)
            if data_df is None:
                logger.error("Failed to load data for Chicago ZIP Summary report.")
                return None

            # Log available columns before rendering
            logging.info(f"ChicagoZipSummaryReport: data_df columns: {list(data_df.columns)}")
            
            # Analyze data to build the content for the 'summary' key
            summary_content = self.analyze_zip_summary(data_df) # This returns a dict like {"zip_summary_data": ..., "notes": ...}
            
            # Structure the context as expected by the template
            context = {
                "summary": summary_content # Nest the analysis results under 'summary'
            }

            logging.info(f"ChicagoZipSummaryReport: context keys: {list(context.keys())}")
            # Check for required keys/columns and add warnings if missing or all zero
            missing = []
            all_zero = []
            for col in REQUIRED_COLS:
                # Check if column exists in the aggregated last year data
                # This logic might need adjustment based on how context is truly built by analyze_zip_summary
                if not any(col in record for record in context.get("summary", {}).get("zip_summary_data", [])):
                    missing.append(col)
                    # Add a placeholder to context if needed by template, or handle in template
                elif all(record.get(col, 0) == 0 for record in context.get("summary", {}).get("zip_summary_data", [])):
                    all_zero.append(col)

            if missing:
                # Add warnings to the 'summary' part of the context if that's where the template expects them
                context["summary"]["warnings"] = context["summary"].get("warnings", []) + [
                    f"Missing columns in aggregated data: {', '.join(missing)}"
                ]
                logging.warning(f"ChicagoZipSummaryReport: Missing columns in aggregated data: {missing}")
            if all_zero:
                context["summary"]["warnings"] = context["summary"].get("warnings", []) + [
                    f"All zero columns in aggregated data: {', '.join(all_zero)}"
                ]
                logging.warning(f"ChicagoZipSummaryReport: All zero columns in aggregated data: {all_zero}")
            
            # Ensure missing_or_defaulted is also part of the summary if template expects it there
            context["summary"]["missing_or_defaulted"] = missing + all_zero
            
            # Render template (assuming a template exists)
            template = self.template_env.get_template("chicago_zip_summary.md") # Ensure this template exists
            rendered_report = template.render(context)

            output_path = self.output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(rendered_report)
            logger.info(f"Chicago ZIP Summary Report generated at {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate chicago zip summary report: {e}")
            return False

    def analyze_zip_summary(self, data_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes the data to create a summary for each ZIP code."""
        if data_df is None or data_df.empty:
            return {"zip_summary_data": [], "notes": ["No data to analyze."]}

        # Example: Get the latest data for each zip code
        latest_data_by_zip = data_df.loc[data_df.groupby('zip_code')['year'].idxmax()]

        summary_list = []
        for _, row in latest_data_by_zip.iterrows():
            summary_item = {"zip_code": row.get("zip_code")}
            for col in REQUIRED_COLS: # Use the defined REQUIRED_COLS
                summary_item[col] = row.get(col, 0) # Default to 0 if missing
            summary_list.append(summary_item)
        
        return {"zip_summary_data": summary_list}

# The following block was outside the class, causing indentation and scope errors.
# It seems like an attempt to call the generate_report method.
# This should typically be done from your main script or a runner, not at the module level.
# I'm commenting it out as its placement is incorrect.
# If you intend to run this report directly for testing, you'd do something like:
# if __name__ == "__main__":
#     report_generator = ChicagoZipSummaryReport()
#     report_generator.generate_report() # Potentially pass data if needed

# try:
#     # ... existing code to load and prepare data ...
#     data = self.load_and_prepare_data() # 'self' is not defined here
#     # Log available columns before rendering
#     for k, v in data.items():
#         if hasattr(v, "columns"):
#             logging.info(f"ChicagoZipSummaryReport: {k} columns: {list(v.columns)}")
#     context = self.analyze_zip_summary(data) # 'self' is not defined here
#     logging.info(f"ChicagoZipSummaryReport: context keys: {list(context.keys())}")
#     # Check for required keys/columns and add warnings if missing or all zero
#     missing = []
#     all_zero = []
#     for col in REQUIRED_COLS:
#         val = context.get(col)
#         if val is None:
#             missing.append(col)
#             context[col] = 0
#         elif isinstance(val, (int, float)) and val == 0:
#             all_zero.append(col)
#     if missing:
#         context["warnings"] = context.get("warnings", []) + [
#             f"Missing columns: {', '.join(missing)}"
#         ]
#         logging.warning(f"ChicagoZipSummaryReport: Missing columns: {missing}")
#     if all_zero:
#         context["warnings"] = context.get("warnings", []) + [
#             f"All zero columns: {', '.join(all_zero)}"
#         ]
#         logging.warning(f"ChicagoZipSummaryReport: All zero columns: {all_zero}")
#     context["missing_or_defaulted"] = missing + all_zero
#     # ... existing code to render template ...
# except Exception as e:
#     logging.error(f"Failed to generate chicago zip summary report: {e}")
#     return False # 'return' outside function
