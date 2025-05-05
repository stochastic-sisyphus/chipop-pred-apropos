from pathlib import Path
import logging

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
    def generate_report(self, output_path: Path = None) -> bool:
        try:
            # ... existing code to load and prepare data ...
            data = self.load_and_prepare_data()
            # Log available columns before rendering
            for k, v in data.items():
                if hasattr(v, "columns"):
                    logging.info(f"ChicagoZipSummaryReport: {k} columns: {list(v.columns)}")
            context = self.analyze_zip_summary(data)
            logging.info(f"ChicagoZipSummaryReport: context keys: {list(context.keys())}")
            # Check for required keys/columns and add warnings if missing or all zero
            missing = []
            all_zero = []
            for col in REQUIRED_COLS:
                val = context.get(col)
                if val is None:
                    missing.append(col)
                    context[col] = 0
                elif isinstance(val, (int, float)) and val == 0:
                    all_zero.append(col)
            if missing:
                context["warnings"] = context.get("warnings", []) + [
                    f"Missing columns: {', '.join(missing)}"
                ]
                logging.warning(f"ChicagoZipSummaryReport: Missing columns: {missing}")
            if all_zero:
                context["warnings"] = context.get("warnings", []) + [
                    f"All zero columns: {', '.join(all_zero)}"
                ]
                logging.warning(f"ChicagoZipSummaryReport: All zero columns: {all_zero}")
            context["missing_or_defaulted"] = missing + all_zero
            # ... existing code to render template ...
        except Exception as e:
            logging.error(f"Failed to generate chicago zip summary report: {e}")
            return False
