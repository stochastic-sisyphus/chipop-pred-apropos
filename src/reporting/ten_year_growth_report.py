"""
Module for generating the ten-year growth analysis report for Chicago.
This module analyzes historical data and generates projections to create
a comprehensive report on Chicago's growth patterns and opportunities.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional # Added Optional
import logging
import jinja2
from datetime import datetime
import numpy as np
import traceback # Added import for traceback

from src.data_processing.processor import DataProcessor
from src.models.population_model import PopulationModel
from src.models.retail_model import RetailModel
from src.models.economic_model import EconomicModel
from src.visualization.visualizer import Visualizer
from src.config import settings

logger = logging.getLogger(__name__)

import logging

# PATCH: Import clean_zip for robust ZIP cleaning
from src.utils.helpers import clean_zip

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


class TenYearGrowthReport:
    """Generates the ten-year growth analysis report for Chicago."""

    def __init__(self, data_dir: Path = None):
        """Initialize the report generator with necessary components."""
        self.data_dir = data_dir or settings.DATA_PROCESSED_DIR
        self.df = None
        self.report_data = {}

        self.data_processor = DataProcessor()
        self.population_model = PopulationModel()
        self.retail_model = RetailModel()
        self.economic_model = EconomicModel()
        self.visualizer = Visualizer()

        # Set up Jinja2 environment for templates
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(settings.REPORT_TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Define template and output paths
        self.templates = {
            "main": "ten_year_growth_analysis.md",
            "executive_summary": "EXECUTIVE_SUMMARY.md",
            "chicago_summary": "chicago_zip_summary.md",
            "economic_impact": "economic_impact_analysis.md",
            "housing_retail": "housing_retail_balance_report.md",
            "retail_deficit": "retail_deficit_analysis.md",
            "void_analysis": "void_analysis.md",
        }

        # Create output paths
        self.output_paths = {
            name: settings.REPORTS_DIR / template_file
            for name, template_file in self.templates.items()
        }

    def _validate_zip_codes(self, df: pd.DataFrame, zip_col: str = "zip_code") -> pd.DataFrame:
        """
        Validate and standardize ZIP codes in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe
            zip_col (str): Name of ZIP code column

        Returns:
            pd.DataFrame: Dataframe with validated ZIP codes
        """
        if zip_col not in df.columns:
            logger.error(f"ZIP code column '{zip_col}' not found in dataframe")
            return df

        # Convert to string and pad with zeros, then robustly clean
        df[zip_col] = df[zip_col].astype(str).apply(clean_zip)

        # Filter to valid Chicago ZIP codes
        valid_zips = df[zip_col].isin(settings.CHICAGO_ZIP_CODES)
        invalid_count = (~valid_zips).sum()

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} records with invalid Chicago ZIP codes")
            df = df[valid_zips].copy()

        return df

    def load_and_prepare_data(self, processed_data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """Load and prepare data for analysis."""
        try:
            census_data, permits_data, economic_data = None, None, None

            if processed_data_dict:
                census_data = processed_data_dict.get("census_data")
                permits_data = processed_data_dict.get("permit_data")
                economic_data = processed_data_dict.get("economic_data")
                logger.info("Attempting to use data from processed_data_dict for TenYearGrowthReport.")

            if census_data is None:
                logger.info("Loading census_data from file for TenYearGrowthReport.")
                census_data = pd.read_csv(settings.PROCESSED_DATA_DIR / "census_processed.csv")
            if permits_data is None:
                logger.info("Loading permit_data from file for TenYearGrowthReport.")
                permits_data = pd.read_csv(settings.PROCESSED_DATA_DIR / "permits_processed.csv")
            if economic_data is None:
                logger.info("Loading economic_data from file for TenYearGrowthReport.")
                economic_data = pd.read_csv(settings.PROCESSED_DATA_DIR / "economic_processed.csv")

            # Validate data loading
            if census_data is None or census_data.empty:
                logger.error("Census data is missing or empty.")
                return None
            if permits_data is None or permits_data.empty:
                logger.error("Permits data is missing or empty.")
                # Depending on requirements, you might allow this and proceed or return None
                # For now, let's assume it's critical for the merge.
                return None
            # Log initial data shapes
            logger.info(
                f"Initial shapes - Census: {census_data.shape}, Permits: {permits_data.shape}"
            )

            # Validate and standardize ZIP codes
            census_data = self._validate_zip_codes(census_data)
            permits_data = self._validate_zip_codes(permits_data)

            # Ensure year column is present and numeric
            for df in [census_data, permits_data]:
                if "year" in df.columns:
                    df["year"] = pd.to_numeric(df["year"], errors="coerce")

            # Merge datasets
            merged = census_data.merge(
                permits_data, on=["zip_code", "year"], how="left", validate="1:1"
            )

            # Validate merge results
            if len(merged) == 0:
                logger.error("Merge resulted in empty dataframe")
                return None

            # Add economic indicators if available
            if economic_data is not None and not economic_data.empty:
                merged = merged.merge(economic_data, on=["year"], how="left")

            # Log merge results
            logger.info(f"Final merged shape: {merged.shape}")
            logger.info(f"Unique ZIP codes: {merged['zip_code'].nunique()}")
            logger.info(f"Year range: {merged['year'].min()} - {merged['year'].max()}")

            # Save debug copy
            debug_path = settings.INTERIM_DATA_DIR / "debug_ten_year_growth_merge.csv"
            merged.to_csv(debug_path, index=False)
            logger.info(f"Debug merge saved to {debug_path}")

            # Set instance dataframe
            self.df = merged

            return merged

        except Exception as e:
            logger.error(f"Error in load_and_prepare_data: {str(e)}")
            return None

    def _calculate_total_growth(
        self,
        df: pd.DataFrame,
        value_column: str,
        time_column: str = "year",
        group_by: str = "zip_code",
    ) -> pd.DataFrame:
        """
        Calculate total growth rate between first and last period for each group.

        Args:
            df (pd.DataFrame): Input dataframe
            value_column (str): Column containing values to calculate growth for
            time_column (str): Column containing time periods
            group_by (str): Column to group by (e.g. zip_code)

        Returns:
            pd.DataFrame: Dataframe with total growth rates
        """
        try:
            # Sort by time column within each group
            df = df.sort_values([group_by, time_column])

            # Get first and last values for each group
            first_values = df.groupby(group_by)[value_column].first()
            last_values = df.groupby(group_by)[value_column].last()

            # Calculate total growth rate
            total_growth = (last_values - first_values) / first_values

            return pd.DataFrame(
                {
                    group_by: total_growth.index,
                    f"{value_column}_total_growth": total_growth.values,
                }
            )
        except Exception as e:
            logger.error(f"Error calculating total growth: {str(e)}")
            return pd.DataFrame()

    def _analyze_historical_trends(self, data: pd.DataFrame) -> dict:
        """Analyze historical trends for population, housing, economic, and development indicators."""
        logger = logging.getLogger(__name__)
        trends = {}
        if "year" not in data.columns or "total_population" not in data.columns:
            logger.warning("Missing required columns for population trend analysis.")
            return trends

        years = data["year"]
        period_start, period_end = int(years.min()), int(years.max())
        pop_by_year = data.groupby("year")["total_population"].sum()
        if (pop_by_year == 0).all():
            logger.warning("All total_population values are zero; marking as insufficient data.")
            trends["population"] = {
                "period_start": period_start,
                "period_end": period_end,
                "total_growth": float("nan"),
                "cagr": float("nan"),
                "start_value": float("nan"),
                "end_value": float("nan"),
                "warning": "insufficient data"
            }
            try:
                population_start = pop_by_year.loc[period_start]
                population_end = pop_by_year.loc[period_end]
                total_growth = (population_end - population_start) / population_start if population_start else float("nan")
            except Exception as e:
                logger.warning(f"Could not compute population growth: {e}")
                population_start = population_end = total_growth = float("nan")
                cagr = ((population_end / population_start) ** (1 / (period_end - period_start)) - 1
                        if population_start and period_end > period_start else float("nan"))
            except Exception as e:
                logger.warning(f"Could not compute population growth: {e}")
                population_start = population_end = total_growth = cagr = float("nan")

            trends["population"] = {
                "period_start": period_start,
                "period_end": period_end,
                "total_growth": total_growth,
                "cagr": cagr,
            "start_value": population_start,
            "end_value": population_end,
            "warning": "insufficient data" if pd.isna(total_growth) else ""
            }
            logger.info(f"Population trends: {trends['population']}")
        return trends

    def _analyze_development_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze development trends from permit data."""
        try:
            permit_types = {
                col.replace("_permits", ""): int(data[col].sum())
                for col in [
                    "residential_permits",
                    "commercial_permits",
                    "retail_permits",
                ]
                if col in data.columns
            }
            construction_costs = {
                col.replace("_construction_cost", ""): float(data[col].sum())
                for col in [
                    "residential_construction_cost",
                    "commercial_construction_cost",
                    "retail_construction_cost",
                ]
                if col in data.columns
            }
            return {
                "permit_types": permit_types,
                "construction_costs": construction_costs,
                "total_permits": (
                    int(data["total_permits"].sum()) if "total_permits" in data.columns else 0
                ),
                "total_cost": (
                    float(data["total_construction_cost"].sum())
                    if "total_construction_cost" in data.columns
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing development trends: {str(e)}")
            return {}

    def _analyze_permit_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze building permit volume trends."""
        try:
            # Calculate yearly permit totals
            yearly_permits = data.groupby("year")["total_permits"].sum()

            # Calculate growth metrics
            volume_metrics = {
                "total_growth": self._calculate_total_growth(data, "total_permits")[
                    "total_permits_total_growth"
                ].mean(),
                "cagr": self._calculate_total_growth(data, "total_permits")[
                    "total_permits_total_growth"
                ].mean()
                / 100,
                "year_over_year": self._calculate_total_growth(data, "total_permits")[
                    "total_permits_total_growth"
                ].mean(),
            }

            # Analyze seasonality
            seasonality = self._analyze_permit_seasonality(data)

            return {
                "yearly_totals": yearly_permits.to_dict(),
                "growth_metrics": volume_metrics,
                "seasonality": seasonality,
            }

        except Exception as e:
            logger.error(f"Error analyzing permit volume: {str(e)}")
            raise

    def _analyze_permit_types(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze permit types and their distribution."""
        try:
            # Get permit columns
            permit_cols = {
                "residential": "residential_permits",
                "commercial": "commercial_permits",
                "retail": "retail_permits",
                "other": "other_permit_id",
            }

            # Initialize results
            results = {"residential": 0, "commercial": 0, "retail": 0, "other": 0, "total": 0}

            # Calculate totals for each type
            for permit_type, col_name in permit_cols.items():
                if col_name in data.columns:
                    results[permit_type] = float(data[col_name].sum())
                else:
                    logger.warning(f"Missing permit column: {col_name}")
                    # Try alternative columns
                    if permit_type == "residential" and "is_residential" in data.columns:
                        results[permit_type] = float(
                            data[data["is_residential"]]["total_permits"].sum()
                        )
                    elif permit_type == "commercial" and "is_commercial" in data.columns:
                        results[permit_type] = float(
                            data[data["is_commercial"]]["total_permits"].sum()
                        )
                    elif permit_type == "retail" and "is_retail" in data.columns:
                        results[permit_type] = float(data[data["is_retail"]]["total_permits"].sum())
                    elif permit_type == "other" and "total_permits" in data.columns:
                        results[permit_type] = float(data["total_permits"].sum())

            # Calculate total
            results["total"] = (
                sum(results.values()) - results["total"]
            )  # Subtract total to avoid double counting

            # Calculate percentages
            if results["total"] > 0:
                for permit_type in ["residential", "commercial", "retail", "other"]:
                    results[f"{permit_type}_pct"] = results[permit_type] / results["total"]
            else:
                logger.warning("No permits found in data")
                for permit_type in ["residential", "commercial", "retail", "other"]:
                    results[f"{permit_type}_pct"] = 0.0

            return results

        except Exception as e:
            logger.error(f"Error analyzing permit types: {str(e)}")
            return {
                "residential": 0,
                "commercial": 0,
                "retail": 0,
                "other": 0,
                "total": 0,
                "residential_pct": 0.0,
                "commercial_pct": 0.0,
                "retail_pct": 0.0,
                "other_pct": 0.0,
            }

    def _analyze_permit_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze permit values and their trends."""
        try:
            # Calculate total values
            total_value = (
                data["total_construction_cost"].sum()
                if "total_construction_cost" in data.columns
                else 0
            )

            # Calculate growth rates
            if "total_construction_cost" in data.columns:
                value_growth = (
                    data.groupby("year")["total_construction_cost"].sum().pct_change().mean()
                )
            else:
                value_growth = 0

            # Calculate per capita values
            if all(col in data.columns for col in ["total_construction_cost", "total_population"]):
                per_capita = data["total_construction_cost"].sum() / data["total_population"].sum()
            else:
                per_capita = 0

            # Calculate distribution by type
            type_values = {
                "residential": (
                    data[data["is_residential"]]["total_construction_cost"].sum()
                    if "is_residential" in data.columns
                    else 0
                ),
                "commercial": (
                    data[data["is_commercial"]]["total_construction_cost"].sum()
                    if "is_commercial" in data.columns
                    else 0
                ),
                "retail": (
                    data[data["is_retail"]]["total_construction_cost"].sum()
                    if "is_retail" in data.columns
                    else 0
                ),
                "other": (
                    data["total_construction_cost"].sum()
                    if "total_construction_cost" in data.columns
                    else 0
                ),
            }

            # Calculate percentages
            total = sum(type_values.values())
            type_percentages = {
                f"{k}_pct": v / total if total > 0 else 0 for k, v in type_values.items()
            }

            return {
                "total_value": total_value,
                "value_growth": value_growth,
                "per_capita": per_capita,
                **type_values,
                **type_percentages,
            }

        except Exception as e:
            logger.error(f"Error analyzing permit values: {str(e)}")
            return {
                "total_value": 0,
                "value_growth": 0,
                "per_capita": 0,
                "residential": 0,
                "commercial": 0,
                "retail": 0,
                "other": 0,
                "residential_pct": 0,
                "commercial_pct": 0,
                "retail_pct": 0,
                "other_pct": 0,
            }

    def _analyze_development_distribution(self, data: pd.DataFrame) -> Dict:
        """Analyze spatial distribution of development."""
        try:
            # Analyze by ZIP code
            zip_metrics = (
                data.groupby("zip_code")
                .agg({"total_permits": "sum", "total_construction_cost": "sum"})
                .to_dict()
            )

            # Calculate development density
            data["development_density"] = data["total_permits"] / data["total_population"]

            # Identify development hotspots (top 10% by permit density)
            density_threshold = data["development_density"].quantile(0.9)
            hotspots = (
                data[data["development_density"] > density_threshold]["zip_code"].unique().tolist()
            )

            # Analyze clustering (adjacent ZIP codes with high activity)
            clusters = []  # This would require spatial data to properly implement

            return {"zip_metrics": zip_metrics, "hotspots": hotspots, "clusters": clusters}

        except Exception as e:
            logger.error(f"Error analyzing development distribution: {str(e)}")
            raise

    def _analyze_permit_seasonality(self, data: pd.DataFrame) -> Dict:
        """Analyze seasonality in permit activity."""
        try:
            # Extract month from year
            data["month"] = pd.to_datetime(data["year"].astype(str), format="%Y").dt.month

            # Calculate monthly averages
            monthly_avg = data.groupby("month")["total_permits"].mean()

            # Calculate seasonal indices
            yearly_avg = monthly_avg.mean()
            seasonal_indices = (monthly_avg / yearly_avg * 100).to_dict()

            # Identify peak and trough months
            peak_month = monthly_avg.idxmax()
            trough_month = monthly_avg.idxmin()

            return {
                "seasonal_indices": seasonal_indices,
                "peak_month": peak_month,
                "trough_month": trough_month,
                "seasonality_strength": (monthly_avg.max() - monthly_avg.min()) / yearly_avg,
            }

        except Exception as e:
            logger.error(f"Error analyzing permit seasonality: {str(e)}")
            raise

    def _analyze_economic_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze economic trends."""
        try:
            # Analyze GDP trends
            gdp_trends = {
                "total_growth": self._calculate_total_growth(data, "real_gdp")[
                    "real_gdp_total_growth"
                ].mean(),
                "cagr": self._calculate_total_growth(data, "real_gdp")[
                    "real_gdp_total_growth"
                ].mean()
                / 100,
                "year_over_year": self._calculate_total_growth(data, "real_gdp")[
                    "real_gdp_total_growth"
                ].mean(),
            }

            # Analyze income trends
            income_trends = {
                "per_capita_growth": self._calculate_total_growth(data, "per_capita_income")[
                    "per_capita_income_total_growth"
                ].mean(),
                "per_capita_cagr": self._calculate_total_growth(data, "per_capita_income")[
                    "per_capita_income_total_growth"
                ].mean()
                / 100,
                "household_growth": self._calculate_total_growth(data, "median_household_income")[
                    "median_household_income_total_growth"
                ].mean(),
                "household_cagr": self._calculate_total_growth(data, "median_household_income")[
                    "median_household_income_total_growth"
                ].mean()
                / 100,
            }

            # Analyze labor force trends
            labor_trends = {
                "total_growth": self._calculate_total_growth(data, "labor_force")[
                    "labor_force_total_growth"
                ].mean(),
                "cagr": self._calculate_total_growth(data, "labor_force")[
                    "labor_force_total_growth"
                ].mean()
                / 100,
                "unemployment_rate": data["unemployment_rate"].mean(),
                "unemployment_change": self._calculate_total_growth(data, "unemployment_rate")[
                    "unemployment_rate_total_growth"
                ].mean(),
            }

            return {"gdp": gdp_trends, "income": income_trends, "labor": labor_trends}

        except Exception as e:
            logger.error(f"Error analyzing economic trends: {str(e)}")
            raise

    def _analyze_employment_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze employment trends."""
        try:
            # Calculate labor force metrics
            labor_metrics = {
                "total_labor_force": data["labor_force"].sum(),
                "labor_force_growth": self._calculate_total_growth(data, "labor_force")[
                    "labor_force_total_growth"
                ].mean(),
                "labor_force_cagr": self._calculate_total_growth(data, "labor_force")[
                    "labor_force_total_growth"
                ].mean()
                / 100,
            }

            # Calculate unemployment metrics
            unemployment_metrics = {
                "current_rate": data.sort_values("year")["unemployment_rate"].iloc[-1],
                "avg_rate": data["unemployment_rate"].mean(),
                "rate_change": self._calculate_total_growth(data, "unemployment_rate")[
                    "unemployment_rate_total_growth"
                ].mean(),
            }

            # Calculate participation rate if population data available
            if "total_population" in data.columns:
                participation_rate = (data["labor_force"] / data["total_population"] * 100).mean()
            else:
                participation_rate = None

            return {
                "labor_force": labor_metrics,
                "unemployment": unemployment_metrics,
                "participation_rate": participation_rate,
            }

        except Exception as e:
            logger.error(f"Error analyzing employment trends: {str(e)}")
            raise

    def _analyze_income_distribution(self, data: pd.DataFrame) -> Dict:
        """Analyze income distribution using available data."""
        if "median_household_income" not in data.columns:
            return {"distribution": None, "summary": {"status": "No income data available"}}
        try:
            current_year = data["year"].max()
            current_data = data[data["year"] == current_year]

            # Calculate income quintiles
            quintiles = current_data["median_household_income"].quantile([0.2, 0.4, 0.6, 0.8, 1.0])

            distribution = {
                "low_income": len(
                    current_data[current_data["median_household_income"] <= quintiles[0.2]]
                ),
                "lower_middle": len(
                    current_data[
                        (current_data["median_household_income"] > quintiles[0.2])
                        & (current_data["median_household_income"] <= quintiles[0.4])
                    ]
                ),
                "middle": len(
                    current_data[
                        (current_data["median_household_income"] > quintiles[0.4])
                        & (current_data["median_household_income"] <= quintiles[0.6])
                    ]
                ),
                "upper_middle": len(
                    current_data[
                        (current_data["median_household_income"] > quintiles[0.6])
                        & (current_data["median_household_income"] <= quintiles[0.8])
                    ]
                ),
                "high_income": len(
                    current_data[current_data["median_household_income"] > quintiles[0.8]]
                ),
            }

            # Calculate summary statistics
            summary = {
                "median": current_data["median_household_income"].median(),
                "mean": current_data["median_household_income"].mean(),
                "std": current_data["median_household_income"].std(),
                "min": current_data["median_household_income"].min(),
                "max": current_data["median_household_income"].max(),
            }

            return {
                "distribution": distribution,
                "summary": summary,
                "quintiles": quintiles.to_dict(),
            }

        except Exception as e:
            logger.error(f"Error analyzing income distribution: {str(e)}")
            return {"distribution": None, "summary": {"status": f"Error: {str(e)}"}}

    def _analyze_current_population(self, data: pd.DataFrame) -> dict:
        """Analyze current population metrics for Chicago ZIPs."""
        logger = logging.getLogger(__name__)
        try:
            pop_col = (
                "total_population"
                if "total_population" in data.columns
                else data.columns[data.columns.str.contains("pop", case=False)][0]
            )
            year_col = (
                "year"
                if "year" in data.columns
                else data.columns[data.columns.str.contains("year", case=False)][0]
            )
            latest_year = data[year_col].max()
            df_latest = data[data[year_col] == latest_year]
            total_pop = df_latest[pop_col].sum()
            avg_pop = df_latest[pop_col].mean()
            pop_by_zip = df_latest.groupby("zip_code")[pop_col].sum().to_dict()
            # Calculate population density if land_area is available
            if "land_area" in df_latest.columns:
                total_land_area = df_latest["land_area"].sum()
                density = float(total_pop) / float(total_land_area) if total_land_area else 0.0
            else:
                density = 0.0
            return {
                "year": int(latest_year),
                "total_population": int(total_pop),
                "average_population_per_zip": float(avg_pop),
                "population_by_zip": pop_by_zip,
                "density": float(density),
            }
        except Exception as e:
            logger.error(f"Error in _analyze_current_population: {e}")
            return {
                "year": None,
                "total_population": 0,
                "average_population_per_zip": 0,
                "population_by_zip": {},
                "density": 0.0,
            }

    def _analyze_current_market(self, data: pd.DataFrame) -> dict:
        """Analyze current housing market metrics."""
        logger = logging.getLogger(__name__)
        try:
            housing_col = (
                "total_housing_units"
                if "total_housing_units" in data.columns
                else data.columns[data.columns.str.contains("housing", case=False)][0]
            )
            occ_col = (
                "occupied_housing_units"
                if "occupied_housing_units" in data.columns
                else data.columns[data.columns.str.contains("occup", case=False)][0]
            )
            vac_col = (
                "vacant_housing_units"
                if "vacant_housing_units" in data.columns
                else data.columns[data.columns.str.contains("vac", case=False)][0]
            )
            year_col = (
                "year"
                if "year" in data.columns
                else data.columns[data.columns.str.contains("year", case=False)][0]
            )
            latest_year = data[year_col].max()
            df_latest = data[data[year_col] == latest_year]
            total_units = df_latest[housing_col].sum()
            occupied = df_latest[occ_col].sum()
            vacant = df_latest[vac_col].sum()
            vacancy_rate = vacant / total_units if total_units else 0
            return {
                "year": int(latest_year),
                "total_housing_units": int(total_units),
                "occupied_units": int(occupied),
                "vacant_units": int(vacant),
                "vacancy_rate": float(vacancy_rate),
            }
        except Exception as e:
            logger.error(f"Error in _analyze_current_market: {e}")
            return {
                "year": None,
                "total_housing_units": 0,
                "occupied_units": 0,
                "vacant_units": 0,
                "vacancy_rate": 0,
            }

    def _analyze_current_economic(self, data: pd.DataFrame) -> dict:
        """Analyze current economic metrics (GDP, income, unemployment)."""
        logger = logging.getLogger(__name__)
        try:
            gdp_col = (
                "real_gdp"
                if "real_gdp" in data.columns
                else data.columns[data.columns.str.contains("gdp", case=False)][0]
            )
            income_col = (
                "median_household_income"
                if "median_household_income" in data.columns
                else data.columns[data.columns.str.contains("income", case=False)][0]
            )
            unemp_col = (
                "unemployment_rate"
                if "unemployment_rate" in data.columns
                else data.columns[data.columns.str.contains("unemp", case=False)][0]
            )
            year_col = (
                "year"
                if "year" in data.columns
                else data.columns[data.columns.str.contains("year", case=False)][0]
            )
            latest_year = data[year_col].max()
            df_latest = data[data[year_col] == latest_year]
            gdp = df_latest[gdp_col].sum() if gdp_col in df_latest else 0
            income = df_latest[income_col].mean() if income_col in df_latest else 0
            unemp = df_latest[unemp_col].mean() if unemp_col in df_latest else 0
            return {
                "year": int(latest_year),
                "real_gdp": float(gdp),
                "median_income": float(income),
                "unemployment_rate": float(unemp),
            }
        except Exception as e:
            logger.error(f"Error in _analyze_current_economic: {e}")
            return {"year": None, "real_gdp": 0, "median_income": 0, "unemployment_rate": 0}

    def _analyze_current_development(self, data: pd.DataFrame) -> dict:
        """Analyze current development activity (permits, construction cost)."""
        logger = logging.getLogger(__name__)
        try:
            permits_col = (
                "total_permits"
                if "total_permits" in data.columns
                else data.columns[data.columns.str.contains("permit", case=False)][0]
            )
            cost_col = (
                "total_construction_cost"
                if "total_construction_cost" in data.columns
                else data.columns[data.columns.str.contains("cost", case=False)][0]
            )
            year_col = (
                "year"
                if "year" in data.columns
                else data.columns[data.columns.str.contains("year", case=False)][0]
            )
            latest_year = data[year_col].max()
            df_latest = data[data[year_col] == latest_year]
            total_permits = df_latest[permits_col].sum() if permits_col in df_latest else 0
            total_cost = df_latest[cost_col].sum() if cost_col in df_latest else 0
            return {
                "year": int(latest_year),
                "total_permits": int(total_permits),
                "total_construction_cost": float(total_cost),
            }
        except Exception as e:
            logger.error(f"Error in _analyze_current_development: {e}")
            return {"year": None, "total_permits": 0, "total_construction_cost": 0}

    def _generate_population_projections(self, data: pd.DataFrame) -> dict:
        """Generate 10-year population projections using recent growth rate."""
        logger = logging.getLogger(__name__)
        try:
            pop_col = (
                "total_population"
                if "total_population" in data.columns
                else data.columns[data.columns.str.contains("pop", case=False)][0]
            )
            year_col = (
                "year"
                if "year" in data.columns
                else data.columns[data.columns.str.contains("year", case=False)][0]
            )
            last_two = data.sort_values(year_col).groupby("zip_code").tail(2)
            growth = last_two.groupby("zip_code").apply(
                lambda g: (
                    (g[pop_col].iloc[-1] - g[pop_col].iloc[0]) / g[pop_col].iloc[0]
                    if g[pop_col].iloc[0]
                    else 0
                )
            )
            avg_growth = growth.mean()
            latest_total = data[data[year_col] == data[year_col].max()][pop_col].sum()
            proj_total = int(latest_total * ((1 + avg_growth) ** 10))
            return {
                "projected_population_10yr": proj_total,
                "average_annual_growth_rate": float(avg_growth),
            }
        except Exception as e:
            logger.error(f"Error in _generate_population_projections: {e}")
            return {"projected_population_10yr": 0, "average_annual_growth_rate": 0}

    def _generate_development_projections(self, data: pd.DataFrame) -> dict:
        """Generate 10-year development projections using recent permit trends."""
        logger = logging.getLogger(__name__)
        try:
            permits_col = (
                "total_permits"
                if "total_permits" in data.columns
                else data.columns[data.columns.str.contains("permit", case=False)][0]
            )
            year_col = (
                "year"
                if "year" in data.columns
                else data.columns[data.columns.str.contains("year", case=False)][0]
            )
            last_two = data.sort_values(year_col).groupby("zip_code").tail(2)
            growth = last_two.groupby("zip_code").apply(
                lambda g: (
                    (g[permits_col].iloc[-1] - g[permits_col].iloc[0]) / g[permits_col].iloc[0]
                    if g[permits_col].iloc[0]
                    else 0
                )
            )
            avg_growth = growth.mean()
            latest_total = data[data[year_col] == data[year_col].max()][permits_col].sum()
            proj_total = int(latest_total * ((1 + avg_growth) ** 10))
            return {
                "projected_permits_10yr": proj_total,
                "average_annual_permit_growth": float(avg_growth),
            }
        except Exception as e:
            logger.error(f"Error in _generate_development_projections: {e}")
            return {"projected_permits_10yr": 0, "average_annual_permit_growth": 0}

    def _generate_economic_projections(self, data: pd.DataFrame) -> Dict:
        """Generate economic projections for the next 10 years."""
        try:
            # Calculate historical growth rates
            data = data.sort_values(["zip_code", "year"])
            data["income_growth"] = (
                data.groupby("zip_code")["median_household_income"].pct_change().fillna(0)
            )
            data["home_value_growth"] = (
                data.groupby("zip_code")["median_home_value"].pct_change().fillna(0)
            )

            # Calculate average growth rates by ZIP code
            avg_income_growth = data.groupby("zip_code")["income_growth"].mean()
            avg_home_value_growth = data.groupby("zip_code")["home_value_growth"].mean()

            # Get current values by ZIP code
            current_year = data["year"].max()
            current_data = data[data["year"] == current_year].set_index("zip_code")
            current_income = current_data["median_household_income"].fillna(0)
            current_home_value = current_data["median_home_value"].fillna(0)

            # Project economic metrics for next 10 years
            projection_years = range(current_year + 1, current_year + 11)
            projections = {}

            for year in projection_years:
                year_proj = {}
                for zip_code in current_income.index:
                    income_growth = avg_income_growth[zip_code]
                    home_value_growth = avg_home_value_growth[zip_code]
                    base_income = current_income[zip_code]
                    base_home_value = current_home_value[zip_code]

                    projected_income = max(
                        0, float(base_income * (1 + income_growth) ** (year - current_year))
                    )
                    projected_home_value = max(
                        0, float(base_home_value * (1 + home_value_growth) ** (year - current_year))
                    )

                    year_proj[zip_code] = {
                        "median_household_income": projected_income,
                        "median_home_value": projected_home_value,
                    }
                projections[year] = year_proj

            return {
                "projections": projections,
                "avg_income_growth": avg_income_growth.to_dict(),
                "avg_home_value_growth": avg_home_value_growth.to_dict(),
                "base_year": current_year,
            }

        except Exception as e:
            logger.error(f"Error generating economic projections: {str(e)}")
            return {
                "projections": {},
                "avg_income_growth": {},
                "avg_home_value_growth": {},
                "base_year": None,
            }

    def _generate_market_projections(self, data: pd.DataFrame) -> Dict:
        """Generate market projections for the next 10 years."""
        try:
            # Calculate historical growth rates
            data = data.sort_values(["zip_code", "year"])
            data["housing_growth"] = (
                data.groupby("zip_code")["total_housing_units"].pct_change().fillna(0)
            )
            data["occupancy_growth"] = (
                data.groupby("zip_code")["occupied_housing_units"].pct_change().fillna(0)
            )
            data["vacancy_growth"] = (
                data.groupby("zip_code")["vacant_housing_units"].pct_change().fillna(0)
            )

            # Calculate average growth rates by ZIP code
            avg_housing_growth = data.groupby("zip_code")["housing_growth"].mean()
            avg_occupancy_growth = data.groupby("zip_code")["occupancy_growth"].mean()
            avg_vacancy_growth = data.groupby("zip_code")["vacancy_growth"].mean()

            # Get current values by ZIP code
            current_year = data["year"].max()
            current_data = data[data["year"] == current_year].set_index("zip_code")
            current_housing = current_data["total_housing_units"].fillna(0)
            current_occupancy = current_data["occupied_housing_units"].fillna(0)
            current_vacancy = current_data["vacant_housing_units"].fillna(0)

            # Project market metrics for next 10 years
            projection_years = range(current_year + 1, current_year + 11)
            projections = {}

            for year in projection_years:
                year_proj = {}
                for zip_code in current_housing.index:
                    housing_growth = avg_housing_growth[zip_code]
                    occupancy_growth = avg_occupancy_growth[zip_code]
                    vacancy_growth = avg_vacancy_growth[zip_code]

                    base_housing = current_housing[zip_code]
                    base_occupancy = current_occupancy[zip_code]
                    base_vacancy = current_vacancy[zip_code]

                    projected_housing = max(
                        0, int(base_housing * (1 + housing_growth) ** (year - current_year))
                    )
                    projected_occupancy = max(
                        0, int(base_occupancy * (1 + occupancy_growth) ** (year - current_year))
                    )
                    projected_vacancy = max(
                        0, int(base_vacancy * (1 + vacancy_growth) ** (year - current_year))
                    )

                    year_proj[zip_code] = {
                        "total_housing_units": projected_housing,
                        "occupied_housing_units": projected_occupancy,
                        "vacant_housing_units": projected_vacancy,
                        "occupancy_rate": (
                            projected_occupancy / projected_housing if projected_housing > 0 else 0
                        ),
                    }
                projections[year] = year_proj

            return {
                "projections": projections,
                "avg_housing_growth": avg_housing_growth.to_dict(),
                "avg_occupancy_growth": avg_occupancy_growth.to_dict(),
                "avg_vacancy_growth": avg_vacancy_growth.to_dict(),
                "base_year": current_year,
            }

        except Exception as e:
            logger.error(f"Error generating market projections: {str(e)}")
            return {
                "projections": {},
                "avg_housing_growth": {},
                "avg_occupancy_growth": {},
                "avg_vacancy_growth": {},
                "base_year": None,
            }

    def _analyze_housing_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze housing market impacts from projections.

        Args:
            current (Dict): Current market conditions
            projections (Dict): Future projections

        Returns:
            Dict: Housing impact analysis
        """
        try:
            # Extract current housing metrics
            current_housing = current.get("market", {}).get("total_housing_units", 0)
            current_occupied = current.get("market", {}).get("occupied_housing_units", 0)
            current_vacant = current.get("market", {}).get("vacant_housing_units", 0)

            # Extract projected housing metrics
            projected_housing = projections.get("market", {}).get("total_housing_units", 0)
            projected_occupied = projections.get("market", {}).get("occupied_housing_units", 0)
            projected_vacant = projections.get("market", {}).get("vacant_housing_units", 0)

            # Calculate changes
            housing_change = projected_housing - current_housing
            occupancy_change = projected_occupied - current_occupied
            vacancy_change = projected_vacant - current_vacant

            # Calculate rates
            current_vacancy_rate = current_vacant / current_housing if current_housing > 0 else 0
            projected_vacancy_rate = (
                projected_vacant / projected_housing if projected_housing > 0 else 0
            )
            vacancy_rate_change = projected_vacancy_rate - current_vacancy_rate

            return {
                "total_change": int(housing_change),
                "occupancy_change": int(occupancy_change),
                "vacancy_change": int(vacancy_change),
                "vacancy_rate_change": float(vacancy_rate_change),
                "current_metrics": {
                    "total_units": int(current_housing),
                    "occupied_units": int(current_occupied),
                    "vacant_units": int(current_vacant),
                    "vacancy_rate": float(current_vacancy_rate),
                },
                "projected_metrics": {
                    "total_units": int(projected_housing),
                    "occupied_units": int(projected_occupied),
                    "vacant_units": int(projected_vacant),
                    "vacancy_rate": float(projected_vacancy_rate),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing housing impacts: {str(e)}")
            return {
                "total_change": 0,
                "occupancy_change": 0,
                "vacancy_change": 0,
                "vacancy_rate_change": 0,
                "current_metrics": {},
                "projected_metrics": {},
            }

    def _analyze_retail_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze retail market impacts from projections.

        Args:
            current (Dict): Current market conditions
            projections (Dict): Future projections

        Returns:
            Dict: Retail impact analysis
        """
        try:
            # Extract current retail metrics
            current_retail_space = current.get("market", {}).get("retail_space", 0)
            current_retail_demand = current.get("market", {}).get("retail_demand", 0)
            current_retail_gap = current.get("market", {}).get("retail_gap", 0)

            # Extract projected retail metrics
            projected_retail_space = projections.get("market", {}).get("retail_space", 0)
            projected_retail_demand = projections.get("market", {}).get("retail_demand", 0)
            projected_retail_gap = projections.get("market", {}).get("retail_gap", 0)

            # Calculate changes
            space_change = projected_retail_space - current_retail_space
            demand_change = projected_retail_demand - current_retail_demand
            gap_change = projected_retail_gap - current_retail_gap

            # Calculate rates
            space_growth = space_change / current_retail_space if current_retail_space > 0 else 0
            demand_growth = (
                demand_change / current_retail_demand if current_retail_demand > 0 else 0
            )

            return {
                "space_change": int(space_change),
                "demand_change": float(demand_change),
                "gap_change": float(gap_change),
                "space_growth": float(space_growth),
                "demand_growth": float(demand_growth),
                "current_metrics": {
                    "retail_space": int(current_retail_space),
                    "retail_demand": float(current_retail_demand),
                    "retail_gap": float(current_retail_gap),
                },
                "projected_metrics": {
                    "retail_space": int(projected_retail_space),
                    "retail_demand": float(projected_retail_demand),
                    "retail_gap": float(projected_retail_gap),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing retail impacts: {str(e)}")
            return {
                "space_change": 0,
                "demand_change": 0,
                "gap_change": 0,
                "space_growth": 0,
                "demand_growth": 0,
                "current_metrics": {},
                "projected_metrics": {},
            }

    def _analyze_economic_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze economic impacts from projections.

        Args:
            current (Dict): Current economic conditions
            projections (Dict): Future projections

        Returns:
            Dict: Economic impact analysis
        """
        try:
            # Extract current economic metrics
            current_gdp = current.get("economic", {}).get("gdp", 0)
            current_income = current.get("economic", {}).get("median_income", 0)
            current_unemployment = current.get("economic", {}).get("unemployment_rate", 0)

            # Extract projected economic metrics
            projected_gdp = projections.get("economic", {}).get("gdp", 0)
            projected_income = projections.get("economic", {}).get("median_income", 0)
            projected_unemployment = projections.get("economic", {}).get("unemployment_rate", 0)

            # Calculate changes
            gdp_change = projected_gdp - current_gdp
            income_change = projected_income - current_income
            unemployment_change = projected_unemployment - current_unemployment

            # Calculate growth rates
            gdp_growth = gdp_change / current_gdp if current_gdp > 0 else 0
            income_growth = income_change / current_income if current_income > 0 else 0

            return {
                "gdp_change": float(gdp_change),
                "income_change": float(income_change),
                "unemployment_change": float(unemployment_change),
                "gdp_growth": float(gdp_growth),
                "income_growth": float(income_growth),
                "current_metrics": {
                    "gdp": float(current_gdp),
                    "median_income": float(current_income),
                    "unemployment_rate": float(current_unemployment),
                },
                "projected_metrics": {
                    "gdp": float(projected_gdp),
                    "median_income": float(projected_income),
                    "unemployment_rate": float(projected_unemployment),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing economic impacts: {str(e)}")
            return {
                "gdp_change": 0,
                "income_change": 0,
                "unemployment_change": 0,
                "gdp_growth": 0,
                "income_growth": 0,
                "current_metrics": {},
                "projected_metrics": {},
            }

    def _analyze_community_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze community impacts from projections.

        Args:
            current (Dict): Current community conditions
            projections (Dict): Future projections

        Returns:
            Dict: Community impact analysis
        """
        try:
            # Extract current community metrics
            current_population = current.get("population", {}).get("total", 0)
            current_density = current.get("population", {}).get("density", 0)
            current_household_size = current.get("population", {}).get("avg_household_size", 0)

            # Extract projected community metrics
            projected_population = projections.get("population", {}).get("total", 0)
            projected_density = projections.get("population", {}).get("density", 0)
            projected_household_size = projections.get("population", {}).get(
                "avg_household_size", 0
            )

            # Calculate changes
            population_change = projected_population - current_population
            density_change = projected_density - current_density
            household_size_change = projected_household_size - current_household_size

            # Calculate growth rates
            population_growth = (
                population_change / current_population if current_population > 0 else 0
            )
            density_growth = density_change / current_density if current_density > 0 else 0

            return {
                "population_change": int(population_change),
                "density_change": float(density_change),
                "household_size_change": float(household_size_change),
                "population_growth": float(population_growth),
                "density_growth": float(density_growth),
                "current_metrics": {
                    "total_population": int(current_population),
                    "population_density": float(current_density),
                    "avg_household_size": float(current_household_size),
                },
                "projected_metrics": {
                    "total_population": int(projected_population),
                    "population_density": float(projected_density),
                    "avg_household_size": float(projected_household_size),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing community impacts: {str(e)}")
            return {
                "population_change": 0,
                "density_change": 0,
                "household_size_change": 0,
                "population_growth": 0,
                "density_growth": 0,
                "current_metrics": {},
                "projected_metrics": {},
            }

    def _identify_primary_growth_centers(self, data: Dict) -> List[Dict]:
        """Identify primary growth centers based on multiple metrics.

        Args:
            data (Dict): Data dictionary containing merged dataset

        Returns:
            List[Dict]: List of identified growth centers with metrics
        """
        try:
            # Get merged data
            merged = data.get("merged", pd.DataFrame())
            if merged.empty:
                return []

            # Calculate growth metrics
            merged["population_growth"] = (
                merged.groupby("zip_code")["total_population"]
                .pct_change(fill_method=None)
                .fillna(0)
            )
            merged["housing_growth"] = (
                merged.groupby("zip_code")["total_housing_units"]
                .pct_change(fill_method=None)
                .fillna(0)
            )
            merged["permit_growth"] = (
                merged.groupby("zip_code")["total_permits"].pct_change(fill_method=None).fillna(0)
            )

            # Calculate composite growth score
            merged["growth_score"] = (
                merged["population_growth"] * 0.4
                + merged["housing_growth"] * 0.3
                + merged["permit_growth"] * 0.3
            )

            # Identify high growth areas (top 20%)
            threshold = merged["growth_score"].quantile(0.8)
            high_growth = merged[merged["growth_score"] > threshold]

            # Get most recent metrics for high growth areas
            current_year = merged["year"].max()
            growth_centers = high_growth[high_growth["year"] == current_year]

            # Format results
            return [
                {
                    "zip_code": row["zip_code"],
                    "total_population": int(row["total_population"]),
                    "total_housing_units": int(row["total_housing_units"]),
                    "total_permits": int(row["total_permits"]),
                    "population_growth": float(row["population_growth"]),
                    "housing_growth": float(row["housing_growth"]),
                    "permit_growth": float(row["permit_growth"]),
                    "growth_score": float(row["growth_score"]),
                }
                for _, row in growth_centers.iterrows()
            ]

        except Exception as e:
            logger.error(f"Error identifying primary growth centers: {str(e)}")
            return []

    def _identify_emerging_markets(self, data: Dict) -> List[Dict]:
        """Identify emerging markets based on recent trends.

        Args:
            data (Dict): Data dictionary containing merged dataset

        Returns:
            List[Dict]: List of identified emerging markets with metrics
        """
        try:
            # Get merged data
            merged = data.get("merged", pd.DataFrame())
            if merged.empty:
                return []

            # Calculate recent trends (last 2 years)
            current_year = merged["year"].max()
            recent_data = merged[merged["year"] >= current_year - 2].copy()

            # Calculate growth metrics
            recent_data.loc[:, "permit_growth"] = (
                recent_data.groupby("zip_code")["total_permits"]
                .pct_change(fill_method=None)
                .fillna(0)
            )
            recent_data.loc[:, "construction_growth"] = (
                recent_data.groupby("zip_code")["total_construction_cost"]
                .pct_change(fill_method=None)
                .fillna(0)
            )
            recent_data.loc[:, "population_growth"] = (
                recent_data.groupby("zip_code")["total_population"]
                .pct_change(fill_method=None)
                .fillna(0)
            )

            # Calculate momentum score
            recent_data.loc[:, "momentum_score"] = (
                recent_data["permit_growth"] * 0.4
                + recent_data["construction_growth"] * 0.3
                + recent_data["population_growth"] * 0.3
            )

            # Identify emerging markets (top 20% by momentum)
            threshold = recent_data["momentum_score"].quantile(0.8)
            emerging = recent_data[recent_data["momentum_score"] > threshold]

            # Get most recent metrics for emerging markets
            current_emerging = emerging[emerging["year"] == current_year]

            # Format results
            return [
                {
                    "zip_code": row["zip_code"],
                    "total_population": int(row["total_population"]),
                    "total_permits": int(row["total_permits"]),
                    "total_construction_cost": float(row["total_construction_cost"]),
                    "permit_growth": float(row["permit_growth"]),
                    "construction_growth": float(row["construction_growth"]),
                    "population_growth": float(row["population_growth"]),
                    "momentum_score": float(row["momentum_score"]),
                }
                for _, row in current_emerging.iterrows()
            ]

        except Exception as e:
            logger.error(f"Error identifying emerging markets: {str(e)}")
            return []

    def _identify_stabilization_zones(self, data: Dict) -> List[Dict]:
        """Identify stabilization zones based on market conditions.

        Args:
            data (Dict): Data dictionary containing merged dataset

        Returns:
            List[Dict]: List of identified stabilization zones with metrics
        """
        try:
            # Get merged data
            merged = data.get("merged", pd.DataFrame())
            if merged.empty:
                return []

            # Calculate market stability metrics
            merged["vacancy_rate"] = (
                merged["vacant_housing_units"] / merged["total_housing_units"]
            ).fillna(0)
            merged["occupancy_rate"] = (
                merged["occupied_housing_units"] / merged["total_housing_units"]
            ).fillna(0)
            merged["population_density"] = merged["total_population"] / len(merged)

            # Calculate stability score
            merged["stability_score"] = (
                (1 - merged["vacancy_rate"]) * 0.4  # Lower vacancy is better
                + merged["occupancy_rate"] * 0.4  # Higher occupancy is better
                + (merged["population_density"] / merged["population_density"].max())
                * 0.2  # Higher density is better
            )

            # Identify stable areas (top 20%)
            threshold = merged["stability_score"].quantile(0.8)
            stable = merged[merged["stability_score"] > threshold]

            # Get most recent metrics for stable areas
            current_year = merged["year"].max()
            stable_areas = stable[stable["year"] == current_year]

            # Format results
            return [
                {
                    "zip_code": row["zip_code"],
                    "total_population": int(row["total_population"]),
                    "total_housing_units": int(row["total_housing_units"]),
                    "occupied_housing_units": int(row["occupied_housing_units"]),
                    "vacant_housing_units": int(row["vacant_housing_units"]),
                    "vacancy_rate": float(row["vacancy_rate"]),
                    "occupancy_rate": float(row["occupancy_rate"]),
                    "population_density": float(row["population_density"]),
                    "stability_score": float(row["stability_score"]),
                }
                for _, row in stable_areas.iterrows()
            ]

        except Exception as e:
            logger.error(f"Error identifying stabilization zones: {str(e)}")
            return []

    def _generate_strategic_priorities(self, impacts: Dict, growth_areas: Dict) -> List[Dict]:
        """Generate strategic priorities based on impacts and growth areas.

        Args:
            impacts (Dict): Impact analysis results
            growth_areas (Dict): Identified growth areas

        Returns:
            List[Dict]: List of strategic priorities with rationale
        """
        try:
            priorities = []

            # Population-driven priorities
            if impacts.get("population", {}).get("total_change", 0) > 0:
                priorities.append(
                    {
                        "priority": "Housing Development",
                        "rationale": "Support projected population growth",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("primary", [])
                        ],
                        "impact_level": "High",
                    }
                )

            # Housing market priorities
            if impacts.get("housing", {}).get("total_units_change", 0) > 0:
                priorities.append(
                    {
                        "priority": "Infrastructure Investment",
                        "rationale": "Support new housing development",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("emerging", [])
                        ],
                        "impact_level": "Medium",
                    }
                )

            # Economic priorities
            if impacts.get("economic", {}).get("gdp_change", 0) > 0:
                priorities.append(
                    {
                        "priority": "Business Development",
                        "rationale": "Capitalize on economic growth",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("primary", [])
                        ],
                        "impact_level": "High",
                    }
                )

            # Community priorities
            if impacts.get("community", {}).get("density_change", 0) > 0:
                priorities.append(
                    {
                        "priority": "Community Services",
                        "rationale": "Support increasing population density",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("stabilization", [])
                        ],
                        "impact_level": "Medium",
                    }
                )

            # Development priorities
            if impacts.get("development", {}).get("total_permits_change", 0) > 0:
                priorities.append(
                    {
                        "priority": "Zoning Updates",
                        "rationale": "Facilitate development activity",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("emerging", [])
                        ],
                        "impact_level": "Medium",
                    }
                )

            # Market priorities
            if impacts.get("market", {}).get("vacancy_rate_change", 0) < 0:
                priorities.append(
                    {
                        "priority": "Market Stabilization",
                        "rationale": "Address declining vacancy rates",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("stabilization", [])
                        ],
                        "impact_level": "High",
                    }
                )

            return priorities

        except Exception as e:
            logger.error(f"Error generating strategic priorities: {str(e)}")
            return []

    def _generate_implementation_steps(self, impacts: Dict, growth_areas: Dict) -> List[Dict]:
        """Generate implementation steps based on impacts and growth areas.

        Args:
            impacts (Dict): Impact analysis results
            growth_areas (Dict): Identified growth areas

        Returns:
            List[Dict]: List of implementation steps with details
        """
        try:
            steps = []

            # Population-driven steps
            if impacts.get("population", {}).get("total_change", 0) > 0:
                steps.append(
                    {
                        "step": "Update Housing Development Guidelines",
                        "timeline": "Q1-Q2 2024",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("primary", [])
                        ],
                        "key_metrics": ["housing_units_added", "population_growth"],
                        "stakeholders": ["Planning Department", "Housing Authority", "Developers"],
                    }
                )

            # Housing market steps
            if impacts.get("housing", {}).get("total_units_change", 0) > 0:
                steps.append(
                    {
                        "step": "Infrastructure Capacity Assessment",
                        "timeline": "Q2-Q3 2024",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("emerging", [])
                        ],
                        "key_metrics": ["infrastructure_capacity", "development_permits"],
                        "stakeholders": ["Public Works", "Utilities", "Planning Department"],
                    }
                )

            # Economic steps
            if impacts.get("economic", {}).get("gdp_change", 0) > 0:
                steps.append(
                    {
                        "step": "Business District Enhancement Program",
                        "timeline": "Q3-Q4 2024",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("primary", [])
                        ],
                        "key_metrics": ["business_licenses", "retail_sales"],
                        "stakeholders": [
                            "Economic Development",
                            "Chamber of Commerce",
                            "Business Owners",
                        ],
                    }
                )

            # Community steps
            if impacts.get("community", {}).get("density_change", 0) > 0:
                steps.append(
                    {
                        "step": "Community Services Expansion Plan",
                        "timeline": "Q1-Q2 2025",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("stabilization", [])
                        ],
                        "key_metrics": ["service_coverage", "community_satisfaction"],
                        "stakeholders": ["Community Services", "Parks & Recreation", "Residents"],
                    }
                )

            # Development steps
            if impacts.get("development", {}).get("total_permits_change", 0) > 0:
                steps.append(
                    {
                        "step": "Zoning Code Updates",
                        "timeline": "Q2-Q3 2025",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("emerging", [])
                        ],
                        "key_metrics": ["zoning_changes", "development_activity"],
                        "stakeholders": ["Zoning Board", "Planning Department", "Developers"],
                    }
                )

            # Market steps
            if impacts.get("market", {}).get("vacancy_rate_change", 0) < 0:
                steps.append(
                    {
                        "step": "Market Stabilization Program",
                        "timeline": "Q3-Q4 2025",
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("stabilization", [])
                        ],
                        "key_metrics": ["vacancy_rates", "property_values"],
                        "stakeholders": [
                            "Housing Authority",
                            "Real Estate Agents",
                            "Property Owners",
                        ],
                    }
                )

            return steps

        except Exception as e:
            logger.error(f"Error generating implementation steps: {str(e)}")
            return []

    def _generate_support_requirements(self, impacts: Dict, growth_areas: Dict) -> List[Dict]:
        """Generate support requirements based on impacts and growth areas.

        Args:
            impacts (Dict): Impact analysis results
            growth_areas (Dict): Identified growth areas

        Returns:
            List[Dict]: List of support requirements with details
        """
        try:
            requirements = []

            # Population support requirements
            if impacts.get("population", {}).get("total_change", 0) > 0:
                requirements.append(
                    {
                        "requirement": "Housing Development Support",
                        "resources": [
                            "Zoning Staff",
                            "Planning Resources",
                            "Development Incentives",
                        ],
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("primary", [])
                        ],
                        "priority": "High",
                        "timeline": "2024-2025",
                    }
                )

            # Housing market support requirements
            if impacts.get("housing", {}).get("total_units_change", 0) > 0:
                requirements.append(
                    {
                        "requirement": "Infrastructure Support",
                        "resources": [
                            "Engineering Staff",
                            "Capital Funding",
                            "Utility Coordination",
                        ],
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("emerging", [])
                        ],
                        "priority": "Medium",
                        "timeline": "2024-2026",
                    }
                )

            # Economic support requirements
            if impacts.get("economic", {}).get("gdp_change", 0) > 0:
                requirements.append(
                    {
                        "requirement": "Business Development Support",
                        "resources": [
                            "Economic Development Staff",
                            "Business Incentives",
                            "Marketing Resources",
                        ],
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("primary", [])
                        ],
                        "priority": "High",
                        "timeline": "2024-2025",
                    }
                )

            # Community support requirements
            if impacts.get("community", {}).get("density_change", 0) > 0:
                requirements.append(
                    {
                        "requirement": "Community Services Support",
                        "resources": ["Service Staff", "Program Funding", "Facility Resources"],
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("stabilization", [])
                        ],
                        "priority": "Medium",
                        "timeline": "2025-2026",
                    }
                )

            # Development support requirements
            if impacts.get("development", {}).get("total_permits_change", 0) > 0:
                requirements.append(
                    {
                        "requirement": "Development Process Support",
                        "resources": ["Permit Staff", "Review Resources", "Technology Systems"],
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("emerging", [])
                        ],
                        "priority": "Medium",
                        "timeline": "2024-2025",
                    }
                )

            # Market support requirements
            if impacts.get("market", {}).get("vacancy_rate_change", 0) < 0:
                requirements.append(
                    {
                        "requirement": "Market Stabilization Support",
                        "resources": ["Housing Staff", "Financial Resources", "Program Support"],
                        "target_areas": [
                            area["zip_code"] for area in growth_areas.get("stabilization", [])
                        ],
                        "priority": "High",
                        "timeline": "2025-2026",
                    }
                )

            return requirements

        except Exception as e:
            logger.error(f"Error generating support requirements: {str(e)}")
            return []

    def _ensure_template_keys(self, context, required_structure):
        """
        Recursively ensure all keys in required_structure exist in context, filling with defaults.
        """
        for key, default_value_or_structure in required_structure.items(): # Renamed for clarity
            if isinstance(default_value_or_structure, dict):
                # Ensure the key exists and is a dictionary
                context.setdefault(key, {})
                # If what's in context is not a dict (e.g. "N/A" from a previous failed render), reset it
                if not isinstance(context[key], dict):
                    context[key] = {}
                self._ensure_template_keys(context[key], default_value_or_structure)
            else:
                context.setdefault(key, default_value_or_structure)

    def generate_report_from_context_dict(self, context: dict) -> str:
        try:
            # Ensure all required context keys for template
            if "historical_trends" not in context:
                context["historical_trends"] = {}
            if "current_analysis" not in context:
                context["current_analysis"] = {}
            if "projections" not in context:
                context["projections"] = {}
            if "impacts" not in context:
                context["impacts"] = {}
            if "growth_areas" not in context:
                context["growth_areas"] = {}
            if "recommendations" not in context:
                context["recommendations"] = {}
            # Log keys for debug
            logging.info(
                f"TenYearGrowthReport: historical_trends keys: {list(context['historical_trends'].keys())}"
            )
            logging.info(
                f"TenYearGrowthReport: current_analysis keys: {list(context['current_analysis'].keys())}"
            )
            logging.info(
                f"TenYearGrowthReport: projections keys: {list(context['projections'].keys())}"
            )
            # Check for missing/zeroed required keys
            missing = []
            all_zero = []
            for col in REQUIRED_COLS:
                val = context.get(col)
                if val is None:
                    missing.append(col)
                    context[col] = 0
                    logging.warning(f"TenYearGrowthReport: Missing key {col}, set to 0.")
                elif isinstance(val, (int, float)) and val == 0:
                    all_zero.append(col)
                    logging.warning(f"TenYearGrowthReport: All values in {col} are zero.")
            notes = []
            if missing:
                notes.append(f"Missing keys: {', '.join(missing)}")
            if all_zero:
                notes.append(f"All zero keys: {', '.join(all_zero)}")
            context["notes"] = notes
            context["missing_or_defaulted"] = missing + all_zero
            # Render template with .get() and missing/defaulted block
            try:
                rendered = self.template_env.get_template("ten_year_growth_analysis.md").render(
                    **{k: context.get(k, "N/A") for k in context}
                )
            except Exception as e:
                logging.error(f"TenYearGrowthReport: Template rendering failed: {e}")
                rendered = (
                    f"Report generation failed. Error: {e}\nNotes: {context.get('notes', [])}"
                )
            # Save the rendered report to the main output path
            with open(self.output_paths["main"], "w") as f:
                f.write(rendered)
            logger.info(f"Ten Year Growth Report generated at {self.output_paths['main']}")
            return rendered # Return the string content
        except Exception as e:
            logging.error(f"TenYearGrowthReport: Failed to generate report: {e}")
            return f"Report generation failed. Error: {e}"

    def generate_report_data(self, processed_data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Populate self.report_data with real, non-empty historical_trends, current_analysis, and projections.
        """
        logger = logging.getLogger(__name__)
        logger.info("Generating data for Ten Year Growth Report...")
        self.report_data = {} # Initialize report_data

        merged_df = self.load_and_prepare_data(processed_data_dict=processed_data_dict)
        
        if merged_df is None or merged_df.empty:
            logger.error("No merged data available for report generation.")
            # Ensure report_data is initialized with minimal structure even on failure
            self.report_data.update({ 
                "current_analysis": {}, "historical_trends": {}, "projections": {},
                "emerging_multifamily_areas": [], "retail_opportunities": [],
                "retail_lagging_zones": [], "downtown_retail_lag": [],
                "notes": ["Failed to load and prepare base data."]
            })
            return self.report_data

        # Historical trends
        historical_trends = self._analyze_historical_trends(merged_df)
        # Current analysis (latest year)
        current_analysis_dict = self._analyze_current_conditions(merged_df) # This is well-structured

        # Build projected_analysis_dict (mirroring current_analysis_dict structure for 10-year projections)
        projected_analysis_dict = {
            "population": {},
            "market": {},  # For housing and retail space projections
            "development": {},
            "economic": {},
            "retail": {} # Explicitly add retail projections if available
        }

        raw_pop_proj = self._generate_population_projections(merged_df)
        projected_analysis_dict["population"]["total"] = raw_pop_proj.get("projected_population_10yr", 0)
        # Note: projected density, avg_household_size would need further calculation if required by impacts.

        raw_dev_proj = self._generate_development_projections(merged_df)
        projected_analysis_dict["development"]["total_permits"] = raw_dev_proj.get("projected_permits_10yr", 0)
        # Note: projected_total_construction_cost would be needed for full development impact.

        # Placeholder for more detailed economic and market projections if _generate_*_projections were enhanced
        # For now, economic and market (housing units, retail space) projections will be minimal or zero
        # leading to minimal reported impacts for those sections.
        # Example: if _generate_economic_projections returned {'gdp': ..., 'median_income': ...}
        # projected_analysis_dict["economic"] = self._generate_economic_projections(merged_df) 
        # Example: if _generate_market_projections returned {'total_housing_units': ..., 'retail_space': ...}
        # projected_analysis_dict["market"] = self._generate_market_projections(merged_df)

        impacts_dict = {
            "housing": self._analyze_housing_impacts(current_analysis_dict, projected_analysis_dict),
            "retail": self._analyze_retail_impacts(current_analysis_dict, projected_analysis_dict),
            "economic": self._analyze_economic_impacts(current_analysis_dict, projected_analysis_dict),
            "community": self._analyze_community_impacts(current_analysis_dict, projected_analysis_dict),
            # development impacts could be added if _analyze_current_development and _generate_development_projections are aligned
        }


        # Load new analysis outputs from DataProcessor
        emerging_multifamily_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "emerging_multifamily_areas.csv") \
            if (settings.PROCESSED_DATA_DIR / "emerging_multifamily_areas.csv").exists() else pd.DataFrame()
        
        retail_opportunities_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_development_opportunities.csv") \
            if (settings.PROCESSED_DATA_DIR / "retail_development_opportunities.csv").exists() else pd.DataFrame()

        retail_lagging_zones_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_lagging_zones.csv") \
            if (settings.PROCESSED_DATA_DIR / "retail_lagging_zones.csv").exists() else pd.DataFrame()
        
        downtown_retail_lag_df = pd.read_csv(settings.PROCESSED_DATA_DIR / "downtown_retail_lag_analysis.csv") \
            if (settings.PROCESSED_DATA_DIR / "downtown_retail_lag_analysis.csv").exists() else pd.DataFrame()

        # Consolidate current analysis from the helper
        # The _analyze_current_conditions already provides a structured dict.
        # We might want to ensure all REQUIRED_COLS are represented if the template expects them directly
        # For now, we'll use the structure from _analyze_current_conditions
        # and ensure the template can handle potentially missing sub-keys.

        # Example of ensuring specific top-level keys from REQUIRED_COLS if needed by template,
        # though it's better if the template adapts to the nested structure.
        # current_analysis_flat = {}
        # if current_analysis_dict.get("population"):
        #     current_analysis_flat["total_population"] = current_analysis_dict["population"].get("total")
        #     current_analysis_flat["median_household_income"] = current_analysis_dict["population"].get("median_income")
        # # ... and so on for other REQUIRED_COLS, mapping from current_analysis_dict structure.
        # This part is commented out as it's preferable for the template to use the nested structure.

        self.report_data = {
            "generation_date": self.report_data.get("generation_date", datetime.now().strftime("%Y-%m-%d")), # Preserve if already set
            "current_analysis": current_analysis_dict,
            "historical_trends": historical_trends,
            "projections": projected_analysis_dict, # Use the structured projections
            "impacts": impacts_dict, # Add the computed impacts
            "emerging_multifamily_areas": [] if emerging_multifamily_df.empty else emerging_multifamily_df.to_dict(orient="records"),
            "retail_opportunities": [] if retail_opportunities_df.empty else retail_opportunities_df.to_dict(orient="records"),
            "retail_lagging_zones": [] if retail_lagging_zones_df.empty else retail_lagging_zones_df.to_dict(orient="records"),
            "downtown_retail_lag": [] if downtown_retail_lag_df.empty else downtown_retail_lag_df.to_dict(orient="records"),
            "notes": self.report_data.get("notes", []) # Preserve existing notes
        }
        
        # Add warnings for missing data files
        if emerging_multifamily_df.empty:
            self.report_data["notes"].append("Emerging multifamily areas data not found or empty.")
            logger.warning("Emerging multifamily areas data not found or empty for report.")
        if retail_opportunities_df.empty:
            self.report_data["notes"].append("Retail development opportunities data not found or empty.")
            logger.warning("Retail development opportunities data not found or empty for report.")
        if retail_lagging_zones_df.empty:
            self.report_data["notes"].append("Retail lagging zones data not found or empty.")
            logger.warning("Retail lagging zones data not found or empty for report.")
        if downtown_retail_lag_df.empty:
            self.report_data["notes"].append("Downtown retail lag analysis data not found or empty.")
            logger.warning("Downtown retail lag analysis data not found or empty for report.")

        logger.info("Populated report_data for Ten Year Growth Report.")
        return self.report_data

    def _analyze_current_conditions(self, merged_df: pd.DataFrame) -> dict:
        """Helper to get various current condition metrics from the merged_df."""
        if merged_df is None or merged_df.empty:
            return {}
        
        latest_year_data = merged_df[merged_df["year"] == merged_df["year"].max()]
        if latest_year_data.empty:
            return {}

        def safe_sum(df, col_name): return int(df[col_name].sum()) if col_name in df.columns and not df[col_name].empty else 0
        def safe_mean(df, col_name): return float(df[col_name].mean()) if col_name in df.columns and not df[col_name].empty else 0.0
        def safe_median(df, col_name): return float(df[col_name].median()) if col_name in df.columns and not df[col_name].empty else 0.0

        return {
            "year": int(latest_year_data["year"].iloc[0]),
            "population": {"total": safe_sum(latest_year_data, "total_population"), "median_income": safe_median(latest_year_data, "median_household_income")},
            "market": {"total_housing_units": safe_sum(latest_year_data, "total_housing_units"), "median_home_value": safe_median(latest_year_data, "median_home_value")},
            "development": {"total_permits": safe_sum(latest_year_data, "total_permits"), "total_construction_cost": safe_sum(latest_year_data, "total_construction_cost")},
            "economic": {"gdp": safe_mean(latest_year_data, "real_gdp"), "unemployment_rate": safe_mean(latest_year_data, "unemployment_rate")},
            "retail": {
                "total_space": safe_sum(latest_year_data, "retail_space"), "total_demand": safe_sum(latest_year_data, "retail_demand"),
                "total_supply": safe_sum(latest_year_data, "retail_supply"), "total_gap": safe_sum(latest_year_data, "retail_gap")
            }
        }

    def generate_ten_year_growth_report(self, df, output_path):
        """
        Generate a ten-year growth report, explicitly flagging ZIPs with missing retail or housing metrics.
        """
        logger = logging.getLogger("src.reporting.ten_year_growth_report")

        # Identify ZIPs with missing metrics
        # Use correct column names and check for existence
        expected_cols = [
            "retail_space", "retail_supply", "retail_demand", "total_housing_units"
        ]
        missing_report = {}
        for col in expected_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' missing from DataFrame. Skipping.")
                continue
            missing_zip_rows = df[df[col].isnull()]
            if "zip_code" not in missing_zip_rows.columns:
                logger.warning(f"'zip_code' column missing when checking for missing {col}.")
                continue
            if missing_zips := missing_zip_rows["zip_code"].tolist():
                logger.warning(f"Missing {col} for ZIPs: {missing_zips}")
                missing_report[col] = missing_zips
            # Check for all-zero column
            if (df[col] == 0).all():
                logger.warning(f"All values in {col} are zero. Downstream metrics may be misleading.")
                missing_report[f"all_zero_{col}"] = True

        # Write missing data section to report
        with open(output_path, "w") as f:
            f.write("# Ten-Year Growth Report\n\n")
            f.write("## ZIPs with Missing Key Metrics\n")
            for col, zips in missing_report.items():
                f.write(f"- {col}: {', '.join(str(z) for z in zips)}\n")
            f.write("\n")
            # ... existing report content ...
            f.write("## Main Analysis\n")
            # (Insert main analysis code here)
            # For demonstration, show summary table
            summary_cols = [
                "zip_code", "total_housing_units", "retail_space", "retail_supply", "retail_demand", "retail_gap"
            ]
            summary_cols = [col for col in summary_cols if col in df.columns]
            summary = df[summary_cols].head(20).to_string(index=False)
            f.write("\n### Sample Data (first 20 rows):\n")
            f.write(summary)
            f.write("\n")

    def generate_report(self, processed_data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> bool:
        """
        Generates the main ten-year growth report and saves it.
        """
        try:
            logger.info("TenYearGrowthReport: Starting generate_report...")
            # Generate the data for the report
            # self.generate_report_data populates self.report_data
            self.generate_report_data(processed_data_dict=processed_data_dict)
            
            if not self.report_data or not self.report_data.get("current_analysis"): # Check if critical data is missing
                logger.error("Failed to generate data for the report. Aborting report generation.")
                return False # Return False for failure

            required_template_structure = {
                "generation_date": datetime.now().strftime("%Y-%m-%d"),
                "current_analysis": {"population": {}, "market": {}, "development": {}, "economic": {}, "retail": {}},
                "historical_trends": {"population": {}},
                "projections": {"population": {}, "market": {}, "development": {}, "economic": {}, "retail": {}},
                "growth_areas": { # Add this structure
                    "primary_centers": [], 
                    "emerging_markets": [], 
                    "stabilization_zones": []
                },
                "recommendations": { # Add default structure for recommendations
                    "strategic": [],
                    "implementation": []
                },
                "impacts": {"housing": {}, "retail": {}, "economic": {}, "community": {}},
                "emerging_multifamily_areas": [],
                "retail_opportunities": [],
                "retail_lagging_zones": [],
                "downtown_retail_lag": [],
                "notes": [],
                "missing_or_defaulted": [] 
            }
            # Use self.report_data as the context
            self._ensure_template_keys(self.report_data, required_template_structure)
            
            if not self.report_data.get("current_analysis", {}).get("population", {}).get("total"):
                 self.report_data["notes"].append("Warning: Current total population data is missing or zero.")
            # Adjust check for projections based on actual structure from generate_report_data
            if not self.report_data.get("projections", {}).get("population", {}).get("total"):
                 self.report_data["notes"].append("Warning: Population projection data is missing or zero.")

            template = self.template_env.get_template(self.templates["main"])
            rendered_report = template.render(self.report_data)

            output_path = self.output_paths["main"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(rendered_report)
            
            logger.info(f"TenYearGrowthReport: Report generated successfully at {output_path}")
            return True # Return True for success

        except Exception as e:
            logger.error(f"TenYearGrowthReport: Failed to generate report: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            try:
                output_path = self.output_paths["main"]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(f"# Report Generation Failed\n\nError: {e}\n\nTraceback:\n{traceback.format_exc()}")
            except Exception as e_write:
                logger.error(f"Failed to write failure report: {e_write}")
            return False # Return False for failure
