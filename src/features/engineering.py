"""
Feature engineering module for deriving insights from Chicago population data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

from ..config import settings

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Handles feature engineering for Chicago population analysis."""

    def __init__(self):
        """Initialize the feature engineering class."""
        self.scaler = StandardScaler()

    def calculate_growth_rates(
        self, df: pd.DataFrame, value_cols: List[str], periods: List[int]
    ) -> pd.DataFrame:
        """
        Calculate growth rates over specified periods.

        Args:
            df: DataFrame with temporal data
            value_cols: Columns to calculate growth for
            periods: List of periods (in years) to calculate growth over

        Returns:
            DataFrame with growth rate features
        """
        try:
            growth_df = df.copy()

            for col in value_cols:
                for period in periods:
                    # Calculate year-over-year growth
                    growth_df[f"{col}_growth_{period}y"] = (
                        growth_df[col] - growth_df[col].shift(period)
                    ) / growth_df[col].shift(period)

            logger.info("Successfully calculated growth rates")
            return growth_df

        except Exception as e:
            logger.error(f"Error calculating growth rates: {str(e)}")
            raise

    def create_density_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create population and development density features.

        Args:
            df: DataFrame with population and area data

        Returns:
            DataFrame with density features
        """
        try:
            density_df = df.copy()

            # Calculate population density
            if all(col in df.columns for col in ["total_population", "land_area"]):
                density_df["population_density"] = df["total_population"] / df["land_area"]

            # Calculate housing density
            if all(col in df.columns for col in ["total_housing_units", "land_area"]):
                density_df["housing_density"] = df["total_housing_units"] / df["land_area"]

            # Calculate business density
            if all(col in df.columns for col in ["total_businesses", "land_area"]):
                density_df["business_density"] = df["total_businesses"] / df["land_area"]

            logger.info("Successfully created density features")
            return density_df

        except Exception as e:
            logger.error(f"Error creating density features: {str(e)}")
            raise

    def create_retail_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create retail development and spending potential features.

        Args:
            df: DataFrame with retail and demographic data

        Returns:
            DataFrame with retail features
        """
        try:
            retail_df = df.copy()

            # Propagate and use all new retail metrics
            required_cols = [
                "retail_permits", "retail_construction_cost", "retail_business_count",
                "retail_space", "retail_demand", "retail_supply", "retail_gap", "retail_lag"
            ]
            for col in required_cols:
                if col not in df.columns:
                    retail_df[col] = np.nan
                    logger.warning(f"Retail feature column {col} is missing in input DataFrame.")
            # Log missing data
            for col in required_cols:
                if retail_df[col].isnull().all():
                    logger.warning(f"Retail feature column {col} is NaN for all ZIPs.")
            # Only keep valid Chicago ZIPs
            retail_df = retail_df[retail_df["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]

            # Calculate retail space per capita
            if all(col in retail_df.columns for col in ["retail_space", "total_population"]):
                retail_df["retail_space_per_capita"] = retail_df["retail_space"] / retail_df["total_population"]

            # Calculate retail spending potential
            if "median_household_income" in retail_df.columns:
                retail_df["retail_spending_potential"] = (
                    retail_df["median_household_income"] * 0.3
                )  # Assumed 30% of income

            # Calculate retail gap
            if all(
                col in retail_df.columns
                for col in ["retail_space_per_capita", "retail_spending_potential"]
            ):
                retail_df["retail_gap"] = retail_df["retail_spending_potential"] - (
                    retail_df["retail_space_per_capita"] * settings.RETAIL_SALES_PER_SF
                )

            logger.info("Successfully created retail features")
            return retail_df

        except Exception as e:
            logger.error(f"Error creating retail features: {str(e)}")
            raise

    def create_development_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create development activity and potential features.

        Args:
            df: DataFrame with permit and zoning data

        Returns:
            DataFrame with development features
        """
        try:
            dev_df = df.copy()

            # Calculate development intensity
            if "estimated_cost" in df.columns:
                dev_df["development_intensity"] = df["estimated_cost"] / df.groupby("year")[
                    "estimated_cost"
                ].transform("mean")

            # Calculate residential development share
            if all(col in df.columns for col in ["is_residential", "is_commercial"]):
                total_development = df["is_residential"] + df["is_commercial"]
                dev_df["residential_share"] = df["is_residential"] / total_development

            # Calculate development potential
            if all(col in df.columns for col in ["is_planned_development", "land_area"]):
                dev_df["development_potential"] = df["is_planned_development"] * df["land_area"]

            logger.info("Successfully created development features")
            return dev_df

        except Exception as e:
            logger.error(f"Error creating development features: {str(e)}")
            raise

    def create_economic_features(
        self, df: pd.DataFrame, fred_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create economic condition and trend features.

        Args:
            df: DataFrame with local economic data
            fred_data: Dictionary of FRED indicator DataFrames

        Returns:
            DataFrame with economic features
        """
        try:
            econ_df = df.copy()

            # Add FRED indicators
            for series_id, fred_df in fred_data.items():
                if series_id == "CUUR0200SA0":  # CPI
                    econ_df["inflation_adjusted_income"] = (
                        df["median_household_income"] / fred_df["value"].iloc[-1] * 100
                    )
                elif series_id == "CHIC917URN":  # Unemployment
                    econ_df["unemployment_impact"] = fred_df["value"].iloc[-1]
                elif series_id == "MSPUS":  # Median Home Price
                    econ_df["price_to_national_ratio"] = (
                        df["median_sale_price"] / fred_df["value"].iloc[-1]
                    )

            # Calculate economic health score
            if all(
                col in econ_df.columns for col in ["median_household_income", "unemployment_impact"]
            ):
                econ_df["economic_health_score"] = econ_df["median_household_income"] * (
                    1 - econ_df["unemployment_impact"] / 100
                )

            logger.info("Successfully created economic features")
            return econ_df

        except Exception as e:
            logger.error(f"Error creating economic features: {str(e)}")
            raise

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features and trends.

        Args:
            df: DataFrame with temporal data

        Returns:
            DataFrame with temporal features
        """
        try:
            temp_df = df.copy()

            # Calculate moving averages
            if "total_population" in df.columns:
                temp_df["population_ma_3y"] = df["total_population"].rolling(window=3).mean()
                temp_df["population_ma_5y"] = df["total_population"].rolling(window=5).mean()

            # Calculate momentum indicators
            value_cols = ["total_population", "median_household_income", "total_housing_units"]
            for col in value_cols:
                if col in df.columns:
                    # 3-year momentum
                    temp_df[f"{col}_momentum_3y"] = (df[col] - df[col].shift(3)) / df[col].shift(3)

                    # 5-year momentum
                    temp_df[f"{col}_momentum_5y"] = (df[col] - df[col].shift(5)) / df[col].shift(5)

            logger.info("Successfully created temporal features")
            return temp_df

        except Exception as e:
            logger.error(f"Error creating temporal features: {str(e)}")
            raise

    def create_spatial_features(self, df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Create location-based and spatial relationship features.

        Args:
            df: DataFrame with ZIP-level data
            gdf: GeoDataFrame with Chicago ZIP code boundaries

        Returns:
            DataFrame with spatial features
        """
        try:
            spatial_df = df.copy()

            # Calculate centrality measures
            centroids = gdf.centroid
            cbd_point = settings.CHICAGO_CBD_COORDS

            # Distance to CBD
            spatial_df["distance_to_cbd"] = centroids.distance(cbd_point)

            # Calculate neighborhood effects
            for col in ["median_household_income", "total_population", "housing_density"]:
                if col in df.columns:
                    # Get adjacency matrix
                    adjacency = gdf.topology.adjacency_matrix()

                    # Calculate neighborhood averages
                    neighborhood_avg = adjacency.dot(df[col]) / adjacency.sum(axis=1)
                    spatial_df[f"{col}_neighborhood_avg"] = neighborhood_avg

                    # Calculate relative position
                    spatial_df[f"{col}_relative_to_neighbors"] = df[col] / neighborhood_avg

            logger.info("Successfully created spatial features")
            return spatial_df

        except Exception as e:
            logger.error(f"Error creating spatial features: {str(e)}")
            raise

    def create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite scores for different aspects of development.

        Args:
            df: DataFrame with engineered features

        Returns:
            DataFrame with composite scores
        """
        try:
            score_df = df.copy()

            # Growth potential score
            growth_features = [
                "population_momentum_5y",
                "development_intensity",
                "development_potential",
                "economic_health_score",
            ]
            if all(col in df.columns for col in growth_features):
                # Normalize features
                normalized_growth = self.scaler.fit_transform(df[growth_features])
                # Calculate weighted sum
                score_df["growth_potential_score"] = np.average(
                    normalized_growth, weights=settings.GROWTH_POTENTIAL_WEIGHTS, axis=1
                )

            # Retail opportunity score
            retail_features = [
                "retail_gap",
                "retail_spending_potential",
                "business_density",
                "population_density",
            ]
            if all(col in df.columns for col in retail_features):
                normalized_retail = self.scaler.fit_transform(df[retail_features])
                score_df["retail_opportunity_score"] = np.average(
                    normalized_retail, weights=settings.RETAIL_OPPORTUNITY_WEIGHTS, axis=1
                )

            # Housing demand score
            housing_features = [
                "population_growth_5y",
                "median_household_income",
                "housing_density",
                "price_to_national_ratio",
            ]
            if all(col in df.columns for col in housing_features):
                normalized_housing = self.scaler.fit_transform(df[housing_features])
                score_df["housing_demand_score"] = np.average(
                    normalized_housing, weights=settings.HOUSING_DEMAND_WEIGHTS, axis=1
                )

            logger.info("Successfully created composite scores")
            return score_df

        except Exception as e:
            logger.error(f"Error creating composite scores: {str(e)}")
            raise

    def engineer_all_features(
        self, df: pd.DataFrame, fred_data: Dict[str, pd.DataFrame], gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps to create final feature matrix.

        Args:
            df: DataFrame with preprocessed data
            fred_data: Dictionary of FRED indicator DataFrames
            gdf: GeoDataFrame with Chicago ZIP code boundaries

        Returns:
            DataFrame with all engineered features
        """
        try:
            # Apply each feature engineering step
            feature_df = df.copy()

            # Growth rates
            value_cols = ["total_population", "median_household_income", "total_housing_units"]
            periods = [1, 3, 5]
            feature_df = self.calculate_growth_rates(feature_df, value_cols, periods)

            # Density features
            feature_df = self.create_density_features(feature_df)

            # Retail features
            feature_df = self.create_retail_features(feature_df)

            # Development features
            feature_df = self.create_development_features(feature_df)

            # Economic features
            feature_df = self.create_economic_features(feature_df, fred_data)

            # Temporal features
            feature_df = self.create_temporal_features(feature_df)

            # Spatial features
            feature_df = self.create_spatial_features(feature_df, gdf)

            # Composite scores
            feature_df = self.create_composite_scores(feature_df)

            # --- Ensure 'development_density' feature is present ---
            if "development_density" not in feature_df.columns:
                if (
                    "total_permits" in feature_df.columns
                    and "total_population" in feature_df.columns
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        feature_df["development_density"] = (
                            feature_df["total_permits"] / feature_df["total_population"]
                        )
                else:
                    feature_df["development_density"] = np.nan

            # --- Ensure 'month' feature is present ---
            if "month" not in feature_df.columns:
                if "date" in feature_df.columns:
                    feature_df["month"] = pd.to_datetime(feature_df["date"]).dt.month
                elif "year" in feature_df.columns:
                    # If only year is present, set month to 1 (or NaN if preferred)
                    feature_df["month"] = 1
                else:
                    feature_df["month"] = np.nan

            # Only keep valid Chicago ZIPs
            feature_df = feature_df[feature_df["zip_code"].isin(settings.CHICAGO_ZIP_CODES)]
            # For all retail metrics, if missing, set to NaN and flag as insufficient data
            required_cols = ["retail_space", "retail_demand", "retail_gap", "retail_supply"]
            for col in required_cols:
                if col not in feature_df.columns:
                    feature_df[col] = np.nan
                if (feature_df[col] == 0).all():
                    feature_df[col] = np.nan
                    feature_df[f"{col}_status"] = "insufficient data"
                else:
                    feature_df[f"{col}_status"] = feature_df[col].apply(lambda x: "insufficient data" if pd.isna(x) or x == 0 else "ok")

            logger.info("Successfully engineered all features")
            return feature_df

        except Exception as e:
            logger.error(f"Error in engineer_all_features: {str(e)}")
            raise
