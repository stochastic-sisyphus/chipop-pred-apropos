"""
Housing analysis model for Chicago ZIP codes.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor # type: ignore
import joblib
from typing import Tuple

from src.config import settings
from src.utils.helpers import (
    match_features,
    resolve_column_name,
    safe_train_model,
    Union, Dict # Added Union, Dict
)
from src.config.column_alias_map import column_aliases # type: ignore

logger = logging.getLogger(__name__)


class HousingModel:
    """Model for analyzing housing trends and making predictions."""

    def __init__(self):
        """Initialize housing model."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(self, data_input: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """Train the housing model."""
        try:
            logger.info("Training housing model...")

            if isinstance(data_input, dict):
                df = data_input.get('merged_data')
                if df is None:
                    logger.error(f"'{self.__class__.__name__}': 'merged_data' not found in input dictionary. Available keys: {list(data_input.keys())}")
                    return False
            elif isinstance(data_input, pd.DataFrame):
                df = data_input
            else:
                logger.error(f"'{self.__class__.__name__}': Invalid data input type: {type(data_input)}")
                return False

            # Prepare features
            X, y = self.prepare_features(df)
            if X is None or y is None:
                logger.error("Failed to prepare features for training")
                return False

            # Train model
            success = safe_train_model(self.model, X, y, model_name="HousingModel")
            if success:
                logger.info("Model trained successfully.")

                # Save model and scaler
                joblib.dump(self.model, settings.TRAINED_MODELS_DIR / "housing_model.joblib")
                joblib.dump(self.scaler, settings.TRAINED_MODELS_DIR / "housing_scaler.joblib")
                logger.info("Model and scaler saved")

            return success

        except Exception as e:
            logger.error(f"HousingModel training failed: {str(e)}")
            return False

    def analyze_housing_trends(self, df):
        """Analyze housing trends in the data."""
        try:
            logger.info("Analyzing housing trends...")

            # Get column names
            year_col = resolve_column_name(df, "year", column_aliases)
            if not year_col:
                logger.error("Year column not found")
                return None

            # Define metrics to analyze
            metrics = [
                "housing_units",
                "housing_density",
                "housing_value_per_unit",
                "residential_construction_cost",
            ]

            # Resolve metric columns
            resolved_metrics = {
                metric: resolve_column_name(df, metric, column_aliases) for metric in metrics
            }

            # Filter out unresolved metrics
            resolved_metrics = {k: v for k, v in resolved_metrics.items() if v is not None}

            if not resolved_metrics:
                logger.error("No metric columns found")
                return None

            # Calculate yearly trends
            yearly_trends = (
                df.groupby(year_col)
                .agg(
                    {
                        col: "mean" if "density" in metric or "value" in metric else "sum"
                        for metric, col in resolved_metrics.items()
                    }
                )
                .reset_index()
            )

            # Calculate year-over-year changes
            for col in yearly_trends.columns:
                if col != year_col:
                    yearly_trends[f"{col}_change"] = yearly_trends[col].pct_change()

            self.trends = yearly_trends
            logger.info("Analyzed housing trends")
            return yearly_trends

        except Exception as e:
            logger.error(f"Error analyzing housing trends: {str(e)}")
            return None

    def predict_housing_demand(self, df: pd.DataFrame, feature_list=None) -> pd.DataFrame:
        """
        Predict housing demand based on demographic and economic factors.

        Args:
            df: Input DataFrame with demographic and economic data
            feature_list: List of features to use (from pipeline inspector)

        Returns:
            DataFrame with housing demand predictions
        """
        try:
            logger.info("Predicting housing demand...")

            # Use dynamic feature selection if feature_list is provided
            if feature_list is None:
                feature_list = [
                    "total_population",
                    "median_household_income",
                    "housing_density",
                    "housing_value_per_unit",
                    "residential_permits",
                    "residential_construction_cost",
                ]

            # Match features using aliases
            feature_cols = match_features(df, feature_list, column_aliases)
            if not feature_cols:
                logger.error("No usable features found")
                return None

            # Resolve target column
            target_col = resolve_column_name(df, "housing_units", column_aliases)
            if not target_col:
                logger.error("Target column 'housing_units' not found")
                return None

            # Prepare features and target
            X = df[feature_cols].copy()
            y = df[target_col]

            # Coerce all features to numeric and handle inf/-inf
            X = X.apply(pd.to_numeric, errors="coerce")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            y = pd.to_numeric(y, errors="coerce")
            y.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Fill missing values with median (for features)
            X = X.fillna(X.median())

            # Log missing value counts before dropping
            logger.info("Missing values per column before drop:\n" + str(X.isnull().sum()))
            logger.info(f"Missing values in target before drop: {y.isnull().sum()}")

            # Remove rows with any NaN or inf in X or y
            mask = (~X.isin([np.nan, np.inf, -np.inf]).any(axis=1)) & (
                ~y.isin([np.nan, np.inf, -np.inf])
            )
            dropped = (~mask).sum()
            if dropped > 0:
                logger.warning(f"Dropping {dropped} rows with NaN or inf in features or target.")
            X = X[mask]
            y = y[mask]

            # Clip values to float64 range
            float_max = np.finfo(np.float64).max
            float_min = np.finfo(np.float64).min
            X = X.clip(lower=float_min, upper=float_max)
            y = y.clip(lower=float_min, upper=float_max)

            # Create prediction model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

            # Train model on historical data
            model.fit(X, y)

            # Generate predictions
            df = df.loc[X.index].copy()
            df["predicted_housing_demand"] = model.predict(X)

            # Calculate additional metrics
            df["demand_gap"] = df["predicted_housing_demand"] - df[target_col]
            df["demand_ratio"] = df["predicted_housing_demand"] / df[target_col]

            # Save predictions
            predictions_file = settings.PREDICTIONS_DIR / "housing_demand_predictions.csv"
            df.to_csv(predictions_file, index=False)

            self.predictions = df
            logger.info("Generated housing demand predictions")
            return df

        except Exception as e:
            logger.error(f"Error predicting housing demand: {str(e)}")
            return None

    def analyze_housing_retail_balance(self, df):
        """Analyze balance between housing and retail development."""
        try:
            logger.info("Analyzing housing-retail balance...")

            # Load retail metrics
            retail_metrics = pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_metrics.csv")

            # Merge housing and retail data
            balance_df = pd.merge(
                df,
                retail_metrics[["year", "zip_code", "retail_construction_cost", "retail_space"]],
                on=["year", "zip_code"],
                how="left",
            )

            # Fill missing retail values with 0
            balance_df["retail_construction_cost"] = balance_df["retail_construction_cost"].fillna(
                0
            )
            balance_df["retail_space"] = balance_df["retail_space"].fillna(0)

            # Calculate balance metrics
            # Avoid division by zero and log(0) issues
            with np.errstate(divide="ignore", invalid="ignore"):
                # housing_retail_ratio: inf if retail_construction_cost == 0
                balance_df["housing_retail_ratio"] = np.where(
                    balance_df["retail_construction_cost"] > 0,
                    balance_df["residential_construction_cost"]
                    / balance_df["retail_construction_cost"],
                    np.nan,  # Use NaN for undefined ratios
                )

                # Cap extremely high/low ratios for log stability
                safe_ratio = balance_df["housing_retail_ratio"].clip(lower=1e-3, upper=1e3)
                # Replace NaN with a neutral value (e.g., 1) for log calculation
                safe_ratio = safe_ratio.fillna(1.0)
                balance_df["balance_score"] = 1 / (1 + np.abs(np.log(safe_ratio)))

            # Categorize balance
            balance_df["balance_category"] = pd.cut(
                balance_df["balance_score"],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=["Severe Imbalance", "Moderate Imbalance", "Slight Imbalance", "Balanced"],
            )

            # Determine primary need
            balance_df["primary_need"] = np.where(
                balance_df["housing_retail_ratio"] > 2,
                "Retail Development",
                np.where(
                    balance_df["housing_retail_ratio"] < 0.5,
                    "Housing Development",
                    "Balanced Development",
                ),
            )

            # Save balance analysis
            balance_df.to_csv(
                settings.PROCESSED_DATA_DIR / "housing_retail_balance.csv", index=False
            )
            logger.info("Saved housing-retail balance analysis")

            logger.info("Analyzed housing-retail balance")
            return balance_df

        except Exception as e:
            logger.error(f"Error analyzing housing-retail balance: {str(e)}")
            return None

    def run_analysis(self, feature_list=None):
        """Run the complete housing analysis pipeline."""
        try:
            # Load processed data
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / "housing_metrics.csv")

            # Analyze trends
            trends = self.analyze_housing_trends(df)
            if trends is not None:
                logger.info("Housing trends analysis completed successfully")
            else:
                logger.warning("Housing trends analysis failed")
                return False

            # Generate predictions
            predictions = self.predict_housing_demand(df, feature_list=feature_list)
            if predictions is not None:
                logger.info("Generated housing demand predictions")
            else:
                logger.error("Failed to generate housing demand predictions")
                return False

            # Analyze balance
            balance = self.analyze_housing_retail_balance(predictions)
            if balance is not None:
                logger.info("Housing-retail balance analysis completed")
            else:
                logger.error("Failed to analyze housing-retail balance")
                return False

            # Store results
            self.results = {"trends": trends, "predictions": predictions, "balance": balance}

            logger.info("Housing analysis pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in housing analysis pipeline: {str(e)}")
            return False

    def get_results(self) -> dict[str, pd.DataFrame]:
        """Get the analysis results."""
        if self.results is None:
            logger.warning("No results available. Run analysis first.")
            return {}
        return self.results

    def prepare_features(
        self, df: pd.DataFrame, feature_list=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for housing modeling."""
        try:
            logger.info("Preparing housing features...")

            # Use dynamic feature selection if feature_list is provided
            if feature_list is not None:
                features = match_features(df, feature_list, column_aliases)
            else:
                # Default features
                default_features = [
                    "total_population",
                    "median_household_income",
                    "median_home_value",
                    "labor_force",
                    "commercial_permits",
                    "retail_permits",
                    "commercial_construction_cost",
                    "retail_construction_cost",
                ]
                features = match_features(df, default_features, column_aliases)

            if not features:
                logger.error("No usable features available for housing modeling.")
                return None, None

            logger.info(f"Features used for housing modeling: {features}")

            # Store feature names for reuse
            self.feature_names = features

            # Resolve target variable
            target_col = resolve_column_name(df, "residential_permits", column_aliases)

            # Create feature matrix and target
            X = df[features].copy()
            y = df[target_col] if target_col else None

            # Log missing value counts before dropping
            logger.info(
                "Missing values per column before drop:\n"
                + str(df[features + ([target_col] if target_col else [])].isnull().sum())
            )

            # Handle missing values and outliers
            if y is not None:
                mask = (~X.isin([np.nan, np.inf, -np.inf]).any(axis=1)) & (
                    ~y.isin([np.nan, np.inf, -np.inf])
                )
            else:
                mask = ~X.isin([np.nan, np.inf, -np.inf]).any(axis=1)
            dropped = (~mask).sum()
            if dropped > 0:
                logger.warning(f"Dropping {dropped} rows with NaN or inf in features or target.")
            X = X[mask]
            if y is not None:
                y = y[mask]

            # Clip values to float64 range
            float_max = np.finfo(np.float64).max
            float_min = np.finfo(np.float64).min
            X = X.clip(lower=float_min, upper=float_max)
            if y is not None:
                y = y.clip(lower=float_min, upper=float_max)

            # Log transform large numeric columns
            large_value_cols = [
                "total_construction_cost",
                "residential_construction_cost",
                "commercial_construction_cost",
                "retail_construction_cost",
                "median_household_income",
                "median_home_value",
            ]
            for col in large_value_cols:
                resolved_col = resolve_column_name(X, col, column_aliases)
                if resolved_col in X.columns:
                    # Ensure column is numeric and handle negatives before log transform
                    X[resolved_col] = pd.to_numeric(X[resolved_col], errors='coerce')
                    X[resolved_col] = X[resolved_col].fillna(0) # Or median, depending on strategy
                    X[resolved_col] = np.log1p(np.maximum(0, X[resolved_col]))
            # Ensure no NaNs/Infs before scaling from log transform
            # This replace should ideally happen after all transformations that might introduce Inf
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            if X.isnull().any().any():
                logger.warning(
                    "NaNs found in features after log transform, before scaling. Imputing with column median."
                )
                for col_with_nan in X.columns[X.isnull().any()]:
                    X[col_with_nan] = X[col_with_nan].fillna(X[col_with_nan].median())

            # Scale features
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

            logger.info("Housing features prepared successfully")
            return X, y
        except Exception as e:
            logger.error(f"Error preparing housing features: {str(e)}")
            return None, None
