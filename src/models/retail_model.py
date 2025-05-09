"""
Retail analysis model for Chicago ZIP codes.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from typing import Tuple, Optional, List, Union, Dict # Added Union, Dict

from src.config import settings
from src.utils.helpers import (
    match_features,
    resolve_column_name,
    safe_train_model,
)
from src.config.column_alias_map import column_aliases # type: ignore

logger = logging.getLogger(__name__)


class RetailModel:
    """Model for analyzing retail trends and making predictions."""

    def __init__(self):
        """Initialize retail model."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_col = None
        self.metrics = {}

        # Create required directories
        for directory in [settings.TRAINED_MODELS_DIR, settings.MODEL_METRICS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_feature_list(
        self, df: pd.DataFrame, feature_list: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get the list of features to use for retail modeling.

        Args:
            df: Input DataFrame
            feature_list: Optional specific features to use

        Returns:
            List of feature names to use
        """
        if feature_list is not None:
            matched = match_features(df, feature_list, column_aliases)
            if not matched:
                logger.error("No valid features found in provided feature list")
                return []
            return matched

        # Core demographic features
        demographic_features = ["total_population", "median_household_income", "labor_force"]

        # Core retail features (aligned with what DataProcessor provides in merged_dataset.csv)
        retail_features = [
            "retail_space",
            "retail_demand",
            "retail_supply",
            # "retail_sales", # Not consistently available in merged_dataset.csv logs
            # "retail_employees", # Not consistently available in merged_dataset.csv logs
            # "retail_establishments", # Not consistently available in merged_dataset.csv logs
        ]

        # Optional features
        optional_features = [
            "retail_space_per_capita",
            "retail_sales_per_capita",
            "retail_spending_potential",
            "retail_gap",
            "retail_leakage",
            "retail_surplus",
        ]

        # Start with core features
        features = []
        for feature in demographic_features:
            if feature in df.columns:
                features.append(feature)
            else:
                logger.error(f"Missing core demographic feature: {feature}")
                return []

        # Add available retail features
        retail_count = 0
        # Ensure retail_gap is considered here if it's a primary retail feature
        for feature in retail_features + ["retail_gap"]: # Add retail_gap here if it's core
            if feature in df.columns and not df[feature].isna().all():
                features.append(feature)
                retail_count += 1

        if retail_count == 0:
            logger.error("No retail features available")
            return []

        # Add optional features if available
        for feature in optional_features:
            # Ensure not to add 'retail_gap' again if it was part of core retail_features
            if feature in df.columns and not df[feature].isna().all() and feature not in features:
                features.append(feature)
                logger.debug(f"Added optional feature: {feature}")

        logger.info(f"Selected features ({len(features)}): {features}")
        return features

    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and impute missing values in features.

        Args:
            X: Feature DataFrame

        Returns:
            Cleaned DataFrame
        """
        X = X.copy()

        # Log initial state
        missing_vals = X.isnull().sum()
        logger.info("Missing values before cleaning:")
        for col, count in missing_vals[missing_vals > 0].items():
            logger.info(f"{col}: {count} missing values")

        for col in X.columns:
            # Convert to numeric first
            X[col] = pd.to_numeric(X[col], errors="coerce")

            # Handle missing values based on column type
            if X[col].isnull().any():
                if col.endswith(("_per_capita", "_potential", "_gap")):
                    # Use ZIP code median for ratio/derived metrics
                    zip_medians = X.groupby("zip_code")[col].transform("median")
                    X[col] = X[col].fillna(zip_medians)
                    if X[col].isnull().any():
                        overall_median = X[col].median()
                        X[col] = X[col].fillna(overall_median)
                        logger.info(
                            f"Used overall median ({overall_median}) for remaining {col} NaN values"
                        )
                else:
                    # For raw counts/values, use 0 for missing
                    X[col] = X[col].fillna(0)
                    logger.debug(f"Filled {col} NaN values with 0")

            # Handle infinite values
            if np.isinf(X[col]).any():
                X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
                logger.warning(f"Replaced infinite values in {col} with median")

        return X

    def _clean_target(self, y: pd.Series) -> pd.Series:
        """
        Clean and validate target variable.

        Args:
            y: Target Series

        Returns:
            Cleaned Series
        """
        y = y.copy()

        # Convert to numeric
        y = pd.to_numeric(y, errors="coerce")

        # Handle missing values
        if y.isnull().any():
            # First try ZIP code median
            zip_medians = y.groupby("zip_code").transform("median")
            y = y.fillna(zip_medians)

            # For any remaining NaN, use overall median
            if y.isnull().any():
                overall_median = y.median()
                y = y.fillna(overall_median)
                logger.info(
                    f"Used overall median ({overall_median}) for remaining target NaN values"
                )

        # Handle negative values
        if (y < 0).any():
            logger.warning("Found negative retail values")
            y = y.clip(lower=0)

        return y

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to features.

        Args:
            X: Feature DataFrame

        Returns:
            Transformed DataFrame
        """
        X = X.copy()

        # Features that should be log-transformed
        log_transform_cols = [
            "total_population",
            "median_household_income",
            "retail_space",
            "retail_sales",
            "retail_employees",
        ]

        # Features that should be normalized by population
        per_capita_cols = [
            ("retail_space", "total_population", "retail_space_per_capita"),
            ("retail_sales", "total_population", "retail_sales_per_capita"),
            ("retail_employees", "total_population", "retail_employees_per_capita"),
        ]

        # Apply log transforms
        for col in log_transform_cols:
            if col in X.columns:
                min_val = X[col].min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    X[col] = X[col] + offset
                X[col] = np.log1p(X[col])
                logger.debug(f"Applied log transform to {col}")

        # Create per capita features
        for numerator, denominator, new_col in per_capita_cols:
            if numerator in X.columns and denominator in X.columns:
                X[new_col] = X[numerator] / X[denominator]
                X[new_col] = X[new_col].fillna(X[new_col].median())
                logger.debug(f"Created per capita feature: {new_col}")

        return X

    def _validate_features(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Validate prepared features and target.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            bool: Whether validation passed
        """
        # Check for NaN values
        if X.isnull().any().any():
            logger.error("NaN values remain in features:")
            logger.error(X.isnull().sum()[X.isnull().sum() > 0])
            return False

        if y.isnull().any():
            logger.error(f"NaN values remain in target: {y.isnull().sum()}")
            return False

        # Check for infinite values
        if np.isinf(X).any().any():
            logger.error("Infinite values found in features:")
            logger.error(np.isinf(X).sum()[np.isinf(X).sum() > 0])
            return False

        if np.isinf(y).any():
            logger.error(f"Infinite values found in target: {np.isinf(y).sum()}")
            return False

        # Verify dimensions
        if len(X) != len(y):
            logger.error(f"Feature/target length mismatch: X={len(X)}, y={len(y)}")
            return False

        return True

    def prepare_features(
        self, df: pd.DataFrame, feature_list: Optional[List[str]] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Prepare features for retail modeling.

        Args:
            df: Input DataFrame
            feature_list: Optional specific features to use

        Returns:
            Tuple of (features DataFrame, target Series) or (None, None) on error
        """
        try:
            logger.info("Preparing retail features...")

            # Get feature list
            features = self._get_feature_list(df, feature_list)
            if not features:
                return None, None

            # Prepare feature matrix and target
            X = df[features].copy()
            y = df["retail_construction_cost"].copy()

            # Store feature names
            self.feature_names = features
            self.target_col = "retail_construction_cost"

            # Clean data
            X = self._clean_features(X)
            y = self._clean_target(y)

            # Transform features
            X = self._transform_features(X)

            # Validate
            if not self._validate_features(X, y):
                return None, None

            # Scale features
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

            logger.info(f"Features prepared successfully: {X.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None, None

    def analyze_retail_trends(self, df):
        """Analyze retail trends in the data."""
        try:
            logger.info("Analyzing retail trends...")

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
                "retail_construction_cost",
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

            logger.info("Analyzed retail trends")
            return yearly_trends

        except Exception as e:
            logger.error(f"Error analyzing retail trends: {str(e)}")
            return None

    def train(self, data_input: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """Train the retail model."""
        try:
            logger.info("Training retail model...")

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
            success = safe_train_model(self.model, X, y, model_name="RetailModel")
            if success:
                logger.info("Model trained successfully.")

                # Save model and scaler
                joblib.dump(self.model, settings.TRAINED_MODELS_DIR / "retail_model.joblib")
                joblib.dump(self.scaler, settings.TRAINED_MODELS_DIR / "retail_scaler.joblib")
                logger.info("Model and scaler saved")

            return success

        except Exception as e:
            logger.error(f"RetailModel training failed: {str(e)}")
            return False

    def predict_retail_demand(self, df: pd.DataFrame, feature_list=None) -> pd.DataFrame:
        """
        Predict retail demand based on demographic and economic factors.

        Args:
            df: Input DataFrame with demographic and economic data
            feature_list: List of features to use (from pipeline inspector)

        Returns:
            DataFrame with retail demand predictions
        """
        try:
            logger.info("Predicting retail demand...")

            # Use dynamic feature selection if feature_list is provided
            if feature_list is None:
                feature_list = [
                    "total_population",
                    "median_household_income",
                    "retail_space_per_capita",
                    "retail_spending_potential",
                    "retail_gap",
                ]

            # Match features using aliases
            feature_cols = match_features(df, feature_list, column_aliases)
            if not feature_cols:
                logger.error("No usable features found")
                return None

            # Resolve target column
            target_col = resolve_column_name(df, "retail_construction_cost", column_aliases)
            if not target_col:
                logger.error("Target column 'retail_construction_cost' not found")
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
            df["predicted_retail_demand"] = model.predict(X)

            # Calculate additional metrics
            df["demand_gap"] = df["predicted_retail_demand"] - df[target_col]
            df["demand_ratio"] = df["predicted_retail_demand"] / df[target_col]

            # Save predictions
            predictions_file = settings.PREDICTIONS_DIR / "retail_demand_predictions.csv"
            df.to_csv(predictions_file, index=False)

            self.predictions = df
            logger.info("Generated retail demand predictions")
            return df

        except Exception as e:
            logger.error(f"Error predicting retail demand: {str(e)}")
            return None

    def analyze_retail_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze retail sales leakage by ZIP code.

        Args:
            df: Input DataFrame with retail data

        Returns:
            DataFrame with leakage analysis
        """
        try:
            logger.info("Analyzing retail leakage...")

            # Calculate leakage metrics
            df["leakage_rate"] = df["retail_gap"] / df["retail_spending_potential"]
            df["leakage_per_capita"] = df["retail_gap"] / df["total_population"]

            # Categorize leakage severity
            df["leakage_severity"] = pd.cut(
                df["leakage_rate"],
                bins=[-np.inf, 0, 0.2, 0.4, np.inf],
                labels=["No Leakage", "Low", "Medium", "High"],
            )

            # Save leakage analysis
            leakage_file = settings.PROCESSED_DATA_DIR / "retail_leakage.csv"
            df.to_csv(leakage_file, index=False)
            logger.info(f"Saved retail leakage analysis to {leakage_file}")

            return df

        except Exception as e:
            logger.error(f"Error analyzing retail leakage: {str(e)}")
            return None

    def calculate_retail_metrics(self, df):
        """Calculate key retail metrics."""
        try:
            metrics = {
                "total_retail_space": float(df["retail_space"].sum()),
                "avg_retail_space_per_capita": float(df["retail_space_per_capita"].mean()),
                "total_retail_gap": float(df["retail_gap"].sum()),
                "avg_leakage_rate": float(df["leakage_rate"].mean()),
                "high_opportunity_areas": int(
                    df[df["retail_opportunity_score"] > 0]["zip_code"].nunique()
                ),
            }

            # Save metrics
            metrics_file = settings.PROCESSED_DATA_DIR / "retail_summary_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved retail summary metrics to {metrics_file}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating retail metrics: {str(e)}")
            return None

    def run_analysis(self, feature_list=None):
        """Run the complete retail analysis pipeline."""
        try:
            # Load retail metrics
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / "retail_metrics.csv")

            # Analyze trends
            trends = self.analyze_retail_trends(df)
            if trends is not None:
                logger.info("Retail trends analysis completed successfully")
            else:
                logger.warning("Retail trends analysis failed")
                return False

            # Generate predictions
            predictions = self.predict_retail_demand(df, feature_list=feature_list)
            if predictions is not None:
                logger.info("Generated retail demand predictions")
            else:
                logger.error("Failed to generate retail demand predictions")
                return False

            # Analyze leakage
            leakage = self.analyze_retail_leakage(predictions)
            if leakage is not None:
                logger.info("Retail leakage analysis completed")
            else:
                logger.error("Failed to analyze retail leakage")
                return False

            # Calculate metrics
            metrics = self.calculate_retail_metrics(leakage)
            if metrics is not None:
                logger.info("Calculated retail metrics")
            else:
                logger.error("Failed to calculate retail metrics")
                return False

            # Store results
            self.results = {
                "trends": trends,
                "predictions": predictions,
                "leakage": leakage,
                "metrics": metrics,
            }

            logger.info("Retail analysis pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in retail analysis pipeline: {str(e)}")
            return False

    def get_results(self) -> dict[str, pd.DataFrame]:
        """Get the analysis results."""
        if self.results is None:
            logger.warning("No results available. Run analysis first.")
            return {}
        return self.results
