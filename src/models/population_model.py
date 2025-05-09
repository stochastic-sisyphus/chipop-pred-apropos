"""
Population analysis model for Chicago ZIP codes.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from typing import Tuple, Optional, List, Union, Dict # Added Union, Dict

from src.config import settings
from src.utils.helpers import (
    sanitize_features,
    match_features,
    resolve_column_name,
    safe_train_model,
)
from src.config.column_alias_map import column_aliases # type: ignore

logger = logging.getLogger(__name__)


class PopulationModel:
    """Model for analyzing population trends and making predictions."""

    def __init__(self):
        """Initialize population model."""
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
        self.predictions = None
        self.scenarios = None
        self.metrics = {}

        # Create required directories
        for directory in [settings.TRAINED_MODELS_DIR, settings.MODEL_METRICS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_feature_list(
        self, df: pd.DataFrame, feature_list: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get the list of features to use for modeling.

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

        # Core features that must be present
        core_features = ["median_household_income", "median_home_value", "labor_force"]

        # Optional permit-related features
        permit_features = [
            "residential_permits",
            "commercial_permits",
            "retail_permits",
            "total_permits",
        ]

        # Optional cost-related features
        cost_features = [
            "residential_construction_cost",
            "commercial_construction_cost",
            "retail_construction_cost",
            "total_construction_cost",
        ]

        # Optional housing-related features
        housing_features = ["total_housing_units", "occupied_housing_units", "vacant_housing_units"]

        # Start with core features
        features = [f for f in core_features if f in df.columns]
        if len(features) != len(core_features):
            missing = set(core_features) - set(features)
            logger.error(f"Missing core features: {missing}")
            return []

        # Add available optional features
        for feature_set in [permit_features, cost_features, housing_features]:
            for feature in feature_set:
                if feature in df.columns and not df[feature].isna().all():
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
                if col.endswith(("_permits", "_construction_cost")):
                    # Fill missing permit/cost data with 0
                    X[col] = X[col].fillna(0)
                else:
                    # For other columns, use ZIP code median then overall median
                    zip_medians = X.groupby("zip_code")[col].transform("median")
                    X[col] = X[col].fillna(zip_medians)
                    if X[col].isnull().any():
                        overall_median = X[col].median()
                        X[col] = X[col].fillna(overall_median)
                        logger.info(
                            f"Used overall median ({overall_median}) for remaining {col} NaN values"
                        )

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

        # Validate range
        if (y <= 0).any():
            logger.error("Found non-positive population values")
            min_val = y[y > 0].min()
            y = y.clip(lower=min_val)
            logger.info(f"Clipped population values to minimum of {min_val}")

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
            "median_household_income",
            "median_home_value",
            "labor_force",
            "total_housing_units",
            "occupied_housing_units",
        ]

        # Features that should be ratio-transformed
        ratio_cols = [
            ("vacant_housing_units", "total_housing_units", "vacancy_rate"),
            ("occupied_housing_units", "total_housing_units", "occupancy_rate"),
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

        # Create ratio features
        for numerator, denominator, new_col in ratio_cols:
            if numerator in X.columns and denominator in X.columns:
                X[new_col] = X[numerator] / X[denominator]
                X[new_col] = X[new_col].fillna(X[new_col].median())
                logger.debug(f"Created ratio feature: {new_col}")

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
        Prepare features for population modeling.

        Args:
            df: Input DataFrame
            feature_list: Optional specific features to use

        Returns:
            Tuple of (features DataFrame, target Series) or (None, None) on error
        """
        try:
            logger.info("Preparing population features...")

            # Get feature list
            features = self._get_feature_list(df, feature_list)
            if not features:
                return None, None

            # Prepare feature matrix and target
            X = df[features].copy()
            y = df["total_population"].copy()

            # Store feature names
            self.feature_names = features
            self.target_col = "total_population"

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

    def train(self, data_input: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """Train the population model."""
        try:
            logger.info("Training population model...")

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
                logger.error(f"Input df shape to prepare_features was: {df.shape if df is not None else 'None'}")
                return False

            # Log training data shape
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

            # Verify no NaN values
            if X.isnull().any().any() or y.isnull().any():
                logger.error("Found NaN values after feature preparation")
                return False

            if safe_train_model(self.model, X, y, model_name="PopulationModel"):
                logger.info("Population model trained successfully")

                # Save model and scaler
                joblib.dump(self.model, settings.TRAINED_MODELS_DIR / "population_model.joblib")
                joblib.dump(self.scaler, settings.TRAINED_MODELS_DIR / "population_scaler.joblib")

                # Save feature names
                feature_names_df = pd.DataFrame({"feature": X.columns})
                feature_names_df.to_csv(
                    settings.TRAINED_MODELS_DIR / "population_feature_names.csv", index=False
                )

                logger.info("Model and scaler saved")
                return True
            else:
                logger.error("Failed to train population model")
                return False

        except Exception as e:
            logger.error(f"Error training population model: {str(e)}")
            return False

    def generate_predictions(self):
        """Generate population predictions for all scenarios."""
        try:
            logger.info("Generating population predictions...")

            # Load processed data
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / "merged_dataset.csv")

            # Get column names
            zip_col = resolve_column_name(df, "zip_code", column_aliases)
            year_col = resolve_column_name(df, "year", column_aliases)

            if not all([zip_col, year_col]):
                logger.error("Required columns not found")
                return None

            # Prepare features using the same feature set used for training
            expected_features = [
                "median_household_income",
                "median_home_value",
                "labor_force",
                "residential_permits",
                "commercial_permits",
                "retail_permits",
                "residential_construction_cost",
                "commercial_construction_cost",
                "retail_construction_cost",
                "total_licenses",
                "unique_businesses",
                "active_licenses",
                "total_permits",
            ]

            feature_cols = match_features(df, expected_features, column_aliases)
            clean_df = sanitize_features(df, feature_cols)
            X = clean_df[feature_cols]

            # Store original index for later use
            original_index = X.index

            # Generate base predictions
            self.model.predict(X)

            # Create scenarios dictionary with adjustments
            scenarios = settings.SCENARIOS
            scenario_predictions = {}

            for scenario_name, adjustments in scenarios.items():
                # Adjust features based on scenario
                X_scenario = X.copy()

                # Apply scenario adjustments
                growth_factor = adjustments.get("growth_factor", 1.0)
                X_scenario = X_scenario * growth_factor

                # Generate predictions for this scenario
                predictions = self.model.predict(X_scenario)

                # Store results
                scenario_predictions[scenario_name] = predictions

            # Create results DataFrame using the correct index
            results_df = pd.DataFrame(
                {
                    zip_col: df.loc[original_index, zip_col],
                    year_col: df.loc[original_index, year_col],
                    "base_prediction": scenario_predictions.get("base"),
                    "optimistic_prediction": scenario_predictions.get("optimistic"),
                    "neutral_prediction": scenario_predictions.get("neutral"),
                    "pessimistic_prediction": scenario_predictions.get("pessimistic"),
                }
            )

            # Save predictions
            results_df.to_csv(settings.PREDICTIONS_DIR / "population_predictions.csv", index=False)
            logger.info("Generated population predictions successfully")

            # Explicit check for DataFrame non-emptiness (prevents ambiguous truth value errors)
            if results_df is None or results_df.empty:
                logger.error("Failed to generate population predictions")
                return False

            return results_df

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return None

    def generate_scenarios(self):
        """Generate scenario predictions."""
        try:
            logger.info("Generating scenario predictions...")

            if self.predictions is None:
                logger.error("No base predictions available. Run generate_predictions first.")
                return False

            # Get column names
            zip_col = resolve_column_name(self.predictions, "zip_code", column_aliases)
            year_col = resolve_column_name(self.predictions, "year", column_aliases)

            if not all([zip_col, year_col]):
                logger.error("Required columns not found")
                return False

            # Create scenario predictions
            scenario_predictions = []

            for scenario_name, factors in settings.SCENARIOS.items():
                # Create scenario DataFrame
                scenario_df = self.predictions.copy()

                # Map scenario name to prediction column
                prediction_col = {
                    "base": "base_prediction",
                    "optimistic": "optimistic_prediction",
                    "neutral": "neutral_prediction",
                    "pessimistic": "pessimistic_prediction",
                }.get(scenario_name)

                if prediction_col not in scenario_df.columns:
                    logger.error(f"Missing prediction column for scenario {scenario_name}")
                    continue

                # Add scenario information
                scenario_df["predicted_population"] = scenario_df[prediction_col]
                scenario_df["scenario"] = scenario_name
                scenario_df["growth_factor"] = factors.get("growth_factor", 1.0)
                scenario_df["confidence"] = factors.get("confidence", 1.0)

                scenario_predictions.append(scenario_df)

            if not scenario_predictions:
                logger.error("No valid scenarios generated")
                return False

            # Combine all scenarios
            all_scenarios = pd.concat(scenario_predictions, ignore_index=True)

            # Save scenario predictions
            all_scenarios.to_csv(settings.PREDICTIONS_DIR / "population_scenarios.csv", index=False)

            self.scenarios = all_scenarios
            logger.info("Generated scenario predictions successfully")
            return True

        except Exception as e:
            logger.error(f"Error generating scenarios: {str(e)}")
            return False

    def run_analysis(self):
        """Run the complete population analysis pipeline."""
        try:
            # Load data
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / "merged_dataset.csv")

            # Train model
            if not self.train(df):
                logger.error("Failed to train population model")
                return False

            # Generate predictions
            predictions_df = self.generate_predictions()
            if predictions_df is None or predictions_df.empty:
                logger.error("Failed to generate population predictions")
                return False

            # Store predictions for scenario generation
            self.predictions = predictions_df

            # Generate scenarios
            if not self.generate_scenarios():
                logger.error("Failed to generate scenario predictions")
                return False

            logger.info("Population analysis pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in population analysis: {str(e)}")
            return False

    def get_results(self) -> pd.DataFrame:
        """Get the analysis results."""
        if self.predictions is None:
            logger.warning("No results available. Run analysis first.")
            return pd.DataFrame()
        return self.predictions

    def get_scenario_results(self) -> pd.DataFrame:
        """Get the scenario analysis results."""
        if self.scenarios is None:
            logger.warning("No scenario results available. Run analysis first.")
            return pd.DataFrame()
        return self.scenarios

    def generate_high_growth_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate high growth scenario projections."""
        try:
            return self._extracted_from_generate_low_growth_scenario_4(
                "Generating high growth scenario...",
                "high_growth",
                "high_growth_prediction",
            )
        except Exception as e:
            logger.error(f"Error generating high growth scenario: {str(e)}")
            return pd.DataFrame()

    def generate_low_growth_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate low growth scenario projections."""
        try:
            return self._extracted_from_generate_low_growth_scenario_4(
                "Generating low growth scenario...",
                "low_growth",
                "low_growth_prediction",
            )
        except Exception as e:
            logger.error(f"Error generating low growth scenario: {str(e)}")
            return pd.DataFrame()

    # TODO Rename this here and in `generate_high_growth_scenario` and `generate_low_growth_scenario`
    def _extracted_from_generate_low_growth_scenario_4(self, arg0, arg1, arg2):
        logger.info(arg0)
        if self.predictions is None:
            logger.error("No base predictions available. Run generate_predictions first.")
            return pd.DataFrame()
        scenario_df = self.predictions.copy()
        growth_factor = settings.SCENARIOS[arg1]["population_growth"]
        scenario_df[arg2] = scenario_df["base_prediction"] * growth_factor
        return scenario_df

    def generate_moderate_growth_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate moderate growth scenario projections."""
        try:
            logger.info("Generating moderate growth scenario...")
            if self.predictions is None:
                logger.error("No base predictions available. Run generate_predictions first.")
                return pd.DataFrame()

            # Calculate moderate growth as average of high and low growth
            high_growth = self.generate_high_growth_scenario(data)
            low_growth = self.generate_low_growth_scenario(data)

            if high_growth.empty or low_growth.empty:
                return pd.DataFrame()

            scenario_df = self.predictions.copy()
            scenario_df["moderate_growth_prediction"] = (
                high_growth["high_growth_prediction"] + low_growth["low_growth_prediction"]
            ) / 2

            return scenario_df

        except Exception as e:
            logger.error(f"Error generating moderate growth scenario: {str(e)}")
            return pd.DataFrame()

    def generate_baseline_projection(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate baseline projection for report generation."""
        try:
            logger.info("Generating baseline population projection...")

            # Explicit check for DataFrame non-emptiness (prevents ambiguous truth value errors)
            if df is None and self.predictions is not None:
                df = self.predictions
            elif df is None:
                df = pd.read_csv(settings.PREDICTIONS_DIR / "population_predictions.csv")
            if df is not None and df.empty:
                logger.warning("Empty dataframe provided for baseline projection")
                return pd.DataFrame()

            required_cols = ["zip_code", "year", "base_prediction"]
            if any(col not in df.columns for col in required_cols):
                logger.error("Missing required columns for baseline projection")
                return pd.DataFrame()

            baseline = df[required_cols].copy()
            baseline.rename(columns={"base_prediction": "baseline_population"}, inplace=True)

            # Validate data
            if baseline["baseline_population"].isna().any():
                logger.warning("Found null values in baseline population predictions")
                baseline = baseline.dropna(subset=["baseline_population"])

            # Validate numeric values
            if not pd.to_numeric(baseline["baseline_population"], errors="coerce").notnull().all():
                logger.error("Non-numeric values found in baseline population")
                return pd.DataFrame()

            # Sort by zip code and year
            baseline = baseline.sort_values(["zip_code", "year"])

            # Validate that values are positive
            if (baseline["baseline_population"] < 0).any():
                logger.error("Negative population values found in baseline")
                return pd.DataFrame()

            return baseline

        except Exception as e:
            logger.error(f"Error generating baseline projection: {str(e)}")
            return pd.DataFrame()
