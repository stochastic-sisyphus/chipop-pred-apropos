"""
Economic modeling module for Chicago economic impact analysis.
Handles scenario-based modeling and economic impact assessment.
"""

import logging
from typing import Dict, Optional, Tuple, Union # Added Union

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import settings
from src.utils.helpers import (
    match_features,
    resolve_column_name,
    safe_train_model,
)
from src.config.column_alias_map import column_aliases # type: ignore

# Set up logging
logger = logging.getLogger(__name__)


class EconomicModel:
    """Handles economic impact modeling and scenario analysis."""

    def __init__(self):
        """Initialize the economic model."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = None

        # Create model directory if it doesn't exist
        settings.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load processed data for economic analysis."""
        try:
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / "merged_dataset.csv")

            # Verify required columns exist using aliases
            required_cols = [
                "total_population",
                "median_household_income",
                "total_permits",
                "total_construction_cost",
            ]
            resolved_cols = [resolve_column_name(df, col, column_aliases) for col in required_cols]

            if missing_cols := [
                col for col, resolved in zip(required_cols, resolved_cols) if not resolved
            ]:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None

            return df

        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            return None

    def prepare_features(
        self, df: pd.DataFrame, feature_list=None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for economic modeling."""
        try:
            logger.info("Preparing economic features...")

            # Use dynamic feature selection if feature_list is provided
            if feature_list is not None:
                features = match_features(df, feature_list, column_aliases)
            else:
                # Default features
                default_features = [
                    "total_population",
                    "median_household_income",
                    "labor_force",
                    "residential_permits",
                    "commercial_permits",
                    "retail_permits",
                ]

                # Optional features - will use if available
                optional_features = [
                    "gdp",
                    "unemployment_rate",
                    "per_capita_income",
                    "personal_income",
                    "residential_construction_cost",
                    "commercial_construction_cost",
                    "retail_construction_cost",
                ]

                # Start with required features
                features = default_features.copy()

                # Add optional features if available
                for feature in optional_features:
                    if feature in df.columns:
                        features.append(feature)
                    else:
                        logger.warning(
                            f"Optional feature '{feature}' not found - model will proceed without it"
                        )

            logger.info(f"Default features used for economic modeling: {features}")

            # Prepare feature matrix X
            X = df[features].copy()

            # Target variable is median_household_income if not specified
            target = "median_household_income"
            if target not in df.columns:
                logger.error(f"Target variable {target} not found in dataset")
                return None, None

            y = df[target]

            # Handle missing values
            for col in X.columns:
                if X[col].isna().any():
                    logger.warning(f"Found {X[col].isna().sum()} missing values in {col}")
                    if col in ["gdp", "unemployment_rate", "per_capita_income", "personal_income"]:
                        # For economic indicators, forward/backward fill
                        X[col] = X[col].fillna(method="ffill").fillna(method="bfill")
                    else:
                        # For other features, fill with median
                        X[col] = X[col].fillna(X[col].median())

            # Log shape of prepared data
            logger.info(f"Prepared {len(X)} samples with {len(features)} features")
            logger.info("Economic features prepared successfully")

            return X, y

        except Exception as e:
            logger.error(f"Error preparing economic features: {str(e)}")
            return None, None

    def train(self, data_input: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """Train the economic model."""
        try:
            logger.info("Training economic model...")

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
            success = safe_train_model(self.model, X, y, model_name="EconomicModel")
            if success:
                logger.info("Economic model trained successfully")

                # Save model and scaler
                joblib.dump(self.model, settings.TRAINED_MODELS_DIR / "economic_model.joblib")
                joblib.dump(self.scaler, settings.TRAINED_MODELS_DIR / "economic_scaler.joblib")
                logger.info("Model and scaler saved")

            return success

        except Exception as e:
            logger.error(f"Error training economic model: {str(e)}")
            return False

    def generate_scenarios(self, base_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate economic impact scenarios."""
        try:
            logger.info("Generating economic impact scenarios...")

            if self.model is None:
                logger.error("Model not trained. Please train the model first.")
                return None

            if self.feature_names is None:
                logger.error("No feature names stored. Please prepare features first.")
                return None

            # Prepare base features using stored feature names
            X_base, _ = self.prepare_features(base_data)
            if X_base is None:
                logger.error("Failed to prepare base features for scenario generation")
                return None

            # Define scenario adjustments with resolved column names
            base_adjustments = {
                "total_population": 1.1,
                "median_household_income": 1.15,
                "labor_force": 1.12,
                "total_permits": 1.2,
                "total_construction_cost": 1.25,
            }

            scenarios = {
                "optimistic": {
                    col: resolve_column_name(X_base, col, column_aliases) or col
                    for col in base_adjustments
                },
                "neutral": {
                    col: resolve_column_name(X_base, col, column_aliases) or col
                    for col in base_adjustments
                },
                "pessimistic": {
                    col: resolve_column_name(X_base, col, column_aliases) or col
                    for col in base_adjustments
                },
            }

            results = {}
            for scenario_name, adjustments in scenarios.items():
                logger.info(f"Generating {scenario_name} scenario...")

                # Apply scenario adjustments
                X_scenario = X_base.copy()
                for feature, resolved_feature in adjustments.items():
                    if resolved_feature in X_scenario.columns:
                        factor = settings.SCENARIOS[scenario_name].get("growth_factor", 1.0)
                        X_scenario[resolved_feature] *= factor

                # Generate predictions
                try:
                    predictions = self.model.predict(X_scenario)
                    results[scenario_name] = pd.DataFrame(
                        {"gdp_prediction": predictions, "scenario": scenario_name},
                        index=X_scenario.index,
                    )
                    logger.info(f"Generated predictions for {scenario_name} scenario")
                except Exception as e:
                    logger.error(
                        f"Failed to generate predictions for {scenario_name} scenario: {str(e)}"
                    )
                    continue

            if not results:
                logger.error("Failed to generate any scenario predictions")
                return None

            # Store results
            self.results = results
            logger.info("Economic impact scenarios generated successfully")

            # Combine all scenarios into one DataFrame
            scenarios_combined = pd.concat(self.results.values(), ignore_index=True)

            # Save scenario results
            output_path = settings.OUTPUT_DIR / "scenario_predictions.csv"
            scenarios_combined.to_csv(output_path, index=False)
            logger.info(f"Saved economic scenario predictions to {output_path}")

            return results

        except Exception as e:
            logger.error(f"Error generating economic scenarios: {str(e)}")
            return None

    def run_analysis(self, feature_list=None) -> bool:
        """Run complete economic analysis pipeline."""
        try:
            # Load data
            df = self.load_data()
            if df is None:
                return False
            # Train model
            if not self.train(df):
                return False
            # Generate scenarios
            if self.generate_scenarios(df) is None:
                return False
            logger.info("Economic analysis pipeline completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error in economic analysis pipeline: {str(e)}")
            return False

    def get_results(self) -> pd.DataFrame:
        """Get economic analysis results for visualization."""
        try:
            # Load merged dataset
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / "merged_dataset.csv")

            # Load scenario predictions
            scenario_df = pd.read_csv(settings.OUTPUT_DIR / "scenario_predictions.csv")

            # Get column names
            zip_col = resolve_column_name(df, "zip_code", column_aliases)
            if not zip_col:
                logger.error("ZIP code column not found")
                return pd.DataFrame()

            return pd.merge(
                df[[zip_col] + self.feature_names],
                scenario_df,
                on=zip_col,
                how="left",
            )
        except Exception as e:
            logger.error(f"Error getting economic results: {str(e)}")
            return pd.DataFrame()
