"""
Population analysis model for Chicago ZIP codes.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.config import settings
from src.utils.helpers import sanitize_features, match_features, resolve_column_name, safe_train_model
from src.config.column_alias_map import column_aliases

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
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_col = None
        self.predictions = None
        self.scenarios = None
        # Create model directory if it doesn't exist
        settings.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame, feature_list=None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for population modeling."""
        try:
            logger.info("Preparing population features...")
            
            # Use dynamic feature selection if feature_list is provided
            if feature_list is not None:
                features = match_features(df, feature_list, column_aliases)
            else:
                # Default features
                default_features = [
                    'median_household_income',
                    'median_home_value',
                    'labor_force',
                    'residential_permits',
                    'commercial_permits',
                    'retail_permits',
                    'residential_construction_cost',
                    'commercial_construction_cost',
                    'retail_construction_cost'
                ]
                features = match_features(df, default_features, column_aliases)
            
            if not features:
                logger.error("No usable features available for population modeling.")
                return None, None
                
            logger.info(f"Features used for population modeling: {features}")
            
            # Store feature names for reuse
            self.feature_names = features
            
            # Resolve target variable
            target_col = resolve_column_name(df, 'total_population', column_aliases)
            
            # Create feature matrix and target
            X = df[features].copy()
            y = df[target_col] if target_col else None
            
            # Convert all columns to numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            if y is not None:
                y = pd.to_numeric(y, errors='coerce')
            
            # Log missing value counts before dropping
            logger.info("Missing values per column before drop:\n" + str(df[features + ([target_col] if target_col else [])].isnull().sum()))
            
            # Handle missing values and outliers
            if y is not None:
                mask = (~X.isnull().any(axis=1)) & (~y.isnull())
            else:
                mask = ~X.isnull().any(axis=1)
            dropped = (~mask).sum()
            if dropped > 0:
                logger.warning(f"Dropping {dropped} rows with NaN values.")
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
                'median_household_income', 'median_home_value',
                'residential_construction_cost', 'commercial_construction_cost',
                'retail_construction_cost'
            ]
            for col in large_value_cols:
                resolved_col = resolve_column_name(X, col, column_aliases)
                if resolved_col in X.columns:
                    X[resolved_col] = np.log1p(X[resolved_col])
            
            # Scale features
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            logger.info("Population features prepared successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing population features: {str(e)}")
            return None, None

    def train(self, df: pd.DataFrame) -> bool:
        """Train the population model."""
        try:
            logger.info("Training population model...")
            
            # Prepare features
            X, y = self.prepare_features(df)
            if X is None or y is None:
                logger.error("Failed to prepare features for training")
                return False
            
            # Train model
            success = safe_train_model(self.model, X, y, model_name="PopulationModel")
            if success:
                logger.info("Population model trained successfully")
                
                # Save model and scaler
                joblib.dump(self.model, settings.TRAINED_MODELS_DIR / 'population_model.joblib')
                joblib.dump(self.scaler, settings.TRAINED_MODELS_DIR / 'population_scaler.joblib')
                logger.info("Model and scaler saved")
                
            return success
            
        except Exception as e:
            logger.error(f"Error training population model: {str(e)}")
            return False

    def generate_predictions(self):
        """Generate population predictions for all scenarios."""
        try:
            logger.info("Generating population predictions...")
            
            # Load processed data
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')
            
            # Get column names
            zip_col = resolve_column_name(df, 'zip_code', column_aliases)
            year_col = resolve_column_name(df, 'year', column_aliases)
            
            if not all([zip_col, year_col]):
                logger.error("Required columns not found")
                return None
            
            # Prepare features using the same feature set used for training
            expected_features = [
                'median_household_income', 'median_home_value', 'labor_force',
                'residential_permits', 'commercial_permits', 'retail_permits',
                'residential_construction_cost', 'commercial_construction_cost',
                'retail_construction_cost', 'total_licenses', 'unique_businesses',
                'active_licenses', 'total_permits'
            ]
            
            feature_cols = match_features(df, expected_features, column_aliases)
            clean_df = sanitize_features(df, feature_cols)
            X = clean_df[feature_cols]
            
            # Store original index for later use
            original_index = X.index
                
            # Generate base predictions
            base_predictions = self.model.predict(X)
            
            # Create scenarios dictionary with adjustments
            scenarios = settings.SCENARIOS
            scenario_predictions = {}
            
            for scenario_name, adjustments in scenarios.items():
                # Adjust features based on scenario
                X_scenario = X.copy()
                
                # Apply scenario adjustments
                growth_factor = adjustments.get('growth_factor', 1.0)
                X_scenario = X_scenario * growth_factor
                
                # Generate predictions for this scenario
                predictions = self.model.predict(X_scenario)
                
                # Store results
                scenario_predictions[scenario_name] = predictions
            
            # Create results DataFrame using the correct index
            results_df = pd.DataFrame({
                zip_col: df.loc[original_index, zip_col],
                year_col: df.loc[original_index, year_col],
                'base_prediction': scenario_predictions.get('base'),
                'optimistic_prediction': scenario_predictions.get('optimistic'),
                'neutral_prediction': scenario_predictions.get('neutral'),
                'pessimistic_prediction': scenario_predictions.get('pessimistic')
            })
            
            # Save predictions
            results_df.to_csv(settings.PREDICTIONS_DIR / 'population_predictions.csv', index=False)
            logger.info("Generated population predictions successfully")
            
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
            zip_col = resolve_column_name(self.predictions, 'zip_code', column_aliases)
            year_col = resolve_column_name(self.predictions, 'year', column_aliases)
            
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
                    'base': 'base_prediction',
                    'optimistic': 'optimistic_prediction',
                    'neutral': 'neutral_prediction',
                    'pessimistic': 'pessimistic_prediction'
                }.get(scenario_name)
                
                if prediction_col not in scenario_df.columns:
                    logger.error(f"Missing prediction column for scenario {scenario_name}")
                    continue
                
                # Add scenario information
                scenario_df['predicted_population'] = scenario_df[prediction_col]
                scenario_df['scenario'] = scenario_name
                scenario_df['growth_factor'] = factors.get('growth_factor', 1.0)
                scenario_df['confidence'] = factors.get('confidence', 1.0)
                
                scenario_predictions.append(scenario_df)
            
            if not scenario_predictions:
                logger.error("No valid scenarios generated")
                return False
            
            # Combine all scenarios
            all_scenarios = pd.concat(scenario_predictions, ignore_index=True)
            
            # Save scenario predictions
            all_scenarios.to_csv(settings.PREDICTIONS_DIR / 'population_scenarios.csv', index=False)
            
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
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')
            
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
                'high_growth',
                'high_growth_prediction',
            )
        except Exception as e:
            logger.error(f"Error generating high growth scenario: {str(e)}")
            return pd.DataFrame()

    def generate_low_growth_scenario(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate low growth scenario projections."""
        try:
            return self._extracted_from_generate_low_growth_scenario_4(
                "Generating low growth scenario...",
                'low_growth',
                'low_growth_prediction',
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
        growth_factor = settings.SCENARIOS[arg1]['population_growth']
        scenario_df[arg2] = scenario_df['base_prediction'] * growth_factor
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
            scenario_df['moderate_growth_prediction'] = (
                high_growth['high_growth_prediction'] + low_growth['low_growth_prediction']
            ) / 2
            
            return scenario_df
            
        except Exception as e:
            logger.error(f"Error generating moderate growth scenario: {str(e)}")
            return pd.DataFrame()

    def generate_baseline_projection(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate baseline projection for report generation."""
        try:
            logger.info("Generating baseline population projection...")
            
            if df is None and self.predictions is not None:
                df = self.predictions
            elif df is None:
                df = pd.read_csv(settings.PREDICTIONS_DIR / 'population_predictions.csv')
            
            if df.empty:
                logger.warning("Empty dataframe provided for baseline projection")
                return pd.DataFrame()
            
            required_cols = ['zip_code', 'year', 'base_prediction']
            if any(col not in df.columns for col in required_cols):
                logger.error("Missing required columns for baseline projection")
                return pd.DataFrame()
            
            baseline = df[required_cols].copy()
            baseline.rename(columns={'base_prediction': 'baseline_population'}, inplace=True)
            
            # Validate data
            if baseline['baseline_population'].isna().any():
                logger.warning("Found null values in baseline population predictions")
                baseline = baseline.dropna(subset=['baseline_population'])
            
            # Validate numeric values
            if not pd.to_numeric(baseline['baseline_population'], errors='coerce').notnull().all():
                logger.error("Non-numeric values found in baseline population")
                return pd.DataFrame()
            
            # Sort by zip code and year
            baseline = baseline.sort_values(['zip_code', 'year'])
            
            # Validate that values are positive
            if (baseline['baseline_population'] < 0).any():
                logger.error("Negative population values found in baseline")
                return pd.DataFrame()
            
            return baseline
            
        except Exception as e:
            logger.error(f"Error generating baseline projection: {str(e)}")
            return pd.DataFrame()