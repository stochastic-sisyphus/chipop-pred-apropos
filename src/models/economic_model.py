"""
Economic modeling module for Chicago economic impact analysis.
Handles scenario-based modeling and economic impact assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class EconomicModel:
    """Handles economic impact modeling and scenario analysis."""
    
    def __init__(self):
        """Initialize the economic model."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = None
        
        # Create model directory if it doesn't exist
        settings.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load processed data for economic analysis."""
        try:
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')

            # Verify required columns exist
            required_cols = [
                'total_population',
                'median_household_income',
                'total_permits',
                'total_construction_cost'
            ]

            if missing_cols := [
                col for col in required_cols if col not in df.columns
            ]:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None

            return df

        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            return None
            
    def prepare_features(self, df: pd.DataFrame, feature_list=None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for economic modeling."""
        try:
            logger.info("Preparing economic features...")
            
            # Use dynamic feature selection if feature_list is provided
            if feature_list is not None:
                features = [f for f in feature_list if f in df.columns and df[f].notna().sum() > 0]
                if not features:
                    logger.error("No usable features available for economic modeling.")
                    return None, None
                logger.info(f"Features used for economic modeling: {features}")
            elif self.feature_names is not None:
                # Reuse previously stored feature list
                features = self.feature_names
                logger.info(f"Reusing stored features for economic modeling: {features}")
            else:
                # Default features
                features = [
                    'total_population',
                    'median_household_income',
                    'labor_force',
                    'total_permits',
                    'residential_permits',
                    'commercial_permits',
                    'retail_permits',
                    'total_construction_cost',
                    'residential_construction_cost',
                    'commercial_construction_cost',
                    'retail_construction_cost'
                ]
                features = [f for f in features if f in df.columns and df[f].notna().sum() > 0]
                if not features:
                    logger.error("No usable default features for economic modeling.")
                    return None, None
                logger.info(f"Default features used for economic modeling: {features}")
            
            # Store feature names for reuse
            self.feature_names = features
            
            # Create feature matrix and target
            X = df[features].copy()
            y = df['gdp'] if 'gdp' in df.columns else None
            
            # Handle missing values and outliers
            if y is not None:
                mask = (~X.isin([np.nan, np.inf, -np.inf]).any(axis=1)) & (~y.isin([np.nan, np.inf, -np.inf]))
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
                'total_construction_cost', 'residential_construction_cost',
                'commercial_construction_cost', 'retail_construction_cost',
                'median_household_income'
            ]
            for col in large_value_cols:
                if col in X.columns:
                    X[col] = np.log1p(X[col])
            
            # Scale features
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            logger.info("Economic features prepared successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing economic features: {str(e)}")
            return None, None
            
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train the economic impact model."""
        try:
            logger.info("Training economic impact model...")
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # Initialize model with better parameters
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=3,
                learning_rate=0.05,
                random_state=42
            )
            # Train model
            self.model.fit(X_train, y_train)
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            logger.info(f"Economic model training complete. Test R² score: {test_score:.4f}")
            # Save feature importances
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            importances.to_csv(settings.MODEL_METRICS_DIR / 'economic_feature_importances.csv', index=False)
            return True
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
                
            # Define scenario adjustments
            scenarios = {
                'optimistic': {
                    'total_population': 1.1,
                    'median_household_income': 1.15,
                    'labor_force': 1.12,
                    'total_permits': 1.2,
                    'total_construction_cost': 1.25
                },
                'neutral': {
                    'total_population': 1.05,
                    'median_household_income': 1.08,
                    'labor_force': 1.06,
                    'total_permits': 1.1,
                    'total_construction_cost': 1.12
                },
                'pessimistic': {
                    'total_population': 0.98,
                    'median_household_income': 0.95,
                    'labor_force': 0.97,
                    'total_permits': 0.9,
                    'total_construction_cost': 0.85
                }
            }
            
            results = {}
            for scenario_name, adjustments in scenarios.items():
                logger.info(f"Generating {scenario_name} scenario...")
                
                # Apply scenario adjustments
                X_scenario = X_base.copy()
                for feature, factor in adjustments.items():
                    if feature in X_scenario.columns:
                        X_scenario[feature] *= factor
                
                # Generate predictions
                try:
                    predictions = self.model.predict(X_scenario)
                    results[scenario_name] = pd.DataFrame({
                        'gdp_prediction': predictions,
                        'scenario': scenario_name
                    }, index=X_scenario.index)
                    logger.info(f"Generated predictions for {scenario_name} scenario")
                except Exception as e:
                    logger.error(f"Failed to generate predictions for {scenario_name} scenario: {str(e)}")
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
            output_path = settings.OUTPUT_DIR / 'scenario_predictions.csv'
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
            # Prepare features
            X, y = self.prepare_features(df, feature_list=feature_list)
            if X is None or y is None:
                logger.error("Failed to prepare features for economic modeling.")
                return False
            # Train model
            if not self.train_model(X, y):
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
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')

            # Load scenario predictions
            scenario_df = pd.read_csv(settings.OUTPUT_DIR / 'scenario_predictions.csv')

            return pd.merge(
                df[['zip_code', 'year'] + self.feature_names],
                scenario_df,
                on='zip_code',
                how='left',
            )
        except Exception as e:
            logger.error(f"Error getting economic results: {str(e)}")
            return pd.DataFrame() 