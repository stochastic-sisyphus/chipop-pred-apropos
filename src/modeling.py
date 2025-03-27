import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import joblib
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PopulationShiftModel:
    def __init__(self):
        self.output_dir = Path(os.getenv('OUTPUT_DIR', 'output'))
        self.model_dir = self.output_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
        self.features = [
            'permit_count',
            'total_cost',
            'total_land_area',
            'treasury_10y',
            'mortgage_30y',
            'consumer_sentiment',
            'housing_starts',
            'house_price_index',
            'middle_class_pct',
            'lower_income_pct',
            'median_household_income'
        ]
        
        # Update target to use future population change
        self.target = 'population_change_next_year'
        # Split year for time-based validation (train on ≤2020, test on >2020)
        self.train_test_split_year = 2020
        
        # Initialize these variables to store data and model
        self.model = None
        self.X = None
        self.y = None
        self.df = None
        
    def prepare_data(self):
        """Load and prepare data for modeling"""
        logger.info("Preparing data for modeling...")

        # Load merged dataset
        merged_dataset_path = self.output_dir / 'merged_dataset.csv'
        if not merged_dataset_path.exists():
            logger.error(f"Merged dataset not found: {merged_dataset_path}")
            logger.error("Run data_processing.py first to create the merged dataset")
            return None, None, None

        try:
            df = pd.read_csv(merged_dataset_path)

            # Check if target column exists
            if self.target not in df.columns:
                available_cols = df.columns.tolist()
                logger.error(f"Target column '{self.target}' not found in dataset")
                logger.error(f"Available columns: {available_cols}")
                return None, None, None

            # Remove rows with missing target (last year of each ZIP)
            df = df.dropna(subset=[self.target])
            logger.info(f"Data after removing rows with missing target: {len(df)} rows")

            # Check if we have enough data
            if len(df) < 30:  # Minimum sample size for reasonable modeling
                logger.warning(f"Not enough data for modeling: only {len(df)} rows after filtering")

            # Check for missing feature columns
            if missing_features := [
                f for f in self.features if f not in df.columns
            ]:
                logger.warning(f"Missing feature columns: {missing_features}")
                # Use only available features
                available_features = [f for f in self.features if f in df.columns]
                if not available_features:
                    logger.error("No valid features found in dataset")
                    return None, None, None
                logger.info(f"Proceeding with available features: {available_features}")
                self.features = available_features

            # Keep the original DataFrame for reference (contains ZIP codes and years)
            df_original = df.copy()
            
            # Extract features and target
            X = df[self.features].copy()
            y = df[self.target].copy()

            # Handle missing values in features
            X = X.fillna(X.mean())

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Save scaler for later use
            joblib.dump(scaler, self.model_dir / 'scaler.joblib')

            # Save feature names
            pd.Series(self.features).to_csv(self.model_dir / 'feature_names.csv', index=False)

            return X_scaled, y, df_original
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None, None
        
    def train_model(self):
        """Train the population shift prediction model"""
        logger.info("Training model...")
        
        X, y, df_original = self.prepare_data()
        
        if X is None or y is None or df_original is None:
            logger.error("Cannot train model: data preparation failed")
            return None, None, None
            
        try:
            # Time-based split: train on data <= split year, test on data > split year
            logger.info(f"Using time-based train-test split: train on ≤{self.train_test_split_year}, test on >{self.train_test_split_year}")
            
            train_mask = df_original['year'] <= self.train_test_split_year
            test_mask = df_original['year'] > self.train_test_split_year
            
            # Check if we have enough test data
            if test_mask.sum() < 10:
                logger.warning(f"Very small test set: only {test_mask.sum()} samples. Consider using random split instead.")
                # Fallback to random split if needed
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]
                logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Store the model and data for later use
            self.model = model
            self.X = pd.DataFrame(X, columns=self.features)
            self.y = y
            self.df = df_original
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            logger.info(f"Train R² score: {train_score:.4f}")
            logger.info(f"Test R² score: {test_score:.4f}")
            logger.info(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(self.output_dir / 'feature_importance.csv', index=False)
            logger.info("\nFeature Importance:")
            logger.info(feature_importance.head(10).to_string())
            
            # Save model
            joblib.dump(model, self.model_dir / 'population_shift_model.joblib')
            
            return model, X, df_original
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None, None, None
        
    def generate_scenarios(self, base_year=None):
        """
        Generate optimistic, neutral, and pessimistic scenario predictions
        
        Args:
            base_year (int, optional): Base year from which to generate predictions. If None, uses max year in data.
            
        Returns:
            DataFrame: Predictions for different scenarios
        """
        try:
            # Load data if not already available
            if not hasattr(self, 'model') or self.model is None:
                logger.warning("Model not trained. Training now...")
                model, X, df_original = self.train_model()
                if model is None:
                    logger.error("Failed to train model for scenarios")
                    return None
            
            # Determine base year for prediction
            if base_year is None:
                base_year = self.df['year'].max()
            
            logger.info(f"Generating scenarios from base year: {base_year}")
            
            # Filter data for the base year
            base_data = self.df[self.df['year'] == base_year].copy()
            
            if len(base_data) == 0:
                logger.error(f"No data available for base year {base_year}")
                return None
                
            # Define scenario adjustments (these could be made configurable)
            scenarios = {
                'optimistic': {
                    'mortgage_30y': -0.5,  # 0.5 percentage point decrease
                    'median_household_income': 0.05,  # 5% increase
                    'consumer_sentiment': 10  # 10 point increase
                },
                'neutral': {
                    'mortgage_30y': 0,  # No change
                    'median_household_income': 0.02,  # 2% increase
                    'consumer_sentiment': 0  # No change
                },
                'pessimistic': {
                    'mortgage_30y': 0.75,  # 0.75 percentage point increase
                    'median_household_income': -0.01,  # 1% decrease
                    'consumer_sentiment': -15  # 15 point decrease
                }
            }
            
            # Define prediction windows (in years)
            prediction_windows = [1, 2]
            
            # Generate predictions for each scenario and prediction window
            all_predictions = []
            
            for scenario_name, adjustments in scenarios.items():
                for window in prediction_windows:
                    prediction_year = base_year + window
                    logger.info(f"Generating {scenario_name} scenario for {window}-year horizon (year {prediction_year})")
                    
                    # Create copy of base data for this scenario
                    scenario_data = base_data.copy()
                    
                    # Apply scenario adjustments
                    for feature, adjustment in adjustments.items():
                        if feature in scenario_data.columns:
                            # For percentage adjustments
                            if feature in ['median_household_income']:
                                scenario_data[feature] = scenario_data[feature] * (1 + adjustment)
                            # For absolute adjustments
                            else:
                                scenario_data[feature] = scenario_data[feature] + adjustment
                    
                    # Extract features for prediction
                    X_scenario = scenario_data[self.features]
                    
                    # Make predictions
                    predictions = self.model.predict(X_scenario)
                    
                    # Add predictions to scenario data
                    scenario_data['predicted_population_change'] = predictions
                    scenario_data['scenario'] = scenario_name
                    scenario_data['prediction_year'] = prediction_year
                    scenario_data['prediction_window'] = f"{window}-Year"
                    
                    # Add to all predictions
                    all_predictions.append(scenario_data[['zip_code', 'year', 'prediction_year', 
                                                         'scenario', 'prediction_window', 
                                                         'predicted_population_change']])
            
            # Combine all scenarios
            if all_predictions:
                all_scenarios_df = pd.concat(all_predictions)
                
                # Save to CSV
                output_path = self.output_dir / 'scenario_predictions.csv'
                all_scenarios_df.to_csv(output_path, index=False)
                logger.info(f"Scenario predictions saved to {output_path}")
                
                # Generate summary statistics
                summary = all_scenarios_df.groupby(['scenario', 'prediction_window'])['predicted_population_change'].agg(
                    ['mean', 'std', 'min', 'max']
                ).reset_index()
                
                # Save summary
                summary_path = self.output_dir / 'scenario_summary.csv'
                summary.to_csv(summary_path, index=False)
                logger.info(f"Scenario summary saved to {summary_path}")
                
                return all_scenarios_df
            else:
                logger.error("Failed to generate scenario predictions")
                return None
                
        except Exception as e:
            logger.error(f"Error generating scenarios: {str(e)}")
            logger.exception("Full traceback:")
            return None

def main():
    model_trainer = PopulationShiftModel()
    model, X, df_original = model_trainer.train_model()
    if model is not None and X is not None and df_original is not None:
        predictions = model_trainer.generate_scenarios()

if __name__ == "__main__":
    main()