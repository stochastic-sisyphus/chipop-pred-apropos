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

from src.config import settings

logger = logging.getLogger(__name__)

class PopulationModel:
    """Model for analyzing population trends and making predictions."""
    
    def __init__(self):
        """Initialize population model."""
        self.full_feature_names = []
        self.scenario_feature_names = [
            'total_population', 'median_household_income', 'labor_force',
            'total_permits', 'residential_permits', 'commercial_permits', 'retail_permits',
            'total_construction_cost', 'residential_construction_cost',
            'commercial_construction_cost', 'retail_construction_cost'
        ]
        self.model = None
        self.feature_names = None
        self.predictions = None
        self.scenarios = None
        # Create model directory if it doesn't exist
        settings.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame, feature_list=None, mode: str = 'full', target_variable: str = 'total_population') -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features for population modeling."""
        try:
            logger.info(f"Preparing population features for target: {target_variable}")

            if feature_list is not None:
                features = [f for f in feature_list if f in df.columns and df[f].notna().sum() > 0]
            elif mode == 'full':
                features = [
                    'total_population', 'median_household_income', 'labor_force',
                    'total_permits', 'residential_permits', 'commercial_permits', 'retail_permits', 
                    'total_construction_cost', 'residential_construction_cost',
                    'commercial_construction_cost', 'retail_construction_cost',
                    'median_age', 'households', 'housing_units', 'vacancy_rate',
                    'owner_occupied_rate', 'poverty_rate', 'unemployment_rate',
                    'education_less_than_hs', 'education_hs', 'education_some_college',
                    'education_bachelors', 'education_graduate',
                    'industry_manufacturing', 'industry_retail', 'industry_professional',
                    'industry_education_health', 'industry_arts_entertainment',
                    'commute_time', 'commute_public_transit', 'commute_carpool',
                    'health_insurance_coverage', 'broadband_internet',
                    'median_rooms', 'median_year_built', 'median_gross_rent',
                    'median_home_value', 'household_family_percent',
                    'foreign_born', 'speak_english_less_than_very_well'
                ]
            elif mode == 'scenario':
                features = [f for f in self.scenario_feature_names if f in df.columns]

            if not features:
                logger.error(f"No usable features for mode: {mode}")
                return None, None

            logger.info(f"Features used for {mode} modeling: {features}")

            # Create feature matrix
            X = df[features].copy()

            # Validate target
            if target_variable not in df.columns:
                logger.error(f"Target variable '{target_variable}' not found in dataframe.")
                return None, None
            y = df[target_variable]

            # Fill missing values with median (for features)
            X = X.fillna(X.median())

            # Remove rows with any NaN or inf in X or y
            mask = (~X.isin([np.nan, np.inf, -np.inf]).any(axis=1)) & (~y.isin([np.nan, np.inf, -np.inf]))
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

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            return pd.DataFrame(X_scaled, columns=features, index=X.index), y.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error preparing population features: {str(e)}")
            return None, None
            
    def train(self, df, feature_list=None, target_variable='total_population'):
        """Train the population model."""
        try:
            logger.info("Training population model...")
            
            # Prepare features
            X, y = self.prepare_features(df, feature_list, target_variable)
            if X is None or y is None:
                logger.error("Failed to prepare features for training")
                return False
                
            # Store feature names for prediction
            self.feature_names = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Save feature importances
            importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importances.to_csv(settings.MODEL_METRICS_DIR / 'population_feature_importances.csv', index=False)
            
            logger.info(f"Model trained successfully. R² score: {test_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
            
    def generate_predictions(self):
        """Generate population predictions for all scenarios."""
        try:
            logger.info("Generating population predictions...")
            
            # Load processed data
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')
            
            # Prepare features using the same feature set used for training
            X, _ = self.prepare_features(df, feature_list=self.feature_names)
            if X is None:
                logger.error("Failed to prepare features for prediction generation")
                return None
                
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
                if scenario_name != 'base':
                    population_factor = adjustments.get('population_growth', 1.0)
                    X_scenario = X_scenario * population_factor
                
                # Generate predictions for this scenario
                predictions = self.model.predict(X_scenario)
                
                # Store results
                scenario_predictions[scenario_name] = predictions
            
            # Create results DataFrame using the correct index
            results_df = pd.DataFrame({
                'zip_code': df.loc[original_index, 'zip_code'],
                'year': df.loc[original_index, 'year'],
                'base_prediction': scenario_predictions['base'],
                'high_growth_prediction': scenario_predictions['high_growth'],
                'low_growth_prediction': scenario_predictions['low_growth']
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
            
            # Create scenario predictions
            scenario_predictions = []
            
            for scenario_name, factors in settings.SCENARIOS.items():
                # Create scenario DataFrame
                scenario_df = self.predictions.copy()
                
                # Map scenario name to prediction column
                prediction_col = {
                    'base': 'base_prediction',
                    'high_growth': 'high_growth_prediction',
                    'low_growth': 'low_growth_prediction'
                }.get(scenario_name)
                
                if prediction_col not in scenario_df.columns:
                    logger.error(f"Missing prediction column for scenario {scenario_name}")
                    continue
                
                # Add scenario information
                scenario_df['predicted_population'] = scenario_df[prediction_col]
                scenario_df['scenario'] = scenario_name
                scenario_df['population_growth_factor'] = factors.get('population_growth', 1.0)
                scenario_df['gdp_growth_factor'] = factors.get('gdp_growth', 1.0)
                scenario_df['employment_growth_factor'] = factors.get('employment_growth', 1.0)
                scenario_df['income_growth_factor'] = factors.get('income_growth', 1.0)
                
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
            
    def run_analysis(self, feature_list=None, target_variable='total_population'):
        """Run the complete population analysis pipeline."""
        try:
            # Load data
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')
            
            # Train model
            if not self.train(df, feature_list=feature_list, target_variable=target_variable):
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