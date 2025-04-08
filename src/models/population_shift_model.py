import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv

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
        """Initialize the population shift model"""
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.models_dir = Path(os.getenv('MODELS_DIR', 'models'))
        self.output_dir = Path(os.getenv('OUTPUT_DIR', 'output'))
        
        # Create necessary directories
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and merge all necessary datasets"""
        try:
            # Load historical population data
            population_df = pd.read_csv(self.data_dir / 'historical_population.csv')
            
            # Load building permits
            permits_df = pd.read_csv(self.data_dir / 'building_permits.csv')
            
            # Load economic indicators
            economic_df = pd.read_csv(self.data_dir / 'economic_indicators.csv')
            
            # Process and merge datasets
            # Convert dates
            permits_df['issue_date'] = pd.to_datetime(permits_df['issue_date'])
            permits_df['year'] = permits_df['issue_date'].dt.year
            
            # Aggregate permits by year and zip code
            permits_agg = permits_df.groupby(['year', 'zip_code']).agg({
                'reported_cost': 'sum',
                'permit_id': 'count'
            }).reset_index()
            permits_agg.columns = ['year', 'zip_code', 'total_permit_cost', 'permit_count']
            
            # Merge datasets
            merged_df = population_df.merge(
                permits_agg,
                left_on=['year', 'zip_code'],
                right_on=['year', 'zip_code'],
                how='left'
            )
            
            # Add economic indicators
            merged_df = merged_df.merge(
                economic_df[['year', 'treasury_10y', 'consumer_sentiment', 'recession_indicator']],
                on='year',
                how='left'
            )
            
            # Fill missing values
            merged_df = merged_df.fillna({
                'total_permit_cost': 0,
                'permit_count': 0
            })
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def prepare_features(self, df):
        """Prepare feature matrix for modeling"""
        # Select features
        feature_columns = [
            'total_permit_cost',
            'permit_count',
            'treasury_10y',
            'consumer_sentiment',
            'recession_indicator',
            'median_household_income',
            'total_households'
        ]
        
        # Create lag features for population
        df['population_lag1'] = df.groupby('zip_code')['population'].shift(1)
        df['population_lag2'] = df.groupby('zip_code')['population'].shift(2)
        
        feature_columns.extend(['population_lag1', 'population_lag2'])
        
        # Drop rows with missing values
        df = df.dropna(subset=feature_columns + ['population'])
        
        # Prepare X and y
        X = df[feature_columns]
        y = df['population']
        
        return X, y, feature_columns
        
    def train_model(self, X_train, y_train):
        """Train the population shift model"""
        try:
            # Initialize and train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            joblib.dump(self.model, self.models_dir / 'population_shift_model.pkl')
            joblib.dump(self.scaler, self.models_dir / 'feature_scaler.pkl')
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
            
    def generate_scenarios(self, X_test):
        """Generate predictions for different scenarios"""
        try:
            scenarios = {
                'optimistic': {
                    'treasury_10y': 0.8,  # 20% lower rates
                    'consumer_sentiment': 1.2,  # 20% higher confidence
                    'recession_indicator': 0  # No recession
                },
                'neutral': {
                    'treasury_10y': 1.0,  # Current rates
                    'consumer_sentiment': 1.0,  # Current confidence
                    'recession_indicator': 0  # No recession
                },
                'pessimistic': {
                    'treasury_10y': 1.2,  # 20% higher rates
                    'consumer_sentiment': 0.8,  # 20% lower confidence
                    'recession_indicator': 1  # Recession
                }
            }
            
            predictions = {}
            
            for scenario, adjustments in scenarios.items():
                # Create copy of test data
                X_scenario = X_test.copy()
                
                # Apply scenario adjustments
                for feature, factor in adjustments.items():
                    if feature in X_scenario.columns:
                        if feature == 'recession_indicator':
                            X_scenario[feature] = factor
                        else:
                            X_scenario[feature] *= factor
                
                # Scale features
                X_scenario_scaled = self.scaler.transform(X_scenario)
                
                # Generate predictions
                predictions[scenario] = self.model.predict(X_scenario_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {str(e)}")
            return None
            
    def run_analysis(self):
        """Run the complete population shift analysis"""
        try:
            # Load data
            logger.info("Loading data...")
            df = self.load_data()
            
            if df is None:
                return False
            
            # Prepare features
            logger.info("Preparing features...")
            X, y, feature_columns = self.prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            logger.info("Training model...")
            if not self.train_model(X_train, y_train):
                return False
            
            # Generate scenarios
            logger.info("Generating scenarios...")
            predictions = self.generate_scenarios(X_test)
            
            if predictions is None:
                return False
            
            # Prepare results
            results = pd.DataFrame({
                'Actual': y_test,
                'Optimistic': predictions['optimistic'],
                'Neutral': predictions['neutral'],
                'Pessimistic': predictions['pessimistic']
            })
            
            # Save results
            results.to_csv(self.output_dir / 'population_predictions.csv')
            logger.info("Analysis complete. Results saved to output directory.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return False


if __name__ == "__main__":
    model = PopulationShiftModel()
    model.run_analysis()