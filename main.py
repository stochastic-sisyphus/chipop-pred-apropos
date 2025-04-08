"""
Main script to run the entire Chicago Population Shift Analysis pipeline.
This script coordinates data collection, processing, modeling, and output generation.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.data_processing.data_collector import DataCollector
from src.models.population_shift_model import PopulationShiftModel
from src.visualization.visualizer import PopulationVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def verify_api_keys():
    """Verify that all required API keys are present"""
    required_keys = {
        'CENSUS_API_KEY': os.getenv('CENSUS_API_KEY'),
        'FRED_API_KEY': os.getenv('FRED_API_KEY')
    }
    
    missing_keys = [k for k, v in required_keys.items() if not v]
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        return False
    return True

def verify_directories():
    """Create and verify required directories"""
    required_dirs = ['data', 'models', 'output', 'output/visualizations']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"Verified directory: {dir_name}")
    return True

def collect_permit_data():
    """Collect building permit data from Chicago Data Portal"""
    try:
        from sodapy import Socrata
        
        client = Socrata("data.cityofchicago.org", None)
        
        # Get building permits (limit to recent years and residential/mixed-use)
        results = client.get(
            "ydr8-5enu",
            limit=50000,
            where="permit_type in ('PERMIT - NEW CONSTRUCTION', 'PERMIT - RENOVATION/ALTERATION')"
        )
        
        if not results:
            logger.error("No permit data retrieved")
            return False
            
        permits_df = pd.DataFrame.from_records(results)
        permits_df.to_csv("data/chicago_permits.csv", index=False)
        logger.info(f"Saved {len(permits_df)} building permits")
        return True
        
    except Exception as e:
        logger.error(f"Error collecting permit data: {str(e)}")
        return False

def collect_census_data():
    """Collect population and demographic data from Census API"""
    try:
        from census import Census
        c = Census(os.getenv('CENSUS_API_KEY'))
        
        # Get data for last 20 years
        current_year = datetime.now().year
        start_year = current_year - 20
        
        all_data = []
        for year in range(start_year, current_year):
            try:
                # B01003_001E is total population
                # B19013_001E is median household income
                data = c.acs5.get(
                    ('B01003_001E', 'B19013_001E'),
                    {'for': 'zip code tabulation area:*'},
                    year=year
                )
                if data:
                    df = pd.DataFrame(data)
                    df['year'] = year
                    all_data.append(df)
                    logger.info(f"Collected Census data for {year}")
            except Exception as e:
                logger.warning(f"Could not collect data for {year}: {str(e)}")
        
        if all_data:
            historical_df = pd.concat(all_data, ignore_index=True)
            historical_df.to_csv("data/historical_population.csv", index=False)
            logger.info("Saved historical population data")
            return True
        else:
            logger.error("No Census data collected")
            return False
            
    except Exception as e:
        logger.error(f"Error collecting Census data: {str(e)}")
        return False

def collect_economic_data():
    """Collect economic indicators from FRED"""
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        
        # Define indicators to collect
        indicators = {
            'DGS10': 'treasury_10y',
            'MORTGAGE30US': 'mortgage_30y',
            'UMCSENT': 'consumer_sentiment',
            'USREC': 'recession_indicator'
        }
        
        data = {}
        for series_id, name in indicators.items():
            series = fred.get_series(series_id)
            data[name] = series
            logger.info(f"Collected {name} data from FRED")
        
        econ_df = pd.DataFrame(data)
        econ_df.to_csv("data/economic_indicators.csv")
        logger.info("Saved economic indicators data")
        return True
        
    except Exception as e:
        logger.error(f"Error collecting economic data: {str(e)}")
        return False

def process_data():
    """Process and merge all collected data"""
    try:
        # Load datasets
        permits_df = pd.read_csv("data/chicago_permits.csv")
        population_df = pd.read_csv("data/historical_population.csv")
        economic_df = pd.read_csv("data/economic_indicators.csv")
        
        # Process permits
        if 'zip_code' in permits_df.columns:
            permits_df['zip_code'] = permits_df['zip_code'].astype(str).str[:5]
        
        # Process dates
        if 'issue_date' in permits_df.columns:
            permits_df['issue_date'] = pd.to_datetime(permits_df['issue_date'])
            permits_df['year'] = permits_df['issue_date'].dt.year
        
        # Merge datasets
        merged_df = permits_df.merge(
            population_df,
            left_on=['zip_code', 'year'],
            right_on=['zip code tabulation area', 'year'],
            how='left'
        )
        
        # Add economic indicators
        economic_df.index = pd.to_datetime(economic_df.index)
        economic_df['year'] = economic_df.index.year
        yearly_economic = economic_df.groupby('year').mean()
        
        final_df = merged_df.merge(yearly_economic, on='year', how='left')
        
        # Save processed data
        final_df.to_csv("data/processed_data.csv", index=False)
        logger.info("Saved processed and merged data")
        return True
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return False

def train_models():
    """Train predictive models"""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        
        # Load processed data
        df = pd.read_csv("data/processed_data.csv")
        
        # Prepare features
        features = ['total_units', 'treasury_10y', 'consumer_sentiment', 'recession_indicator']
        target = 'B01003_001E'  # population
        
        # Drop rows with missing values
        model_df = df[features + [target]].dropna()
        
        X = model_df[features]
        y = model_df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Save model
        joblib.dump(rf, "models/population_model.pkl")
        logger.info("Saved trained model")
        
        # Generate predictions for different scenarios
        scenarios = {
            'optimistic': {
                'treasury_10y': X['treasury_10y'].quantile(0.25),
                'consumer_sentiment': X['consumer_sentiment'].quantile(0.75),
                'recession_indicator': 0
            },
            'neutral': {
                'treasury_10y': X['treasury_10y'].mean(),
                'consumer_sentiment': X['consumer_sentiment'].mean(),
                'recession_indicator': 0
            },
            'pessimistic': {
                'treasury_10y': X['treasury_10y'].quantile(0.75),
                'consumer_sentiment': X['consumer_sentiment'].quantile(0.25),
                'recession_indicator': 1
            }
        }
        
        predictions = {}
        for scenario, values in scenarios.items():
            X_scenario = X_test.copy()
            for feature, value in values.items():
                X_scenario[feature] = value
            predictions[scenario] = rf.predict(X_scenario)
        
        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv("output/scenario_predictions.csv", index=False)
        logger.info("Saved scenario predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return False

def generate_visualizations():
    """Generate visualizations of results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create visualizations directory
        Path("output/visualizations").mkdir(exist_ok=True)
        
        # Load predictions
        predictions = pd.read_csv("output/scenario_predictions.csv")
        
        # Plot scenario comparisons
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=predictions)
        plt.title("Population Predictions by Scenario")
        plt.ylabel("Predicted Population")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/visualizations/scenario_predictions.png")
        plt.close()
        
        logger.info("Generated visualizations")
        return True
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return False

def run_pipeline():
    """Run the complete analysis pipeline"""
    steps = [
        ("Verifying API keys", verify_api_keys),
        ("Verifying directories", verify_directories),
        ("Collecting permit data", collect_permit_data),
        ("Collecting Census data", collect_census_data),
        ("Collecting economic data", collect_economic_data),
        ("Processing data", process_data),
        ("Training models", train_models),
        ("Generating visualizations", generate_visualizations)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"Starting: {step_name}")
        if not step_func():
            logger.error(f"Pipeline failed at: {step_name}")
            return False
        logger.info(f"Completed: {step_name}")
    
    logger.info("Pipeline completed successfully")
    return True

if __name__ == "__main__":
    if run_pipeline():
        sys.exit(0)
    else:
        sys.exit(1)