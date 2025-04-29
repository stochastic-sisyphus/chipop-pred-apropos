"""
Housing analysis model for Chicago ZIP codes.
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

from src.config import settings

logger = logging.getLogger(__name__)

class HousingModel:
    """Model for analyzing housing trends and making predictions."""
    
    def __init__(self):
        """Initialize housing model."""
        self.results = None
        self.trends = None
        self.predictions = None
        
    def analyze_housing_trends(self, df):
        """Analyze housing trends in the data."""
        try:
            logger.info("Analyzing housing trends...")
            
            # Calculate housing metrics by year
            yearly_trends = df.groupby('year').agg({
                'housing_units': 'sum',
                'housing_density': 'mean',
                'housing_value_per_unit': 'mean',
                'residential_construction_cost': 'sum'
            }).reset_index()
            
            # Calculate year-over-year changes
            for col in yearly_trends.columns:
                if col != 'year':
                    yearly_trends[f'{col}_change'] = yearly_trends[col].pct_change()
            
            self.trends = yearly_trends
            logger.info("Analyzed housing trends")
            return yearly_trends
            
        except Exception as e:
            logger.error(f"Error analyzing housing trends: {str(e)}")
            return None
            
    def predict_housing_demand(self, df, feature_list=None):
        """Predict housing demand based on demographic and economic factors."""
        try:
            logger.info("Predicting housing demand...")
            # Use dynamic feature selection if feature_list is provided
            if feature_list is not None:
                features = [f for f in feature_list if f in df.columns and df[f].notna().sum() > 0]
                if not features:
                    logger.error("No usable features available for housing demand modeling.")
                    return None
                logger.info(f"Features used for housing modeling: {features}")
            else:
                features = [
                    'total_population',
                    'median_household_income',
                    'housing_density',
                    'housing_value_per_unit',
                    'residential_permits',
                    'residential_construction_cost'
                ]
                features = [f for f in features if f in df.columns and df[f].notna().sum() > 0]
                if not features:
                    logger.error("No usable default features for housing demand modeling.")
                    return None
                logger.info(f"Default features used for housing modeling: {features}")
            # Prepare features and target
            X = df[features].copy()
            y = df['housing_units']  # Use housing_units as target
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
            # Create prediction model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            # Train model on historical data
            model.fit(X, y)
            # Generate predictions
            df = df.loc[X.index].copy()
            df['predicted_housing_demand'] = model.predict(X)
            # Calculate additional metrics
            df['demand_gap'] = df['predicted_housing_demand'] - df['housing_units']
            df['demand_ratio'] = df['predicted_housing_demand'] / df['housing_units']
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
            retail_metrics = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_metrics.csv')
            
            # Merge housing and retail data
            balance_df = pd.merge(
                df,
                retail_metrics[['year', 'zip_code', 'retail_construction_cost', 'retail_space']],
                on=['year', 'zip_code'],
                how='left'
            )
            
            # Fill missing retail values with 0
            balance_df['retail_construction_cost'] = balance_df['retail_construction_cost'].fillna(0)
            balance_df['retail_space'] = balance_df['retail_space'].fillna(0)
            
            # Calculate balance metrics
            # Avoid division by zero and log(0) issues
            with np.errstate(divide='ignore', invalid='ignore'):
                # housing_retail_ratio: inf if retail_construction_cost == 0
                balance_df['housing_retail_ratio'] = np.where(
                    balance_df['retail_construction_cost'] > 0,
                    balance_df['residential_construction_cost'] / balance_df['retail_construction_cost'],
                    np.nan  # Use NaN for undefined ratios
                )

                # Cap extremely high/low ratios for log stability
                safe_ratio = balance_df['housing_retail_ratio'].clip(lower=1e-3, upper=1e3)
                # Replace NaN with a neutral value (e.g., 1) for log calculation
                safe_ratio = safe_ratio.fillna(1.0)
                balance_df['balance_score'] = 1 / (1 + np.abs(np.log(safe_ratio)))

            # Categorize balance
            balance_df['balance_category'] = pd.cut(
                balance_df['balance_score'],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['Severe Imbalance', 'Moderate Imbalance', 'Slight Imbalance', 'Balanced']
            )

            # Determine primary need
            balance_df['primary_need'] = np.where(
                balance_df['housing_retail_ratio'] > 2,
                'Retail Development',
                np.where(
                    balance_df['housing_retail_ratio'] < 0.5,
                    'Housing Development',
                    'Balanced Development'
                )
            )

            # Save balance analysis
            balance_df.to_csv(settings.PROCESSED_DATA_DIR / 'housing_retail_balance.csv', index=False)
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
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'housing_metrics.csv')
            
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
            self.results = {
                'trends': trends,
                'predictions': predictions,
                'balance': balance
            }
            
            logger.info("Housing analysis pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in housing analysis pipeline: {str(e)}")
            return False
            
    def get_results(self) -> dict[str, pd.DataFrame]:
        """Get the analysis results."""
        if self.results is None:
            logger.warning("No results available. Run analysis first.")
            return pd.DataFrame()
        return self.results