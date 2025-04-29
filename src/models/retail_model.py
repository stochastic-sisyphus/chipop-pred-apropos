"""
Retail analysis model for Chicago ZIP codes.
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
import json

from src.config import settings

logger = logging.getLogger(__name__)

class RetailModel:
    """Model for analyzing retail trends and making predictions."""
    
    def __init__(self):
        """Initialize retail model."""
        self.results = None
        self.trends = None
        self.predictions = None
        
    def analyze_retail_trends(self, df):
        """Analyze retail trends in the data."""
        try:
            logger.info("Analyzing retail trends...")
            
            # Calculate retail metrics by year
            yearly_trends = df.groupby('year').agg({
                'retail_permits': 'sum',
                'retail_construction_cost': 'sum',
                'retail_space': 'sum',
                'retail_space_per_capita': 'mean',
                'retail_spending_potential': 'sum',
                'retail_gap': 'sum'
            }).reset_index()
            
            # Calculate year-over-year changes
            for col in yearly_trends.columns:
                if col != 'year':
                    yearly_trends[f'{col}_change'] = yearly_trends[col].pct_change()
            
            self.trends = yearly_trends
            logger.info("Retail trends analysis completed")
            
            # Save trends
            trends_file = settings.PROCESSED_DATA_DIR / 'retail_trends.csv'
            yearly_trends.to_csv(trends_file, index=False)
            logger.info(f"Saved retail trends to {trends_file}")
            
            return yearly_trends
            
        except Exception as e:
            logger.error(f"Error analyzing retail trends: {str(e)}")
            return None
            
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
            if feature_list is not None:
                features = [f for f in feature_list if f in df.columns and df[f].notna().sum() > 0]
                if not features:
                    logger.error("No usable features available for retail demand modeling.")
                    return None
                logger.info(f"Features used for retail modeling: {features}")
            else:
                features = [
                    'total_population',
                    'median_household_income',
                    'retail_space_per_capita',
                    'retail_spending_potential',
                    'retail_gap'
                ]
                features = [f for f in features if f in df.columns and df[f].notna().sum() > 0]
                if not features:
                    logger.error("No usable default features for retail demand modeling.")
                    return None
                logger.info(f"Default features used for retail modeling: {features}")
            
            # Prepare features and target
            X = df[features].copy()
            y = df['retail_construction_cost']  # Always use retail_construction_cost as target
            
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
            df['predicted_retail_demand'] = model.predict(X)
            
            # Calculate additional metrics
            df['demand_gap'] = df['predicted_retail_demand'] - df['retail_construction_cost']
            df['demand_ratio'] = df['predicted_retail_demand'] / df['retail_construction_cost']
            
            # Save predictions
            predictions_file = settings.PREDICTIONS_DIR / 'retail_demand_predictions.csv'
            df.to_csv(predictions_file, index=False)
            logger.info(f"Saved retail demand predictions to {predictions_file}")
            
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
            df['leakage_rate'] = df['retail_gap'] / df['retail_spending_potential']
            df['leakage_per_capita'] = df['retail_gap'] / df['total_population']
            
            # Categorize leakage severity
            df['leakage_severity'] = pd.cut(
                df['leakage_rate'],
                bins=[-np.inf, 0, 0.2, 0.4, np.inf],
                labels=['No Leakage', 'Low', 'Medium', 'High']
            )
            
            # Save leakage analysis
            leakage_file = settings.PROCESSED_DATA_DIR / 'retail_leakage.csv'
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
                'total_retail_space': float(df['retail_space'].sum()),
                'avg_retail_space_per_capita': float(df['retail_space_per_capita'].mean()),
                'total_retail_gap': float(df['retail_gap'].sum()),
                'avg_leakage_rate': float(df['leakage_rate'].mean()),
                'high_opportunity_areas': int(df[df['retail_opportunity_score'] > 0]['zip_code'].nunique())
            }
            
            # Save metrics
            metrics_file = settings.PROCESSED_DATA_DIR / 'retail_summary_metrics.json'
            with open(metrics_file, 'w') as f:
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
            df = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_metrics.csv')
            
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
                'trends': trends,
                'predictions': predictions,
                'leakage': leakage,
                'metrics': metrics
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