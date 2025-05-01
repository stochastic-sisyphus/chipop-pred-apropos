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
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.config import settings
from src.utils.helpers import sanitize_features, match_features, resolve_column_name, safe_train_model
from src.config.column_alias_map import column_aliases

logger = logging.getLogger(__name__)

class RetailModel:
    """Model for analyzing retail trends and making predictions."""
    
    def __init__(self):
        """Initialize retail model."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def analyze_retail_trends(self, df):
        """Analyze retail trends in the data."""
        try:
            logger.info("Analyzing retail trends...")
            
            # Get column names
            year_col = resolve_column_name(df, 'year', column_aliases)
            if not year_col:
                logger.error("Year column not found")
                return None
            
            # Define metrics to analyze
            metrics = [
                'housing_units', 'housing_density', 'housing_value_per_unit',
                'retail_construction_cost'
            ]
            
            # Resolve metric columns
            resolved_metrics = {
                metric: resolve_column_name(df, metric, column_aliases)
                for metric in metrics
            }
            
            # Filter out unresolved metrics
            resolved_metrics = {
                k: v for k, v in resolved_metrics.items() if v is not None
            }
            
            if not resolved_metrics:
                logger.error("No metric columns found")
                return None
            
            # Calculate yearly trends
            yearly_trends = df.groupby(year_col).agg({
                col: 'mean' if 'density' in metric or 'value' in metric else 'sum'
                for metric, col in resolved_metrics.items()
            }).reset_index()
            
            # Calculate year-over-year changes
            for col in yearly_trends.columns:
                if col != year_col:
                    yearly_trends[f'{col}_change'] = yearly_trends[col].pct_change()
            
            logger.info("Analyzed retail trends")
            return yearly_trends
            
        except Exception as e:
            logger.error(f"Error analyzing retail trends: {str(e)}")
            return None
            
    def train(self, df: pd.DataFrame) -> bool:
        """Train the retail model."""
        try:
            logger.info("Training retail model...")
            
            # Prepare features
            X, y = self.prepare_features(df)
            if X is None or y is None:
                logger.error("Failed to prepare features for training")
                return False
            
            # Train model
            success = safe_train_model(self.model, X, y, model_name="RetailModel")
            if success:
                logger.info("Model trained successfully.")
                
                # Save model and scaler
                joblib.dump(self.model, settings.TRAINED_MODELS_DIR / 'retail_model.joblib')
                joblib.dump(self.scaler, settings.TRAINED_MODELS_DIR / 'retail_scaler.joblib')
                logger.info("Model and scaler saved")
                
            return success
            
        except Exception as e:
            logger.error(f"RetailModel training failed: {str(e)}")
            return False
            
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
            if feature_list is None:
                feature_list = [
                    'total_population',
                    'median_household_income',
                    'retail_space_per_capita',
                    'retail_spending_potential',
                    'retail_gap'
                ]
            
            # Match features using aliases
            feature_cols = match_features(df, feature_list, column_aliases)
            if not feature_cols:
                logger.error("No usable features found")
                return None
            
            # Resolve target column
            target_col = resolve_column_name(df, 'retail_construction_cost', column_aliases)
            if not target_col:
                logger.error("Target column 'retail_construction_cost' not found")
                return None
            
            # Prepare features and target
            X = df[feature_cols].copy()
            y = df[target_col]
            
            # Coerce all features to numeric and handle inf/-inf
            X = X.apply(pd.to_numeric, errors='coerce')
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            y = pd.to_numeric(y, errors='coerce')
            y.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill missing values with median (for features)
            X = X.fillna(X.median())
            
            # Log missing value counts before dropping
            logger.info("Missing values per column before drop:\n" + str(X.isnull().sum()))
            logger.info(f"Missing values in target before drop: {y.isnull().sum()}")
            
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
            df['demand_gap'] = df['predicted_retail_demand'] - df[target_col]
            df['demand_ratio'] = df['predicted_retail_demand'] / df[target_col]
            
            # Save predictions
            predictions_file = settings.PREDICTIONS_DIR / 'retail_demand_predictions.csv'
            df.to_csv(predictions_file, index=False)
            
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

    def prepare_features(self, df: pd.DataFrame, feature_list=None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for retail modeling."""
        try:
            logger.info("Preparing retail features...")
            
            # Use dynamic feature selection if feature_list is provided
            if feature_list is not None:
                features = match_features(df, feature_list, column_aliases)
            else:
                # Default features
                default_features = [
                    'total_population',
                    'median_household_income',
                    'median_home_value',
                    'labor_force',
                    'residential_permits',
                    'commercial_permits',
                    'residential_construction_cost',
                    'commercial_construction_cost'
                ]
                features = match_features(df, default_features, column_aliases)
            
            if not features:
                logger.error("No usable features available for retail modeling.")
                return None, None
                
            logger.info(f"Features used for retail modeling: {features}")
            
            # Store feature names for reuse
            self.feature_names = features
            
            # Resolve target variable
            target_col = resolve_column_name(df, 'retail_permits', column_aliases)
            
            # Create feature matrix and target
            X = df[features].copy()
            y = df[target_col] if target_col else None
            
            # Log missing value counts before dropping
            logger.info("Missing values per column before drop:\n" + str(df[features + ([target_col] if target_col else [])].isnull().sum()))
            
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
                'median_household_income', 'median_home_value'
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
            
            logger.info("Retail features prepared successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing retail features: {str(e)}")
            return None, None