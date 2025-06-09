"""
Population Forecast Model for Chicago Housing Pipeline.

This module provides advanced time series forecasting for population trends by ZIP code.
"""

import os
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

from src.models.base_model import BaseModel
from src.config import settings

logger = logging.getLogger(__name__)

class PopulationForecastModel(BaseModel):
    """
    Population Forecast Model for predicting future population trends by ZIP code.
    
    Uses advanced time series forecasting techniques to predict population growth
    and demographic shifts across Chicago ZIP codes.
    """
    
    def __init__(self, output_dir=None, forecast_years=5):
        """
        Initialize the Population Forecast Model.
        
        Args:
            output_dir (Path, optional): Directory to save model outputs
            forecast_years (int): Number of years to forecast into the future
        """
        super().__init__("PopulationForecast", output_dir)
        self.forecast_years = forecast_years
        self.time_series_models = {}
        self.feature_models = {}
        self.forecast_data = None
        self.top_growth_zips = []
        self.demographic_shifts = {}
        self.feature_columns = []
        self.global_model = None
        self.global_model_metrics = {}
    
    def preprocess_data(self, data):
        """
        Preprocess data for population forecasting.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            logger.info("Preprocessing data for population forecasting")
            
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Validate required columns
            required_columns = ["zip_code", "year", "population"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Ensure year is integer
            df["year"] = df["year"].astype(int)
            
            # Ensure population is numeric
            df["population"] = pd.to_numeric(df["population"], errors="coerce")
            
            # Filter out rows with missing population
            df = df[~df["population"].isna()]
            
            # Sort by ZIP code and year
            df = df.sort_values(["zip_code", "year"])
            
            # Create additional features for forecasting
            if "housing_units" in df.columns:
                df["population_per_housing_unit"] = df["population"] / df["housing_units"].replace(0, np.nan)
            
            if "median_income" in df.columns:
                df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")
            
            # Calculate year-over-year growth rate
            df["population_growth_rate"] = df.groupby("zip_code")["population"].pct_change()
            
            # Fill missing growth rates with 0
            df["population_growth_rate"] = df["population_growth_rate"].fillna(0)
            
            # Calculate rolling average growth rate (3-year window)
            df["rolling_growth_rate"] = df.groupby("zip_code")["population_growth_rate"].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # Identify features for forecasting
            self.feature_columns = [
                "population", "population_growth_rate", "rolling_growth_rate"
            ]
            
            # Add additional features if available
            additional_features = [
                "median_income", "housing_units", "population_per_housing_unit",
                "total_permits", "unit_count", "multifamily_permits",
                "total_licenses", "vacancy_rate", "housing_growth_score",
                "economic_prosperity_score", "growth_potential_score"
            ]
            
            for feature in additional_features:
                if feature in df.columns:
                    self.feature_columns.append(feature)
            
            logger.info(f"Preprocessed data: {len(df)} records with {len(self.feature_columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def train(self, data):
        """
        Train population forecast models.
        
        Args:
            data (pd.DataFrame): Training data
            
        Returns:
            dict: Trained models
        """
        try:
            logger.info("Training population forecast models")
            
            # Preprocess data
            preprocessed_data = self.preprocess_data(data)
            
            # Get unique ZIP codes
            zip_codes = preprocessed_data["zip_code"].unique()
            
            # Train time series models for each ZIP code
            for zip_code in zip_codes:
                logger.info(f"Training models for ZIP code {zip_code}")
                
                # Filter data for this ZIP code
                zip_data = preprocessed_data[preprocessed_data["zip_code"] == zip_code].sort_values("year")
                
                if len(zip_data) < 3:
                    logger.warning(f"Insufficient data for ZIP code {zip_code}, skipping")
                    continue
                
                # Train time series model
                self._train_time_series_model(zip_code, zip_data)
                
                # Train feature-based model
                self._train_feature_model(zip_code, zip_data)
            
            # Train a global model using all data
            self._train_global_model(preprocessed_data)
            
            # Save feature importance
            if hasattr(self, "global_model") and hasattr(self.global_model, "feature_importances_"):
                self.feature_importance = dict(zip(self.feature_columns, self.global_model.feature_importances_))
            
            logger.info(f"Trained models for {len(self.time_series_models)} ZIP codes")
            return {"time_series_models": self.time_series_models, "feature_models": self.feature_models}
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _train_time_series_model(self, zip_code, data):
        """
        Train time series model for a specific ZIP code.
        
        Args:
            zip_code (str): ZIP code
            data (pd.DataFrame): Training data for this ZIP code
            
        Returns:
            object: Trained time series model
        """
        try:
            # Extract population time series
            population_series = data.set_index("year")["population"]
            
            # Try different time series models and select the best one
            models = {}
            errors = {}
            
            # Simple exponential smoothing
            try:
                model_ets = ExponentialSmoothing(
                    population_series,
                    trend="add",
                    seasonal=None,
                    damped=True
                ).fit()
                
                # Make in-sample predictions
                predictions_ets = model_ets.predict(start=population_series.index[0], end=population_series.index[-1])
                
                # Calculate error
                error_ets = mean_squared_error(population_series, predictions_ets)
                
                models["ets"] = model_ets
                errors["ets"] = error_ets
            except Exception as e:
                logger.warning(f"Error fitting ETS model for ZIP {zip_code}: {str(e)}")
            
            # Auto ARIMA
            try:
                model_arima = pm.auto_arima(
                    population_series,
                    start_p=1, start_q=1,
                    max_p=3, max_q=3,
                    d=None, max_d=2,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False
                )
                
                # Make in-sample predictions
                predictions_arima = model_arima.predict_in_sample()
                
                # Calculate error
                error_arima = mean_squared_error(population_series, predictions_arima)
                
                models["arima"] = model_arima
                errors["arima"] = error_arima
            except Exception as e:
                logger.warning(f"Error fitting ARIMA model for ZIP {zip_code}: {str(e)}")
            
            # Select the best model
            if errors:
                best_model_type = min(errors, key=errors.get)
                best_model = models[best_model_type]
                best_error = errors[best_model_type]
                
                logger.info(f"Selected {best_model_type} model for ZIP {zip_code} with MSE: {best_error:.2f}")
                
                # Store the model
                self.time_series_models[zip_code] = {
                    "model": best_model,
                    "type": best_model_type,
                    "error": best_error,
                    "last_year": population_series.index[-1]
                }
                
                return best_model
            else:
                logger.warning(f"No time series models could be fit for ZIP {zip_code}")
                return None
            
        except Exception as e:
            logger.error(f"Error training time series model for ZIP {zip_code}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _train_feature_model(self, zip_code, data):
        """
        Train feature-based model for a specific ZIP code.
        
        Args:
            zip_code (str): ZIP code
            data (pd.DataFrame): Training data for this ZIP code
            
        Returns:
            object: Trained feature model
        """
        try:
            # Check if we have enough data and features
            if len(data) < 5 or len(self.feature_columns) < 2:
                logger.warning(f"Insufficient data or features for ZIP {zip_code}, skipping feature model")
                return None
            
            # Prepare features and target
            X = data[self.feature_columns].values
            y = data["population"].values
            
            # Train a random forest regressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X, y)
            
            # Make in-sample predictions
            predictions = model.predict(X)
            
            # Calculate error
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            logger.info(f"Trained feature model for ZIP {zip_code} with MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Store the model
            self.feature_models[zip_code] = {
                "model": model,
                "mse": mse,
                "r2": r2,
                "feature_importance": dict(zip(self.feature_columns, model.feature_importances_))
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error training feature model for ZIP {zip_code}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _train_global_model(self, data):
        """
        Train a global model using all data.
        
        Args:
            data (pd.DataFrame): All training data
            
        Returns:
            object: Trained global model
        """
        try:
            # Prepare features and target
            X = data[self.feature_columns].values
            y = data["population"].values
            
            # Train a random forest regressor
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42
            )
            
            model.fit(X, y)
            
            # Make in-sample predictions
            predictions = model.predict(X)
            
            # Calculate error
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            logger.info(f"Trained global model with MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Store the model
            self.global_model = model
            self.global_model_metrics = {
                "mse": mse,
                "r2": r2
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error training global model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def predict(self, data):
        """
        Generate population forecasts.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Forecast data
        """
        try:
            logger.info("Generating population forecasts")
            
            # Preprocess data
            preprocessed_data = self.preprocess_data(data)
            
            # Get unique ZIP codes
            zip_codes = preprocessed_data["zip_code"].unique()
            
            # Get the maximum year in the data
            max_year = preprocessed_data["year"].max()
            
            # Create a list to store forecast results
            forecast_results = []
            
            # Generate forecasts for each ZIP code
            for zip_code in zip_codes:
                logger.info(f"Generating forecast for ZIP code {zip_code}")
                
                # Filter data for this ZIP code
                zip_data = preprocessed_data[preprocessed_data["zip_code"] == zip_code].sort_values("year")
                
                if len(zip_data) < 3:
                    logger.warning(f"Insufficient data for ZIP code {zip_code}, skipping")
                    continue
                
                # Get the last row of data for this ZIP code
                last_data = zip_data.iloc[-1].copy()
                
                # Generate time series forecast
                ts_forecast = self._forecast_time_series(zip_code, max_year)
                
                # Generate feature-based forecast
                feature_forecast = self._forecast_features(zip_code, zip_data, max_year)
                
                # Combine forecasts (simple average if both are available)
                combined_forecast = {}
                
                for year in range(max_year + 1, max_year + self.forecast_years + 1):
                    # Initialize with time series forecast if available
                    if ts_forecast and year in ts_forecast:
                        combined_forecast[year] = ts_forecast[year]
                    else:
                        combined_forecast[year] = None
                    
                    # Add feature forecast if available
                    if feature_forecast and year in feature_forecast:
                        if combined_forecast[year] is None:
                            combined_forecast[year] = feature_forecast[year]
                        else:
                            # Simple average
                            combined_forecast[year] = (combined_forecast[year] + feature_forecast[year]) / 2
                    
                    # If still None, use last known value
                    if combined_forecast[year] is None:
                        combined_forecast[year] = last_data["population"]
                
                # Add historical data to results
                for _, row in zip_data.iterrows():
                    forecast_results.append({
                        "zip_code": zip_code,
                        "year": row["year"],
                        "population": row["population"],
                        "forecast_type": "historical"
                    })
                
                # Add forecast data to results
                for year, population in combined_forecast.items():
                    forecast_results.append({
                        "zip_code": zip_code,
                        "year": year,
                        "population": population,
                        "forecast_type": "forecast"
                    })
            
            # Create DataFrame from results
            self.forecast_data = pd.DataFrame(forecast_results)
            
            # Add combined forecast type
            self.forecast_data["forecast_type"] = np.where(
                self.forecast_data["year"] <= max_year, "historical", "forecast"
            )
            
            # Create a combined series for visualization
            combined_df = self.forecast_data.copy()
            combined_df["forecast_type"] = "combined"
            self.forecast_data = pd.concat([self.forecast_data, combined_df], ignore_index=True)
            
            logger.info(f"Generated population forecasts for {len(zip_codes)} ZIP codes")
            return self.forecast_data
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _forecast_time_series(self, zip_code, max_year):
        """
        Generate time series forecast for a specific ZIP code.
        
        Args:
            zip_code (str): ZIP code
            max_year (int): Maximum year in historical data
            
        Returns:
            dict: Time series forecast {year: population}
        """
        try:
            if zip_code not in self.time_series_models:
                logger.warning(f"No time series model found for ZIP {zip_code}")
                return None
            
            model_info = self.time_series_models[zip_code]
            model = model_info["model"]
            model_type = model_info["type"]
            last_year = model_info["last_year"]
            
            # Calculate number of steps to forecast
            steps = max_year + self.forecast_years - last_year
            
            if steps <= 0:
                logger.warning(f"No future years to forecast for ZIP {zip_code}")
                return None
            
            # Generate forecast
            if model_type == "ets":
                forecast = model.forecast(steps=steps)
            elif model_type == "arima":
                forecast = model.predict(n_periods=steps)
            else:
                logger.warning(f"Unknown model type {model_type} for ZIP {zip_code}")
                return None
            
            # Create forecast dictionary
            forecast_dict = {}
            for i, pop in enumerate(forecast):
                year = last_year + i + 1
                forecast_dict[year] = pop
            
            return forecast_dict
            
        except Exception as e:
            logger.error(f"Error forecasting time series for ZIP {zip_code}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _forecast_features(self, zip_code, data, max_year):
        """
        Generate feature-based forecast for a specific ZIP code.
        
        Args:
            zip_code (str): ZIP code
            data (pd.DataFrame): Historical data for this ZIP code
            max_year (int): Maximum year in historical data
            
        Returns:
            dict: Feature-based forecast {year: population}
        """
        try:
            if zip_code not in self.feature_models:
                logger.warning(f"No feature model found for ZIP {zip_code}")
                return None
            
            model_info = self.feature_models[zip_code]
            model = model_info["model"]
            
            # Get the last row of data
            last_data = data.iloc[-1].copy()
            
            # Create future feature data (simple extrapolation)
            future_features = []
            current_features = last_data[self.feature_columns].values
            
            for i in range(self.forecast_years):
                # Predict next year population
                predicted_pop = model.predict([current_features])[0]
                
                # Update features for next prediction
                # Simple approach: assume other features remain constant or follow trend
                next_features = current_features.copy()
                next_features[0] = predicted_pop # Update population
                
                # Update growth rates
                if len(data) > 0:
                    last_pop = data.iloc[-1]["population"]
                    growth_rate = (predicted_pop - last_pop) / last_pop if last_pop else 0
                    next_features[1] = growth_rate # Update growth rate
                    
                    # Update rolling growth rate (simple average of last 3)
                    recent_growth = data["population_growth_rate"].tolist()[-2:] + [growth_rate]
                    next_features[2] = np.mean(recent_growth)
                
                future_features.append(next_features)
                current_features = next_features
            
            # Make predictions for future years
            future_predictions = model.predict(future_features)
            
            # Create forecast dictionary
            forecast_dict = {}
            for i, pop in enumerate(future_predictions):
                year = max_year + i + 1
                forecast_dict[year] = pop
            
            return forecast_dict
            
        except Exception as e:
            logger.error(f"Error forecasting features for ZIP {zip_code}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def evaluate(self, data, predictions=None):
        """
        Evaluate population forecast models.
        
        Args:
            data (pd.DataFrame): Evaluation data
            predictions (pd.DataFrame, optional): Model predictions
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info("Evaluating population forecast models")
            
            # If predictions not provided, generate them
            if predictions is None:
                predictions = self.predict(data)
            
            if predictions is None or len(predictions) == 0:
                logger.error("No predictions available for evaluation")
                return {"error": "No predictions available for evaluation"}
            
            # Evaluate global model performance
            global_metrics = self.global_model_metrics
            
            # Evaluate individual ZIP code models
            zip_metrics = {}
            for zip_code, model_info in self.time_series_models.items():
                zip_metrics[zip_code] = {
                    "time_series_mse": model_info.get("error", None)
                }
            
            for zip_code, model_info in self.feature_models.items():
                if zip_code not in zip_metrics:
                    zip_metrics[zip_code] = {}
                zip_metrics[zip_code]["feature_mse"] = model_info.get("mse", None)
                zip_metrics[zip_code]["feature_r2"] = model_info.get("r2", None)
            
            # Calculate overall forecast accuracy (e.g., MAE on historical fit)
            historical_data = predictions[predictions["forecast_type"] == "historical"]
            
            # Need to align historical data with in-sample predictions
            # This requires storing in-sample predictions during training
            # For now, we use the global model metrics as a proxy
            overall_mae = global_metrics.get("mae", None) # Assuming MAE was calculated
            
            # Store metrics
            self.model_metrics = {
                "global_model_metrics": global_metrics,
                "zip_code_metrics": zip_metrics,
                "overall_mae": overall_mae
            }
            
            logger.info("Population forecast model evaluation completed")
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def analyze_results(self):
        """
        Analyze population forecast results.
        
        Returns:
            dict: Analysis results
        """
        try:
            logger.info("Analyzing population forecast results")
            
            if self.forecast_data is None or len(self.forecast_data) == 0:
                logger.error("No forecast data available for analysis")
                return {"error": "No forecast data available for analysis"}
            
            # Filter to forecast data
            forecast_only = self.forecast_data[self.forecast_data["forecast_type"] == "forecast"].copy()
            
            # Calculate growth rates for forecast period
            forecast_only["forecast_growth_rate"] = forecast_only.groupby("zip_code")["population"].pct_change()
            
            # Calculate average growth rate per ZIP code
            avg_growth_rates = forecast_only.groupby("zip_code")["forecast_growth_rate"].mean().reset_index()
            
            # Identify top 5 growing ZIP codes
            self.top_growth_zips = avg_growth_rates.sort_values("forecast_growth_rate", ascending=False).head(5)["zip_code"].tolist()
            
            # Analyze demographic shifts (if data available)
            # This requires additional demographic data (e.g., age, income)
            # Placeholder for now
            self.demographic_shifts = {
                "summary": "Demographic shift analysis requires additional data (e.g., age, income distribution)."
            }
            
            # Generate summary
            summary = f"Population forecast generated for {self.forecast_years} years. Top 5 projected growth ZIP codes: {self.top_growth_zips}."
            
            # Store results
            self.results = {
                "top_growth_zips": self.top_growth_zips,
                "demographic_shifts": self.demographic_shifts,
                "summary": summary,
                "forecast_data": self.forecast_data.to_dict(orient="records") # Include forecast data in results
            }
            
            logger.info("Population forecast analysis completed")
            return self.results
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def run(self, data):
        """
        Run population forecast analysis.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Analysis results
        """
        try:
            logger.info("Running population forecast analysis")
            
            # Train models
            self.train(data)
            
            # Generate predictions (forecasts)
            self.predict(data)
            
            # Evaluate models
            self.evaluate(data)
            
            # Analyze results
            analysis_results = self.analyze_results()
            
            # Save results
            self._save_results()
            
            logger.info("Population forecast analysis completed successfully")
            
            # Return results with required keys for validation
            return {
                "forecast_data": self.forecast_data.to_dict(orient="records") if self.forecast_data is not None else [],
                "top_growth_zips": self.top_growth_zips,
                "demographic_shifts": self.demographic_shifts,
                "summary": analysis_results.get("summary", ""),
                "visualizations": self.visualization_paths # Assuming visualizations are generated and paths stored
            }
            
        except Exception as e:
            logger.error(f"Error running population forecast analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _save_results(self):
        """
        Save analysis results to disk.
        """
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save forecast data
            if self.forecast_data is not None:
                forecast_path = self.output_dir / "population_forecast_data.csv"
                self.forecast_data.to_csv(forecast_path, index=False)
                logger.info(f"Saved forecast data to {forecast_path}")
            
            # Save results as JSON
            results_path = self.output_dir / "population_forecast_results.json"
            
            # Prepare results for JSON serialization
            json_results = {
                "top_growth_zips": self.top_growth_zips,
                "demographic_shifts": self.demographic_shifts,
                "summary": self.results.get("summary", ""),
                "visualization_paths": self.visualization_paths,
                "analysis_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            with open(results_path, "w") as f:
                json.dump(json_results, f, indent=2)
            
            self.output_file = str(results_path)
            logger.info(f"Saved results to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            logger.error(traceback.format_exc())

