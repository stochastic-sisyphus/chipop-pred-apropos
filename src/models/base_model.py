"""
Base model class for Chicago Housing Pipeline.

This module provides the base model class for all analytical models.
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
from abc import ABC, abstractmethod

from src.config import settings

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Base model class for all analytical models.
    
    Provides common functionality for model training, evaluation, and prediction.
    """
    
    def __init__(self, model_name, output_dir=None):
        """
        Initialize the base model.
        
        Args:
            model_name (str): Name of the model
            output_dir (Path, optional): Directory to save model outputs
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path(settings.OUTPUT_DIR) / "models" / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir = Path(settings.OUTPUT_DIR) / "visualizations" / model_name
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.model_params = {}
        self.model_metrics = {}
        self.feature_importance = {}
        self.predictions = None
        self.results = {}
    
    @abstractmethod
    def preprocess_data(self, data):
        """
        Preprocess data for model training and prediction.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        pass
    
    @abstractmethod
    def train(self, data):
        """
        Train the model.
        
        Args:
            data (pd.DataFrame): Training data
            
        Returns:
            object: Trained model
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Make predictions with the trained model.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            
        Returns:
            pd.DataFrame: Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, data, predictions=None):
        """
        Evaluate model performance.
        
        Args:
            data (pd.DataFrame): Evaluation data
            predictions (pd.DataFrame, optional): Model predictions
            
        Returns:
            dict: Evaluation metrics
        """
        pass
    
    @abstractmethod
    def analyze_results(self):
        """
        Analyze model results and generate insights.
        
        Returns:
            dict: Analysis results
        """
        pass
    
    def save_model(self):
        """
        Save the trained model to disk.
        
        Returns:
            str: Path to saved model
        """
        try:
            if self.model is None:
                logger.warning(f"No model to save for {self.model_name}")
                return None
            
            model_path = self.output_dir / f"{self.model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            params_path = self.output_dir / f"{self.model_name}_params.json"
            with open(params_path, 'w') as f:
                json.dump(self.model_params, f, indent=2)
            
            metrics_path = self.output_dir / f"{self.model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
            
            logger.info(f"Saved {self.model_name} model to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def load_model(self, model_path=None):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str, optional): Path to saved model
            
        Returns:
            object: Loaded model
        """
        try:
            if model_path is None:
                model_path = self.output_dir / f"{self.model_name}_model.pkl"
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            params_path = self.output_dir / f"{self.model_name}_params.json"
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.model_params = json.load(f)
            
            metrics_path = self.output_dir / f"{self.model_name}_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.model_metrics = json.load(f)
            
            logger.info(f"Loaded {self.model_name} model from {model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def save_results(self, results=None):
        """
        Save model results to disk.
        
        Args:
            results (dict, optional): Results to save
            
        Returns:
            str: Path to saved results
        """
        try:
            if results is None:
                results = self.results
            
            if not results:
                logger.warning(f"No results to save for {self.model_name}")
                return None
            
            # Save predictions if available
            if self.predictions is not None and isinstance(self.predictions, pd.DataFrame):
                predictions_path = self.output_dir / f"{self.model_name}_predictions.csv"
                self.predictions.to_csv(predictions_path, index=False)
                logger.info(f"Saved predictions to {predictions_path}")
            
            # Save feature importance if available
            if self.feature_importance:
                importance_path = self.output_dir / f"{self.model_name}_feature_importance.json"
                with open(importance_path, 'w') as f:
                    json.dump(self.feature_importance, f, indent=2)
                logger.info(f"Saved feature importance to {importance_path}")
            
            # Save results
            results_path = self.output_dir / f"{self.model_name}_results.json"
            
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                    serializable_results[key] = value
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, pd.DataFrame):
                    csv_path = self.output_dir / f"{self.model_name}_{key}.csv"
                    value.to_csv(csv_path, index=False)
                    serializable_results[key] = f"Saved to {csv_path.name}"
                else:
                    serializable_results[key] = str(value)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved {self.model_name} results to {results_path}")
            return str(results_path)
            
        except Exception as e:
            logger.error(f"Error saving results for {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def load_results(self, results_path=None):
        """
        Load model results from disk.
        
        Args:
            results_path (str, optional): Path to saved results
            
        Returns:
            dict: Loaded results
        """
        try:
            if results_path is None:
                results_path = self.output_dir / f"{self.model_name}_results.json"
            
            if not os.path.exists(results_path):
                logger.warning(f"Results file not found: {results_path}")
                return None
            
            with open(results_path, 'r') as f:
                self.results = json.load(f)
            
            # Load predictions if available
            predictions_path = self.output_dir / f"{self.model_name}_predictions.csv"
            if os.path.exists(predictions_path):
                self.predictions = pd.read_csv(predictions_path)
            
            # Load feature importance if available
            importance_path = self.output_dir / f"{self.model_name}_feature_importance.json"
            if os.path.exists(importance_path):
                with open(importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
            
            logger.info(f"Loaded {self.model_name} results from {results_path}")
            return self.results
            
        except Exception as e:
            logger.error(f"Error loading results for {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def plot_feature_importance(self, top_n=10):
        """
        Plot feature importance.
        
        Args:
            top_n (int): Number of top features to plot
            
        Returns:
            str: Path to saved plot
        """
        try:
            if not self.feature_importance:
                logger.warning(f"No feature importance available for {self.model_name}")
                return None
            
            # Convert feature importance to DataFrame
            if isinstance(self.feature_importance, dict):
                importance_df = pd.DataFrame({
                    'Feature': list(self.feature_importance.keys()),
                    'Importance': list(self.feature_importance.values())
                })
            else:
                logger.warning(f"Feature importance format not supported: {type(self.feature_importance)}")
                return None
            
            # Sort by importance and take top N
            importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Top {top_n} Feature Importance - {self.model_name}')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.visualization_dir / f"{self.model_name}_feature_importance.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved feature importance plot to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting feature importance for {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def plot_prediction_distribution(self):
        """
        Plot distribution of predictions.
        
        Returns:
            str: Path to saved plot
        """
        try:
            if self.predictions is None:
                logger.warning(f"No predictions available for {self.model_name}")
                return None
            
            # Identify the target column
            target_cols = [col for col in self.predictions.columns if 'predicted' in col.lower()]
            if not target_cols:
                logger.warning(f"No prediction columns found in {self.model_name} predictions")
                return None
            
            target_col = target_cols[0]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.histplot(self.predictions[target_col], kde=True)
            plt.title(f'Prediction Distribution - {self.model_name}')
            plt.xlabel(target_col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.visualization_dir / f"{self.model_name}_prediction_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved prediction distribution plot to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting prediction distribution for {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def plot_actual_vs_predicted(self, actual_col=None, predicted_col=None):
        """
        Plot actual vs predicted values.
        
        Args:
            actual_col (str, optional): Column name for actual values
            predicted_col (str, optional): Column name for predicted values
            
        Returns:
            str: Path to saved plot
        """
        try:
            if self.predictions is None:
                logger.warning(f"No predictions available for {self.model_name}")
                return None
            
            # Identify the target columns if not provided
            if predicted_col is None:
                predicted_cols = [col for col in self.predictions.columns if 'predicted' in col.lower()]
                if not predicted_cols:
                    logger.warning(f"No prediction columns found in {self.model_name} predictions")
                    return None
                predicted_col = predicted_cols[0]
            
            if actual_col is None:
                # Try to find the actual column by removing 'predicted' from the predicted column name
                actual_col = predicted_col.replace('predicted_', '').replace('_predicted', '')
                if actual_col not in self.predictions.columns:
                    logger.warning(f"Actual column {actual_col} not found in {self.model_name} predictions")
                    return None
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.scatter(self.predictions[actual_col], self.predictions[predicted_col], alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(self.predictions[actual_col].min(), self.predictions[predicted_col].min())
            max_val = max(self.predictions[actual_col].max(), self.predictions[predicted_col].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f'Actual vs Predicted - {self.model_name}')
            plt.xlabel(f'Actual ({actual_col})')
            plt.ylabel(f'Predicted ({predicted_col})')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.visualization_dir / f"{self.model_name}_actual_vs_predicted.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved actual vs predicted plot to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting actual vs predicted for {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def plot_geographic_distribution(self, column=None, title=None, cmap='viridis'):
        """
        Plot geographic distribution of results by ZIP code.
        
        Args:
            column (str, optional): Column to plot
            title (str, optional): Plot title
            cmap (str): Colormap name
            
        Returns:
            str: Path to saved plot
        """
        try:
            if self.predictions is None or 'zip_code' not in self.predictions.columns:
                logger.warning(f"No ZIP code data available for {self.model_name}")
                return None
            
            # Identify the column to plot if not provided
            if column is None:
                target_cols = [col for col in self.predictions.columns if 'predicted' in col.lower()]
                if not target_cols:
                    logger.warning(f"No prediction columns found in {self.model_name} predictions")
                    return None
                column = target_cols[0]
            
            if column not in self.predictions.columns:
                logger.warning(f"Column {column} not found in {self.model_name} predictions")
                return None
            
            # Set title if not provided
            if title is None:
                title = f'{column} by ZIP Code - {self.model_name}'
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Group by ZIP code and calculate mean
            zip_data = self.predictions.groupby('zip_code')[column].mean().reset_index()
            
            # Create bar plot
            sns.barplot(x='zip_code', y=column, data=zip_data, palette=cmap)
            plt.title(title)
            plt.xlabel('ZIP Code')
            plt.ylabel(column)
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            # Save plot
            plot_name = f"{self.model_name}_{column}_by_zipcode.png"
            plot_path = self.visualization_dir / plot_name
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved geographic distribution plot to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error plotting geographic distribution for {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def run_pipeline(self, data):
        """
        Run the full model pipeline: preprocess, train, predict, evaluate, analyze.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Model results
        """
        try:
            logger.info(f"Running full pipeline for {self.model_name}")
            
            # Preprocess data
            preprocessed_data = self.preprocess_data(data)
            
            # Train model
            self.model = self.train(preprocessed_data)
            
            # Make predictions
            self.predictions = self.predict(preprocessed_data)
            
            # Evaluate model
            self.model_metrics = self.evaluate(preprocessed_data, self.predictions)
            
            # Analyze results
            self.results = self.analyze_results()
            
            # Save model and results
            self.save_model()
            self.save_results()
            
            # Generate visualizations
            if hasattr(self, 'feature_importance') and self.feature_importance:
                self.plot_feature_importance()
            
            if self.predictions is not None:
                self.plot_prediction_distribution()
                
                # Check if we can plot actual vs predicted
                predicted_cols = [col for col in self.predictions.columns if 'predicted' in col.lower()]
                if predicted_cols:
                    predicted_col = predicted_cols[0]
                    actual_col = predicted_col.replace('predicted_', '').replace('_predicted', '')
                    if actual_col in self.predictions.columns:
                        self.plot_actual_vs_predicted(actual_col, predicted_col)
                
                if 'zip_code' in self.predictions.columns:
                    self.plot_geographic_distribution()
            
            logger.info(f"Completed pipeline for {self.model_name}")
            return self.results
            
        except Exception as e:
            logger.error(f"Error running pipeline for {self.model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
