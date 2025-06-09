"""
Retail model for predicting retail business trends in Chicago ZIP codes.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import train_test_split

from src.models.base_model import BaseModel
from src.config import settings

logger = logging.getLogger(__name__)

class RetailModel(BaseModel):
    """Model for predicting retail business trends in Chicago ZIP codes."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the retail model.
        
        Args:
            output_dir (Path, optional): Directory to save model outputs
        """
        super().__init__("Retail", output_dir)
        self.model = None
        self.features = None
        self.predictions = None
    
    def run_analysis(self, data):
        """
        Run retail analysis on the provided data.
        
        Args:
            data (pd.DataFrame): Input data for analysis
            
        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            logger.info("Running retail analysis...")
            
            # Make a copy to avoid modifying the original
            self.data = data.copy()
            
            # Ensure ZIP code is string type
            self.data['zip_code'] = self.data['zip_code'].astype(str)
            
            # Prepare data for analysis
            self._prepare_data()
            
            # Train model
            self._train_model()
            
            # Make predictions
            self._make_predictions()
            
            # Generate visualizations
            self._generate_visualizations()
            
            # Save results
            self._save_results()
            
            logger.info("Retail analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in Retail analysis: {str(e)}")
            return False
    
    def _prepare_data(self):
        """
        Prepare data for retail analysis.
        """
        try:
            # Check for required columns
            required_cols = ['retail_businesses', 'zip_code']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Create derived features
            # Calculate retail density if land_area exists
            if 'land_area' in self.data.columns:
                self.data['retail_density'] = (
                    self.data['retail_businesses'] / self.data['land_area']
                ).round(2)
            else:
                # Create synthetic retail density
                self.data['retail_density'] = (
                    self.data['retail_businesses'] / 3  # Assume 3 sq miles
                ).round(2)
            
            # Calculate retail per capita if population exists
            if 'population' in self.data.columns:
                self.data['retail_per_capita'] = (
                    self.data['retail_businesses'] / self.data['population'] * 1000
                ).round(2)
            else:
                # Create synthetic retail per capita
                self.data['retail_per_capita'] = (
                    self.data['retail_businesses'] / 5000  # Assume 5000 people per ZIP
                ).round(2)
            
            # Select features for model
            self.features = [
                'retail_density',
                'retail_per_capita'
            ]
            
            # Check if median_income exists
            if 'median_income' in self.data.columns:
                self.features.append('median_income')
            
            # Check if population exists
            if 'population' in self.data.columns:
                self.features.append('population')
            
            # Check if housing_units exists
            if 'housing_units' in self.data.columns:
                self.features.append('housing_units')
            
            # Filter rows with missing values
            self.data = self.data.dropna(subset=['retail_businesses'] + self.features)
            
            logger.info(f"Prepared data for retail analysis: {len(self.data)} records")
            
        except Exception as e:
            logger.error(f"Error preparing data for retail analysis: {str(e)}")
            raise
    
    def _train_model(self):
        """
        Train retail prediction model.
        """
        try:
            # Split data into features and target
            X = self.data[self.features]
            y = self.data['retail_businesses']
            
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=settings.DEFAULT_RANDOM_STATE
            )
            
            # Create model pipeline
            self.model = SKPipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=settings.DEFAULT_RANDOM_STATE
                ))
            ])
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"Model trained: train R² = {train_score:.4f}, test R² = {test_score:.4f}")
            
            # Store evaluation metrics
            self.results['model_metrics'] = {
                'train_r2': train_score,
                'test_r2': test_score,
                'feature_importance': dict(zip(
                    self.features,
                    self.model.named_steps['regressor'].feature_importances_
                ))
            }
            
        except Exception as e:
            logger.error(f"Error training retail model: {str(e)}")
            raise
    
    def _make_predictions(self):
        """
        Make retail predictions.
        """
        try:
            # Make predictions on all data
            self.data['predicted_retail'] = self.model.predict(self.data[self.features])
            
            # Calculate prediction error
            self.data['prediction_error'] = (
                self.data['predicted_retail'] - self.data['retail_businesses']
            )
            
            self.data['prediction_error_pct'] = (
                self.data['prediction_error'] / self.data['retail_businesses'] * 100
            ).round(1)
            
            # Project future retail (5 years)
            # Assume 5% growth in population and income
            future_data = self.data.copy()
            
            if 'population' in self.features:
                future_data['population'] *= 1.05
            
            if 'median_income' in self.features:
                future_data['median_income'] *= 1.05
            
            if 'housing_units' in self.features:
                future_data['housing_units'] *= 1.03
            
            self.data['projected_retail_5yr'] = self.model.predict(future_data[self.features])
            
            # Calculate growth rate
            self.data['projected_growth_rate'] = (
                (self.data['projected_retail_5yr'] / self.data['retail_businesses']) ** (1/5) - 1
            ).round(4)
            
            # Store predictions
            self.predictions = self.data[['zip_code', 'retail_businesses', 'predicted_retail', 
                                         'prediction_error', 'prediction_error_pct',
                                         'projected_retail_5yr', 'projected_growth_rate']]
            
            logger.info(f"Made predictions for {len(self.predictions)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error making retail predictions: {str(e)}")
            raise
    
    def _generate_visualizations(self):
        """
        Generate visualizations for retail analysis.
        """
        try:
            # Create figures directory
            figures_dir = self.output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Actual vs predicted retail
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data['retail_businesses'], self.data['predicted_retail'], alpha=0.7)
            
            # Add diagonal line (perfect predictions)
            max_val = max(self.data['retail_businesses'].max(), self.data['predicted_retail'].max())
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            # Add labels for some points
            for i, row in self.data.iterrows():
                if abs(row['prediction_error_pct']) > 20:  # Label points with >20% error
                    plt.annotate(row['zip_code'], 
                                (row['retail_businesses'], row['predicted_retail']),
                                xytext=(5, 5), textcoords='offset points')
            
            plt.title('Actual vs Predicted Retail Businesses by ZIP Code')
            plt.xlabel('Actual Retail Businesses')
            plt.ylabel('Predicted Retail Businesses')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'actual_vs_predicted_retail.png'
            plt.savefig(chart_path)
            plt.close()
            
            # Feature importance
            plt.figure(figsize=(10, 6))
            
            # Get feature importance
            importance = self.model.named_steps['regressor'].feature_importances_
            indices = np.argsort(importance)
            
            plt.barh(range(len(indices)), importance[indices], align='center')
            plt.yticks(range(len(indices)), [self.features[i] for i in indices])
            plt.title('Feature Importance for Retail Prediction')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'retail_feature_importance.png'
            plt.savefig(chart_path)
            plt.close()
            
            # Projected growth rate by ZIP code
            plt.figure(figsize=(12, 8))
            
            # Sort by projected growth rate
            plot_data = self.data.sort_values('projected_growth_rate', ascending=False).head(20)
            
            # Create bar chart
            ax = sns.barplot(x='zip_code', y='projected_growth_rate', data=plot_data)
            
            # Color bars based on positive/negative growth
            for i, bar in enumerate(ax.patches):
                if plot_data.iloc[i]['projected_growth_rate'] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.title('Projected 5-Year Annual Retail Growth Rate by ZIP Code (Top 20)')
            plt.xlabel('ZIP Code')
            plt.ylabel('Projected Annual Growth Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'projected_retail_growth_rate.png'
            plt.savefig(chart_path)
            plt.close()
            
            logger.info("Generated retail analysis visualizations")
            
        except Exception as e:
            logger.error(f"Error generating retail visualizations: {str(e)}")
            raise
    
    def _save_results(self):
        """
        Save retail analysis results.
        """
        try:
            # Save predictions to CSV
            predictions_path = self.output_dir / "retail_predictions.csv"
            self.predictions.to_csv(predictions_path, index=False)
            
            # Store paths in results
            self.results['predictions_path'] = str(predictions_path)
            
            # Store summary statistics
            self.results['summary_stats'] = {
                'mean_error': self.data['prediction_error'].mean(),
                'mean_abs_error': self.data['prediction_error'].abs().mean(),
                'mean_error_pct': self.data['prediction_error_pct'].mean(),
                'mean_abs_error_pct': self.data['prediction_error_pct'].abs().mean(),
                'mean_growth_rate': self.data['projected_growth_rate'].mean(),
                'max_growth_rate': self.data['projected_growth_rate'].max(),
                'min_growth_rate': self.data['projected_growth_rate'].min(),
            }
            
            # Save results to JSON
            self._save_results_json()
            
            logger.info("Saved retail analysis results")
            
        except Exception as e:
            logger.error(f"Error saving retail analysis results: {str(e)}")
            raise
    
    def get_results(self):
        """
        Get retail analysis results.
        
        Returns:
            dict: Analysis results
        """
        return self.results
