"""
Housing model for predicting housing trends in Chicago ZIP codes.
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

class HousingModel(BaseModel):
    """Model for predicting housing trends in Chicago ZIP codes."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the housing model.
        
        Args:
            output_dir (Path, optional): Directory to save model outputs
        """
        super().__init__("Housing", output_dir)
        self.model = None
        self.features = None
        self.predictions = None
    
    def run_analysis(self, data):
        """
        Run housing analysis on the provided data.
        
        Args:
            data (pd.DataFrame): Input data for analysis
            
        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            logger.info("Running housing analysis...")
            
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
            
            logger.info("Housing analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in Housing analysis: {str(e)}")
            return False
    
    def _prepare_data(self):
        """
        Prepare data for housing analysis.
        """
        try:
            # Check for required columns
            required_cols = ['housing_units', 'zip_code']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Create derived features
            # Calculate housing density if land_area exists
            if 'land_area' in self.data.columns:
                self.data['housing_density'] = (
                    self.data['housing_units'] / self.data['land_area']
                ).round(2)
            else:
                # Create synthetic housing density
                self.data['housing_density'] = (
                    self.data['housing_units'] / 3  # Assume 3 sq miles
                ).round(2)
            
            # Calculate housing per capita if population exists
            if 'population' in self.data.columns:
                self.data['housing_per_capita'] = (
                    self.data['housing_units'] / self.data['population']
                ).round(4)
            else:
                # Create synthetic housing per capita
                self.data['housing_per_capita'] = 0.4  # Assume 0.4 housing units per person
            
            # Calculate new construction if permit data exists
            if 'new_construction_permits' in self.data.columns:
                self.data['new_construction_ratio'] = (
                    self.data['new_construction_permits'] / self.data['housing_units']
                ).round(4)
            else:
                # Create synthetic new construction ratio
                self.data['new_construction_ratio'] = 0.01  # Assume 1% new construction
            
            # Select features for model
            self.features = [
                'housing_density',
                'housing_per_capita',
                'new_construction_ratio'
            ]
            
            # Check if median_income exists
            if 'median_income' in self.data.columns:
                self.features.append('median_income')
            
            # Check if population exists
            if 'population' in self.data.columns:
                self.features.append('population')
            
            # Check if retail_businesses exists
            if 'retail_businesses' in self.data.columns:
                self.features.append('retail_businesses')
            
            # Filter rows with missing values
            self.data = self.data.dropna(subset=['housing_units'] + self.features)
            
            logger.info(f"Prepared data for housing analysis: {len(self.data)} records")
            
        except Exception as e:
            logger.error(f"Error preparing data for housing analysis: {str(e)}")
            raise
    
    def _train_model(self):
        """
        Train housing prediction model.
        """
        try:
            # Split data into features and target
            X = self.data[self.features]
            y = self.data['housing_units']
            
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
            logger.error(f"Error training housing model: {str(e)}")
            raise
    
    def _make_predictions(self):
        """
        Make housing predictions.
        """
        try:
            # Make predictions on all data
            self.data['predicted_housing'] = self.model.predict(self.data[self.features])
            
            # Calculate prediction error
            self.data['prediction_error'] = (
                self.data['predicted_housing'] - self.data['housing_units']
            )
            
            self.data['prediction_error_pct'] = (
                self.data['prediction_error'] / self.data['housing_units'] * 100
            ).round(1)
            
            # Project future housing (5 years)
            # Assume 5% growth in population and income
            future_data = self.data.copy()
            
            if 'population' in self.features:
                future_data['population'] *= 1.05
            
            if 'median_income' in self.features:
                future_data['median_income'] *= 1.05
            
            if 'retail_businesses' in self.features:
                future_data['retail_businesses'] *= 1.03
            
            future_data['new_construction_ratio'] *= 1.1  # 10% increase in construction
            
            self.data['projected_housing_5yr'] = self.model.predict(future_data[self.features])
            
            # Calculate growth rate
            self.data['projected_growth_rate'] = (
                (self.data['projected_housing_5yr'] / self.data['housing_units']) ** (1/5) - 1
            ).round(4)
            
            # Store predictions
            self.predictions = self.data[['zip_code', 'housing_units', 'predicted_housing', 
                                         'prediction_error', 'prediction_error_pct',
                                         'projected_housing_5yr', 'projected_growth_rate']]
            
            logger.info(f"Made predictions for {len(self.predictions)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error making housing predictions: {str(e)}")
            raise
    
    def _generate_visualizations(self):
        """
        Generate visualizations for housing analysis.
        """
        try:
            # Create figures directory
            figures_dir = self.output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Actual vs predicted housing
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data['housing_units'], self.data['predicted_housing'], alpha=0.7)
            
            # Add diagonal line (perfect predictions)
            max_val = max(self.data['housing_units'].max(), self.data['predicted_housing'].max())
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            # Add labels for some points
            for i, row in self.data.iterrows():
                if abs(row['prediction_error_pct']) > 20:  # Label points with >20% error
                    plt.annotate(row['zip_code'], 
                                (row['housing_units'], row['predicted_housing']),
                                xytext=(5, 5), textcoords='offset points')
            
            plt.title('Actual vs Predicted Housing Units by ZIP Code')
            plt.xlabel('Actual Housing Units')
            plt.ylabel('Predicted Housing Units')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'actual_vs_predicted_housing.png'
            plt.savefig(chart_path)
            plt.close()
            
            # Feature importance
            plt.figure(figsize=(10, 6))
            
            # Get feature importance
            importance = self.model.named_steps['regressor'].feature_importances_
            indices = np.argsort(importance)
            
            plt.barh(range(len(indices)), importance[indices], align='center')
            plt.yticks(range(len(indices)), [self.features[i] for i in indices])
            plt.title('Feature Importance for Housing Prediction')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'housing_feature_importance.png'
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
            
            plt.title('Projected 5-Year Annual Housing Growth Rate by ZIP Code (Top 20)')
            plt.xlabel('ZIP Code')
            plt.ylabel('Projected Annual Growth Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'projected_housing_growth_rate.png'
            plt.savefig(chart_path)
            plt.close()
            
            logger.info("Generated housing analysis visualizations")
            
        except Exception as e:
            logger.error(f"Error generating housing visualizations: {str(e)}")
            raise
    
    def _save_results(self):
        """
        Save housing analysis results.
        """
        try:
            # Save predictions to CSV
            predictions_path = self.output_dir / "housing_predictions.csv"
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
            
            logger.info("Saved housing analysis results")
            
        except Exception as e:
            logger.error(f"Error saving housing analysis results: {str(e)}")
            raise
    
    def get_results(self):
        """
        Get housing analysis results.
        
        Returns:
            dict: Analysis results
        """
        return self.results
