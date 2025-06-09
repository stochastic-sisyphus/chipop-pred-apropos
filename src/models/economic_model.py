"""
Economic model for predicting economic trends in Chicago ZIP codes.
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

class EconomicModel(BaseModel):
    """Model for predicting economic trends in Chicago ZIP codes."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the economic model.
        
        Args:
            output_dir (Path, optional): Directory to save model outputs
        """
        super().__init__("Economic", output_dir)
        self.model = None
        self.features = None
        self.predictions = None
    
    def run_analysis(self, data):
        """
        Run economic analysis on the provided data.
        
        Args:
            data (pd.DataFrame): Input data for analysis
            
        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            logger.info("Running economic analysis...")
            
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
            
            logger.info("Economic analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in Economic analysis: {str(e)}")
            return False
    
    def _prepare_data(self):
        """
        Prepare data for economic analysis.
        """
        try:
            # Check for required columns
            required_cols = ['zip_code']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Create or use median_income
            if 'median_income' not in self.data.columns:
                # Create synthetic median income based on other factors
                if 'housing_units' in self.data.columns and 'population' in self.data.columns:
                    # Higher housing density often correlates with higher income in urban areas
                    housing_density = self.data['housing_units'] / self.data['population']
                    self.data['median_income'] = (
                        50000 + housing_density * 10000
                    ).round(0)
                else:
                    # Default synthetic value
                    self.data['median_income'] = 50000
            
            # Create or use employment_rate
            if 'employment_rate' not in self.data.columns:
                # Create synthetic employment rate
                self.data['employment_rate'] = 95.0  # Default 95% employment
            
            # Create economic growth indicators
            # Business growth if retail data exists
            if 'retail_businesses' in self.data.columns:
                self.data['business_growth_indicator'] = (
                    self.data['retail_businesses'] / self.data['retail_businesses'].mean()
                ).round(2)
            else:
                # Synthetic business growth
                self.data['business_growth_indicator'] = 1.0
            
            # Housing value growth if housing data exists
            if 'housing_units' in self.data.columns and 'median_income' in self.data.columns:
                self.data['housing_value_indicator'] = (
                    self.data['median_income'] / 50000 * 
                    (self.data['housing_units'] / self.data['housing_units'].mean())
                ).round(2)
            else:
                # Synthetic housing value
                self.data['housing_value_indicator'] = 1.0
            
            # Select features for model
            self.features = [
                'median_income',
                'employment_rate',
                'business_growth_indicator',
                'housing_value_indicator'
            ]
            
            # Add population if available
            if 'population' in self.data.columns:
                self.features.append('population')
            
            # Filter rows with missing values
            self.data = self.data.dropna(subset=self.features)
            
            # Create target variable: economic health index
            # This is a synthetic measure combining income, employment, and growth indicators
            self.data['economic_health_index'] = (
                (self.data['median_income'] / 50000) * 0.4 +
                (self.data['employment_rate'] / 100) * 0.3 +
                self.data['business_growth_indicator'] * 0.15 +
                self.data['housing_value_indicator'] * 0.15
            ).round(2)
            
            logger.info(f"Prepared data for economic analysis: {len(self.data)} records")
            
        except Exception as e:
            logger.error(f"Error preparing data for economic analysis: {str(e)}")
            raise
    
    def _train_model(self):
        """
        Train economic prediction model.
        """
        try:
            # Split data into features and target
            X = self.data[self.features]
            y = self.data['economic_health_index']
            
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
            logger.error(f"Error training economic model: {str(e)}")
            raise
    
    def _make_predictions(self):
        """
        Make economic predictions.
        """
        try:
            # Make predictions on all data
            self.data['predicted_economic_index'] = self.model.predict(self.data[self.features])
            
            # Calculate prediction error
            self.data['prediction_error'] = (
                self.data['predicted_economic_index'] - self.data['economic_health_index']
            )
            
            self.data['prediction_error_pct'] = (
                self.data['prediction_error'] / self.data['economic_health_index'] * 100
            ).round(1)
            
            # Project future economic health (5 years)
            # Assume modest growth in key indicators
            future_data = self.data.copy()
            
            future_data['median_income'] *= 1.10  # 10% income growth over 5 years
            future_data['employment_rate'] = future_data['employment_rate'].clip(upper=99.0)  # Cap at 99%
            future_data['business_growth_indicator'] *= 1.05  # 5% business growth
            future_data['housing_value_indicator'] *= 1.08  # 8% housing value growth
            
            if 'population' in self.features:
                future_data['population'] *= 1.03  # 3% population growth
            
            self.data['projected_economic_index_5yr'] = self.model.predict(future_data[self.features])
            
            # Calculate growth rate
            self.data['projected_growth_rate'] = (
                (self.data['projected_economic_index_5yr'] / self.data['economic_health_index']) ** (1/5) - 1
            ).round(4)
            
            # Classify ZIP codes by economic health
            self.data['economic_category'] = pd.cut(
                self.data['economic_health_index'],
                bins=[0, 0.8, 0.9, 1.0, 1.1, float('inf')],
                labels=['Struggling', 'Below Average', 'Average', 'Above Average', 'Thriving']
            )
            
            # Store predictions
            self.predictions = self.data[['zip_code', 'economic_health_index', 'predicted_economic_index', 
                                         'prediction_error', 'prediction_error_pct',
                                         'projected_economic_index_5yr', 'projected_growth_rate',
                                         'economic_category']]
            
            logger.info(f"Made predictions for {len(self.predictions)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error making economic predictions: {str(e)}")
            raise
    
    def _generate_visualizations(self):
        """
        Generate visualizations for economic analysis.
        """
        try:
            # Create figures directory
            figures_dir = self.output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # Actual vs predicted economic health
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data['economic_health_index'], self.data['predicted_economic_index'], alpha=0.7)
            
            # Add diagonal line (perfect predictions)
            max_val = max(self.data['economic_health_index'].max(), self.data['predicted_economic_index'].max())
            min_val = min(self.data['economic_health_index'].min(), self.data['predicted_economic_index'].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Add labels for some points
            for i, row in self.data.iterrows():
                if abs(row['prediction_error_pct']) > 10:  # Label points with >10% error
                    plt.annotate(row['zip_code'], 
                                (row['economic_health_index'], row['predicted_economic_index']),
                                xytext=(5, 5), textcoords='offset points')
            
            plt.title('Actual vs Predicted Economic Health Index by ZIP Code')
            plt.xlabel('Actual Economic Health Index')
            plt.ylabel('Predicted Economic Health Index')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'actual_vs_predicted_economic.png'
            plt.savefig(chart_path)
            plt.close()
            
            # Feature importance
            plt.figure(figsize=(10, 6))
            
            # Get feature importance
            importance = self.model.named_steps['regressor'].feature_importances_
            indices = np.argsort(importance)
            
            plt.barh(range(len(indices)), importance[indices], align='center')
            plt.yticks(range(len(indices)), [self.features[i] for i in indices])
            plt.title('Feature Importance for Economic Health Prediction')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'economic_feature_importance.png'
            plt.savefig(chart_path)
            plt.close()
            
            # Economic health by ZIP code
            plt.figure(figsize=(12, 8))
            
            # Sort by economic health index
            plot_data = self.data.sort_values('economic_health_index', ascending=False).head(20)
            
            # Create bar chart
            ax = sns.barplot(x='zip_code', y='economic_health_index', data=plot_data)
            
            # Color bars based on economic category
            category_colors = {
                'Struggling': 'red',
                'Below Average': 'orange',
                'Average': 'yellow',
                'Above Average': 'lightgreen',
                'Thriving': 'darkgreen'
            }
            
            for i, bar in enumerate(ax.patches):
                category = plot_data.iloc[i]['economic_category']
                bar.set_color(category_colors.get(category, 'blue'))
            
            plt.title('Economic Health Index by ZIP Code (Top 20)')
            plt.xlabel('ZIP Code')
            plt.ylabel('Economic Health Index')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'economic_health_by_zip.png'
            plt.savefig(chart_path)
            plt.close()
            
            # Economic category distribution
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of economic categories
            category_counts = self.data['economic_category'].value_counts().sort_index()
            category_counts.plot(kind='bar')
            
            plt.title('Distribution of ZIP Codes by Economic Health Category')
            plt.xlabel('Economic Health Category')
            plt.ylabel('Number of ZIP Codes')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'economic_category_distribution.png'
            plt.savefig(chart_path)
            plt.close()
            
            logger.info("Generated economic analysis visualizations")
            
        except Exception as e:
            logger.error(f"Error generating economic visualizations: {str(e)}")
            raise
    
    def _save_results(self):
        """
        Save economic analysis results.
        """
        try:
            # Save predictions to CSV
            predictions_path = self.output_dir / "economic_predictions.csv"
            self.predictions.to_csv(predictions_path, index=False)
            
            # Store paths in results
            self.results['predictions_path'] = str(predictions_path)
            
            # Store summary statistics
            self.results['summary_stats'] = {
                'mean_economic_index': self.data['economic_health_index'].mean().round(2),
                'median_economic_index': self.data['economic_health_index'].median().round(2),
                'max_economic_index': self.data['economic_health_index'].max().round(2),
                'min_economic_index': self.data['economic_health_index'].min().round(2),
                'mean_growth_rate': self.data['projected_growth_rate'].mean().round(4),
                'max_growth_rate': self.data['projected_growth_rate'].max().round(4),
                'min_growth_rate': self.data['projected_growth_rate'].min().round(4),
                'category_counts': self.data['economic_category'].value_counts().to_dict()
            }
            
            # Save results to JSON
            self._save_results_json()
            
            logger.info("Saved economic analysis results")
            
        except Exception as e:
            logger.error(f"Error saving economic analysis results: {str(e)}")
            raise
    
    def get_results(self):
        """
        Get economic analysis results.
        
        Returns:
            dict: Analysis results
        """
        return self.results
