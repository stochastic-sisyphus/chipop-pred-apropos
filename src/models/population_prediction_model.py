"""
Population Prediction Model for Chicago ZIP Codes

This model predicts population shifts over the next 10 years based on:
- Multi-family and single-family housing developments
- Historical population trends (20 years)
- Income distribution changes
- External factors (interest rates, economic indicators)
- Zoning regulations and their impact
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PopulationPredictionModel:
    """
    Predicts population shifts at ZIP code level over 10 years with 95% confidence intervals.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("output/models/population_prediction")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Key neighborhoods of interest
        self.focus_neighborhoods = {
            '60615': 'Bronzeville',
            '60637': 'Woodlawn', 
            '60607': 'West Loop',
            '60654': 'River North',
            '60601': 'Loop',
            '60602': 'Loop',
            '60603': 'Loop',
            '60604': 'South Loop',
            '60605': 'South Loop'
        }
        
        # Model parameters
        self.forecast_years = 10  # 10-year forecast as requested
        self.confidence_level = 0.95  # 95% confidence intervals
        self.historical_years = 20  # 20 years of historical data
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive features for population prediction.
        """
        logger.info("Preparing features for population prediction")
        
        features = pd.DataFrame()
        
        # Group by ZIP code for time series analysis
        for zip_code in data['zip_code'].unique():
            zip_data = data[data['zip_code'] == zip_code].copy()
            
            # Historical features (20 years)
            if 'year' in zip_data.columns:
                zip_data = zip_data.sort_values('year')
                
                # Population trends
                if 'population' in zip_data.columns:
                    pop_values = zip_data.groupby('year')['population'].first().values
                    if len(pop_values) > 1:
                        pop_growth_rate = np.mean(np.diff(pop_values) / pop_values[:-1])
                        pop_volatility = np.std(np.diff(pop_values) / pop_values[:-1])
                    else:
                        pop_growth_rate = 0
                        pop_volatility = 0
                else:
                    pop_growth_rate = 0
                    pop_volatility = 0
                
                # Housing unit trends
                if 'housing_units' in zip_data.columns:
                    housing_values = zip_data.groupby('year')['housing_units'].first().values
                    if len(housing_values) > 1:
                        housing_growth_rate = np.mean(np.diff(housing_values) / housing_values[:-1])
                        housing_acceleration = np.mean(np.diff(np.diff(housing_values)))
                    else:
                        housing_growth_rate = 0
                        housing_acceleration = 0
                else:
                    housing_growth_rate = 0
                    housing_acceleration = 0
            else:
                pop_growth_rate = 0
                pop_volatility = 0
                housing_growth_rate = 0
                housing_acceleration = 0
            
            # Permit-based features
            permit_data = zip_data[zip_data.get('permit_year', 0) > 0] if 'permit_year' in zip_data.columns else pd.DataFrame()
            
            # Multi-family vs single-family ratio
            if len(permit_data) > 0 and 'permit_type' in permit_data.columns:
                multifamily_permits = len(permit_data[permit_data['permit_type'] == 'multifamily'])
                singlefamily_permits = len(permit_data[permit_data['permit_type'] == 'single_family'])
                total_permits = multifamily_permits + singlefamily_permits
                multifamily_ratio = multifamily_permits / total_permits if total_permits > 0 else 0
                
                # Recent permit activity (last 3 years)
                recent_years = permit_data['permit_year'].max() - 2 if len(permit_data) > 0 else 0
                recent_permits = permit_data[permit_data['permit_year'] >= recent_years]
                recent_units = recent_permits['unit_count'].sum() if 'unit_count' in recent_permits.columns else 0
                
                # Permit momentum
                permit_counts = permit_data.groupby('permit_year').size()
                if len(permit_counts) > 2:
                    permit_momentum = np.polyfit(range(len(permit_counts)), permit_counts.values, 1)[0]
                else:
                    permit_momentum = 0
            else:
                multifamily_ratio = 0
                recent_units = 0
                permit_momentum = 0
            
            # Income distribution features
            if 'median_income' in zip_data.columns:
                income_values = zip_data.groupby('year')['median_income'].first().values
                if len(income_values) > 1:
                    income_growth = np.mean(np.diff(income_values) / income_values[:-1])
                    income_inequality = np.std(income_values) / np.mean(income_values) if np.mean(income_values) > 0 else 0
                else:
                    income_growth = 0
                    income_inequality = 0
                
                # Income tier (for gentrification analysis)
                latest_income = income_values[-1] if len(income_values) > 0 else 0
                income_tier = self._calculate_income_tier(latest_income)
            else:
                income_growth = 0
                income_inequality = 0
                income_tier = 0
            
            # Economic indicators (external factors)
            if 'employment_rate' in zip_data.columns:
                employment_trend = zip_data['employment_rate'].mean()
                employment_stability = 1 - zip_data['employment_rate'].std() if len(zip_data) > 1 else 1
            else:
                employment_trend = 0.9  # Default assumption
                employment_stability = 0.9
            
            # Retail development indicator
            if 'retail_businesses' in zip_data.columns:
                retail_values = zip_data.groupby('year')['retail_businesses'].first().values
                if len(retail_values) > 1:
                    retail_growth = np.mean(np.diff(retail_values) / retail_values[:-1])
                else:
                    retail_growth = 0
            else:
                retail_growth = 0
            
            # Neighborhood characteristics
            is_focus_neighborhood = 1 if zip_code in self.focus_neighborhoods else 0
            is_downtown = 1 if zip_code in ['60601', '60602', '60603', '60604', '60605'] else 0
            is_north_side = 1 if zip_code.startswith('606') and int(zip_code) > 60630 else 0
            is_south_side = 1 if zip_code.startswith('606') and int(zip_code) < 60615 else 0
            
            # Current values for baseline
            current_population = zip_data['population'].iloc[-1] if 'population' in zip_data.columns and len(zip_data) > 0 else 0
            current_housing = zip_data['housing_units'].iloc[-1] if 'housing_units' in zip_data.columns and len(zip_data) > 0 else 0
            
            # Compile features
            feature_row = pd.DataFrame({
                'zip_code': [zip_code],
                'current_population': [current_population],
                'current_housing_units': [current_housing],
                'pop_growth_rate': [pop_growth_rate],
                'pop_volatility': [pop_volatility],
                'housing_growth_rate': [housing_growth_rate],
                'housing_acceleration': [housing_acceleration],
                'multifamily_ratio': [multifamily_ratio],
                'recent_units': [recent_units],
                'permit_momentum': [permit_momentum],
                'income_growth': [income_growth],
                'income_inequality': [income_inequality],
                'income_tier': [income_tier],
                'employment_trend': [employment_trend],
                'employment_stability': [employment_stability],
                'retail_growth': [retail_growth],
                'is_focus_neighborhood': [is_focus_neighborhood],
                'is_downtown': [is_downtown],
                'is_north_side': [is_north_side],
                'is_south_side': [is_south_side],
                'persons_per_unit': [current_population / current_housing if current_housing > 0 else 2.5]
            })
            
            features = pd.concat([features, feature_row], ignore_index=True)
        
        return features
    
    def _calculate_income_tier(self, income: float) -> int:
        """Calculate income tier for gentrification analysis."""
        if income < 30000:
            return 1  # Low income
        elif income < 50000:
            return 2  # Lower middle
        elif income < 75000:
            return 3  # Middle
        elif income < 100000:
            return 4  # Upper middle
        else:
            return 5  # High income
    
    def train_models(self, features: pd.DataFrame, data: pd.DataFrame):
        """
        Train ensemble models for population and income prediction.
        """
        logger.info("Training predictive models with confidence intervals")
        
        # Prepare target variables
        targets = self._prepare_targets(features, data)
        
        # Feature columns for modeling
        feature_cols = [
            'pop_growth_rate', 'pop_volatility', 'housing_growth_rate', 
            'housing_acceleration', 'multifamily_ratio', 'recent_units',
            'permit_momentum', 'income_growth', 'income_inequality',
            'income_tier', 'employment_trend', 'employment_stability',
            'retail_growth', 'is_focus_neighborhood', 'is_downtown',
            'is_north_side', 'is_south_side', 'persons_per_unit'
        ]
        
        X = features[feature_cols]
        
        # Train population growth model
        logger.info("Training population growth model")
        self.models['population'] = self._train_ensemble(
            X, targets['population_growth'],
            model_name='population_growth'
        )
        
        # Train income change model
        logger.info("Training income distribution model")
        self.models['income'] = self._train_ensemble(
            X, targets['income_change'],
            model_name='income_change'
        )
        
        # Train housing unit model
        logger.info("Training housing unit projection model")
        self.models['housing'] = self._train_ensemble(
            X, targets['housing_growth'],
            model_name='housing_growth'
        )
        
        # Calculate feature importance
        self._calculate_feature_importance(X, feature_cols)
    
    def _prepare_targets(self, features: pd.DataFrame, data: pd.DataFrame) -> dict:
        """Prepare target variables for training."""
        targets = {}
        
        # Population growth rate
        pop_growth = []
        income_change = []
        housing_growth = []
        
        for _, row in features.iterrows():
            zip_code = row['zip_code']
            zip_data = data[data['zip_code'] == zip_code]
            
            # Calculate historical growth rates as targets
            if 'population' in zip_data.columns and len(zip_data) > 1:
                pop_values = zip_data.groupby('year')['population'].first().values
                growth_rate = np.mean(np.diff(pop_values) / pop_values[:-1]) if len(pop_values) > 1 else 0
                pop_growth.append(growth_rate)
            else:
                pop_growth.append(0)
            
            if 'median_income' in zip_data.columns and len(zip_data) > 1:
                income_values = zip_data.groupby('year')['median_income'].first().values
                change_rate = np.mean(np.diff(income_values) / income_values[:-1]) if len(income_values) > 1 else 0
                income_change.append(change_rate)
            else:
                income_change.append(0)
            
            if 'housing_units' in zip_data.columns and len(zip_data) > 1:
                housing_values = zip_data.groupby('year')['housing_units'].first().values
                growth_rate = np.mean(np.diff(housing_values) / housing_values[:-1]) if len(housing_values) > 1 else 0
                housing_growth.append(growth_rate)
            else:
                housing_growth.append(0)
        
        targets['population_growth'] = np.array(pop_growth)
        targets['income_change'] = np.array(income_change)
        targets['housing_growth'] = np.array(housing_growth)
        
        return targets
    
    def _train_ensemble(self, X: pd.DataFrame, y: np.ndarray, model_name: str) -> dict:
        """Train ensemble of models for robust predictions."""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[model_name] = scaler
        
        # Train multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        
        trained_models = {}
        scores = {}
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            scores[name] = cv_scores.mean()
            
            # Train on full data
            model.fit(X_scaled, y)
            trained_models[name] = model
            
            logger.info(f"{model_name} - {name} model R2: {scores[name]:.3f}")
        
        return {
            'models': trained_models,
            'scores': scores,
            'scaler': scaler
        }
    
    def predict_with_confidence(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with 95% confidence intervals.
        """
        logger.info("Generating 10-year predictions with confidence intervals")
        
        predictions = features.copy()
        
        # Predict for each target
        for target, model_dict in self.models.items():
            feature_cols = [col for col in features.columns if col not in ['zip_code', 'current_population', 'current_housing_units']]
            X = features[feature_cols]
            X_scaled = model_dict['scaler'].transform(X)
            
            # Get predictions from each model
            all_predictions = []
            for model_name, model in model_dict['models'].items():
                pred = model.predict(X_scaled)
                all_predictions.append(pred)
            
            # Ensemble prediction (mean)
            ensemble_pred = np.mean(all_predictions, axis=0)
            
            # Calculate confidence intervals using prediction variance
            pred_std = np.std(all_predictions, axis=0)
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            
            # Bootstrap for more robust confidence intervals
            bootstrap_preds = self._bootstrap_predictions(X_scaled, model_dict['models'], n_bootstrap=100)
            lower_bound = np.percentile(bootstrap_preds, (1 - self.confidence_level) * 100 / 2, axis=0)
            upper_bound = np.percentile(bootstrap_preds, (1 + self.confidence_level) * 100 / 2, axis=0)
            
            predictions[f'{target}_rate'] = ensemble_pred
            predictions[f'{target}_lower'] = lower_bound
            predictions[f'{target}_upper'] = upper_bound
            predictions[f'{target}_std'] = pred_std
        
        # Calculate 10-year projections
        predictions = self._calculate_projections(predictions)
        
        return predictions
    
    def _bootstrap_predictions(self, X: np.ndarray, models: dict, n_bootstrap: int = 100) -> np.ndarray:
        """Generate bootstrap predictions for confidence intervals."""
        n_samples = X.shape[0]
        bootstrap_preds = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Get predictions for each model
            model_preds = []
            for model in models.values():
                # Use out-of-bag samples for prediction
                oob_indices = np.setdiff1d(range(n_samples), indices)
                if len(oob_indices) > 0:
                    pred = model.predict(X[oob_indices])
                    # Extend to full size
                    full_pred = np.zeros(n_samples)
                    full_pred[oob_indices] = pred
                    model_preds.append(full_pred)
            
            if model_preds:
                bootstrap_preds.append(np.mean(model_preds, axis=0))
        
        return np.array(bootstrap_preds)
    
    def _calculate_projections(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Calculate 10-year projections from growth rates."""
        # Population projections
        for year in range(1, self.forecast_years + 1):
            # Mean projection
            predictions[f'population_year_{year}'] = (
                predictions['current_population'] * 
                (1 + predictions['population_rate']) ** year
            )
            
            # Lower bound
            predictions[f'population_year_{year}_lower'] = (
                predictions['current_population'] * 
                (1 + predictions['population_lower']) ** year
            )
            
            # Upper bound
            predictions[f'population_year_{year}_upper'] = (
                predictions['current_population'] * 
                (1 + predictions['population_upper']) ** year
            )
            
            # Housing projections
            predictions[f'housing_year_{year}'] = (
                predictions['current_housing_units'] * 
                (1 + predictions['housing_rate']) ** year
            )
            
            # Income projections (as multiplier of current)
            predictions[f'income_multiplier_year_{year}'] = (
                (1 + predictions['income_rate']) ** year
            )
        
        # Add specific neighborhood flags
        predictions['neighborhood_name'] = predictions['zip_code'].map(self.focus_neighborhoods)
        
        return predictions
    
    def _calculate_feature_importance(self, X: pd.DataFrame, feature_cols: list):
        """Calculate and visualize feature importance."""
        logger.info("Calculating feature importance")
        
        importance_df = pd.DataFrame()
        
        for target, model_dict in self.models.items():
            # Get feature importance from Random Forest
            rf_model = model_dict['models']['rf']
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_,
                'target': target
            })
            importance_df = pd.concat([importance_df, importance], ignore_index=True)
        
        # Average importance across targets
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        avg_importance.plot(kind='barh')
        plt.xlabel('Feature Importance')
        plt.title('Average Feature Importance Across All Models')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['feature_importance'] = avg_importance.to_dict()
    
    def create_visualizations(self, predictions: pd.DataFrame, data: pd.DataFrame):
        """
        Create comprehensive visualizations including time-lapse capabilities.
        """
        logger.info("Creating visualizations for population predictions")
        
        # 1. 10-year population projection with confidence intervals
        self._plot_population_projections(predictions)
        
        # 2. Focus neighborhood analysis
        self._plot_neighborhood_focus(predictions)
        
        # 3. Income distribution changes
        self._plot_income_distribution(predictions, data)
        
        # 4. Housing vs population growth
        self._plot_housing_population_correlation(predictions)
        
        # 5. Confidence interval analysis
        self._plot_confidence_analysis(predictions)
        
        # 6. Time-lapse preparation data
        self._prepare_timelapse_data(predictions)
    
    def _plot_population_projections(self, predictions: pd.DataFrame):
        """Plot 10-year population projections with confidence intervals."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top 10 growth ZIP codes
        top_growth = predictions.nlargest(10, 'population_rate')
        
        # Plot 1: Population projections
        years = list(range(2025, 2025 + self.forecast_years + 1))
        
        for _, row in top_growth.iterrows():
            zip_code = row['zip_code']
            neighborhood = row.get('neighborhood_name', zip_code)
            
            # Mean projection
            pop_projection = [row['current_population']]
            pop_lower = [row['current_population']]
            pop_upper = [row['current_population']]
            
            for year in range(1, self.forecast_years + 1):
                pop_projection.append(row[f'population_year_{year}'])
                pop_lower.append(row[f'population_year_{year}_lower'])
                pop_upper.append(row[f'population_year_{year}_upper'])
            
            ax1.plot(years, pop_projection, label=f'{neighborhood} ({zip_code})', linewidth=2)
            ax1.fill_between(years, pop_lower, pop_upper, alpha=0.2)
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Population')
        ax1.set_title('10-Year Population Projections with 95% Confidence Intervals')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Growth rates
        ax2.barh(range(len(top_growth)), top_growth['population_rate'] * 100)
        ax2.set_yticks(range(len(top_growth)))
        ax2.set_yticklabels([f"{row['neighborhood_name'] or row['zip_code']} ({row['zip_code']})" 
                            for _, row in top_growth.iterrows()])
        ax2.set_xlabel('Annual Population Growth Rate (%)')
        ax2.set_title('Top 10 ZIP Codes by Projected Population Growth')
        ax2.grid(True, alpha=0.3)
        
        # Add error bars
        errors = (top_growth['population_upper'] - top_growth['population_lower']) * 50
        ax2.errorbar(top_growth['population_rate'] * 100, range(len(top_growth)), 
                    xerr=errors, fmt='none', color='black', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'population_projections_10year.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_neighborhood_focus(self, predictions: pd.DataFrame):
        """Create focused analysis for key neighborhoods."""
        focus_data = predictions[predictions['neighborhood_name'].notna()]
        
        if len(focus_data) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot 1: Population growth comparison
        ax = axes[0]
        x = range(len(focus_data))
        ax.bar(x, focus_data['population_rate'] * 100, 
               color=['#1f77b4' if 'Woodlawn' in name or 'Bronzeville' in name else '#ff7f0e' 
                      for name in focus_data['neighborhood_name']])
        ax.set_xticks(x)
        ax.set_xticklabels(focus_data['neighborhood_name'], rotation=45)
        ax.set_ylabel('Annual Growth Rate (%)')
        ax.set_title('Population Growth Rate by Focus Neighborhood')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Housing unit projections
        ax = axes[1]
        for _, row in focus_data.iterrows():
            housing_projection = [row['current_housing_units']]
            for year in range(1, self.forecast_years + 1):
                housing_projection.append(row[f'housing_year_{year}'])
            
            years = list(range(2025, 2025 + self.forecast_years + 1))
            ax.plot(years, housing_projection, label=row['neighborhood_name'], linewidth=2, marker='o')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Housing Units')
        ax.set_title('Housing Unit Projections - Focus Neighborhoods')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Income tier changes
        ax = axes[2]
        current_tiers = focus_data['income_tier']
        future_tiers = focus_data['income_tier'] + focus_data['income_rate'] * 10 * 2  # Rough projection
        
        x = np.arange(len(focus_data))
        width = 0.35
        ax.bar(x - width/2, current_tiers, width, label='Current', alpha=0.8)
        ax.bar(x + width/2, future_tiers, width, label='Projected (10yr)', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(focus_data['neighborhood_name'], rotation=45)
        ax.set_ylabel('Income Tier (1-5)')
        ax.set_title('Income Tier Changes - Gentrification Indicator')
        ax.legend()
        ax.set_ylim(0, 6)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Confidence interval width
        ax = axes[3]
        confidence_width = (focus_data['population_upper'] - focus_data['population_lower']) * 100
        ax.bar(x, confidence_width)
        ax.set_xticks(x)
        ax.set_xticklabels(focus_data['neighborhood_name'], rotation=45)
        ax.set_ylabel('Confidence Interval Width (%)')
        ax.set_title('Prediction Uncertainty by Neighborhood')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'neighborhood_focus_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_income_distribution(self, predictions: pd.DataFrame, data: pd.DataFrame):
        """Visualize income distribution changes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Current vs projected income distribution
        current_income_dist = []
        projected_income_dist = []
        
        for _, row in predictions.iterrows():
            zip_data = data[data['zip_code'] == row['zip_code']]
            if 'median_income' in zip_data.columns and len(zip_data) > 0:
                current_income = zip_data['median_income'].iloc[-1]
                projected_income = current_income * row[f'income_multiplier_year_{self.forecast_years}']
                current_income_dist.append(current_income)
                projected_income_dist.append(projected_income)
        
        # Plot 1: Distribution comparison
        ax1.hist(current_income_dist, bins=20, alpha=0.5, label='Current', density=True)
        ax1.hist(projected_income_dist, bins=20, alpha=0.5, label=f'Projected ({self.forecast_years}yr)', density=True)
        ax1.set_xlabel('Median Income ($)')
        ax1.set_ylabel('Density')
        ax1.set_title('Income Distribution: Current vs Projected')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Income mobility by ZIP
        top_mobility = predictions.nlargest(15, 'income_rate')
        y_pos = np.arange(len(top_mobility))
        
        ax2.barh(y_pos, top_mobility['income_rate'] * 100)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{row.get('neighborhood_name', row['zip_code'])} ({row['zip_code']})" 
                            for _, row in top_mobility.iterrows()])
        ax2.set_xlabel('Annual Income Growth Rate (%)')
        ax2.set_title('Top 15 ZIP Codes by Income Growth Potential')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'income_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_housing_population_correlation(self, predictions: pd.DataFrame):
        """Analyze correlation between housing and population growth."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Scatter plot of growth rates
        ax1.scatter(predictions['housing_rate'] * 100, 
                   predictions['population_rate'] * 100,
                   s=predictions['current_population'] / 1000,  # Size by current population
                   alpha=0.6)
        
        # Add trend line with error handling
        try:
            # Filter out NaN and infinite values
            mask = (np.isfinite(predictions['housing_rate']) & 
                   np.isfinite(predictions['population_rate']) &
                   ~predictions['housing_rate'].isna() &
                   ~predictions['population_rate'].isna())
            
            x_data = predictions.loc[mask, 'housing_rate'].values
            y_data = predictions.loc[mask, 'population_rate'].values
            
            if len(x_data) > 1 and np.std(x_data) > 0:  # Check for variation
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                ax1.plot(x_trend * 100, p(x_trend) * 100, "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        except Exception as e:
            logger.warning(f"Could not fit trend line: {e}")
        
        # Highlight focus neighborhoods
        focus = predictions[predictions['neighborhood_name'].notna()]
        ax1.scatter(focus['housing_rate'] * 100, 
                   focus['population_rate'] * 100,
                   s=focus['current_population'] / 1000,
                   color='red', alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add labels for focus neighborhoods
        for _, row in focus.iterrows():
            ax1.annotate(row['neighborhood_name'], 
                        (row['housing_rate'] * 100, row['population_rate'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Housing Growth Rate (%/year)')
        ax1.set_ylabel('Population Growth Rate (%/year)')
        ax1.set_title('Housing vs Population Growth Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Persons per unit analysis
        ax2.scatter(predictions['persons_per_unit'], 
                   predictions['population_rate'] * 100,
                   s=60, alpha=0.6)
        ax2.set_xlabel('Current Persons per Housing Unit')
        ax2.set_ylabel('Population Growth Rate (%/year)')
        ax2.set_title('Household Density vs Population Growth')
        ax2.grid(True, alpha=0.3)
        
        # Add median lines
        ax2.axvline(predictions['persons_per_unit'].median(), color='red', linestyle='--', alpha=0.5, label='Median')
        ax2.axhline(predictions['population_rate'].median() * 100, color='red', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'housing_population_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_analysis(self, predictions: pd.DataFrame):
        """Analyze prediction confidence and uncertainty."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Confidence interval width by prediction magnitude
        ci_width = (predictions['population_upper'] - predictions['population_lower']) * 100
        ax1.scatter(predictions['population_rate'] * 100, ci_width, alpha=0.6)
        ax1.set_xlabel('Population Growth Rate (%/year)')
        ax1.set_ylabel('95% CI Width (%)')
        ax1.set_title('Prediction Uncertainty vs Growth Rate')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line with error handling
        try:
            # Filter out NaN and infinite values
            mask = (np.isfinite(predictions['population_rate']) & 
                   np.isfinite(ci_width) &
                   ~predictions['population_rate'].isna() &
                   ~ci_width.isna())
            
            x_data = (predictions.loc[mask, 'population_rate'] * 100).values
            y_data = ci_width[mask].values
            
            if len(x_data) > 1 and np.std(x_data) > 0:  # Check for variation
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        except Exception as e:
            logger.warning(f"Could not fit trend line: {e}")
        
        # Plot 2: Model agreement (standard deviation of predictions)
        model_agreement = predictions.nsmallest(15, 'population_std')
        y_pos = np.arange(len(model_agreement))
        
        ax2.barh(y_pos, model_agreement['population_std'] * 100)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{row.get('neighborhood_name', row['zip_code'])} ({row['zip_code']})" 
                            for _, row in model_agreement.iterrows()])
        ax2.set_xlabel('Model Disagreement (Std Dev %)')
        ax2.set_title('ZIP Codes with Highest Model Agreement (Lowest Uncertainty)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _prepare_timelapse_data(self, predictions: pd.DataFrame):
        """Prepare data for time-lapse visualization."""
        timelapse_data = []
        
        for _, row in predictions.iterrows():
            zip_code = row['zip_code']
            
            # Create year-by-year data
            for year in range(self.forecast_years + 1):
                if year == 0:
                    population = row['current_population']
                    housing = row['current_housing_units']
                    income_mult = 1.0
                else:
                    population = row[f'population_year_{year}']
                    housing = row[f'housing_year_{year}']
                    income_mult = row[f'income_multiplier_year_{year}']
                
                timelapse_data.append({
                    'zip_code': zip_code,
                    'year': 2025 + year,
                    'population': population,
                    'population_lower': row[f'population_year_{year}_lower'] if year > 0 else population,
                    'population_upper': row[f'population_year_{year}_upper'] if year > 0 else population,
                    'housing_units': housing,
                    'income_multiplier': income_mult,
                    'neighborhood_name': row.get('neighborhood_name', ''),
                    'is_focus': row.get('neighborhood_name', '') != ''
                })
        
        timelapse_df = pd.DataFrame(timelapse_data)
        timelapse_df.to_csv(self.output_dir / 'timelapse_data.csv', index=False)
        
        # Also save as JSON for web visualization
        timelapse_json = timelapse_df.to_dict('records')
        with open(self.output_dir / 'timelapse_data.json', 'w') as f:
            json.dump(timelapse_json, f, indent=2)
        
        logger.info("Saved time-lapse data for visualization")
    
    def generate_report(self, predictions: pd.DataFrame) -> dict:
        """Generate comprehensive report with confidence levels."""
        logger.info("Generating population prediction report")
        
        # Summary statistics
        summary = {
            'forecast_horizon': f'{self.forecast_years} years',
            'confidence_level': f'{self.confidence_level * 100}%',
            'total_zip_codes': len(predictions),
            'focus_neighborhoods': list(self.focus_neighborhoods.values()),
            
            # Population projections
            'population_growth': {
                'mean_annual_rate': f"{predictions['population_rate'].mean() * 100:.2f}%",
                'median_annual_rate': f"{predictions['population_rate'].median() * 100:.2f}%",
                'top_growth_zips': predictions.nlargest(5, 'population_rate')[
                    ['zip_code', 'neighborhood_name', 'population_rate']
                ].to_dict('records'),
                'total_population_2025': int(predictions['current_population'].sum()),
                'total_population_2035': int(predictions[f'population_year_{self.forecast_years}'].sum()),
                'net_population_change': int(
                    predictions[f'population_year_{self.forecast_years}'].sum() - 
                    predictions['current_population'].sum()
                )
            },
            
            # Income distribution
            'income_changes': {
                'mean_annual_change': f"{predictions['income_rate'].mean() * 100:.2f}%",
                'gentrification_risk_zips': predictions[
                    (predictions['income_rate'] > predictions['income_rate'].quantile(0.75)) &
                    (predictions['income_tier'] <= 2)
                ][['zip_code', 'neighborhood_name']].to_dict('records')
            },
            
            # Housing projections
            'housing_growth': {
                'mean_annual_rate': f"{predictions['housing_rate'].mean() * 100:.2f}%",
                'total_units_2025': int(predictions['current_housing_units'].sum()),
                'total_units_2035': int(predictions[f'housing_year_{self.forecast_years}'].sum()),
                'net_new_units': int(
                    predictions[f'housing_year_{self.forecast_years}'].sum() - 
                    predictions['current_housing_units'].sum()
                )
            },
            
            # Model performance
            'model_performance': self.results.get('feature_importance', {}),
            
            # Focus neighborhood details
            'neighborhood_analysis': self._generate_neighborhood_analysis(predictions)
        }
        
        self.results['summary'] = summary
        
        # Save detailed predictions
        predictions.to_csv(self.output_dir / 'population_predictions_10year.csv', index=False)
        
        # Save summary report
        with open(self.output_dir / 'prediction_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _generate_neighborhood_analysis(self, predictions: pd.DataFrame) -> dict:
        """Generate detailed analysis for focus neighborhoods."""
        analysis = {}
        
        focus_data = predictions[predictions['neighborhood_name'].notna()]
        
        for _, row in focus_data.iterrows():
            neighborhood = row['neighborhood_name']
            analysis[neighborhood] = {
                'zip_code': row['zip_code'],
                'current_population': int(row['current_population']),
                'projected_population_10yr': int(row[f'population_year_{self.forecast_years}']),
                'population_change': int(row[f'population_year_{self.forecast_years}'] - row['current_population']),
                'population_change_pct': f"{(row[f'population_year_{self.forecast_years}'] / row['current_population'] - 1) * 100:.1f}%",
                'confidence_interval': f"[{int(row[f'population_year_{self.forecast_years}_lower'])}, {int(row[f'population_year_{self.forecast_years}_upper'])}]",
                'annual_growth_rate': f"{row['population_rate'] * 100:.2f}%",
                'housing_units_added': int(row[f'housing_year_{self.forecast_years}'] - row['current_housing_units']),
                'income_tier_change': row['income_rate'] * 10 * 2,  # Rough estimate
                'gentrification_risk': 'High' if row['income_rate'] > 0.03 and row['income_tier'] <= 2 else 'Low'
            }
        
        return analysis
    
    def run_analysis(self, data: pd.DataFrame) -> bool:
        """
        Run the complete population prediction analysis.
        """
        try:
            logger.info("Starting population prediction analysis")
            
            # Prepare features
            features = self.prepare_features(data)
            
            # Train models
            self.train_models(features, data)
            
            # Make predictions with confidence intervals
            predictions = self.predict_with_confidence(features)
            
            # Create visualizations
            self.create_visualizations(predictions, data)
            
            # Generate report
            report = self.generate_report(predictions)
            
            # Store results
            self.results['predictions'] = predictions
            self.results['report'] = report
            
            logger.info("Population prediction analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in population prediction analysis: {str(e)}")
            return False 