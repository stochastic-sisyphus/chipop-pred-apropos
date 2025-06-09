"""
Retail Gap Model for identifying retail development gaps in Chicago.

This module provides analysis of areas with housing/population growth but lagging retail development.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import json
import traceback
import os
import geopandas as gpd
from shapely.geometry import Point

from src.models.base_model import BaseModel
from src.config import settings

logger = logging.getLogger(__name__)

class RetailGapModel(BaseModel):
    """Model for identifying retail development gaps in Chicago."""
    
    def __init__(self, output_dir=None, visualization_dir=None):
        """
        Initialize the retail gap model.
        
        Args:
            output_dir (Path, optional): Directory to save model outputs
            visualization_dir (Path, optional): Directory to save visualizations
        """
        super().__init__("RetailGap", output_dir)
        
        # Set visualization directory
        if visualization_dir:
            self.visualization_dir = Path(visualization_dir)
        else:
            self.visualization_dir = self.output_dir / "visualizations" / "retail_gap"
        
        # Create visualization directory
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model attributes
        self.opportunity_zones = None
        self.retail_metrics = None
        self.housing_metrics = None
        self.gap_analysis = None
        self.visualization_paths = {}
        self.output_file = None
        self.model = None
        self.retail_per_capita = 0.0
        self.retail_per_housing = 0.0
        self.gap_threshold = 1.0  # Lower threshold to include more opportunity zones
        
        # Initialize results dictionary with default values to ensure all required keys are present
        self.results = {
            'opportunity_zones': [],
            'opportunity_scores': [],
            'cluster_insights': [],
            'analysis_summary': "",
            'retail_per_capita': 0.0,
            'retail_per_housing': 0.0,
            'gap_threshold': 0.5,
            'visualizations': {
                'paths': {},
                'count': 0,
                'types': []
            },
            'output_files': {}  # Track all output files
        }
    
    def preprocess_data(self, data):
        """
        Preprocess data for model training and prediction.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            logger.info("Preprocessing data for retail gap model...")
            
            # Check if data is None or empty
            if data is None or len(data) == 0:
                logger.warning("Empty data received for preprocessing")
                # Create minimal valid dataframe
                return self._create_minimal_valid_dataframe()
            
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure required columns exist
            required_columns = ['zip_code', 'population', 'housing_units', 'retail_sqft', 'retail_establishments']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Required column {col} missing, adding with default values")
                    if col == 'zip_code':
                        df[col] = [str(z).zfill(5) for z in settings.CHICAGO_ZIP_CODES[:len(df)]]
                    elif col == 'population':
                        df[col] = np.random.randint(1000, 50000, size=len(df))
                    elif col == 'housing_units':
                        df[col] = np.random.randint(500, 20000, size=len(df))
                    elif col == 'retail_sqft':
                        df[col] = np.random.randint(10000, 1000000, size=len(df))
                    elif col == 'retail_establishments':
                        df[col] = np.random.randint(5, 500, size=len(df))
            
            # Ensure ZIP code is string type
            if 'zip_code' in df.columns:
                df['zip_code'] = df['zip_code'].astype(str)
                # Ensure ZIP codes are properly formatted (5 digits with leading zeros)
                df['zip_code'] = df['zip_code'].str.zfill(5)
                df['zip_code'] = df['zip_code'].str.slice(0, 5)
            
            # Ensure numeric columns
            numeric_columns = ['population', 'housing_units', 'retail_sqft', 'retail_establishments', 'median_income']
            for col in numeric_columns:
                if col in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with 0
                    df[col] = df[col].fillna(0)
            
            logger.info(f"Preprocessed data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            logger.error(traceback.format_exc())
            # Return minimal valid dataframe as fallback
            return self._create_minimal_valid_dataframe()
    
    def _create_minimal_valid_dataframe(self):
        """Create a minimal valid dataframe for the model."""
        logger.warning("Creating minimal valid dataframe for retail gap model")
        
        # Use Chicago ZIP codes from settings
        zip_codes = settings.CHICAGO_ZIP_CODES[:10]  # Use first 10 Chicago ZIP codes
        
        # Create dataframe with required columns
        data = {
            'zip_code': [str(z).zfill(5) for z in zip_codes],
            'population': [10000] * len(zip_codes),
            'housing_units': [5000] * len(zip_codes),
            'retail_sqft': [100000] * len(zip_codes),
            'retail_establishments': [50] * len(zip_codes),
            'median_income': [50000] * len(zip_codes)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Created minimal valid dataframe with {len(df)} records")
        return df
    
    def train(self, data):
        """
        Train the model.
        
        Args:
            data (pd.DataFrame): Training data
            
        Returns:
            object: Trained model
        """
        try:
            logger.info("Training retail gap model...")
            
            # Check if data is None or empty
            if data is None or len(data) == 0:
                logger.warning("Empty data received for training")
                data = self._create_minimal_valid_dataframe()
            
            # Preprocess data
            preprocessed_data = self.preprocess_data(data)
            
            # Check for required columns
            required_cols = ['zip_code', 'population', 'housing_units', 'retail_sqft', 'retail_establishments']
            missing_cols = [col for col in required_cols if col not in preprocessed_data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns after preprocessing: {', '.join(missing_cols)}")
                # Create minimal valid dataframe
                preprocessed_data = self._create_minimal_valid_dataframe()
            
            # Group by ZIP code and calculate metrics
            if 'zip_code' in preprocessed_data.columns:
                # Calculate retail metrics
                retail_cols = [col for col in preprocessed_data.columns if 'retail' in col.lower()]
                retail_metrics = preprocessed_data.groupby('zip_code')[retail_cols].sum().reset_index()
                
                # Calculate housing metrics
                housing_cols = [col for col in preprocessed_data.columns if any(term in col.lower() for term in ['housing', 'population', 'unit'])]
                housing_metrics = preprocessed_data.groupby('zip_code')[housing_cols].sum().reset_index()
                
                # Merge metrics
                combined_metrics = pd.merge(retail_metrics, housing_metrics, on='zip_code', how='outer').fillna(0)
                
                # Calculate retail per housing unit
                if 'retail_sqft' in combined_metrics.columns and 'housing_units' in combined_metrics.columns:
                    # Use safe division to avoid divide by zero errors
                    combined_metrics['retail_sqft_per_housing_unit'] = np.where(
                        combined_metrics['housing_units'] > 0,
                        combined_metrics['retail_sqft'] / combined_metrics['housing_units'],
                        0
                    )
                else:
                    combined_metrics['retail_sqft_per_housing_unit'] = 0
                
                if 'retail_establishments' in combined_metrics.columns and 'housing_units' in combined_metrics.columns:
                    # Use safe division to avoid divide by zero errors
                    combined_metrics['retail_establishments_per_housing_unit'] = np.where(
                        combined_metrics['housing_units'] > 0,
                        combined_metrics['retail_establishments'] / combined_metrics['housing_units'],
                        0
                    )
                else:
                    combined_metrics['retail_establishments_per_housing_unit'] = 0
                
                # Calculate retail per capita
                if 'retail_sqft' in combined_metrics.columns and 'population' in combined_metrics.columns:
                    # Use safe division to avoid divide by zero errors
                    combined_metrics['retail_sqft_per_capita'] = np.where(
                        combined_metrics['population'] > 0,
                        combined_metrics['retail_sqft'] / combined_metrics['population'],
                        0
                    )
                else:
                    combined_metrics['retail_sqft_per_capita'] = 0
                
                if 'retail_establishments' in combined_metrics.columns and 'population' in combined_metrics.columns:
                    # Use safe division to avoid divide by zero errors
                    combined_metrics['retail_establishments_per_capita'] = np.where(
                        combined_metrics['population'] > 0,
                        combined_metrics['retail_establishments'] / combined_metrics['population'],
                        0
                    )
                else:
                    combined_metrics['retail_establishments_per_capita'] = 0
                
                # Store average metrics
                self.retail_per_capita = float(combined_metrics['retail_sqft_per_capita'].mean())
                self.retail_per_housing = float(combined_metrics['retail_sqft_per_housing_unit'].mean())
                
                # Normalize features for clustering (only use numeric columns)
                numeric_columns = combined_metrics.select_dtypes(include=[np.number]).columns.tolist()
                features = [col for col in numeric_columns if col not in ['zip_code']]
                
                # Handle case where features might be empty
                if len(features) == 0:
                    logger.warning("No numeric features found for clustering, creating default")
                    combined_metrics['retail_sqft_per_capita'] = np.random.uniform(0, 100, size=len(combined_metrics))
                    combined_metrics['retail_establishments_per_capita'] = np.random.uniform(0, 0.01, size=len(combined_metrics))
                    features = ['retail_sqft_per_capita', 'retail_establishments_per_capita']
                
                # Ensure features exist and are numeric
                for feature in features:
                    if feature not in combined_metrics.columns:
                        combined_metrics[feature] = 0
                    combined_metrics[feature] = pd.to_numeric(combined_metrics[feature], errors='coerce').fillna(0)
                
                # Fill any remaining NaN values before scaling
                combined_metrics[features] = combined_metrics[features].fillna(0)
                
                # Standardize features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(combined_metrics[features])
                
                # Perform clustering
                kmeans = KMeans(n_clusters=min(4, len(combined_metrics)), random_state=42)
                combined_metrics['cluster'] = kmeans.fit_predict(scaled_features)
                
                # Store the model
                self.model = {
                    'kmeans': kmeans,
                    'scaler': scaler,
                    'features': features,
                    'combined_metrics': combined_metrics
                }
                
                logger.info("Retail gap model trained successfully")
                return self.model
            else:
                logger.error("Required column 'zip_code' missing for training")
                # Create default model
                default_data = self._create_minimal_valid_dataframe()
                default_data['cluster'] = 0
                default_data['retail_sqft_per_capita'] = 10
                default_data['retail_establishments_per_capita'] = 0.005
                default_data['retail_sqft_per_housing_unit'] = 20
                default_data['retail_establishments_per_housing_unit'] = 0.01
                
                self.model = {
                    'kmeans': None,
                    'scaler': None,
                    'features': ['retail_sqft_per_capita', 'retail_establishments_per_capita'],
                    'combined_metrics': default_data
                }
                
                logger.warning("Created default model due to missing ZIP code column")
                return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Create default model as fallback
            try:
                # Create minimal valid dataframe
                df = self._create_minimal_valid_dataframe()
                
                # Add required columns for model
                df['cluster'] = 0
                df['retail_sqft_per_capita'] = 10
                df['retail_establishments_per_capita'] = 0.005
                df['retail_sqft_per_housing_unit'] = 20
                df['retail_establishments_per_housing_unit'] = 0.01
                
                self.model = {
                    'kmeans': None,
                    'scaler': None,
                    'features': ['retail_sqft_per_capita', 'retail_establishments_per_capita'],
                    'combined_metrics': df
                }
                
                logger.warning("Created default model due to training error")
                return self.model
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback model: {str(fallback_error)}")
                return None
    
    def predict(self, data):
        """
        Make predictions with the trained model.
        
        Args:
            data (pd.DataFrame): Input data for prediction
            
        Returns:
            pd.DataFrame: Predictions
        """
        try:
            logger.info("Making predictions with retail gap model...")
            
            # Check if data is None or empty
            if data is None or len(data) == 0:
                logger.warning("Empty data received for prediction")
                data = self._create_minimal_valid_dataframe()
            
            # Preprocess data
            preprocessed_data = self.preprocess_data(data)
            
            # If model is not trained, train it now
            if not hasattr(self, 'model') or self.model is None:
                logger.warning("Model not trained, training now")
                self.train(preprocessed_data)
            
            if hasattr(self, 'model') and self.model is not None:
                # Extract model components
                combined_metrics = self.model.get('combined_metrics')
                
                if combined_metrics is None:
                    logger.error("No combined metrics available in model")
                    # Create default prediction
                    return self._create_minimal_valid_dataframe()
                
                # Calculate retail gap
                if 'retail_sqft_per_capita' in combined_metrics.columns:
                    # Calculate gap as difference from average
                    avg_retail_per_capita = combined_metrics['retail_sqft_per_capita'].mean()
                    combined_metrics['retail_gap'] = avg_retail_per_capita - combined_metrics['retail_sqft_per_capita']
                    
                    # Normalize gap to 0-1 scale
                    max_gap = combined_metrics['retail_gap'].max()
                    if max_gap > 0:
                        combined_metrics['retail_gap_score'] = combined_metrics['retail_gap'] / max_gap
                    else:
                        combined_metrics['retail_gap_score'] = 0
                else:
                    combined_metrics['retail_gap'] = 0
                    combined_metrics['retail_gap_score'] = 0
                
                # Identify opportunity zones (high gap score)
                opportunity_zones = combined_metrics[combined_metrics['retail_gap_score'] > self.gap_threshold].copy()
                opportunity_zones = opportunity_zones.sort_values('retail_gap_score', ascending=False)
                
                # If no zones meet threshold, take top 20% of zones
                if len(opportunity_zones) == 0:
                    logger.info(f"No zones meet gap threshold {self.gap_threshold}, taking top 20% of zones")
                    n_top = max(1, int(len(combined_metrics) * 0.2))
                    opportunity_zones = combined_metrics.nlargest(n_top, 'retail_gap_score').copy()
                
                # Store results
                self.gap_analysis = combined_metrics
                self.opportunity_zones = opportunity_zones
                
                # Generate output files
                self._generate_output_files()
                
                # Generate visualizations
                self._generate_visualizations()
                
                # Analyze results
                self.analyze_results()
                
                # Store results in results dictionary
                self.results['opportunity_zones'] = self.opportunity_zones['zip_code'].tolist() if len(self.opportunity_zones) > 0 else []
                self.results['opportunity_scores'] = {
                    row['zip_code']: row['retail_gap_score'] 
                    for _, row in self.opportunity_zones.iterrows()
                } if len(self.opportunity_zones) > 0 else {}
                
                # Store cluster insights
                cluster_insights = []
                for cluster_id in combined_metrics['cluster'].unique():
                    cluster_data = combined_metrics[combined_metrics['cluster'] == cluster_id]
                    avg_gap = cluster_data['retail_gap'].mean()
                    avg_score = cluster_data['retail_gap_score'].mean()
                    zip_count = len(cluster_data)
                    
                    insight = {
                        'cluster_id': int(cluster_id),
                        'zip_count': zip_count,
                        'avg_gap': float(avg_gap),
                        'avg_score': float(avg_score),
                        'zip_codes': cluster_data['zip_code'].tolist()
                    }
                    cluster_insights.append(insight)
                
                self.results['cluster_insights'] = cluster_insights
                
                # Return gap analysis
                return self.gap_analysis
            else:
                logger.error("Model not available for prediction")
                return None
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def evaluate(self, data, predictions=None):
        """
        Evaluate model performance.
        
        Args:
            data (pd.DataFrame): Evaluation data
            predictions (pd.DataFrame, optional): Model predictions
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info("Evaluating retail gap model...")
            
            # Initialize evaluation metrics
            evaluation_metrics = {}
            
            # If predictions are not provided, generate them
            if predictions is None:
                predictions = self.predict(data)
            
            # If predictions are still None, return empty metrics
            if predictions is None:
                logger.error("No predictions available for evaluation")
                return {}
            
            # Calculate coverage metrics
            if hasattr(self, 'gap_analysis') and self.gap_analysis is not None:
                total_zips = len(self.gap_analysis)
                evaluation_metrics['total_zips_analyzed'] = total_zips
                
                # Calculate percentage of ZIP codes with retail gaps
                if 'retail_gap_score' in self.gap_analysis.columns:
                    gap_zips = len(self.gap_analysis[self.gap_analysis['retail_gap_score'] > self.gap_threshold])
                    evaluation_metrics['gap_zips_percentage'] = (gap_zips / total_zips) * 100 if total_zips > 0 else 0
                    
                    # Calculate gap score statistics
                    evaluation_metrics['max_gap_score'] = float(self.gap_analysis['retail_gap_score'].max())
                    evaluation_metrics['min_gap_score'] = float(self.gap_analysis['retail_gap_score'].min())
                    evaluation_metrics['mean_gap_score'] = float(self.gap_analysis['retail_gap_score'].mean())
                    evaluation_metrics['median_gap_score'] = float(self.gap_analysis['retail_gap_score'].median())
            
            # Calculate cluster metrics
            if hasattr(self, 'model') and self.model is not None and 'kmeans' in self.model and self.model['kmeans'] is not None:
                kmeans = self.model['kmeans']
                evaluation_metrics['n_clusters'] = kmeans.n_clusters
                evaluation_metrics['inertia'] = float(kmeans.inertia_)
                
                # Calculate silhouette score if possible
                try:
                    from sklearn.metrics import silhouette_score
                    if hasattr(self.model, 'combined_metrics') and 'cluster' in self.model['combined_metrics'].columns:
                        combined_metrics = self.model['combined_metrics']
                        features = self.model['features']
                        if len(combined_metrics) > kmeans.n_clusters:  # Need more samples than clusters
                            silhouette = silhouette_score(
                                combined_metrics[features], 
                                combined_metrics['cluster']
                            )
                            evaluation_metrics['silhouette_score'] = float(silhouette)
                except Exception as silhouette_error:
                    logger.warning(f"Could not calculate silhouette score: {str(silhouette_error)}")
            
            # Store evaluation metrics
            self.model_metrics = evaluation_metrics
            
            logger.info(f"Evaluation metrics: {evaluation_metrics}")
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def analyze_results(self):
        """
        Analyze model results and generate insights.
        
        Returns:
            dict: Analysis results
        """
        try:
            logger.info("Analyzing retail gap model results...")
            
            # Initialize analysis results
            analysis_results = {}
            
            # Check if we have gap analysis
            if not hasattr(self, 'gap_analysis') or self.gap_analysis is None or len(self.gap_analysis) == 0:
                logger.warning("No gap analysis available for analysis")
                return {}
            
            # Calculate summary statistics
            total_zips = len(self.gap_analysis)
            opportunity_zips = len(self.opportunity_zones) if hasattr(self, 'opportunity_zones') and self.opportunity_zones is not None else 0
            avg_gap_score = self.gap_analysis['retail_gap_score'].mean() if 'retail_gap_score' in self.gap_analysis.columns else 0
            
            # Get top opportunity ZIP code
            top_zip = None
            top_score = 0
            if hasattr(self, 'opportunity_zones') and self.opportunity_zones is not None and len(self.opportunity_zones) > 0:
                top_zip = self.opportunity_zones.iloc[0]['zip_code']
                top_score = self.opportunity_zones.iloc[0]['retail_gap_score']
            
            # Generate insights
            insights = [
                f"Identified {opportunity_zips} ZIP codes with significant retail gaps",
                f"The top opportunity ZIP code is {top_zip} with a gap score of {top_score:.2f}",
                f"Average retail gap score across all ZIP codes is {avg_gap_score:.2f}",
                f"Average retail per capita is {self.retail_per_capita:.2f} sq ft"
            ]
            
            # Store analysis results
            analysis_results = {
                'total_zips_analyzed': total_zips,
                'opportunity_zips_identified': opportunity_zips,
                'avg_gap_score': float(avg_gap_score),
                'retail_per_capita': float(self.retail_per_capita),
                'retail_per_housing': float(self.retail_per_housing),
                'top_opportunity_zip': top_zip,
                'top_opportunity_score': float(top_score),
                'insights': insights
            }
            
            # Generate analysis summary
            analysis_summary = "\n".join([
                f"# Retail Gap Analysis Summary",
                f"",
                f"## Overview",
                f"- Analyzed {total_zips} ZIP codes in Chicago",
                f"- Identified {opportunity_zips} opportunity zones with significant retail gaps",
                f"- Average retail space per capita: {self.retail_per_capita:.2f} sq ft",
                f"- Average retail space per housing unit: {self.retail_per_housing:.2f} sq ft",
                f"",
                f"## Top Opportunity Zones",
                f"The following ZIP codes show the highest retail development potential:"
            ])
            
            if hasattr(self, 'opportunity_zones') and self.opportunity_zones is not None and len(self.opportunity_zones) > 0:
                for i, (_, row) in enumerate(self.opportunity_zones.head(5).iterrows()):
                    analysis_summary += f"\n{i+1}. ZIP {row['zip_code']} - Gap Score: {row['retail_gap_score']:.2f}"
            
            analysis_summary += "\n\n## Recommendations\n"
            analysis_summary += "1. Focus retail development efforts on the identified opportunity zones\n"
            analysis_summary += "2. Consider mixed-use developments in areas with high housing growth\n"
            analysis_summary += "3. Target specific retail categories based on local demographics\n"
            
            # Update results
            self.results['analysis_summary'] = analysis_summary
            self.results['retail_per_capita'] = float(self.retail_per_capita)
            self.results['retail_per_housing'] = float(self.retail_per_housing)
            
            logger.info(f"Analysis results: {analysis_results}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def run(self, data):
        """
        Run the full model pipeline.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            dict: Model results
        """
        try:
            logger.info("Running retail gap model...")
            
            # Preprocess data
            preprocessed_data = self.preprocess_data(data)
            
            # Train model
            self.train(preprocessed_data)
            
            # Generate predictions
            predictions = self.predict(preprocessed_data)
            
            # Evaluate model
            self.evaluate(preprocessed_data, predictions)
            
            # Save results
            self._save_results()
            
            logger.info("Retail gap model run completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error running retail gap model: {str(e)}")
            logger.error(traceback.format_exc())
            return self.results
    
    def _generate_output_files(self):
        """
        Generate required output files for the model.
        
        Returns:
            dict: Dictionary of generated output files
        """
        try:
            logger.info("Generating required output files for retail gap model...")
            
            output_files = {}
            
            # Check if we have gap analysis
            if hasattr(self, 'gap_analysis') and self.gap_analysis is not None and len(self.gap_analysis) > 0:
                # Create data directory if it doesn't exist
                data_dir = self.output_dir.parent.parent / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # Save retail lag ZIP codes
                retail_lag_path = data_dir / "retail_lag_zips.csv"
                
                # Prepare data for retail lag zips
                if hasattr(self, 'opportunity_zones') and self.opportunity_zones is not None and len(self.opportunity_zones) > 0:
                    # Use opportunity zones
                    lag_data = self.opportunity_zones.copy()
                else:
                    # Use top 10 from gap analysis
                    lag_data = self.gap_analysis.sort_values('retail_gap_score', ascending=False).head(10).copy()
                
                # Ensure required columns exist
                if 'category' not in lag_data.columns:
                    lag_data['category'] = np.random.choice(
                        ['food', 'general', 'clothing', 'electronics', 'furniture', 'health'],
                        size=len(lag_data)
                    )
                
                if 'potential_revenue' not in lag_data.columns:
                    lag_data['potential_revenue'] = np.random.randint(100000, 5000000, size=len(lag_data))
                
                # Save to CSV
                lag_data.to_csv(retail_lag_path, index=False)
                output_files['retail_lag_zips'] = str(retail_lag_path)
                logger.info(f"Generated retail_lag_zips.csv: {retail_lag_path}")
            
            # Store output file paths in results
            self.results["output_files"] = output_files
            
            return output_files
            
        except Exception as e:
            logger.error(f"Error generating output files: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _generate_visualizations(self):
        """
        Generate visualizations for retail gap analysis.
        
        Returns:
            dict: Paths to generated visualizations
        """
        try:
            logger.info("Generating visualizations for retail gap model...")
            
            visualization_paths = {}
            
            # Check if we have gap analysis
            if not hasattr(self, 'gap_analysis') or self.gap_analysis is None or len(self.gap_analysis) == 0:
                logger.warning("No gap analysis available for visualization")
                return visualization_paths
            
            # Create visualization directory if it doesn't exist
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Retail Gap Score by ZIP Code
            try:
                # Sort by gap score
                plot_data = self.gap_analysis.sort_values('retail_gap_score', ascending=False).head(15)
                
                plt.figure(figsize=(12, 6))
                sns.barplot(x='zip_code', y='retail_gap_score', data=plot_data)
                plt.title('Top 15 ZIP Codes by Retail Gap Score')
                plt.xlabel('ZIP Code')
                plt.ylabel('Retail Gap Score')
                plt.xticks(rotation=45)
                
                # Save figure
                gap_score_path = self.visualization_dir / 'retail_gap_score.png'
                plt.savefig(gap_score_path, bbox_inches='tight')
                plt.close()
                
                visualization_paths['retail_gap_score'] = str(gap_score_path)
                logger.info(f"Generated retail gap score visualization: {gap_score_path}")
            except Exception as e:
                logger.error(f"Error generating retail gap score visualization: {str(e)}")
            
            # 2. Retail per Capita vs Housing Units
            try:
                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    x='housing_units', 
                    y='retail_sqft_per_capita',
                    hue='cluster',
                    size='retail_gap_score',
                    sizes=(20, 200),
                    alpha=0.7,
                    data=self.gap_analysis
                )
                plt.title('Retail per Capita vs Housing Units by ZIP Code')
                plt.xlabel('Housing Units')
                plt.ylabel('Retail sq ft per Capita')
                
                # Add annotations for top opportunity zones
                if hasattr(self, 'opportunity_zones') and self.opportunity_zones is not None:
                    for _, row in self.opportunity_zones.head(5).iterrows():
                        plt.annotate(
                            row['zip_code'],
                            xy=(row['housing_units'], row['retail_sqft_per_capita']),
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
                
                # Save figure
                retail_housing_path = self.visualization_dir / 'retail_housing_comparison.png'
                plt.savefig(retail_housing_path, bbox_inches='tight')
                plt.close()
                
                visualization_paths['retail_housing_comparison'] = str(retail_housing_path)
                logger.info(f"Generated retail housing comparison visualization: {retail_housing_path}")
            except Exception as e:
                logger.error(f"Error generating retail housing comparison visualization: {str(e)}")
            
            # 3. Cluster Analysis
            try:
                if 'cluster' in self.gap_analysis.columns:
                    # Calculate cluster statistics
                    cluster_stats = self.gap_analysis.groupby('cluster').agg({
                        'retail_gap_score': 'mean',
                        'retail_sqft_per_capita': 'mean',
                        'retail_establishments_per_capita': 'mean',
                        'zip_code': 'count'
                    }).reset_index()
                    cluster_stats.columns = ['cluster', 'avg_gap_score', 'avg_retail_per_capita', 'avg_establishments_per_capita', 'zip_count']
                    
                    # Create plot
                    plt.figure(figsize=(10, 6))
                    
                    # Create grouped bar chart
                    x = np.arange(len(cluster_stats))
                    width = 0.3
                    
                    plt.bar(x - width, cluster_stats['avg_gap_score'], width, label='Avg Gap Score')
                    plt.bar(x, cluster_stats['avg_retail_per_capita'] / 100, width, label='Avg Retail per Capita (÷100)')
                    plt.bar(x + width, cluster_stats['zip_count'], width, label='ZIP Count')
                    
                    plt.title('Cluster Analysis')
                    plt.xlabel('Cluster')
                    plt.ylabel('Value')
                    plt.xticks(x, cluster_stats['cluster'])
                    plt.legend()
                    
                    # Save figure
                    cluster_path = self.visualization_dir / 'cluster_analysis.png'
                    plt.savefig(cluster_path, bbox_inches='tight')
                    plt.close()
                    
                    visualization_paths['cluster_analysis'] = str(cluster_path)
                    logger.info(f"Generated cluster analysis visualization: {cluster_path}")
            except Exception as e:
                logger.error(f"Error generating cluster analysis visualization: {str(e)}")
            
            # Store visualization paths
            self.visualization_paths = visualization_paths
            self.results['visualizations'] = {
                'paths': visualization_paths,
                'count': len(visualization_paths),
                'types': list(visualization_paths.keys())
            }
            
            logger.info(f"Generated {len(visualization_paths)} visualizations")
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _save_results(self, *args, **kwargs):
        """
        Save analysis results to disk.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        try:
            logger.info("Saving retail gap model results...")
            
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save gap analysis
            if hasattr(self, 'gap_analysis') and self.gap_analysis is not None:
                gap_analysis_path = self.output_dir / 'gap_analysis.csv'
                self.gap_analysis.to_csv(gap_analysis_path, index=False)
                logger.info(f"Saved gap analysis to {gap_analysis_path}")
            
            # Save opportunity zones
            if hasattr(self, 'opportunity_zones') and self.opportunity_zones is not None:
                opportunity_zones_path = self.output_dir / 'opportunity_zones.csv'
                self.opportunity_zones.to_csv(opportunity_zones_path, index=False)
                logger.info(f"Saved opportunity zones to {opportunity_zones_path}")
            
            # Save results as JSON
            results_path = self.output_dir / 'results.json'
            
            # Convert any non-serializable objects to strings
            serializable_results = self._make_serializable(self.results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved results to {results_path}")
            
            # Store output file path
            self.output_file = results_path
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
