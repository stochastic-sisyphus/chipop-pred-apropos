"""
Module for generating retail deficit analysis reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import jinja2
from datetime import datetime
from jinja2 import Template

from src.config import settings
from src.models.retail_model import RetailModel
from src.visualization.visualizer import Visualizer
from src.utils.helpers import resolve_column_name
from src.config.column_alias_map import column_aliases

logger = logging.getLogger(__name__)

class RetailDeficitReport:
    """Generates retail deficit analysis reports."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.retail_model = RetailModel()
        self.visualizer = Visualizer()
        
        # Set up Jinja2 environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(settings.REPORT_TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Define output path
        self.output_path = settings.REPORTS_DIR / 'retail_deficit_analysis.md'
        
    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare data for the report."""
        try:
            # Load data files
            retail_metrics = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_metrics.csv')
            census_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'census_processed.csv')
            economic_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'economic_processed.csv')
            retail_leakage = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_leakage.csv')
            retail_predictions = pd.read_csv(settings.PREDICTIONS_DIR / 'retail_demand_predictions.csv')
            
            # Resolve column names
            zip_col = resolve_column_name(retail_metrics, 'zip_code', column_aliases)
            year_col = resolve_column_name(retail_metrics, 'year', column_aliases)
            
            if not all([zip_col, year_col]):
                logger.error("Required columns not found")
                return None
            
            # Merge datasets using resolved column names
            merged = retail_metrics.merge(
                census_data,
                on=[zip_col, year_col] if year_col in census_data.columns else [zip_col],
                how='left'
            )
            
            merged = merged.merge(
                economic_data,
                on=[zip_col, year_col] if year_col in economic_data.columns else [zip_col],
                how='left'
            )
            
            # Add retail leakage data if available
            if retail_leakage is not None:
                merged = merged.merge(
                    retail_leakage,
                    on=zip_col,
                    how='left'
                )
            
            # Add predictions if available
            if retail_predictions is not None:
                merged = merged.merge(
                    retail_predictions,
                    on=zip_col,
                    how='left'
                )
            
            return {
                'retail': retail_metrics,
                'demographic': census_data,
                'economic': economic_data,
                'leakage': retail_leakage,
                'predictions': retail_predictions,
                'merged': merged
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
            
    def analyze_retail_deficit(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze retail deficit patterns."""
        try:
            merged_data = data['merged']
            
            # Resolve column names
            zip_col = resolve_column_name(merged_data, 'zip_code', column_aliases)
            pop_col = resolve_column_name(merged_data, 'total_population', column_aliases)
            income_col = resolve_column_name(merged_data, 'median_household_income', column_aliases)
            retail_space_col = resolve_column_name(merged_data, 'retail_space', column_aliases)
            
            if not all([zip_col, pop_col, income_col, retail_space_col]):
                logger.error("Required columns not found for retail deficit analysis")
                return {}
            
            # Calculate per capita metrics
            merged_data['retail_space_per_capita'] = merged_data[retail_space_col] / merged_data[pop_col]
            merged_data['retail_spending_potential'] = merged_data[income_col] * settings.RETAIL_SPENDING_FACTOR
            
            # Group by ZIP code for analysis
            zip_metrics = merged_data.groupby(zip_col).agg({
                'retail_space_per_capita': 'mean',
                'retail_spending_potential': 'sum',
                pop_col: 'sum',
                retail_space_col: 'sum'
            }).reset_index()
            
            # Calculate deficit metrics
            zip_metrics['retail_gap'] = (
                zip_metrics['retail_spending_potential'] - 
                zip_metrics[retail_space_col] * settings.RETAIL_SPENDING_FACTOR
            )
            
            # Identify high deficit areas
            high_deficit = zip_metrics[
                zip_metrics['retail_gap'] > zip_metrics['retail_gap'].quantile(0.75)
            ][zip_col].tolist()
            
            # Calculate summary statistics
            summary_stats = {
                'total_deficit': zip_metrics['retail_gap'].sum(),
                'avg_deficit': zip_metrics['retail_gap'].mean(),
                'median_deficit': zip_metrics['retail_gap'].median(),
                'high_deficit_zips': high_deficit,
                'total_retail_space': zip_metrics[retail_space_col].sum(),
                'avg_space_per_capita': zip_metrics['retail_space_per_capita'].mean()
            }
            
            return {
                'zip_metrics': zip_metrics.to_dict('records'),
                'summary': summary_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing retail deficit: {str(e)}")
            return {}
            
    def generate_report(self) -> bool:
        """Generate the retail deficit analysis report."""
        try:
            # Load and prepare data
            data = self.load_and_prepare_data()
            if not data:
                logger.error("Failed to load required data")
                return False
            
            # Analyze retail deficit
            analysis = self.analyze_retail_deficit(data)
            if not analysis:
                logger.error("Failed to analyze retail deficit")
                return False
            
            # Load report template
            template = self.template_env.get_template('retail_deficit_analysis.md')
            
            # Generate report content
            report_content = template.render(
                date=datetime.now().strftime('%Y-%m-%d'),
                analysis=analysis,
                data=data
            )
            
            # Save report
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Retail deficit report generated at {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating retail deficit report: {str(e)}")
            return False 