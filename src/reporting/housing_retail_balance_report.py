"""
Module for generating housing-retail balance analysis reports.
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
from src.models.housing_model import HousingModel
from src.models.retail_model import RetailModel
from src.visualization.visualizer import Visualizer
from src.utils.helpers import resolve_column_name
from src.config.column_alias_map import column_aliases

logger = logging.getLogger(__name__)

class HousingRetailBalanceReport:
    """Generates housing-retail balance analysis reports."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.housing_model = HousingModel()
        self.retail_model = RetailModel()
        self.visualizer = Visualizer()
        
        # Set up Jinja2 environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(settings.REPORT_TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Define output path
        self.output_path = settings.REPORTS_DIR / 'housing_retail_balance_report.md'
        
    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare data for the report."""
        try:
            # Load processed data
            permits = pd.read_csv(settings.PROCESSED_DATA_DIR / 'permits_processed.csv')
            retail_metrics = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_metrics.csv')
            census_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'census_processed.csv')

            # Resolve column names
            zip_col = resolve_column_name(permits, 'zip_code', column_aliases)
            res_permits_col = resolve_column_name(permits, 'residential_permits', column_aliases)
            total_permits_col = resolve_column_name(permits, 'total_permits', column_aliases)
            res_cost_col = resolve_column_name(permits, 'residential_construction_cost', column_aliases)
            total_cost_col = resolve_column_name(permits, 'total_construction_cost', column_aliases)
            pop_col = resolve_column_name(census_data, 'total_population', column_aliases)
            housing_units_col = resolve_column_name(census_data, 'housing_units', column_aliases)
            retail_deficit_col = resolve_column_name(retail_metrics, 'retail_deficit', column_aliases)

            # Validate required columns
            required_cols = {
                'permits': [zip_col, res_permits_col, total_permits_col, res_cost_col, total_cost_col],
                'census': [zip_col, pop_col, housing_units_col],
                'retail': [zip_col, retail_deficit_col]
            }

            for dataset_name, cols in required_cols.items():
                if missing := [col for col in cols if not col]:
                    logger.error(f"Missing required columns in {dataset_name}: {missing}")
                    return None

            # Calculate ratios
            permits['residential_permit_ratio'] = permits[res_permits_col] / permits[total_permits_col]
            permits['residential_cost_ratio'] = permits[res_cost_col] / permits[total_cost_col]

            # Merge datasets
            merged = permits.merge(
                retail_metrics[[zip_col, retail_deficit_col]], 
                on=zip_col, 
                how='left'
            )
            merged = merged.merge(
                census_data[[zip_col, pop_col, housing_units_col]], 
                on=zip_col, 
                how='left'
            )

            # Fill missing values
            merged = merged.fillna({
                retail_deficit_col: 0,
                'residential_permit_ratio': 0,
                'residential_cost_ratio': 0
            })

            # Calculate balance metrics
            merged['housing_retail_ratio'] = merged['residential_permit_ratio'] / (1 - merged[retail_deficit_col])
            merged['balance_score'] = 1 - abs(merged['housing_retail_ratio'] - 1)

            # Categorize areas
            merged['balance_category'] = pd.cut(
                merged['balance_score'],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=['Severe Imbalance', 'Moderate Imbalance', 'Slight Imbalance', 'Balanced']
            )

            logger.info("Successfully prepared data for housing retail balance report")
            return {
                'merged': merged,
                'permits': permits,
                'retail': retail_metrics,
                'census': census_data
            }

        except Exception as e:
            logger.error(f"Error preparing data for housing retail balance report: {str(e)}")
            return None
            
    def analyze_balance(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Analyze the balance between housing and retail development."""
        try:
            merged_data = data['merged']
            
            # Resolve column names
            zip_col = resolve_column_name(merged_data, 'zip_code', column_aliases)
            pop_col = resolve_column_name(merged_data, 'total_population', column_aliases)
            housing_col = resolve_column_name(merged_data, 'housing_units', column_aliases)
            retail_space_col = resolve_column_name(merged_data, 'retail_space', column_aliases)
            
            if not all([zip_col, pop_col, housing_col, retail_space_col]):
                logger.error("Required columns not found for balance analysis")
                return {}
            
            # Calculate metrics by ZIP code
            zip_metrics = merged_data.groupby(zip_col).agg({
                pop_col: 'sum',
                housing_col: 'sum',
                retail_space_col: 'sum',
                'balance_score': 'mean',
                'balance_category': lambda x: 'Unknown' if x.empty else x.mode()[0]
            }).reset_index()
            
            # Calculate housing density
            zip_metrics['housing_density'] = zip_metrics[housing_col] / zip_metrics[pop_col]
            zip_metrics['retail_density'] = zip_metrics[retail_space_col] / zip_metrics[pop_col]
            
            # Identify imbalanced areas
            severe_imbalance = zip_metrics[
                zip_metrics['balance_score'] < 0.3
            ][zip_col].tolist()
            
            balanced = zip_metrics[
                zip_metrics['balance_score'] > 0.8
            ][zip_col].tolist()
            
            # Calculate summary statistics
            summary_stats = {
                'avg_balance_score': zip_metrics['balance_score'].mean(),
                'median_balance_score': zip_metrics['balance_score'].median(),
                'severe_imbalance_count': len(severe_imbalance),
                'balanced_count': len(balanced),
                'avg_housing_density': zip_metrics['housing_density'].mean(),
                'avg_retail_density': zip_metrics['retail_density'].mean()
            }
            
            return {
                'zip_metrics': zip_metrics.to_dict('records'),
                'summary': summary_stats,
                'severe_imbalance_zips': severe_imbalance,
                'balanced_zips': balanced
            }
            
        except Exception as e:
            logger.error(f"Error analyzing housing-retail balance: {str(e)}")
            return {}
            
    def generate_report(self) -> bool:
        """Generate the housing-retail balance report."""
        try:
            # Load and prepare data
            data = self.load_and_prepare_data()
            if not data:
                logger.error("Failed to load required data")
                return False
            
            # Analyze housing-retail balance
            analysis = self.analyze_balance(data)
            if not analysis:
                logger.error("Failed to analyze housing-retail balance")
                return False
            
            # Create visualizations
            self.visualizer.create_balance_analysis_charts(data['merged'])
            
            # Load report template
            template = self.template_env.get_template('housing_retail_balance_report.md')
            
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
            
            logger.info(f"Housing-retail balance report generated at {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating housing-retail balance report: {str(e)}")
            return False 