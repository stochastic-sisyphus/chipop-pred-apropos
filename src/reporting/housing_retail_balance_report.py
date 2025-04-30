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
            
            # Create housing metrics from permits data
            housing_metrics = permits[['year', 'zip_code', 'residential_permits', 
                                    'residential_construction_cost', 'residential_permit_ratio',
                                    'residential_cost_ratio']].copy()
            
            # Merge with census data
            housing_metrics = housing_metrics.merge(
                census_data[['year', 'zip_code', 'total_population', 'median_household_income']],
                on=['year', 'zip_code'],
                how='left'
            )
            
            # Calculate housing metrics
            housing_metrics['housing_units'] = housing_metrics['residential_permits'] * 1.5  # Assume 1.5 units per permit
            housing_metrics['housing_density'] = housing_metrics['housing_units'] / housing_metrics['total_population']
            housing_metrics['housing_value_per_unit'] = housing_metrics['residential_construction_cost'] / housing_metrics['housing_units']
            
            # Calculate year-over-year changes
            for col in ['housing_units', 'housing_density', 'housing_value_per_unit']:
                housing_metrics[f'{col}_change'] = housing_metrics.groupby('zip_code')[col].pct_change()
            
            # Fill NaN values
            housing_metrics = housing_metrics.fillna(0)
            
            # Save housing metrics for future use
            housing_metrics.to_csv(settings.PROCESSED_DATA_DIR / 'housing_metrics.csv', index=False)
            
            return {
                'housing': housing_metrics,
                'retail': retail_metrics,
                'demographic': census_data
            }
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def analyze_balance(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Analyze the balance between housing and retail development."""
        try:
            # Initialize housing model
            housing_model = HousingModel()
            housing_trends = housing_model.analyze_housing_trends(data['housing'])
            
            # Initialize retail model
            retail_model = RetailModel()
            retail_trends = retail_model.analyze_retail_trends(data['retail'])
            
            # Analyze housing-retail balance
            balance_analysis = housing_model.analyze_housing_retail_balance(data['housing'])
            
            # Ensure all required fields are present
            required_fields = {
                'housing': {
                    'total_units': 0,
                    'density': 0.0,
                    'pipeline_units': 0,
                    'total_value': 0.0,
                    'growth_rate': 0.0,
                    'vacancy_rate': 0.0
                },
                'retail': {
                    'total_space': 0,
                    'per_capita': 0.0,
                    'vacancy_rate': 0.0,
                    'total_value': 0.0,
                    'growth_rate': 0.0
                },
                'balance': {
                    'score': 0.0,
                    'ratio': 0.0,
                    'target_ratio': 0.0,
                    'deviation': 0.0
                },
                'trends': {
                    'housing_growth': 0.0,
                    'retail_growth': 0.0,
                    'correlation': 0.0
                }
            }
            
            # Merge analysis results with required fields
            analysis_results = {}
            for section, defaults in required_fields.items():
                section_data = {}
                if section == 'housing':
                    section_data = housing_trends.get('summary', {})
                elif section == 'retail':
                    section_data = retail_trends.get('summary', {})
                elif section == 'balance':
                    section_data = balance_analysis.get('metrics', {})
                elif section == 'trends':
                    section_data = balance_analysis.get('trends', {})
                
                # Ensure all required fields exist
                for field, default in defaults.items():
                    if field not in section_data or pd.isna(section_data[field]):
                        section_data[field] = default
                
                analysis_results[section] = section_data
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing housing-retail balance: {str(e)}")
            return {}
            
    def identify_opportunities(self, analysis_results: Dict) -> Dict[str, List[Dict]]:
        """Identify mixed-use development opportunities."""
        logger.info("Identifying development opportunities")
        
        opportunities = {
            'high_priority': [],
            'medium_priority': [],
            'long_term': []
        }
        
        try:
            # Analyze imbalanced areas
            imbalanced_areas = analysis_results['balance'][
                analysis_results['balance']['balance_score'] < 0.7
            ]
            
            # Generate opportunities for each area
            for _, area in imbalanced_areas.iterrows():
                if area['balance_score'] < 0.5:
                    opportunities['high_priority'].append({
                        'zip_code': area['zip_code'],
                        'action': f"Develop mixed-use project in {area['zip_code']} to address severe imbalance",
                        'priority': 'High',
                        'timeline': '0-12 months'
                    })
                elif area['balance_score'] < 0.6:
                    opportunities['medium_priority'].append({
                        'zip_code': area['zip_code'],
                        'action': f"Plan mixed-use development in {area['zip_code']} to improve balance",
                        'priority': 'Medium',
                        'timeline': '12-24 months'
                    })
                else:
                    opportunities['long_term'].append({
                        'zip_code': area['zip_code'],
                        'action': f"Monitor development balance in {area['zip_code']}",
                        'priority': 'Low',
                        'timeline': '24-36 months'
                    })
                    
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {str(e)}")
            raise
            
    def generate_report(self) -> None:
        """Generate the housing-retail balance report."""
        try:
            logger.info("Starting housing-retail balance report generation")
            
            # Load and prepare data
            data = self.load_and_prepare_data()
            if not all(df.size > 0 for df in data.values()):
                logger.error("One or more required datasets are empty")
                return
            
            # Analyze housing-retail balance
            logger.info("Analyzing housing-retail balance")
            analysis_results = self.analyze_balance(data)
            if not analysis_results:
                logger.error("Failed to analyze housing-retail balance")
                return
            
            # Identify development opportunities
            logger.info("Identifying development opportunities")
            opportunities = self.identify_opportunities(analysis_results)
            if not opportunities:
                logger.error("Failed to identify development opportunities")
                return
            
            # Create visualizations
            logger.info("Creating housing-retail balance analysis charts...")
            visualizer = Visualizer()
            visualizer.create_balance_analysis_charts(data['housing'])
            
            # Load report template
            template_path = settings.REPORT_TEMPLATES_DIR / 'housing_retail_balance_report.md'
            with open(template_path, 'r') as f:
                template = f.read()
            
            # Prepare context
            context = {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'current_analysis': analysis_results.get('housing', {}),
                'analysis_results': analysis_results,
                'recommendations': opportunities
            }
            
            # Generate report
            report = Template(template).render(context)
            
            # Save report
            output_path = settings.REPORTS_DIR / 'housing_retail_balance_report.md'
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Generated housing-retail balance report at {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating housing-retail balance report: {str(e)}")
            raise 