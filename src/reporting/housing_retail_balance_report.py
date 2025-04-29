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
        """Analyze housing-retail balance patterns."""
        logger.info("Analyzing housing-retail balance")
        
        try:
            housing_analysis = self.housing_model.analyze_housing_trends(data['housing'])
            retail_analysis = self.retail_model.analyze_retail_trends(data['retail'])
            balance_analysis = self.housing_model.analyze_housing_retail_balance(data['housing'])
            
            return {
                'housing': housing_analysis,
                'retail': retail_analysis,
                'balance': balance_analysis
            }
        except Exception as e:
            logger.error(f"Error analyzing housing-retail balance: {str(e)}")
            raise
            
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
        """Generate the complete housing-retail balance report."""
        logger.info("Starting housing-retail balance report generation")

        try:
            # Load and analyze data
            data = self.load_and_prepare_data()
            analysis_results = self.analyze_balance(data)
            opportunities = self.identify_opportunities(analysis_results)

            # Generate visualizations
            self.visualizer.create_balance_analysis_charts(analysis_results['balance'])

            # Load template
            template = self.template_env.get_template('housing_retail_balance_report.md')

            # Prepare template context
            imbalance_areas = analysis_results['balance']
            if not isinstance(imbalance_areas, pd.DataFrame) or imbalance_areas.empty:
                imbalance_areas_records = []
            else:
                imbalance_areas_records = imbalance_areas.to_dict('records')

            # Check if high_priority opportunities exist and are not empty
            high_priority_opps = opportunities.get('high_priority', [])
            mixed_use_opps = list(high_priority_opps) if high_priority_opps else []

            context = {
                'current_analysis': {
                    'housing': analysis_results['housing'],
                    'retail': analysis_results['retail'],
                },
                'analysis_results': {
                    'imbalance_areas': imbalance_areas_records,
                    'mixed_use_opportunities': mixed_use_opps,
                },
                'recommendations': {
                    'development': [opp['action'] for opp in high_priority_opps],
                    'policy': [
                        "Update zoning to encourage mixed-use development",
                        "Streamline permits for balanced development",
                        "Create incentives for retail in residential areas",
                    ],
                },
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
            }

            # Generate report
            report_content = template.render(**context)

            # Save report
            with open(self.output_path, 'w') as f:
                f.write(report_content)

            logger.info(f"Generated housing-retail balance report at {self.output_path}")

        except Exception as e:
            logger.error(f"Error generating housing-retail balance report: {str(e)}")
            raise 