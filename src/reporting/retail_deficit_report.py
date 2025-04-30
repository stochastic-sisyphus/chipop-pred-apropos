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
            retail_metrics = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_metrics.csv')
            census_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'census_processed.csv')
            economic_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'economic_processed.csv')
            retail_leakage = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_leakage.csv')
            retail_predictions = pd.read_csv(settings.PREDICTIONS_DIR / 'retail_demand_predictions.csv')
            
            return {
                'retail': retail_metrics,
                'demographic': census_data,
                'economic': economic_data,
                'leakage': retail_leakage,
                'predictions': retail_predictions
            }
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def analyze_retail_deficit(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze retail deficit patterns and opportunities."""
        try:
            # Initialize retail model
            retail_model = RetailModel()
            
            # Get retail trends and leakage analysis
            retail_trends = retail_model.analyze_retail_trends(data)
            retail_leakage = retail_model.analyze_retail_leakage(data)
            
            # Required fields with default values
            required_fields = {
                'current': {
                    'retail': {
                        'total_space': 0,
                        'density': 0.0,
                        'vacancy_rate': 0.0,
                        'total_value': 0.0
                    }
                },
                'retail_deficit': [],  # Will be populated from leakage analysis
                'retail_categories': [],  # Will be populated from trends
                'development_potential': []  # Will be populated from analysis
            }
            
            # Process retail deficit areas
            deficit_areas = []
            if not retail_leakage.empty:
                for _, row in retail_leakage.iterrows():
                    area = {
                        'zip_code': row.get('zip_code', ''),
                        'spending_potential': float(row.get('spending_potential', 0.0)),
                        'current_provision': float(row.get('current_provision', 0.0)),
                        'retail_gap': float(row.get('retail_gap', 0.0)),
                        'leakage_rate': float(row.get('leakage_rate', 0.0))
                    }
                    deficit_areas.append(area)
            
            # Process retail categories
            categories = []
            if 'retail_categories' in retail_trends:
                for cat in retail_trends['retail_categories']:
                    category = {
                        'name': cat.get('name', ''),
                        'market_gap': float(cat.get('market_gap', 0.0)),
                        'required_space': float(cat.get('required_space', 0.0)),
                        'potential_stores': int(cat.get('potential_stores', 0))
                    }
                    categories.append(category)
            
            # Process development potential
            development_areas = []
            if 'development_sites' in retail_trends:
                for site in retail_trends['development_sites']:
                    area = {
                        'code': site.get('zip_code', ''),
                        'available_sites': int(site.get('available_sites', 0)),
                        'potential_gla': float(site.get('potential_gla', 0.0)),
                        'investment_value': float(site.get('investment_value', 0.0))
                    }
                    development_areas.append(area)
            
            # Combine all results
            analysis_results = {
                'current': {
                    'retail': {
                        'total_space': float(retail_trends.get('total_space', 0.0)),
                        'density': float(retail_trends.get('density', 0.0)),
                        'vacancy_rate': float(retail_trends.get('vacancy_rate', 0.0)),
                        'total_value': float(retail_trends.get('total_value', 0.0))
                    }
                },
                'retail_deficit': deficit_areas,
                'retail_categories': categories,
                'development_potential': development_areas
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing retail deficit: {str(e)}")
            return {}
            
    def generate_recommendations(self, deficit_data: Dict) -> Dict[str, list]:
        """Generate recommendations based on retail deficit analysis."""
        try:
            # Initialize recommendations
            recommendations = {
                'retail': [],
                'development': [],
                'policy': []
            }
            
            # Process retail deficit areas
            if 'retail_deficit' in deficit_data:
                deficit_areas = deficit_data['retail_deficit']
                if isinstance(deficit_areas, list) and deficit_areas:
                    # Sort by leakage rate if available
                    sorted_areas = sorted(
                        deficit_areas,
                        key=lambda x: float(x.get('leakage_rate', 0)),
                        reverse=True
                    )
                    
                    # Generate recommendations for high deficit areas
                    for area in sorted_areas[:3]:  # Top 3 areas
                        zip_code = area.get('zip_code', '')
                        leakage = float(area.get('retail_gap', 0))
                        if leakage > 0:
                            recommendations['retail'].append(
                                f"Address retail gap of ${leakage:,.2f} in ZIP {zip_code}"
                            )
            
            # Process retail categories
            if 'retail_categories' in deficit_data:
                categories = deficit_data['retail_categories']
                if isinstance(categories, list) and categories:
                    # Sort by market gap if available
                    sorted_cats = sorted(
                        categories,
                        key=lambda x: float(x.get('market_gap', 0)),
                        reverse=True
                    )
                    
                    # Generate recommendations for underserved categories
                    for cat in sorted_cats[:3]:  # Top 3 categories
                        name = cat.get('name', '')
                        gap = float(cat.get('market_gap', 0))
                        if gap > 0:
                            recommendations['development'].append(
                                f"Develop {name} retail space to capture ${gap:,.2f} market opportunity"
                            )
            
            # Add policy recommendations
            recommendations['policy'] = [
                "Update zoning regulations to encourage retail development in high-deficit areas",
                "Create incentives for retailers in underserved categories",
                "Streamline permitting process for retail development in target areas"
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                'retail': [],
                'development': [],
                'policy': []
            }
            
    def generate_report(self) -> None:
        """Generate the retail deficit analysis report."""
        try:
            logger.info("Starting retail deficit report generation")
            
            # Load and prepare data
            data = self.load_and_prepare_data()
            if not all(df.size > 0 for df in data.values()):
                logger.error("One or more required datasets are empty")
                return
            
            # Analyze retail deficit
            logger.info("Analyzing retail deficit patterns")
            analysis_results = self.analyze_retail_deficit(data['retail'])
            if not analysis_results:
                logger.error("Failed to analyze retail deficit")
                return
            
            # Generate recommendations
            logger.info("Generating recommendations")
            recommendations = self.generate_recommendations(analysis_results)
            if not recommendations:
                logger.error("Failed to generate recommendations")
                return
            
            # Create visualizations
            visualizer = Visualizer()
            visualizer.create_retail_deficit_map(data['retail'])
            
            # Load report template
            template_path = settings.REPORT_TEMPLATES_DIR / 'retail_deficit_analysis.md'
            with open(template_path, 'r') as f:
                template = f.read()
            
            # Prepare context
            context = {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'current_analysis': analysis_results.get('current', {}),
                'analysis_results': analysis_results,
                'recommendations': recommendations
            }
            
            # Generate report
            report = Template(template).render(context)
            
            # Save report
            output_path = settings.REPORTS_DIR / 'retail_deficit_analysis.md'
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Generated retail deficit report at {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating retail deficit report: {str(e)}")
            raise 