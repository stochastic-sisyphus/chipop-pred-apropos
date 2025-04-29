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
        """Analyze retail deficit patterns."""
        logger.info("Analyzing retail deficit patterns")
        
        try:
            retail_analysis = self.retail_model.analyze_retail_trends(data['retail'])
            retail_demand = self.retail_model.predict_retail_demand(data['retail'])
            retail_leakage = self.retail_model.analyze_retail_leakage(data['retail'])
            
            return {
                'trends': retail_analysis,
                'demand': retail_demand,
                'leakage': retail_leakage
            }
        except Exception as e:
            logger.error(f"Error analyzing retail deficit: {str(e)}")
            raise
            
    def generate_recommendations(self, deficit_data: Dict) -> Dict[str, list]:
        """Generate recommendations based on deficit analysis."""
        logger.info("Generating recommendations")
        
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'long_term': []
        }
        
        try:
            # Analyze high deficit areas
            high_deficit_areas = deficit_data['leakage'][
                deficit_data['leakage']['leakage_rate'] > 0.2
            ]
            
            # Generate recommendations for each area
            for _, area in high_deficit_areas.iterrows():
                if area['leakage_rate'] > 0.4:
                    recommendations['high_priority'].append({
                        'zip_code': area['zip_code'],
                        'action': f"Develop retail space in {area['zip_code']} to capture ${area['retail_gap']:,.0f} in retail opportunity",
                        'priority': 'High',
                        'timeline': '0-12 months'
                    })
                elif area['leakage_rate'] > 0.3:
                    recommendations['medium_priority'].append({
                        'zip_code': area['zip_code'],
                        'action': f"Plan retail development in {area['zip_code']} to address ${area['retail_gap']:,.0f} retail gap",
                        'priority': 'Medium',
                        'timeline': '12-24 months'
                    })
                else:
                    recommendations['long_term'].append({
                        'zip_code': area['zip_code'],
                        'action': f"Monitor retail opportunity in {area['zip_code']} (${area['retail_gap']:,.0f} potential)",
                        'priority': 'Low',
                        'timeline': '24-36 months'
                    })
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
            
    def generate_report(self) -> None:
        """Generate the complete retail deficit analysis report."""
        logger.info("Starting retail deficit report generation")
        
        try:
            # Load and analyze data
            data = self.load_and_prepare_data()
            deficit_analysis = self.analyze_retail_deficit(data)
            recommendations = self.generate_recommendations(deficit_analysis)
            
            # Generate visualizations
            self.visualizer.create_retail_deficit_map(deficit_analysis['leakage'])
            
            # Calculate current state metrics with safe defaults
            def safe_float(val, default=0.0):
                try:
                    return default if pd.isna(val) or val is None else float(val)
                except Exception:
                    return default

            retail_df = data['retail']
            total_space = safe_float(retail_df['retail_space'].sum())
            density = safe_float(retail_df['retail_space_per_capita'].mean())
            vacancy_rate = 0.1  # Placeholder - would need actual vacancy data

            current_metrics = {
                'retail': {
                    'total_space': total_space,
                    'density': density,
                    'vacancy_rate': vacancy_rate
                }
            }
            
            # Load template
            template = self.template_env.get_template('retail_deficit_analysis.md')
            
            # Patch: ensure all fields are present and correct type for template
            def clean_record(record, keys_defaults):
                out = {}
                for k, default in keys_defaults.items():
                    v = record.get(k, default)
                    # Patch: ensure correct type for numeric fields
                    if isinstance(default, (int, float)):
                        try:
                            v = float(v)
                        except Exception:
                            v = default
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        out[k] = default
                    else:
                        out[k] = v
                # Add any extra keys from record
                for k, v in record.items():
                    if k not in out:
                        if isinstance(v, (int, float)):
                            try:
                                v = float(v)
                            except Exception:
                                v = 0
                        out[k] = 0 if (v is None or (isinstance(v, float) and pd.isna(v))) else v
                return out

            # Define expected keys for each section
            retail_deficit_keys = {
                'zip_code': '', 'spending_potential': 0.0, 'current_provision': 0.0, 'retail_gap': 0.0, 'leakage_rate': 0.0
            }
            retail_categories_keys = {
                'name': '', 'market_gap': 0.0, 'required_space': 0.0, 'potential_stores': 0
            }
            development_potential_keys = {
                'code': '', 'available_sites': 0, 'potential_gla': 0.0, 'investment_value': 0.0
            }

            # Prepare lists, ensuring at least one record with safe defaults if empty
            retail_deficit_list = [clean_record(r, retail_deficit_keys) for r in deficit_analysis['leakage'].to_dict('records')]
            if not retail_deficit_list:
                retail_deficit_list = [retail_deficit_keys]
            retail_categories_list = [clean_record(r, retail_categories_keys) for r in deficit_analysis['trends'].to_dict('records')]
            if not retail_categories_list:
                retail_categories_list = [retail_categories_keys]
            # Patch: ensure 'code' key is present for development_potential
            dev_pot_records = []
            for r in deficit_analysis['demand'].to_dict('records'):
                if 'zip_code' in r and 'code' not in r:
                    r['code'] = r['zip_code']
                dev_pot_records.append(r)
            development_potential_list = [clean_record(r, development_potential_keys) for r in dev_pot_records]
            if not development_potential_list:
                development_potential_list = [development_potential_keys]

            # Prepare template context with safe values
            context = {
                'current_analysis': current_metrics,
                'analysis_results': {
                    'retail_deficit': retail_deficit_list,
                    'retail_categories': retail_categories_list,
                    'development_potential': development_potential_list
                },
                'recommendations': {
                    'retail': [rec['action'] for rec in recommendations['high_priority']] if recommendations.get('high_priority') else []
                },
                'generation_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Generate report
            report_content = template.render(**context)
            
            # Save report
            with open(self.output_path, 'w') as f:
                f.write(report_content)
                
            logger.info(f"Generated retail deficit report at {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error generating retail deficit report: {str(e)}")
            raise 