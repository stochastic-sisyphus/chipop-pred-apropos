"""
Module for generating housing-retail balance analysis reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import jinja2
from datetime import datetime
from jinja2 import Template
import traceback

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
        """Initialize the housing-retail balance report generator."""
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
        self.census_data_path = settings.PROCESSED_DATA_DIR / 'census_processed.csv'
        self.permits_data_path = settings.PROCESSED_DATA_DIR / 'permits_processed.csv'
        self.retail_data_path = settings.PROCESSED_DATA_DIR / 'retail_metrics.csv'
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        try:
            # Load data
            census_data = pd.read_csv(self.census_data_path)
            permits_data = pd.read_csv(self.permits_data_path)
            retail_data = pd.read_csv(self.retail_data_path)
            
            # Ensure ZIP codes are strings
            census_data['zip_code'] = census_data['zip_code'].astype(str).str.zfill(5)
            permits_data['zip_code'] = permits_data['zip_code'].astype(str).str.zfill(5)
            retail_data['zip_code'] = retail_data['zip_code'].astype(str).str.zfill(5)
            
            # Get most recent year
            current_year = census_data['year'].max()
            
            # Filter for current year
            census_current = census_data[census_data['year'] == current_year].copy()
            permits_current = permits_data[permits_data['year'] == current_year].copy()
            retail_current = retail_data[retail_data['year'] == current_year].copy()
            
            # Ensure required columns exist
            required_columns = {
                'census': ['total_population', 'total_housing_units'],
                'retail': ['retail_space', 'retail_demand', 'retail_gap']
            }
            
            for col in required_columns['census']:
                if col not in census_current.columns:
                    logger.error(f"Missing required census column: {col}")
                    return None
                    
            for col in required_columns['retail']:
                if col not in retail_current.columns:
                    logger.warning(f"Missing required retail column: {col}")
                    retail_current[col] = 0
            
            # Merge data
            merged = census_current.merge(permits_current, on='zip_code', how='left', suffixes=('', '_permits'))
            merged = merged.merge(retail_current, on='zip_code', how='left', suffixes=('', '_retail'))
            
            # Fill missing values
            merged = merged.fillna({
                'total_population': merged['total_population'].mean(),
                'total_housing_units': merged['total_housing_units'].mean(),
                'retail_space': 0,
                'retail_demand': 0,
                'retail_gap': 0
            })
            
            # Calculate per capita metrics
            merged['retail_per_capita'] = merged['retail_space'] / merged['total_population']
            merged['housing_per_capita'] = merged['total_housing_units'] / merged['total_population']
            
            # Calculate balance score (0 = perfect balance, higher = more imbalanced)
            merged['balance_score'] = abs(merged['retail_per_capita'] - merged['housing_per_capita'])
            
            # Determine balance categories
            merged['balance_category'] = pd.cut(
                merged['balance_score'],
                bins=[-float('inf'), 0.1, 0.3, 0.5, float('inf')],
                labels=['Balanced', 'Slightly Imbalanced', 'Moderately Imbalanced', 'Severely Imbalanced']
            )
            
            return merged
            
        except Exception as e:
            logger.error(f"Error loading and preparing data: {str(e)}")
            return None
            
    def analyze_balance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze housing-retail balance."""
        try:
            # Ensure required columns exist
            required_cols = {
                'total_population': 'total_population',
                'total_housing_units': 'total_housing_units',
                'retail_space': 'retail_space',
                'vacancy_rate': 'vacancy_rate'
            }
            
            # Map column names
            for target_col, source_col in required_cols.items():
                if source_col not in data.columns:
                    logger.warning(f"Missing required column: {source_col}")
                    if source_col == 'retail_space':
                        data[source_col] = 0
                    elif source_col == 'vacancy_rate':
                        data[source_col] = 0.1  # Default 10% vacancy
                    elif source_col == 'total_population':
                        data[source_col] = data['total_housing_units'] * 2.5 if 'total_housing_units' in data.columns else 0
                    elif source_col == 'total_housing_units':
                        data[source_col] = data['total_population'] / 2.5 if 'total_population' in data.columns else 0
            
            # Calculate balance metrics
            metrics = {
                'total_population': int(data['total_population'].sum()),
                'total_housing_units': int(data['total_housing_units'].sum()),
                'total_retail_space': float(data['retail_space'].sum()),
                'avg_retail_density': float(data['retail_space'].sum() / data['total_population'].sum()),
                'avg_housing_density': float(data['total_housing_units'].sum() / data['total_population'].sum()),
                'retail_vacancy_rate': float(data['vacancy_rate'].mean())
            }
            
            # Calculate balance scores by ZIP
            zip_metrics = []
            for zip_code in data['zip_code'].unique():
                zip_data = data[data['zip_code'] == zip_code]
                
                # Calculate metrics
                housing_units = int(zip_data['total_housing_units'].sum())
                population = int(zip_data['total_population'].sum())
                retail_space = float(zip_data['retail_space'].sum())
                
                # Calculate balance score (0-1, higher is better balanced)
                retail_per_capita = retail_space / population if population > 0 else 0
                housing_per_capita = housing_units / population if population > 0 else 0
                balance_score = min(retail_per_capita / metrics['avg_retail_density'], 1) if metrics['avg_retail_density'] > 0 else 0
                
                zip_metrics.append({
                    'zip_code': zip_code,
                    'total_population': population,
                    'total_housing_units': housing_units,
                    'retail_space': retail_space,
                    'retail_per_capita': retail_per_capita,
                    'housing_per_capita': housing_per_capita,
                    'balance_score': balance_score
                })
            
            # Identify severe imbalance areas
            severe_imbalance_zips = [
                metrics for metrics in zip_metrics
                if metrics['balance_score'] < 0.5 and metrics['total_population'] > 1000
            ]
            
            return {
                'summary': metrics,
                'zip_metrics': zip_metrics,
                'severe_imbalance_zips': severe_imbalance_zips
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze housing-retail balance: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def generate_report(self) -> bool:
        """Generate the housing-retail balance report."""
        try:
            # Load and prepare data
            data = self.load_and_prepare_data()
            if data is None or data.empty:
                logger.error("Failed to load required data")
                return False
            
            # Analyze housing-retail balance
            analysis = self.analyze_balance(data)
            if analysis is None:
                logger.error("Failed to analyze housing-retail balance")
                return False
            
            # Create visualizations
            try:
                self.visualizer.create_balance_analysis_charts(data)
            except Exception as e:
                logger.error(f"Failed to create visualizations: {str(e)}")
            
            # Load report template
            template = self.template_env.get_template('housing_retail_balance_report.md')
            
            # Prepare template variables with defaults
            template_vars = {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'current_analysis': {
                    'housing': {
                        'total_units': analysis['summary']['total_housing_units'],
                        'density': analysis['summary']['avg_housing_density'],
                        'pipeline_units': data['residential_permits'].sum() if 'residential_permits' in data.columns else 0
                    },
                    'retail': {
                        'total_space': analysis['summary']['total_retail_space'],
                        'per_capita': analysis['summary']['avg_retail_density'],
                        'vacancy_rate': analysis['summary']['retail_vacancy_rate']
                    }
                },
                'analysis_results': {
                    'imbalance_areas': [
                        {
                            'zip_code': metrics['zip_code'],
                            'total_population': metrics['total_population'],
                            'total_housing_units': metrics['total_housing_units'],
                            'retail_space': metrics['retail_space'],
                            'retail_per_capita': metrics['retail_per_capita'],
                            'housing_per_capita': metrics['housing_per_capita'],
                            'balance_score': metrics['balance_score'],
                            'primary_need': 'Retail Development' if metrics['retail_per_capita'] < metrics['housing_per_capita'] else 'Housing Development'
                        }
                        for metrics in analysis['severe_imbalance_zips']
                    ],
                    'development_patterns': [
                        {
                            'name': 'High Growth Areas',
                            'trend': 'Increasing housing development',
                            'impact': 'High retail demand potential',
                            'risk_level': 'Low'
                        }
                    ],
                    'mixed_use_opportunities': [
                        {
                            'location': f"ZIP {metrics['zip_code']}",
                            'potential': 'High',
                            'housing_units': metrics['total_housing_units'],
                            'retail_space': metrics['retail_space'],
                            'investment_value': metrics['retail_space'] * 300  # Assuming $300/sqft
                        }
                        for metrics in sorted(
                            analysis['zip_metrics'],
                            key=lambda x: x['balance_score']
                        )[:5]  # Top 5 opportunities
                    ]
                },
                'recommendations': {
                    'development': [
                        "Focus retail development in high-growth residential areas",
                        "Prioritize mixed-use development in opportunity zones",
                        "Incentivize retail in underserved neighborhoods"
                    ],
                    'policy': [
                        "Update zoning to encourage mixed-use development",
                        "Streamline retail development permits",
                        "Create retail attraction programs"
                    ]
                }
            }
            
            # Generate report content
            report_content = template.render(**template_vars)
            
            # Save report
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Housing-retail balance report generated at {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate housing retail balance report: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def generate_report(census_data, permit_data, economic_data, zoning_data, retail_metrics, retail_deficit):
    """Generate housing retail balance report."""
    try:
        # Load report template
        template_path = settings.TEMPLATES_DIR / 'reports/housing_retail_balance_report.md'
        with open(template_path, 'r') as f:
            template = Template(f.read())
        
        # Calculate balance metrics
        metrics = {
            'total_housing_units': int(census_data['total_housing_units'].sum()),
            'occupied_units': int(census_data['occupied_housing_units'].sum()),
            'vacant_units': int(census_data['vacant_housing_units'].sum()),
            'total_population': int(census_data['total_population'].sum()),
            'total_retail_space': retail_metrics['retail_space'].sum() if 'retail_space' in retail_metrics.columns else 0,
            'retail_per_capita': retail_metrics['retail_space'].sum() / census_data['total_population'].sum() if 'retail_space' in retail_metrics.columns else 0,
            'retail_vacancy_rate': retail_metrics['vacancy_rate'].mean() if 'vacancy_rate' in retail_metrics.columns else 0
        }
        
        # Calculate balance scores
        if 'retail_space' in retail_metrics.columns:
            retail_metrics['balance_score'] = retail_metrics.apply(
                lambda row: min(
                    row['retail_space'] / (row['total_population'] * 20),  # 20 sq ft per person benchmark
                    1.0
                ),
                axis=1
            )
        else:
            retail_metrics['balance_score'] = 0
            
        # Identify imbalanced areas
        imbalanced_areas = []
        if 'balance_score' in retail_metrics.columns:
            imbalanced_areas = retail_metrics[
                retail_metrics['balance_score'] < 0.5
            ].sort_values('balance_score').to_dict('records')
            
        # Generate report content
        report = template.render(
            generation_date=datetime.now().strftime('%Y-%m-%d'),
            current_analysis={
                'housing': {
                    'total_units': metrics['total_housing_units'],
                    'density': metrics['total_housing_units'] / metrics['total_population'],
                    'pipeline_units': permit_data['residential_permits'].sum() if 'residential_permits' in permit_data.columns else 0
                },
                'retail': {
                    'total_space': metrics['total_retail_space'],
                    'per_capita': metrics['retail_per_capita'],
                    'vacancy_rate': metrics['retail_vacancy_rate']
                }
            },
            analysis_results={
                'imbalance_areas': imbalanced_areas,
                'development_patterns': [],
                'mixed_use_opportunities': []
            },
            recommendations={
                'development': [],
                'policy': []
            }
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate housing retail balance report: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None 