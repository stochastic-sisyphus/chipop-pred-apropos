"""
Module for generating retail deficit analysis reports.
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
from src.models.retail_model import RetailModel
from src.visualization.visualizer import Visualizer
from src.utils.helpers import resolve_column_name
from src.config.column_alias_map import column_aliases

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    'total_population', 'median_household_income', 'retail_space', 'vacancy_rate',
    'retail_gap', 'retail_demand', 'retail_supply'
]

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
            # Load processed data
            census_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'census_processed.csv')
            permit_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'permits_processed.csv')
            retail_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_deficit.csv')
            retail_deficit = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_deficit.csv')
            
            # Ensure required columns exist
            required_columns = {
                'census': ['zip_code', 'year', 'total_population', 'median_household_income'],
                'permits': ['zip_code', 'year', 'total_permits', 'total_construction_cost'],
                'retail': ['zip_code', 'retail_space', 'vacancy_rate'],
                'deficit': ['zip_code', 'retail_gap', 'retail_demand', 'retail_supply']
            }
            
            # Validate and standardize zip_code columns
            for df_name, df in {
                'census': census_data,
                'permits': permit_data,
                'retail': retail_data,
                'deficit': retail_deficit
            }.items():
                # Ensure zip_code exists
                if 'zip_code' not in df.columns:
                    logger.error(f"zip_code column missing from {df_name} data")
                    return None
                
                # Standardize zip_code format
                df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
                
                # Validate other required columns
                for col in required_columns[df_name]:
                    if col not in df.columns and col != 'zip_code':
                        logger.warning(f"{col} missing from {df_name} data")
                        if col in ['retail_space', 'vacancy_rate', 'retail_gap', 'retail_demand', 'retail_supply']:
                            df[col] = 0
                        elif col == 'total_population':
                            df[col] = df['total_housing_units'] * 2.5 if 'total_housing_units' in df.columns else 0
                        elif col == 'median_household_income':
                            df[col] = 50000  # Default value
            
            # Merge datasets
            merged = pd.merge(census_data, permit_data, on=['zip_code', 'year'], how='left')
            merged = pd.merge(merged, retail_data, on=['zip_code', 'year'], how='left')  # ✅ FIX
            merged = pd.merge(merged, retail_deficit, on=['zip_code', 'year'], how='left')  # ✅ FIX
            
            # Fill missing values
            merged = merged.fillna({
                'total_permits': 0,
                'total_construction_cost': 0,
                'retail_space': 0,
                'vacancy_rate': 0.1,
                'retail_gap': 0,
                'retail_demand': 0,
                'retail_supply': 0
            })
            
            # Log data shapes
            logger.info(f"Census data: {census_data.shape}")
            logger.info(f"Permits data: {permit_data.shape}")
            logger.info(f"Retail data: {retail_data.shape}")
            logger.info(f"Retail deficit data: {retail_deficit.shape}")
            logger.info(f"Merged data: {merged.shape}")
            
            return {
                'census': census_data,
                'permits': permit_data,
                'retail': retail_data,
                'deficit': retail_deficit,
                'merged': merged
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def analyze_retail_deficit(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze retail deficit patterns."""
        try:
            merged_data = data['merged']

            # Ensure required columns exist
            required_cols = {
                'total_population': 'total_population',
                'median_household_income': 'median_household_income',
                'retail_space': 'retail_space',
                'vacancy_rate': 'vacancy_rate',
                'retail_gap': 'retail_gap',
                'retail_demand': 'retail_demand',
                'retail_supply': 'retail_supply'
            }

            # Map column names and provide defaults
            for source_col in required_cols.values():
                if source_col not in merged_data.columns:
                    logger.warning(f"Missing required column: {source_col}")
                    if source_col in ['retail_gap', 'retail_demand']:
                        # Calculate retail gap as 30% of total income
                        merged_data[source_col] = merged_data['median_household_income'] * merged_data['total_population'] * 0.3
                    elif source_col in ['retail_space', 'retail_supply']:
                        merged_data[source_col] = 0
                    elif source_col == 'vacancy_rate':
                        merged_data[source_col] = 0.1  # Default 10% vacancy
            # Calculate summary metrics
            metrics = {
                'total_retail_space': float(merged_data['retail_space'].sum()),
                'avg_space_per_capita': float(merged_data['retail_space'].sum() / merged_data['total_population'].sum()),
                'total_deficit': float(merged_data['retail_gap'].sum()),
                'avg_deficit_per_capita': float(merged_data['retail_gap'].sum() / merged_data['total_population'].sum())
            }

            # Calculate metrics by ZIP
            zip_metrics = []
            for zip_code in merged_data['zip_code'].unique():
                zip_data = merged_data[merged_data['zip_code'] == zip_code]

                # Calculate metrics
                population = int(zip_data['total_population'].sum())
                retail_space = float(zip_data['retail_space'].sum())
                retail_gap = float(zip_data['retail_gap'].sum())
                retail_demand = float(zip_data['retail_demand'].sum())
                retail_supply = float(zip_data['retail_supply'].sum())

                # Calculate per capita metrics
                retail_per_capita = retail_space / population if population > 0 else 0
                gap_per_capita = retail_gap / population if population > 0 else 0

                zip_metrics.append({
                    'location': f"ZIP {zip_code}",
                    'total_population': population,
                    'retail_space': retail_space,
                    'retail_per_capita': retail_per_capita,
                    'market_gap': retail_gap,
                    'gap_per_capita': gap_per_capita,
                    'retail_demand': retail_demand,
                    'retail_supply': retail_supply,
                    'required_space': int(retail_gap / 300),  # Assuming $300/sqft
                    'potential_stores': int(retail_gap / 1000000)  # Assuming $1M per store
                })

            # Sort by market gap
            zip_metrics.sort(key=lambda x: x['market_gap'], reverse=True)

            return {
                'summary': metrics,
                'zip_metrics': zip_metrics[:5]  # Top 5 deficit areas
            }

        except Exception as e:
            logger.error(f"Failed to analyze retail deficit: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def ensure_merged_retail_columns(self, merged_data: pd.DataFrame, retail_deficit: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure merged_data has all required retail columns, filling from retail_deficit if possible, else with 0.
        """
        required_cols = ["retail_space", "vacancy_rate", "retail_gap", "retail_demand", "retail_supply"]
        for col in required_cols:
            if col not in merged_data.columns:
                if retail_deficit is not None and col in retail_deficit.columns:
                    merged_data[col] = retail_deficit[col]
                    logging.info(f"Filled missing column '{col}' in merged_data from retail_deficit.")
                else:
                    merged_data[col] = 0
                    logging.warning(f"Filled missing column '{col}' in merged_data with 0 (not found in retail_deficit).")
        return merged_data

    def generate_report(self, census_data, permit_data, economic_data, zoning_data, retail_metrics, retail_deficit):
        try:
            # Load report template
            template_path = settings.TEMPLATES_DIR / 'reports/retail_deficit_analysis.md'
            with open(template_path, 'r') as f:
                template = Template(f.read())

            # Prepare opportunity dataframe
            opportunity_df = pd.merge(
                census_data[['zip_code', 'total_population', 'median_household_income']],
                retail_metrics[['zip_code', 'retail_space', 'vacancy_rate']] if 'retail_space' in retail_metrics.columns else pd.DataFrame({'zip_code': census_data['zip_code'].unique()}),
                on='zip_code', how='left'
            )
            # Fill missing retail metrics with 0
            for col in ['retail_space', 'vacancy_rate']:
                if col not in opportunity_df.columns:
                    opportunity_df[col] = 0
            # Log available columns
            logging.info(f"RetailDeficitReport: opportunity_df columns: {list(opportunity_df.columns)}")
            # Check for missing/zeroed required columns
            missing = []
            all_zero = []
            for col in REQUIRED_COLS:
                if col not in opportunity_df.columns:
                    missing.append(col)
                    opportunity_df[col] = 0
                    logging.warning(f"RetailDeficitReport: Missing column {col}, set to 0.")
                elif (opportunity_df[col] == 0).all():
                    all_zero.append(col)
                    logging.warning(f"RetailDeficitReport: All values in {col} are zero.")
            notes = []
            if missing:
                notes.append(f"Missing columns: {', '.join(missing)}")
            if all_zero:
                notes.append(f"All zero columns: {', '.join(all_zero)}")
            # Prepare context
            context = {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'opportunity_areas': {},
                'high_deficit_areas': [],
                'recommendations': [],
                'methodology_notes': '',
                'notes': notes,
                'missing_or_defaulted': missing + all_zero
            }
            # Always provide opportunity_areas and high_deficit_areas as empty or defaulted
            try:
                rendered = template.render(**{k: context.get(k, 'N/A') for k in context})
            except Exception as e:
                logging.error(f"RetailDeficitReport: Template rendering failed: {e}")
                rendered = f"Report generation failed. Error: {e}\nNotes: {context.get('notes', [])}"
            return rendered
        except Exception as e:
            logging.error(f"RetailDeficitReport: Failed to generate report: {e}")
            return f"Report generation failed. Error: {e}"

    def ensure_retail_columns(self, retail_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required retail columns exist, adding with default 0 if missing. Log only once per column."""
        REQUIRED_RETAIL_COLUMNS = [
            "retail_space", "retail_demand", "retail_gap", "vacancy_rate", "retail_supply"
        ]
        logged_missing_retail_columns = set()
        for col in REQUIRED_RETAIL_COLUMNS:
            if col not in retail_data.columns:
                if col not in logged_missing_retail_columns:
                    logging.warning(f"Added missing column {col} to retail_data with default value 0")
                    logged_missing_retail_columns.add(col)
                retail_data[col] = 0
        return retail_data

def generate_report(census_data, permit_data, economic_data, zoning_data, retail_metrics, retail_deficit):
    """Generate retail deficit analysis report."""
    try:
        # Load report template
        template_path = settings.TEMPLATES_DIR / 'reports/retail_deficit_analysis.md'
        with open(template_path, 'r') as f:
            template = Template(f.read())

        # Prepare opportunity dataframe
        opportunity_df = pd.merge(
            census_data[['zip_code', 'total_population', 'median_household_income']],
            retail_metrics[['zip_code', 'retail_space', 'vacancy_rate']] if 'retail_space' in retail_metrics.columns else pd.DataFrame({'zip_code': census_data['zip_code'].unique()}),
            on='zip_code',
            how='left'
        )

        # Fill missing retail metrics with 0
        if 'retail_space' not in opportunity_df.columns:
            opportunity_df['retail_space'] = 0
        if 'vacancy_rate' not in opportunity_df.columns:
            opportunity_df['vacancy_rate'] = 0

        # Calculate retail metrics
        metrics = {
            'total_population': int(census_data['total_population'].sum()),
            'total_retail_space': opportunity_df['retail_space'].sum(),
            'retail_per_capita': opportunity_df['retail_space'].sum() / opportunity_df['total_population'].sum(),
            'vacancy_rate': opportunity_df['vacancy_rate'].mean(),
            'total_spending_potential': (opportunity_df['total_population'] * opportunity_df['median_household_income'] * 0.3).sum(),
            'total_retail_gap': (opportunity_df['total_population'] * opportunity_df['median_household_income'] * 0.3 - opportunity_df['retail_space'] * 300).sum()
        }

        # Calculate retail deficit areas
        deficit_areas = opportunity_df.copy()
        deficit_areas['spending_potential'] = deficit_areas['total_population'] * deficit_areas['median_household_income'] * 0.3
        deficit_areas['current_provision'] = deficit_areas['retail_space'] * 300  # $300 sales per sq ft
        deficit_areas['retail_gap'] = deficit_areas['spending_potential'] - deficit_areas['current_provision']
        deficit_areas['leakage_rate'] = deficit_areas['retail_gap'] / deficit_areas['spending_potential']

        # Sort and get top deficit areas
        top_deficit_areas = deficit_areas.sort_values('retail_gap', ascending=False).head(5).to_dict('records')

        return template.render(
            generation_date=datetime.now().strftime('%Y-%m-%d'),
            current_analysis={
                'retail': {
                    'total_space': metrics['total_retail_space'],
                    'density': metrics['retail_per_capita'],
                    'vacancy_rate': metrics['vacancy_rate'],
                }
            },
            analysis_results={
                'retail_deficit': top_deficit_areas,
                'retail_categories': [],
                'development_potential': [],
            },
        )
    except Exception as e:
        logger.error(f"Failed to generate retail deficit report: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None 