"""
Zoning data processing module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union

from src.config import settings
from src.utils.helpers import resolve_column_name
from src.config.column_alias_map import column_aliases

logger = logging.getLogger(__name__)

class ZoningProcessor:
    """Handles zoning data processing and analysis."""
    
    def __init__(self):
        """Initialize the zoning processor."""
        self.processed_data_dir = settings.PROCESSED_DATA_DIR
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.zoning_data = None
        self.property_data = None
        
    def process_zoning_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process zoning data."""
        try:
            # Resolve column names
            zip_col = resolve_column_name(df, 'zip_code', column_aliases)

            if not zip_col:
                logger.error("ZIP code column not found in zoning data")
                return pd.DataFrame()

            # Clean ZIP codes
            df[zip_col] = df[zip_col].astype(str).str.extract(r'(\d{5})').fillna('00000')

            # Calculate zoning metrics
            metrics = df.groupby(zip_col).agg({
                'zoning_area': 'sum',
                'residential_area': 'sum',
                'commercial_area': 'sum',
                'retail_area': 'sum'
            }).reset_index()

            # Calculate percentages
            for col in ['residential_area', 'commercial_area', 'retail_area']:
                metrics[f'{col}_pct'] = metrics[col] / metrics['zoning_area'] * 100

            return self._extracted_from_process_property_data_27(
                'zoning_processed.csv',
                metrics,
                "Successfully processed zoning data",
            )
        except Exception as e:
            logger.error(f"Error processing zoning data: {str(e)}")
            return pd.DataFrame()
            
    def process_property_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process property data."""
        try:
            # Resolve column names
            zip_col = resolve_column_name(df, 'zip_code', column_aliases)

            if not zip_col:
                logger.error("ZIP code column not found in property data")
                return pd.DataFrame()

            # Clean ZIP codes
            df[zip_col] = df[zip_col].astype(str).str.extract(r'(\d{5})').fillna('00000')

            # Calculate property metrics
            metrics = df.groupby(zip_col).agg({
                'property_area': 'sum',
                'building_area': 'sum',
                'assessed_value': 'mean',
                'market_value': 'mean',
                'year_built': 'mean'
            }).reset_index()

            # Calculate derived metrics
            metrics['building_density'] = metrics['building_area'] / metrics['property_area']
            metrics['value_per_sqft'] = metrics['market_value'] / metrics['building_area']

            return self._extracted_from_process_property_data_27(
                'property_processed.csv',
                metrics,
                "Successfully processed property data",
            )
        except Exception as e:
            logger.error(f"Error processing property data: {str(e)}")
            return pd.DataFrame()

    # TODO Rename this here and in `process_zoning_data` and `process_property_data`
    def _extracted_from_process_property_data_27(self, arg0, metrics, arg2):
        output_path = self.processed_data_dir / arg0
        metrics.to_csv(output_path, index=False)
        logger.info(arg2)
        return metrics
            
    def analyze_zoning_patterns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze zoning patterns and trends."""
        try:
            # Resolve column names
            zip_col = resolve_column_name(df, 'zip_code', column_aliases)
            
            if not zip_col:
                logger.error("ZIP code column not found for zoning analysis")
                return {}
            
            # Calculate zoning mix
            zoning_mix = df.groupby(zip_col).agg({
                'residential_area_pct': 'mean',
                'commercial_area_pct': 'mean',
                'retail_area_pct': 'mean'
            }).reset_index()
            
            # Identify predominant use
            for col in ['residential', 'commercial', 'retail']:
                pct_col = f'{col}_area_pct'
                zoning_mix[f'{col}_dominant'] = zoning_mix[pct_col] > 50
            
            # Identify mixed-use areas
            zoning_mix['mixed_use'] = ~(
                zoning_mix['residential_dominant'] |
                zoning_mix['commercial_dominant'] |
                zoning_mix['retail_dominant']
            )
            
            # Calculate development potential
            development_potential = df.groupby(zip_col).agg({
                'zoning_area': 'sum',
                'building_area': 'sum'
            }).reset_index()
            
            development_potential['development_capacity'] = (
                development_potential['zoning_area'] -
                development_potential['building_area']
            )
            
            return {
                'zoning_mix': zoning_mix,
                'development_potential': development_potential
            }
            
        except Exception as e:
            logger.error(f"Error analyzing zoning patterns: {str(e)}")
            return {}
            
    def analyze_property_values(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze property values and trends."""
        try:
            # Resolve column names
            zip_col = resolve_column_name(df, 'zip_code', column_aliases)
            
            if not zip_col:
                logger.error("ZIP code column not found for property analysis")
                return {}
            
            # Calculate value metrics
            value_metrics = df.groupby(zip_col).agg({
                'market_value': ['mean', 'median', 'std'],
                'value_per_sqft': ['mean', 'median', 'std']
            }).reset_index()
            
            # Flatten column names
            value_metrics.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0]
                for col in value_metrics.columns
            ]
            
            # Calculate value tiers
            value_metrics['value_tier'] = pd.qcut(
                value_metrics['market_value_mean'],
                q=5,
                labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            )
            
            # Calculate value density
            value_density = df.groupby(zip_col).agg({
                'market_value': 'sum',
                'property_area': 'sum'
            }).reset_index()
            
            value_density['value_density'] = (
                value_density['market_value'] /
                value_density['property_area']
            )
            
            return {
                'value_metrics': value_metrics,
                'value_density': value_density
            }
            
        except Exception as e:
            logger.error(f"Error analyzing property values: {str(e)}")
            return {}
            
    def process_all(self) -> bool:
        """Run the complete zoning and property analysis pipeline."""
        try:
            logger.info("Starting zoning and property analysis...")
            
            # Process zoning data if available
            if settings.ZONING_RAW_PATH.exists():
                zoning_data = pd.read_csv(settings.ZONING_RAW_PATH)
                self.zoning_data = self.process_zoning_data(zoning_data)
                if not self.zoning_data.empty:
                    zoning_analysis = self.analyze_zoning_patterns(self.zoning_data)
                    
                    # Save analysis results
                    for name, df in zoning_analysis.items():
                        output_path = self.processed_data_dir / f'zoning_{name}.csv'
                        df.to_csv(output_path, index=False)
            else:
                logger.warning("Zoning data file not found - skipping")
            
            # Process property data if available
            if settings.PROPERTY_RAW_PATH.exists():
                property_data = pd.read_csv(settings.PROPERTY_RAW_PATH)
                self.property_data = self.process_property_data(property_data)
                if not self.property_data.empty:
                    property_analysis = self.analyze_property_values(self.property_data)
                    
                    # Save analysis results
                    for name, df in property_analysis.items():
                        output_path = self.processed_data_dir / f'property_{name}.csv'
                        df.to_csv(output_path, index=False)
            else:
                logger.warning("Property data file not found - skipping")
            
            logger.info("Zoning and property analysis completed")
            return True
        
            
        except Exception as e:
            logger.error(f"Error in zoning and property analysis: {str(e)}")
            return False 