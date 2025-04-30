import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

class ZoningProcessor:
    """Processes zoning and property data for Chicago neighborhoods."""
    
    def __init__(self):
        """Initialize the ZoningProcessor."""
        self.zoning_data: Optional[pd.DataFrame] = None
        self.property_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> Tuple[bool, bool]:
        """Load zoning and property data from files.
        
        Returns:
            Tuple[bool, bool]: Status of (zoning_loaded, property_loaded)
        """
        zoning_loaded = False
        property_loaded = False
        
        try:
            self.zoning_data = pd.read_csv(settings.ZONING_DATA_PATH)
            logger.info(f"Loaded zoning data from {settings.ZONING_DATA_PATH}")
            zoning_loaded = True
        except FileNotFoundError:
            logger.error(f"Zoning data file not found at {settings.ZONING_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error loading zoning data: {str(e)}")
            
        try:
            self.property_data = pd.read_csv(settings.PROPERTY_DATA_PATH)
            logger.info(f"Loaded property data from {settings.PROPERTY_DATA_PATH}")
            property_loaded = True
        except FileNotFoundError:
            logger.error(f"Property data file not found at {settings.PROPERTY_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error loading property data: {str(e)}")
            
        return zoning_loaded, property_loaded
    
    def clean_zoning_data(self) -> bool:
        """Clean and preprocess zoning data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.zoning_data is None:
            logger.error("No zoning data loaded to clean")
            return False
            
        try:
            # Clean column names
            self.zoning_data.columns = self.zoning_data.columns.str.lower().str.replace(' ', '_')
            
            # Remove duplicates
            self.zoning_data = self.zoning_data.drop_duplicates()
            
            # Aggregate metrics by zip code
            agg_dict = {
                'zone_type': lambda x: x.mode().iloc[0] if not x.empty else None,
                'lot_area': 'sum',
                'floor_area_ratio': 'mean',
                'height_limit': 'mean'
            }
            
            self.zoning_data = self.zoning_data.groupby('zip_code').agg(agg_dict).reset_index()
            
            logger.info("Successfully cleaned zoning data")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning zoning data: {str(e)}")
            return False
    
    def clean_property_data(self) -> bool:
        """Clean and preprocess property data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.property_data is None:
            logger.error("No property data loaded to clean")
            return False
            
        try:
            # Clean column names
            self.property_data.columns = self.property_data.columns.str.lower().str.replace(' ', '_')
            
            # Remove duplicates
            self.property_data = self.property_data.drop_duplicates()
            
            # Aggregate metrics by zip code
            agg_dict = {
                'building_area': 'sum',
                'land_value': 'mean',
                'improvement_value': 'mean',
                'year_built': 'mean'
            }
            
            self.property_data = self.property_data.groupby('zip_code').agg(agg_dict).reset_index()
            
            logger.info("Successfully cleaned property data")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning property data: {str(e)}")
            return False
    
    def merge_data(self) -> bool:
        """Merge zoning and property data and calculate derived metrics.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.zoning_data is None or self.property_data is None:
            logger.error("Missing data for merge operation")
            return False
            
        try:
            # Merge on zip code
            self.merged_data = pd.merge(
                self.zoning_data,
                self.property_data,
                on='zip_code',
                how='outer'
            )
            
            # Calculate derived metrics
            self.merged_data['building_density'] = (
                self.merged_data['building_area'] / self.merged_data['lot_area']
            )
            
            self.merged_data['zoning_utilization'] = (
                self.merged_data['building_density'] / self.merged_data['floor_area_ratio']
            )
            
            # Fill missing values with appropriate defaults
            self.merged_data = self.merged_data.fillna({
                'building_density': 0,
                'zoning_utilization': 0
            })
            
            logger.info("Successfully merged zoning and property data")
            return True
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return False
    
    def save_processed_data(self) -> bool:
        """Save processed data to files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save individual datasets
            if self.zoning_data is not None:
                self.zoning_data.to_csv(settings.ZONING_PROCESSED_PATH, index=False)
                logger.info(f"Saved processed zoning data to {settings.ZONING_PROCESSED_PATH}")
                
            if self.property_data is not None:
                self.property_data.to_csv(settings.PROPERTY_PROCESSED_PATH, index=False)
                logger.info(f"Saved processed property data to {settings.PROPERTY_PROCESSED_PATH}")
                
            if self.merged_data is not None:
                self.merged_data.to_csv(settings.ZONING_PROPERTY_MERGED_PATH, index=False)
                logger.info(f"Saved merged data to {settings.ZONING_PROPERTY_MERGED_PATH}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            return False
    
    def process_all(self) -> Dict[str, bool]:
        """Run the complete zoning and property data processing pipeline.
        
        Returns:
            Dict[str, bool]: Status of each processing step
        """
        results = {}
        
        # Load data
        zoning_loaded, property_loaded = self.load_data()
        results['load_zoning'] = zoning_loaded
        results['load_property'] = property_loaded
        
        # Clean data if loaded successfully
        if zoning_loaded:
            results['clean_zoning'] = self.clean_zoning_data()
        if property_loaded:
            results['clean_property'] = self.clean_property_data()
            
        # Merge data if both datasets are available
        if zoning_loaded and property_loaded:
            results['merge'] = self.merge_data()
            
        # Save processed data
        results['save'] = self.save_processed_data()
        
        # Log overall status
        success = all(results.values())
        if success:
            logger.info("Zoning processor completed all steps successfully")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.error(f"Failed steps in zoning processor: {', '.join(failed)}")
        
        return results 