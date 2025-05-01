import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import settings
from src.data_processing.processor import DataProcessor
from src.reporting.ten_year_growth_report import TenYearGrowthReport
from src.visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)

class Pipeline:
    """Main pipeline runner for the Chicago Population Analysis."""
    
    def __init__(self):
        """Initialize the pipeline components."""
        self.processor = DataProcessor()
        self.report_generator = TenYearGrowthReport()
        self.visualizer = Visualizer()
        
        # Ensure all required directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all required directories if they don't exist."""
        directories = [
            settings.DATA_RAW_DIR,
            settings.DATA_INTERIM_DIR,
            settings.DATA_PROCESSED_DIR,
            settings.OUTPUT_DIR,
            settings.REPORTS_DIR,
            settings.VISUALIZATIONS_DIR,
            settings.MODELS_DIR,
            settings.ANALYSIS_RESULTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def _validate_processed_data(self) -> bool:
        """Validate processed data files and their contents."""
        try:
            # Check processed files exist
            required_processed = {
                'Census': settings.CENSUS_PROCESSED_PATH,
                'Permits': settings.PERMITS_PROCESSED_PATH,
                'Business Licenses': settings.BUSINESS_LICENSES_PROCESSED_PATH,
                'Merged Data': settings.MERGED_DATA_PATH,
                'Retail Metrics': settings.PROCESSED_DATA_DIR / 'retail_metrics.csv',
                'Retail Deficit': settings.PROCESSED_DATA_DIR / 'retail_deficit_processed.csv'
            }
            
            for name, path in required_processed.items():
                if not path.exists():
                    logger.error(f"Missing {name} processed file: {path}")
                    return False
                
                # Validate file contents
                try:
                    df = pd.read_csv(path)
                    if 'zip_code' not in df.columns:
                        logger.error(f"Missing zip_code column in {name} file")
                        return False
                except Exception as e:
                    logger.error(f"Error reading {name} file: {str(e)}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating processed data: {str(e)}")
            return False
    
    def run(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            bool: True if pipeline completed successfully, False otherwise.
        """
        try:
            # Step 1: Check required directories exist
            logger.info("Checking required directories...")
            self._create_directories()
            
            # Step 2: Check required input files exist
            required_files = {
                'Census': settings.CENSUS_DATA_PATH,
                'Permits': settings.PERMITS_DATA_PATH,
                'Business Licenses': settings.BUSINESS_LICENSES_PATH,
                'Economic': settings.ECONOMIC_DATA_PATH
            }
            
            missing_files = []
            for name, path in required_files.items():
                if not path.exists():
                    missing_files.append(f"{name}: {path}")
            
            if missing_files:
                logger.error("Missing required input files:")
                for missing in missing_files:
                    logger.error(f"  - {missing}")
                return False
            
            # Step 3: Process all data
            logger.info("Starting data processing...")
            if not self.processor.process_all():
                logger.error("Data processing failed")
                return False
            logger.info("Data processing completed successfully")
            
            # Step 4: Validate processed data
            logger.info("Validating processed data...")
            if not self._validate_processed_data():
                logger.error("Data validation failed")
                return False
            logger.info("Data validation completed successfully")
            
            # Step 5: Generate reports
            logger.info("Generating reports...")
            try:
                if not self.report_generator.generate_report():
                    logger.error("Report generation failed")
                    return False
            except Exception as e:
                logger.error(f"Error generating reports: {str(e)}")
                return False
            logger.info("Reports generated successfully")
            
            # Step 6: Create visualizations
            logger.info("Creating visualizations...")
            try:
                # Initialize visualizer with validated data
                self.visualizer = Visualizer(
                    population_data=pd.read_csv(settings.CENSUS_PROCESSED_PATH),
                    permit_data=pd.read_csv(settings.PERMITS_PROCESSED_PATH),
                    economic_data=pd.read_csv(settings.ECONOMIC_PROCESSED_PATH),
                    business_data=pd.read_csv(settings.BUSINESS_LICENSES_PROCESSED_PATH)
                )
                
                if not self.visualizer.create_all_visualizations():
                    logger.error("Visualization creation failed")
                    return False
            except Exception as e:
                logger.error(f"Error creating visualizations: {str(e)}")
                return False
            logger.info("Visualizations created successfully")
            
            logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False
    
    @staticmethod
    def get_output_path(filename: str) -> Path:
        """
        Get the full path for an output file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path: Full path to the output file
        """
        return settings.OUTPUT_DIR / filename

def main():
    """Main entry point for running the pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = Pipeline()
    success = pipeline.run()
    
    if success:
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed")
        exit(1)

if __name__ == "__main__":
    main() 