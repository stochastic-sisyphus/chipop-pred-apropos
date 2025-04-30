import logging
from pathlib import Path
from typing import Optional

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
            settings.VISUALIZATIONS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
    
    def run(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            bool: True if pipeline completed successfully, False otherwise.
        """
        try:
            # Step 1: Process all data
            logger.info("Starting data processing...")
            if not self.processor.process_all():
                logger.error("Data processing failed")
                return False
            logger.info("Data processing completed successfully")
            
            # Step 2: Generate reports
            logger.info("Generating reports...")
            if not self.report_generator.generate_report():
                logger.error("Report generation failed")
                return False
            logger.info("Reports generated successfully")
            
            # Step 3: Create visualizations
            logger.info("Creating visualizations...")
            if not self.visualizer.create_dashboard():
                logger.error("Visualization creation failed")
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