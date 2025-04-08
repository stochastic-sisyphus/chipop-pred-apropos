import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from data.data_collector import DataCollector
from models.population_shift_model import PopulationShiftModel
from visualization.visualizer import PopulationVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'output', 'output/visualizations']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def run_pipeline():
    """Run the complete analysis pipeline"""
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Step 1: Collect Data
        logger.info("Starting data collection...")
        collector = DataCollector()
        
        # Collect all required data
        collector.get_census_data()
        collector.get_building_permits()
        collector.get_zoning_data()
        collector.get_economic_indicators()
        logger.info("Data collection complete.")
        
        # Step 2: Run Population Shift Analysis
        logger.info("Starting population shift analysis...")
        model = PopulationShiftModel()
        if not model.run_analysis():
            logger.error("Population shift analysis failed.")
            return False
        logger.info("Population shift analysis complete.")
        
        # Step 3: Generate Visualizations
        logger.info("Generating visualizations...")
        visualizer = PopulationVisualizer()
        if not visualizer.generate_all_visualizations():
            logger.error("Visualization generation failed.")
            return False
        logger.info("Visualizations complete.")
        
        logger.info("Pipeline completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False

if __name__ == "__main__":
    if run_pipeline():
        logger.info("Analysis complete. Check the output directory for results.")
    else:
        logger.error("Analysis failed. Check the logs for details.")