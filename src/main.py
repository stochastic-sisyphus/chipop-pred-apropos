import logging
import os
import pandas as pd
import joblib
from pathlib import Path
from dotenv import load_dotenv
from data_collection import DataCollector
from data_processing import ChicagoDataProcessor
from modeling import PopulationShiftModel
from visualization import ChicagoDataVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    try:
        # Step 1: Collect Data
        logger.info("Starting data collection...")
        collector = DataCollector()
        
        # Check if data already exists, if not, collect it
        data_dir = Path(os.getenv('DATA_DIR', 'data'))
        historical_pop_path = data_dir / 'historical_population.csv'
        
        if not historical_pop_path.exists():
            logger.info("Historical population data not found, fetching from Census...")
            collector.get_census_data(start_year=2005)
        else:
            logger.info("Using existing historical population data")
        
        # Get other data sources if not already available
        if not (data_dir / 'building_permits.csv').exists():
            collector.get_building_permits()
        else:
            logger.info("Using existing building permits data")
            
        if not (data_dir / 'zoning.geojson').exists():
            collector.get_zoning_data()
        else:
            logger.info("Using existing zoning data")
            
        if not (data_dir / 'economic_indicators.csv').exists():
            collector.get_economic_indicators()
        else:
            logger.info("Using existing economic indicators data")
        
        # Step 2: Process Data
        logger.info("Starting data processing...")
        processor = ChicagoDataProcessor()
        merged_data = processor.merge_datasets()
        
        # Step 3: Train Model and Generate Predictions
        logger.info("Starting modeling...")
        model_trainer = PopulationShiftModel()
        model, original_df = model_trainer.train_model()
        
        # Generate scenarios
        predictions = model_trainer.generate_scenarios(model, original_df)
        
        # Step 4: Visualize results
        logger.info("Creating visualizations...")
        visualizer = ChicagoDataVisualizer()
        visualization_success = visualizer.generate_all_visualizations()
        
        if visualization_success:
            logger.info("Visualizations created successfully in the output directory")
        else:
            logger.warning("Some visualizations could not be created. Check logs for details.")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()