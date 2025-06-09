"""
Data pipeline module for the Chicago Population Analysis project.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime

from src.data_processing.data_preparer import DataPreparer
from src.pipeline.pipeline import Pipeline
from src.config import settings

logger = logging.getLogger(__name__)

class ChicagoPipelineRunner:
    """Main runner for the Chicago Population Analysis pipeline."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the pipeline runner.
        
        Args:
            output_dir (Path, optional): Directory to save outputs
        """
        # Set output directory
        if output_dir is None:
            self.output_dir = settings.OUTPUT_DIR
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_preparer = DataPreparer()
        self.pipeline = Pipeline(output_dir=self.output_dir)
        
        # Initialize results
        self.results = {
            "start_time": datetime.now().isoformat(),
            "status": "initialized"
        }
    
    def run(self):
        """
        Run the full Chicago Population Analysis pipeline.
        
        Returns:
            bool: True if pipeline execution successful, False otherwise
        """
        try:
            logger.info("Starting Chicago Population Analysis pipeline runner...")
            
            # Load data
            data = self._load_data()
            if data is None:
                logger.error("Failed to load data")
                self.results["status"] = "failed"
                self.results["error"] = "Data loading failed"
                return False
            
            # Prepare data
            prepared_data = self._prepare_data(data)
            if prepared_data is None:
                logger.error("Failed to prepare data")
                self.results["status"] = "failed"
                self.results["error"] = "Data preparation failed"
                return False
            
            # Run pipeline
            success = self._run_pipeline(prepared_data)
            if not success:
                logger.error("Pipeline execution failed")
                self.results["status"] = "failed"
                self.results["error"] = "Pipeline execution failed"
                return False
            
            # Save results
            self._save_results()
            
            logger.info("Chicago Population Analysis pipeline runner completed successfully")
            self.results["status"] = "completed"
            return True
            
        except Exception as e:
            logger.error(f"Error running Chicago Population Analysis pipeline: {str(e)}")
            self.results["status"] = "failed"
            self.results["error"] = str(e)
            return False
    
    def _load_data(self):
        """
        Load data for analysis.
        
        Returns:
            pd.DataFrame: Loaded data or None if data cannot be loaded
        """
        try:
            # Check if data directory exists
            if not settings.DATA_DIR.exists():
                logger.error(f"Data directory not found: {settings.DATA_DIR}")
                return None
            
            # Look for CSV files in data directory
            csv_files = list(settings.DATA_DIR.glob("*.csv"))
            if not csv_files:
                logger.error("No CSV files found in data directory")
                return None
            
            # Load first CSV file
            data_path = csv_files[0]
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            if data.empty:
                logger.error(f"Data file {data_path} is empty")
                return None
                
            logger.info(f"Successfully loaded {len(data)} records from {data_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def _prepare_data(self, data):
        """
        Prepare data for analysis.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Prepared data or None if preparation fails
        """
        try:
            logger.info("Preparing data for analysis...")
            
            # Use data preparer to ensure all required columns are present
            prepared_data = self.data_preparer.prepare_data(data)
            
            logger.info(f"Successfully prepared {len(prepared_data)} records for analysis")
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None
    
    def _run_pipeline(self, data):
        """
        Run the pipeline with prepared data.
        
        Args:
            data (pd.DataFrame): Prepared data
            
        Returns:
            bool: True if pipeline execution successful, False otherwise
        """
        try:
            logger.info("Running pipeline with prepared data...")
            
            # Run pipeline with prepared data
            success = self.pipeline.run_with_data(data)
            
            if success:
                logger.info("Pipeline execution completed successfully")
                self.results["pipeline_results"] = self.pipeline.results
            else:
                logger.error("Pipeline execution failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            return False
    
    def _save_results(self):
        """
        Save runner results.
        """
        try:
            # Add end time
            self.results["end_time"] = datetime.now().isoformat()
            
            # Calculate duration
            start_time = datetime.fromisoformat(self.results["start_time"])
            end_time = datetime.fromisoformat(self.results["end_time"])
            duration = (end_time - start_time).total_seconds()
            self.results["duration_seconds"] = duration
            
            # Save results to JSON
            results_path = self.output_dir / "runner_summary.json"
            
            # Convert all values to JSON-serializable types
            def json_serialize(obj):
                if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                if isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (Path, os.PathLike)):
                    return str(obj)
                return obj
            
            # Recursively convert all values
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return json_serialize(obj)
            
            # Convert results to serializable format
            serializable_results = convert_to_serializable(self.results)
            
            # Save to JSON
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Runner results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving runner results: {str(e)}")
