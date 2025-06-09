"""
Modified validation script for Chicago Housing Pipeline.

This script validates the entire pipeline against project objectives and quality standards,
with flexibility for environments where certain dependencies may not be available.
"""

import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PipelineValidator:
    """
    Validator for Chicago Housing Pipeline.
    
    Validates all pipeline components and outputs against project objectives.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the pipeline validator.
        
        Args:
            output_dir (Path, optional): Directory to save validation results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("validation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validation results
        self.validation_results = {
            "data_collection": {},
            "data_processing": {},
            "models": {},
            "visualizations": {},
            "reports": {},
            "pipeline": {},
            "overall": {
                "status": "Not Started",
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": None,
                "duration": None
            }
        }
        
        # Check available dependencies
        self.available_dependencies = self._check_dependencies()
    
    def _check_dependencies(self):
        """
        Check which dependencies are available.
        
        Returns:
            dict: Dictionary of available dependencies
        """
        dependencies = {
            "pdfkit": False,
            "wkhtmltopdf": False,
            "matplotlib": False,
            "seaborn": False,
            "pandas": False,
            "numpy": False
        }
        
        # Check Python packages
        try:
            import pdfkit
            dependencies["pdfkit"] = True
        except ImportError:
            logger.warning("pdfkit not available. PDF report validation will be limited.")
        
        try:
            import matplotlib
            dependencies["matplotlib"] = True
        except ImportError:
            logger.warning("matplotlib not available. Visualization validation will be limited.")
        
        try:
            import seaborn
            dependencies["seaborn"] = True
        except ImportError:
            logger.warning("seaborn not available. Advanced visualization validation will be limited.")
        
        try:
            import pandas
            dependencies["pandas"] = True
        except ImportError:
            logger.warning("pandas not available. Data processing validation will be limited.")
        
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            logger.warning("numpy not available. Numerical analysis validation will be limited.")
        
        # Check system dependencies
        try:
            import subprocess
            result = subprocess.run(['which', 'wkhtmltopdf'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                dependencies["wkhtmltopdf"] = True
            else:
                logger.warning("wkhtmltopdf not available. PDF report generation will be limited.")
        except Exception:
            logger.warning("Could not check for wkhtmltopdf. PDF report generation will be limited.")
        
        return dependencies
    
    def validate_all(self):
        """
        Validate all pipeline components.
        
        Returns:
            dict: Validation results
        """
        try:
            logger.info("Starting pipeline validation")
            start_time = time.time()
            
            # Import required modules
            try:
                from src.config import settings
                from src.data_collection.collector import DataCollector
                from src.data_processing.processor import DataProcessor
                from src.models.multifamily_growth_model import MultifamilyGrowthModel
                from src.models.retail_gap_model import RetailGapModel
                from src.models.retail_void_model import RetailVoidModel
                from src.visualization.visualization_generator import VisualizationGenerator
                
                # Conditionally import report generator if pdfkit is available
                if self.available_dependencies["pdfkit"]:
                    try:
                        from src.reports.report_generator import ReportGenerator
                    except ImportError:
                        logger.warning("Could not import ReportGenerator. Report validation will be skipped.")
            except ImportError as e:
                logger.error(f"Error importing required modules: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.validation_results["overall"]["status"] = "Failed"
                self.validation_results["overall"]["error"] = f"Error importing required modules: {str(e)}"
                self.validation_results["overall"]["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                return self.validation_results
            
            # Validate data collection
            self.validate_data_collection()
            
            # Validate data processing
            self.validate_data_processing()
            
            # Validate models
            self.validate_models()
            
            # Validate visualizations
            self.validate_visualizations()
            
            # Validate reports if pdfkit is available
            if self.available_dependencies["pdfkit"]:
                self.validate_reports()
            else:
                logger.warning("Skipping report validation due to missing dependencies.")
                self.validation_results["reports"] = {
                    "status": "Skipped",
                    "message": "Report validation skipped due to missing dependencies.",
                    "components": {},
                    "passed": 0,
                    "failed": 0,
                    "warnings": 0
                }
            
            # Validate full pipeline
            self.validate_pipeline()
            
            # Calculate overall results
            self._calculate_overall_results()
            
            # Save validation results
            self._save_validation_results()
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.validation_results["overall"]["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.validation_results["overall"]["duration"] = f"{duration:.2f} seconds"
            
            logger.info(f"Pipeline validation completed in {duration:.2f} seconds")
            logger.info(f"Passed: {self.validation_results['overall']['passed']}, Failed: {self.validation_results['overall']['failed']}, Warnings: {self.validation_results['overall']['warnings']}")
            
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Error during pipeline validation: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.validation_results["overall"]["status"] = "Failed"
            self.validation_results["overall"]["error"] = str(e)
            self.validation_results["overall"]["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return self.validation_results
    
    def validate_data_collection(self):
        """
        Validate data collection components.
        """
        try:
            logger.info("Validating data collection components")
            
            # Initialize results
            self.validation_results["data_collection"] = {
                "status": "In Progress",
                "components": {},
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
            
            # Import required modules
            try:
                from src.data_collection.collector import DataCollector
                
                # Try to import specialized collectors if available
                try:
                    from src.data_collection.census_collector import CensusCollector
                    self._validate_component(
                        "CensusCollector",
                        self._validate_census_collector,
                        self.validation_results["data_collection"]["components"]
                    )
                except ImportError:
                    logger.warning("CensusCollector not available. Skipping validation.")
                
                try:
                    from src.data_collection.fred_collector import FredCollector
                    self._validate_component(
                        "FredCollector",
                        self._validate_fred_collector,
                        self.validation_results["data_collection"]["components"]
                    )
                except ImportError:
                    logger.warning("FredCollector not available. Skipping validation.")
                
                try:
                    from src.data_collection.chicago_collector import ChicagoCollector
                    self._validate_component(
                        "ChicagoCollector",
                        self._validate_chicago_collector,
                        self.validation_results["data_collection"]["components"]
                    )
                except ImportError:
                    logger.warning("ChicagoCollector not available. Skipping validation.")
                
                try:
                    from src.data_collection.bea_collector import BEACollector
                    self._validate_component(
                        "BEACollector",
                        self._validate_bea_collector,
                        self.validation_results["data_collection"]["components"]
                    )
                except ImportError:
                    logger.warning("BEACollector not available. Skipping validation.")
                
                # Validate DataCollector
                self._validate_component(
                    "DataCollector",
                    self._validate_data_collector,
                    self.validation_results["data_collection"]["components"]
                )
                
            except ImportError as e:
                logger.error(f"Error importing data collection modules: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.validation_results["data_collection"]["status"] = "Failed"
                self.validation_results["data_collection"]["error"] = f"Error importing data collection modules: {str(e)}"
                return
            
            # Calculate component results
            self._calculate_component_results(self.validation_results["data_collection"])
            
            logger.info(f"Data collection validation completed: {self.validation_results['data_collection']['status']}")
            
        except Exception as e:
            logger.error(f"Error validating data collection: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.validation_results["data_collection"]["status"] = "Failed"
            self.validation_results["data_collection"]["error"] = str(e)
    
    def _validate_data_collector(self):
        """
        Validate DataCollector class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.data_collection.collector import DataCollector
            
            # Test initialization
            collector = DataCollector()
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test collect_data method
            try:
                data = collector.collect_data(use_sample=True)
                if data is not None and isinstance(data, pd.DataFrame):
                    results["tests"]["collect_data"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["collect_data"] = {
                        "status": "Warning",
                        "message": "collect_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["collect_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test sample data loading
            try:
                sample_data = collector.load_sample_data()
                if sample_data is not None and isinstance(sample_data, pd.DataFrame):
                    results["tests"]["load_sample_data"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["load_sample_data"] = {
                        "status": "Warning",
                        "message": "load_sample_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["load_sample_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating DataCollector: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_census_collector(self):
        """
        Validate CensusCollector class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.data_collection.census_collector import CensusCollector
            
            # Test initialization
            try:
                collector = CensusCollector()
                results["tests"]["initialization"] = {"status": "Passed"}
                results["passed"] += 1
            except Exception as e:
                results["tests"]["initialization"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
                results["status"] = "Failed"
                return results
            
            # Test collect_data method
            try:
                # Use sample mode to avoid API calls
                data = collector.collect_data(use_sample=True)
                if data is not None and isinstance(data, pd.DataFrame):
                    results["tests"]["collect_data"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["collect_data"] = {
                        "status": "Warning",
                        "message": "collect_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["collect_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating CensusCollector: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_fred_collector(self):
        """
        Validate FredCollector class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.data_collection.fred_collector import FredCollector
            
            # Test initialization
            try:
                collector = FredCollector()
                results["tests"]["initialization"] = {"status": "Passed"}
                results["passed"] += 1
            except Exception as e:
                results["tests"]["initialization"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
                results["status"] = "Failed"
                return results
            
            # Test collect_data method
            try:
                # Use sample mode to avoid API calls
                data = collector.collect_data(use_sample=True)
                if data is not None and isinstance(data, pd.DataFrame):
                    results["tests"]["collect_data"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["collect_data"] = {
                        "status": "Warning",
                        "message": "collect_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["collect_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating FredCollector: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_chicago_collector(self):
        """
        Validate ChicagoCollector class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.data_collection.chicago_collector import ChicagoCollector
            
            # Test initialization
            try:
                collector = ChicagoCollector()
                results["tests"]["initialization"] = {"status": "Passed"}
                results["passed"] += 1
            except Exception as e:
                results["tests"]["initialization"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
                results["status"] = "Failed"
                return results
            
            # Test collect_data method
            try:
                # Use sample mode to avoid API calls
                data = collector.collect_data(use_sample=True)
                if data is not None and isinstance(data, pd.DataFrame):
                    results["tests"]["collect_data"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["collect_data"] = {
                        "status": "Warning",
                        "message": "collect_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["collect_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating ChicagoCollector: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_bea_collector(self):
        """
        Validate BEACollector class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.data_collection.bea_collector import BEACollector
            
            # Test initialization
            try:
                collector = BEACollector()
                results["tests"]["initialization"] = {"status": "Passed"}
                results["passed"] += 1
            except Exception as e:
                results["tests"]["initialization"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
                results["status"] = "Failed"
                return results
            
            # Test collect_data method
            try:
                # Use sample mode to avoid API calls
                data = collector.collect_data(use_sample=True)
                if data is not None and isinstance(data, pd.DataFrame):
                    results["tests"]["collect_data"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["collect_data"] = {
                        "status": "Warning",
                        "message": "collect_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["collect_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating BEACollector: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def validate_data_processing(self):
        """
        Validate data processing components.
        """
        try:
            logger.info("Validating data processing components")
            
            # Initialize results
            self.validation_results["data_processing"] = {
                "status": "In Progress",
                "components": {},
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
            
            # Import required modules
            try:
                from src.data_processing.processor import DataProcessor
                
                # Try to import specialized processors if available
                try:
                    from src.data_processing.data_cleaner import DataCleaner
                    self._validate_component(
                        "DataCleaner",
                        self._validate_data_cleaner,
                        self.validation_results["data_processing"]["components"]
                    )
                except ImportError:
                    logger.warning("DataCleaner not available. Skipping validation.")
                
                # Validate DataProcessor
                self._validate_component(
                    "DataProcessor",
                    self._validate_data_processor,
                    self.validation_results["data_processing"]["components"]
                )
                
            except ImportError as e:
                logger.error(f"Error importing data processing modules: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.validation_results["data_processing"]["status"] = "Failed"
                self.validation_results["data_processing"]["error"] = f"Error importing data processing modules: {str(e)}"
                return
            
            # Calculate component results
            self._calculate_component_results(self.validation_results["data_processing"])
            
            logger.info(f"Data processing validation completed: {self.validation_results['data_processing']['status']}")
            
        except Exception as e:
            logger.error(f"Error validating data processing: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.validation_results["data_processing"]["status"] = "Failed"
            self.validation_results["data_processing"]["error"] = str(e)
    
    def _validate_data_processor(self):
        """
        Validate DataProcessor class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.data_processing.processor import DataProcessor
            
            # Test initialization
            processor = DataProcessor()
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test process_data method
            try:
                # Create sample data
                sample_data = pd.DataFrame({
                    'zip_code': [60601, 60602, 60603, 60604, 60605],
                    'population': [1000, 2000, 3000, 4000, 5000],
                    'year': [2020, 2020, 2020, 2020, 2020]
                })
                
                processed_data = processor.process_data(sample_data)
                if processed_data is not None and isinstance(processed_data, pd.DataFrame):
                    results["tests"]["process_data"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["process_data"] = {
                        "status": "Warning",
                        "message": "process_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["process_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating DataProcessor: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_data_cleaner(self):
        """
        Validate DataCleaner class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.data_processing.data_cleaner import DataCleaner
            
            # Test initialization
            cleaner = DataCleaner()
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test clean_data method
            try:
                # Create sample data with issues
                sample_data = pd.DataFrame({
                    'zip_code': [60601, 60602, None, 60604, 'invalid'],
                    'population': [1000, -1, 3000, 4000, 5000],
                    'year': [2020, 2020, 2020, 2020, 2020]
                })
                
                cleaned_data = cleaner.clean_data(sample_data)
                if cleaned_data is not None and isinstance(cleaned_data, pd.DataFrame):
                    # Check if issues were fixed
                    if cleaned_data['zip_code'].isna().sum() == 0 and (cleaned_data['population'] >= 0).all():
                        results["tests"]["clean_data"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["clean_data"] = {
                            "status": "Warning",
                            "message": "clean_data did not fix all issues"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["clean_data"] = {
                        "status": "Warning",
                        "message": "clean_data returned None or non-DataFrame"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["clean_data"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating DataCleaner: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def validate_models(self):
        """
        Validate model components.
        """
        try:
            logger.info("Validating model components")
            
            # Initialize results
            self.validation_results["models"] = {
                "status": "In Progress",
                "components": {},
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
            
            # Import required modules
            try:
                from src.models.multifamily_growth_model import MultifamilyGrowthModel
                from src.models.retail_gap_model import RetailGapModel
                from src.models.retail_void_model import RetailVoidModel
                
                # Try to import specialized models if available
                try:
                    from src.models.population_forecast_model import PopulationForecastModel
                    self._validate_component(
                        "PopulationForecastModel",
                        self._validate_population_forecast_model,
                        self.validation_results["models"]["components"]
                    )
                except ImportError:
                    logger.warning("PopulationForecastModel not available. Skipping validation.")
                
                # Validate MultifamilyGrowthModel
                self._validate_component(
                    "MultifamilyGrowthModel",
                    self._validate_multifamily_growth_model,
                    self.validation_results["models"]["components"]
                )
                
                # Validate RetailGapModel
                self._validate_component(
                    "RetailGapModel",
                    self._validate_retail_gap_model,
                    self.validation_results["models"]["components"]
                )
                
                # Validate RetailVoidModel
                self._validate_component(
                    "RetailVoidModel",
                    self._validate_retail_void_model,
                    self.validation_results["models"]["components"]
                )
                
            except ImportError as e:
                logger.error(f"Error importing model modules: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.validation_results["models"]["status"] = "Failed"
                self.validation_results["models"]["error"] = f"Error importing model modules: {str(e)}"
                return
            
            # Calculate component results
            self._calculate_component_results(self.validation_results["models"])
            
            logger.info(f"Model validation completed: {self.validation_results['models']['status']}")
            
        except Exception as e:
            logger.error(f"Error validating models: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.validation_results["models"]["status"] = "Failed"
            self.validation_results["models"]["error"] = str(e)
    
    def _validate_multifamily_growth_model(self):
        """
        Validate MultifamilyGrowthModel class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.models.multifamily_growth_model import MultifamilyGrowthModel
            
            # Test initialization
            model = MultifamilyGrowthModel()
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test run method
            try:
                # Create sample data
                sample_data = pd.DataFrame({
                    'zip_code': [60601, 60602, 60603, 60604, 60605],
                    'population': [1000, 2000, 3000, 4000, 5000],
                    'year': [2020, 2020, 2020, 2020, 2020],
                    'housing_units': [500, 1000, 1500, 2000, 2500],
                    'building_permits': [50, 100, 150, 200, 250]
                })
                
                model_results = model.run(sample_data)
                if model_results is not None and isinstance(model_results, dict):
                    # Check for required keys
                    required_keys = ['top_growth_zips', 'summary']
                    if all(key in model_results for key in required_keys):
                        results["tests"]["run"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["run"] = {
                            "status": "Warning",
                            "message": f"run results missing required keys: {[key for key in required_keys if key not in model_results]}"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["run"] = {
                        "status": "Warning",
                        "message": "run returned None or non-dict"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["run"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test _save_results method
            try:
                output_dir = self.output_dir / "models" / "multifamily_growth"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample results
                sample_results = {
                    'top_growth_zips': [
                        {'zip_code': 60605, 'population_growth_rate': 5.0},
                        {'zip_code': 60604, 'population_growth_rate': 4.0}
                    ],
                    'summary': 'Sample summary'
                }
                
                model._save_results(sample_results, output_dir)
                
                # Check if results file exists
                results_file = output_dir / "results.json"
                if results_file.exists():
                    results["tests"]["_save_results"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["_save_results"] = {
                        "status": "Warning",
                        "message": "results file not created"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["_save_results"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating MultifamilyGrowthModel: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_retail_gap_model(self):
        """
        Validate RetailGapModel class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.models.retail_gap_model import RetailGapModel
            
            # Test initialization
            model = RetailGapModel()
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test run method
            try:
                # Create sample data
                sample_data = pd.DataFrame({
                    'zip_code': [60601, 60602, 60603, 60604, 60605],
                    'population': [1000, 2000, 3000, 4000, 5000],
                    'retail_establishments': [10, 20, 30, 40, 50],
                    'income': [50000, 60000, 70000, 80000, 90000]
                })
                
                model_results = model.run(sample_data)
                if model_results is not None and isinstance(model_results, dict):
                    # Check for required keys
                    required_keys = ['opportunity_zones', 'summary']
                    if all(key in model_results for key in required_keys):
                        results["tests"]["run"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["run"] = {
                            "status": "Warning",
                            "message": f"run results missing required keys: {[key for key in required_keys if key not in model_results]}"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["run"] = {
                        "status": "Warning",
                        "message": "run returned None or non-dict"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["run"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test _save_results method
            try:
                output_dir = self.output_dir / "models" / "retail_gap"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample results
                sample_results = {
                    'opportunity_zones': [
                        {'zip_code': 60601, 'retail_gap_score': -0.5},
                        {'zip_code': 60602, 'retail_gap_score': -0.4}
                    ],
                    'summary': 'Sample summary'
                }
                
                model._save_results(sample_results, output_dir)
                
                # Check if results file exists
                results_file = output_dir / "results.json"
                if results_file.exists():
                    results["tests"]["_save_results"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["_save_results"] = {
                        "status": "Warning",
                        "message": "results file not created"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["_save_results"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating RetailGapModel: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_retail_void_model(self):
        """
        Validate RetailVoidModel class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.models.retail_void_model import RetailVoidModel
            
            # Test initialization
            model = RetailVoidModel()
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test run method
            try:
                # Create sample data
                sample_data = pd.DataFrame({
                    'zip_code': [60601, 60602, 60603, 60604, 60605],
                    'population': [1000, 2000, 3000, 4000, 5000],
                    'retail_establishments': [10, 20, 30, 40, 50],
                    'income': [50000, 60000, 70000, 80000, 90000],
                    'spending': [40000, 50000, 60000, 70000, 80000]
                })
                
                model_results = model.run(sample_data)
                if model_results is not None and isinstance(model_results, dict):
                    # Check for required keys
                    required_keys = ['void_zones', 'summary']
                    if all(key in model_results for key in required_keys):
                        results["tests"]["run"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["run"] = {
                            "status": "Warning",
                            "message": f"run results missing required keys: {[key for key in required_keys if key not in model_results]}"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["run"] = {
                        "status": "Warning",
                        "message": "run returned None or non-dict"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["run"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test _save_results method
            try:
                output_dir = self.output_dir / "models" / "retail_void"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample results
                sample_results = {
                    'void_zones': [
                        {'zip_code': 60601, 'leakage_ratio': 0.5},
                        {'zip_code': 60602, 'leakage_ratio': 0.4}
                    ],
                    'summary': 'Sample summary'
                }
                
                model._save_results(sample_results, output_dir)
                
                # Check if results file exists
                results_file = output_dir / "results.json"
                if results_file.exists():
                    results["tests"]["_save_results"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["_save_results"] = {
                        "status": "Warning",
                        "message": "results file not created"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["_save_results"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating RetailVoidModel: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_population_forecast_model(self):
        """
        Validate PopulationForecastModel class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.models.population_forecast_model import PopulationForecastModel
            
            # Test initialization
            model = PopulationForecastModel()
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test run method
            try:
                # Create sample data with multiple years
                years = [2015, 2016, 2017, 2018, 2019, 2020]
                zip_codes = [60601, 60602, 60603]
                
                data = []
                for year in years:
                    for zip_code in zip_codes:
                        # Create some growth pattern
                        base_pop = 1000 * (zip_codes.index(zip_code) + 1)
                        growth_factor = 1.02 ** (years.index(year))
                        population = int(base_pop * growth_factor)
                        
                        data.append({
                            'zip_code': zip_code,
                            'year': year,
                            'population': population
                        })
                
                sample_data = pd.DataFrame(data)
                
                model_results = model.run(sample_data)
                if model_results is not None and isinstance(model_results, dict):
                    # Check for required keys
                    required_keys = ['forecast_data', 'summary']
                    if all(key in model_results for key in required_keys):
                        results["tests"]["run"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["run"] = {
                            "status": "Warning",
                            "message": f"run results missing required keys: {[key for key in required_keys if key not in model_results]}"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["run"] = {
                        "status": "Warning",
                        "message": "run returned None or non-dict"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["run"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test _save_results method
            try:
                output_dir = self.output_dir / "models" / "population_forecast"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create sample results
                forecast_data = pd.DataFrame({
                    'zip_code': [60601, 60602, 60603],
                    'year': [2025, 2025, 2025],
                    'population': [1500, 2500, 3500],
                    'forecast_type': ['combined', 'combined', 'combined']
                })
                
                sample_results = {
                    'forecast_data': forecast_data,
                    'summary': 'Sample summary'
                }
                
                model._save_results(sample_results, output_dir)
                
                # Check if results file exists
                results_file = output_dir / "results.json"
                if results_file.exists():
                    results["tests"]["_save_results"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["_save_results"] = {
                        "status": "Warning",
                        "message": "results file not created"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["_save_results"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating PopulationForecastModel: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def validate_visualizations(self):
        """
        Validate visualization components.
        """
        try:
            logger.info("Validating visualization components")
            
            # Initialize results
            self.validation_results["visualizations"] = {
                "status": "In Progress",
                "components": {},
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
            
            # Import required modules
            try:
                from src.visualization.visualization_generator import VisualizationGenerator
                
                # Validate VisualizationGenerator
                self._validate_component(
                    "VisualizationGenerator",
                    self._validate_visualization_generator,
                    self.validation_results["visualizations"]["components"]
                )
                
            except ImportError as e:
                logger.error(f"Error importing visualization modules: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.validation_results["visualizations"]["status"] = "Failed"
                self.validation_results["visualizations"]["error"] = f"Error importing visualization modules: {str(e)}"
                return
            
            # Calculate component results
            self._calculate_component_results(self.validation_results["visualizations"])
            
            logger.info(f"Visualization validation completed: {self.validation_results['visualizations']['status']}")
            
        except Exception as e:
            logger.error(f"Error validating visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.validation_results["visualizations"]["status"] = "Failed"
            self.validation_results["visualizations"]["error"] = str(e)
    
    def _validate_visualization_generator(self):
        """
        Validate VisualizationGenerator class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.visualization.visualization_generator import VisualizationGenerator
            
            # Test initialization
            output_dir = self.output_dir / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            viz_generator = VisualizationGenerator(output_dir=output_dir)
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Skip visualization tests if matplotlib is not available
            if not self.available_dependencies["matplotlib"]:
                results["tests"]["visualization_generation"] = {
                    "status": "Skipped",
                    "message": "Visualization tests skipped due to missing matplotlib dependency."
                }
                results["status"] = "Warning"
                return results
            
            # Test population forecast visualizations
            try:
                # Create sample forecast data
                years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
                zip_codes = [60601, 60602, 60603]
                
                data = []
                for year in years:
                    for zip_code in zip_codes:
                        # Create some growth pattern
                        base_pop = 1000 * (zip_codes.index(zip_code) + 1)
                        growth_factor = 1.02 ** (years.index(year))
                        population = int(base_pop * growth_factor)
                        
                        # Historical vs forecast
                        forecast_type = 'historical' if year <= 2020 else 'combined'
                        
                        data.append({
                            'zip_code': zip_code,
                            'year': year,
                            'population': population,
                            'forecast_type': forecast_type
                        })
                
                forecast_data = pd.DataFrame(data)
                
                # Create sample model results
                model_results = {
                    'top_growth_zips': [
                        {'zip_code': 60603, 'population_growth_rate': 2.5, 'current_population': 3000, 'forecast_population': 3500},
                        {'zip_code': 60602, 'population_growth_rate': 2.0, 'current_population': 2000, 'forecast_population': 2300},
                        {'zip_code': 60601, 'population_growth_rate': 1.5, 'current_population': 1000, 'forecast_population': 1100}
                    ],
                    'summary': 'Sample summary'
                }
                
                viz_paths = viz_generator.generate_population_forecast_visualizations(model_results, forecast_data)
                
                if viz_paths and isinstance(viz_paths, dict) and len(viz_paths) > 0:
                    # Check if at least one visualization was created
                    results["tests"]["generate_population_forecast_visualizations"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["generate_population_forecast_visualizations"] = {
                        "status": "Warning",
                        "message": "No visualizations generated"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["generate_population_forecast_visualizations"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test retail gap visualizations
            try:
                # Create sample gap data
                zip_codes = [60601, 60602, 60603, 60604, 60605]
                
                gap_data = pd.DataFrame({
                    'zip_code': zip_codes,
                    'retail_gap_score': [-0.8, -0.4, 0.0, 0.4, 0.8],
                    'retail_per_capita': [0.01, 0.02, 0.03, 0.04, 0.05],
                    'predicted_retail_per_capita': [0.02, 0.03, 0.03, 0.03, 0.04],
                    'population': [1000, 2000, 3000, 4000, 5000]
                })
                
                # Create sample model results
                model_results = {
                    'opportunity_zones': [
                        {'zip_code': 60601, 'retail_gap_score': -0.8, 'retail_per_capita': 0.01, 'predicted_retail_per_capita': 0.02},
                        {'zip_code': 60602, 'retail_gap_score': -0.4, 'retail_per_capita': 0.02, 'predicted_retail_per_capita': 0.03}
                    ],
                    'retail_clusters': [
                        {'retail_cluster': 0, 'zip_count': 2, 'retail_per_capita': 0.015, 'population': 1500},
                        {'retail_cluster': 1, 'zip_count': 1, 'retail_per_capita': 0.03, 'population': 3000},
                        {'retail_cluster': 2, 'zip_count': 2, 'retail_per_capita': 0.045, 'population': 4500}
                    ],
                    'category_gaps': {
                        'food': [{'zip_code': 60601, 'gap_size': 0.5}],
                        'clothing': [{'zip_code': 60602, 'gap_size': 0.4}]
                    },
                    'summary': 'Sample summary'
                }
                
                viz_paths = viz_generator.generate_retail_gap_visualizations(model_results, gap_data)
                
                if viz_paths and isinstance(viz_paths, dict) and len(viz_paths) > 0:
                    # Check if at least one visualization was created
                    results["tests"]["generate_retail_gap_visualizations"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["generate_retail_gap_visualizations"] = {
                        "status": "Warning",
                        "message": "No visualizations generated"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["generate_retail_gap_visualizations"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test retail void visualizations
            try:
                # Create sample void data
                zip_codes = [60601, 60602, 60603, 60604, 60605]
                
                void_data = pd.DataFrame({
                    'zip_code': zip_codes,
                    'leakage_ratio': [0.5, 0.3, 0.0, -0.3, -0.5],
                    'is_retail_void': [True, True, False, False, False],
                    'retail_per_capita': [0.01, 0.02, 0.03, 0.04, 0.05],
                    'population': [1000, 2000, 3000, 4000, 5000]
                })
                
                # Create sample model results
                model_results = {
                    'void_zones': [
                        {'zip_code': 60601, 'leakage_ratio': 0.5, 'retail_per_capita': 0.01, 'population': 1000},
                        {'zip_code': 60602, 'leakage_ratio': 0.3, 'retail_per_capita': 0.02, 'population': 2000}
                    ],
                    'category_voids': {
                        'food': [{'zip_code': 60601, 'void_size': 0.5}],
                        'clothing': [{'zip_code': 60602, 'void_size': 0.4}]
                    },
                    'leakage_patterns': {
                        'mean_leakage': 0.0,
                        'median_leakage': 0.0,
                        'max_leakage': 0.5,
                        'min_leakage': -0.5,
                        'std_leakage': 0.3,
                        'high_leakage_zips': [60601, 60602],
                        'low_leakage_zips': [60604, 60605]
                    },
                    'summary': 'Sample summary'
                }
                
                viz_paths = viz_generator.generate_retail_void_visualizations(model_results, void_data)
                
                if viz_paths and isinstance(viz_paths, dict) and len(viz_paths) > 0:
                    # Check if at least one visualization was created
                    results["tests"]["generate_retail_void_visualizations"] = {"status": "Passed"}
                    results["passed"] += 1
                else:
                    results["tests"]["generate_retail_void_visualizations"] = {
                        "status": "Warning",
                        "message": "No visualizations generated"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["generate_retail_void_visualizations"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating VisualizationGenerator: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def validate_reports(self):
        """
        Validate report components.
        """
        try:
            logger.info("Validating report components")
            
            # Initialize results
            self.validation_results["reports"] = {
                "status": "In Progress",
                "components": {},
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
            
            # Skip report validation if pdfkit is not available
            if not self.available_dependencies["pdfkit"]:
                logger.warning("Skipping report validation due to missing pdfkit dependency.")
                self.validation_results["reports"]["status"] = "Skipped"
                self.validation_results["reports"]["message"] = "Report validation skipped due to missing pdfkit dependency."
                return
            
            # Import required modules
            try:
                from src.reports.report_generator import ReportGenerator
                
                # Validate ReportGenerator
                self._validate_component(
                    "ReportGenerator",
                    self._validate_report_generator,
                    self.validation_results["reports"]["components"]
                )
                
            except ImportError as e:
                logger.error(f"Error importing report modules: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.validation_results["reports"]["status"] = "Failed"
                self.validation_results["reports"]["error"] = f"Error importing report modules: {str(e)}"
                return
            
            # Calculate component results
            self._calculate_component_results(self.validation_results["reports"])
            
            logger.info(f"Report validation completed: {self.validation_results['reports']['status']}")
            
        except Exception as e:
            logger.error(f"Error validating reports: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.validation_results["reports"]["status"] = "Failed"
            self.validation_results["reports"]["error"] = str(e)
    
    def _validate_report_generator(self):
        """
        Validate ReportGenerator class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.reports.report_generator import ReportGenerator
            
            # Test initialization
            output_dir = self.output_dir / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_generator = ReportGenerator(output_dir=output_dir)
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Skip PDF generation tests if wkhtmltopdf is not available
            if not self.available_dependencies["wkhtmltopdf"]:
                results["tests"]["pdf_generation"] = {
                    "status": "Skipped",
                    "message": "PDF generation tests skipped due to missing wkhtmltopdf dependency."
                }
                results["warnings"] += 1
            
            # Test template creation
            try:
                # Check if templates directory exists
                templates_dir = Path(report_generator.templates_dir)
                if templates_dir.exists():
                    # Check if at least one template exists
                    template_files = list(templates_dir.glob("*.html"))
                    if len(template_files) > 0:
                        results["tests"]["_create_templates"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["_create_templates"] = {
                            "status": "Warning",
                            "message": "No template files created"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["_create_templates"] = {
                        "status": "Warning",
                        "message": "Templates directory does not exist"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["_create_templates"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Test markdown template creation
            try:
                # Check if markdown templates directory exists
                md_templates_dir = templates_dir / "markdown"
                if md_templates_dir.exists():
                    # Check if at least one markdown template exists
                    md_template_files = list(md_templates_dir.glob("*.md"))
                    if len(md_template_files) > 0:
                        results["tests"]["_create_markdown_templates"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["_create_markdown_templates"] = {
                            "status": "Warning",
                            "message": "No markdown template files created"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["_create_markdown_templates"] = {
                        "status": "Warning",
                        "message": "Markdown templates directory does not exist"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["_create_markdown_templates"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating ReportGenerator: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def validate_pipeline(self):
        """
        Validate full pipeline.
        """
        try:
            logger.info("Validating full pipeline")
            
            # Initialize results
            self.validation_results["pipeline"] = {
                "status": "In Progress",
                "components": {},
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
            
            # Import required modules
            try:
                from src.pipeline.pipeline import Pipeline
                
                # Validate Pipeline
                self._validate_component(
                    "Pipeline",
                    self._validate_pipeline_class,
                    self.validation_results["pipeline"]["components"]
                )
                
            except ImportError as e:
                logger.error(f"Error importing pipeline modules: {str(e)}")
                logger.error(traceback.format_exc())
                
                self.validation_results["pipeline"]["status"] = "Failed"
                self.validation_results["pipeline"]["error"] = f"Error importing pipeline modules: {str(e)}"
                return
            
            # Calculate component results
            self._calculate_component_results(self.validation_results["pipeline"])
            
            logger.info(f"Pipeline validation completed: {self.validation_results['pipeline']['status']}")
            
        except Exception as e:
            logger.error(f"Error validating pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.validation_results["pipeline"]["status"] = "Failed"
            self.validation_results["pipeline"]["error"] = str(e)
    
    def _validate_pipeline_class(self):
        """
        Validate Pipeline class.
        
        Returns:
            dict: Validation results
        """
        results = {
            "status": "Not Started",
            "tests": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
        
        try:
            # Import required modules
            from src.pipeline.pipeline import Pipeline
            
            # Test initialization
            output_dir = self.output_dir / "pipeline_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            pipeline = Pipeline(output_dir=output_dir)
            results["tests"]["initialization"] = {"status": "Passed"}
            results["passed"] += 1
            
            # Test run method with sample data
            try:
                # Run pipeline with sample data
                pipeline_results = pipeline.run(use_sample_data=True)
                
                if pipeline_results and isinstance(pipeline_results, dict):
                    # Check for required keys
                    required_keys = ['status', 'models']
                    if all(key in pipeline_results for key in required_keys):
                        results["tests"]["run"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["run"] = {
                            "status": "Warning",
                            "message": f"run results missing required keys: {[key for key in required_keys if key not in pipeline_results]}"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["run"] = {
                        "status": "Warning",
                        "message": "run returned None or non-dict"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["run"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Check output files
            try:
                # Check if output directory exists
                if output_dir.exists():
                    # Check if model outputs exist
                    model_outputs = list(output_dir.glob("models/**/*.json"))
                    if len(model_outputs) > 0:
                        results["tests"]["model_outputs"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["model_outputs"] = {
                            "status": "Warning",
                            "message": "No model output files found"
                        }
                        results["warnings"] += 1
                    
                    # Check if visualization outputs exist
                    viz_outputs = list(output_dir.glob("visualizations/**/*.png"))
                    viz_outputs.extend(list(output_dir.glob("visualizations/**/*.html")))
                    if len(viz_outputs) > 0:
                        results["tests"]["visualization_outputs"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["visualization_outputs"] = {
                            "status": "Warning",
                            "message": "No visualization output files found"
                        }
                        results["warnings"] += 1
                    
                    # Check if report outputs exist
                    report_outputs = list(output_dir.glob("reports/**/*.html"))
                    report_outputs.extend(list(output_dir.glob("reports/**/*.pdf")))
                    report_outputs.extend(list(output_dir.glob("reports/**/*.md")))
                    if len(report_outputs) > 0:
                        results["tests"]["report_outputs"] = {"status": "Passed"}
                        results["passed"] += 1
                    else:
                        results["tests"]["report_outputs"] = {
                            "status": "Warning",
                            "message": "No report output files found"
                        }
                        results["warnings"] += 1
                else:
                    results["tests"]["output_directory"] = {
                        "status": "Warning",
                        "message": "Output directory does not exist"
                    }
                    results["warnings"] += 1
            except Exception as e:
                results["tests"]["output_files"] = {
                    "status": "Failed",
                    "message": str(e)
                }
                results["failed"] += 1
            
            # Calculate overall status
            if results["failed"] > 0:
                results["status"] = "Failed"
            elif results["warnings"] > 0:
                results["status"] = "Warning"
            else:
                results["status"] = "Passed"
            
        except Exception as e:
            logger.error(f"Error validating Pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            
            results["status"] = "Failed"
            results["error"] = str(e)
        
        return results
    
    def _validate_component(self, component_name, validation_func, results_dict):
        """
        Validate a component and update results.
        
        Args:
            component_name (str): Component name
            validation_func (callable): Validation function
            results_dict (dict): Results dictionary to update
        """
        try:
            logger.info(f"Validating {component_name}")
            
            # Run validation function
            component_results = validation_func()
            
            # Update results dictionary
            results_dict[component_name] = component_results
            
            logger.info(f"{component_name} validation completed: {component_results['status']}")
            
        except Exception as e:
            logger.error(f"Error validating {component_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            results_dict[component_name] = {
                "status": "Failed",
                "error": str(e),
                "passed": 0,
                "failed": 1,
                "warnings": 0
            }
    
    def _calculate_component_results(self, component_results):
        """
        Calculate overall results for a component.
        
        Args:
            component_results (dict): Component results
        """
        # Initialize counters
        passed = 0
        failed = 0
        warnings = 0
        
        # Count results
        for component_name, results in component_results["components"].items():
            passed += results.get("passed", 0)
            failed += results.get("failed", 0)
            warnings += results.get("warnings", 0)
        
        # Update component results
        component_results["passed"] = passed
        component_results["failed"] = failed
        component_results["warnings"] = warnings
        
        # Calculate overall status
        if failed > 0:
            component_results["status"] = "Failed"
        elif warnings > 0:
            component_results["status"] = "Warning"
        else:
            component_results["status"] = "Passed"
    
    def _calculate_overall_results(self):
        """
        Calculate overall validation results.
        """
        # Initialize counters
        passed = 0
        failed = 0
        warnings = 0
        
        # Count results from all components
        for component_type in ["data_collection", "data_processing", "models", "visualizations", "reports", "pipeline"]:
            if component_type in self.validation_results:
                if self.validation_results[component_type].get("status") != "Skipped":
                    passed += self.validation_results[component_type].get("passed", 0)
                    failed += self.validation_results[component_type].get("failed", 0)
                    warnings += self.validation_results[component_type].get("warnings", 0)
        
        # Update overall results
        self.validation_results["overall"]["passed"] = passed
        self.validation_results["overall"]["failed"] = failed
        self.validation_results["overall"]["warnings"] = warnings
        
        # Calculate overall status
        if failed > 0:
            self.validation_results["overall"]["status"] = "Failed"
        elif warnings > 0:
            self.validation_results["overall"]["status"] = "Warning"
        else:
            self.validation_results["overall"]["status"] = "Passed"
    
    def _save_validation_results(self):
        """
        Save validation results to file.
        """
        try:
            # Save as JSON
            results_file = self.output_dir / "validation_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            logger.info(f"Validation results saved to {results_file}")
            
            # Save as Markdown
            md_file = self.output_dir / "validation_results.md"
            with open(md_file, 'w') as f:
                f.write("# Chicago Housing Pipeline Validation Results\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Overall results
                f.write("## Overall Results\n\n")
                f.write(f"Status: **{self.validation_results['overall']['status']}**\n\n")
                f.write(f"- Passed: {self.validation_results['overall']['passed']}\n")
                f.write(f"- Failed: {self.validation_results['overall']['failed']}\n")
                f.write(f"- Warnings: {self.validation_results['overall']['warnings']}\n")
                f.write(f"- Duration: {self.validation_results['overall'].get('duration', 'N/A')}\n\n")
                
                # Component results
                for component_type in ["data_collection", "data_processing", "models", "visualizations", "reports", "pipeline"]:
                    if component_type in self.validation_results:
                        component_results = self.validation_results[component_type]
                        
                        f.write(f"## {component_type.replace('_', ' ').title()} Results\n\n")
                        f.write(f"Status: **{component_results['status']}**\n\n")
                        
                        if component_results.get("message"):
                            f.write(f"Message: {component_results['message']}\n\n")
                        
                        if component_results["status"] != "Skipped":
                            f.write(f"- Passed: {component_results['passed']}\n")
                            f.write(f"- Failed: {component_results['failed']}\n")
                            f.write(f"- Warnings: {component_results['warnings']}\n\n")
                            
                            # Component details
                            for component_name, results in component_results["components"].items():
                                f.write(f"### {component_name}\n\n")
                                f.write(f"Status: **{results['status']}**\n\n")
                                f.write(f"- Passed: {results['passed']}\n")
                                f.write(f"- Failed: {results['failed']}\n")
                                f.write(f"- Warnings: {results['warnings']}\n\n")
                                
                                # Test details
                                if "tests" in results:
                                    f.write("#### Tests\n\n")
                                    for test_name, test_results in results["tests"].items():
                                        status = test_results["status"]
                                        message = test_results.get("message", "")
                                        
                                        f.write(f"- **{test_name}**: {status}")
                                        if message:
                                            f.write(f" - {message}")
                                        f.write("\n")
                                    
                                    f.write("\n")
            
            logger.info(f"Validation results saved to {md_file}")
            
        except Exception as e:
            logger.error(f"Error saving validation results: {str(e)}")
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    validator = PipelineValidator()
    validation_results = validator.validate_all()
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Status: {validation_results['overall']['status']}")
    print(f"Passed: {validation_results['overall']['passed']}")
    print(f"Failed: {validation_results['overall']['failed']}")
    print(f"Warnings: {validation_results['overall']['warnings']}")
    print(f"Duration: {validation_results['overall'].get('duration', 'N/A')}")
    
    # Exit with appropriate code
    if validation_results['overall']['status'] == "Failed":
        sys.exit(1)
    else:
        sys.exit(0)
