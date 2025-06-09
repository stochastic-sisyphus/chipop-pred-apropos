"""
Main dynamic runner module for the Chicago Population Analysis project.
Dynamically discovers and runs models and reports.
"""

import logging
import importlib
import inspect
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

class DynamicRunner:
    """Dynamically discovers and runs models and reports."""
    
    def __init__(self, base_dir=None):
        """
        Initialize the dynamic runner.
        
        Args:
            base_dir (Path, optional): Base directory for module discovery
        """
        # Fix: Use the correct base directory for module discovery
        # The base_dir should be the project root, not the src directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Get the absolute path to the project root
            self.base_dir = Path(__file__).resolve().parent.parent.parent
            
        logger.info(f"Dynamic runner initialized with base_dir: {self.base_dir}")
        self.models = {}
        self.reports = {}
    
    def discover_modules(self, package_name):
        """
        Discover modules in a package.
        
        Args:
            package_name (str): Name of the package to discover modules in
            
        Returns:
            list: List of discovered module names
        """
        try:
            # Fix: Correctly construct the package path
            # For package 'src.models', the path should be '{base_dir}/src/models'
            package_path = self.base_dir / package_name.replace('.', '/')
            logger.info(f"Looking for modules in: {package_path}")
            
            if not package_path.exists() or not package_path.is_dir():
                logger.error(f"Package path not found: {package_path}")
                return []
            
            module_files = list(package_path.glob('*.py'))
            module_names = [
                f"{package_name}.{f.stem}" 
                for f in module_files 
                if f.stem != '__init__' and not f.stem.startswith('_')
            ]
            
            logger.info(f"Discovered {len(module_names)} modules in {package_name}: {module_names}")
            return module_names
            
        except Exception as e:
            logger.error(f"Error discovering modules in {package_name}: {str(e)}")
            return []
    
    def discover_classes(self, module_name, base_class=None):
        """
        Discover classes in a module.
        
        Args:
            module_name (str): Name of the module to discover classes in
            base_class (class, optional): Base class to filter by
            
        Returns:
            dict: Dictionary of discovered class names and classes
        """
        try:
            logger.info(f"Attempting to import module: {module_name}")
            module = importlib.import_module(module_name)
            
            classes = {}
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip imported classes
                if obj.__module__ != module_name:
                    continue
                    
                # Filter by base class if provided
                if base_class and not issubclass(obj, base_class):
                    continue
                    
                classes[name] = obj
            
            logger.info(f"Discovered {len(classes)} classes in {module_name}: {list(classes.keys())}")
            return classes
            
        except Exception as e:
            logger.error(f"Error discovering classes in {module_name}: {str(e)}")
            return {}
    
    def load_all(self):
        """
        Load all models and reports.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Discover model modules
            model_modules = self.discover_modules('src.models')
            
            # Load model classes
            from src.models.base_model import BaseModel
            for module_name in model_modules:
                model_classes = self.discover_classes(module_name, BaseModel)
                self.models.update(model_classes)
            
            # Discover report modules
            report_modules = self.discover_modules('src.reporting')
            
            # Load report classes
            from src.reporting.base_report import BaseReport
            for module_name in report_modules:
                report_classes = self.discover_classes(module_name, BaseReport)
                self.reports.update(report_classes)
            
            logger.info(f"Loaded {len(self.models)} models and {len(self.reports)} reports")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models and reports: {str(e)}")
            return False
    
    def run_all_models(self, data):
        """
        Run all models.
        
        Args:
            data (pd.DataFrame): Input data for models
            
        Returns:
            dict: Dictionary of model results
        """
        try:
            if not self.models:
                logger.warning("No models loaded. Attempting to load models...")
                self.load_all()
                
            if not self.models:
                logger.error("No models available to run")
                return {}
                
            results = {}
            
            for name, model_class in self.models.items():
                logger.info(f"Running model: {name}")
                
                try:
                    # Skip base class
                    if name == 'BaseModel':
                        continue
                        
                    # Initialize model
                    model = model_class()
                    
                    # Run analysis
                    if model.run_analysis(data):
                        results[name] = model.get_results()
                        logger.info(f"Model {name} completed successfully")
                    else:
                        logger.error(f"Model {name} failed to run")
                        
                except Exception as e:
                    logger.error(f"Error running model {name}: {str(e)}")
                    continue
            
            if not results:
                logger.error("No models ran successfully")
                return {}
                
            logger.info(f"Successfully ran {len(results)} models")
            return results
            
        except Exception as e:
            logger.error(f"Error running all models: {str(e)}")
            return {}
    
    def run_all_reports(self, data):
        """
        Run all reports.
        
        Args:
            data (pd.DataFrame): Input data for reports
            
        Returns:
            dict: Dictionary of report results
        """
        try:
            if not self.reports:
                logger.warning("No reports loaded. Attempting to load reports...")
                self.load_all()
                
            if not self.reports:
                logger.error("No reports available to run")
                return {}
                
            results = {}
            
            for name, report_class in self.reports.items():
                logger.info(f"Running report: {name}")
                
                try:
                    # Skip base class
                    if name == 'BaseReport':
                        continue
                        
                    # Initialize report
                    report = report_class()
                    
                    # Load and prepare data
                    prepared_data = report.load_and_prepare_data(data)
                    if prepared_data is None:
                        logger.error(f"Report {name} failed to prepare data")
                        continue
                    
                    # Generate report
                    if report.generate_report():
                        results[name] = True
                        logger.info(f"Report {name} generated successfully")
                    else:
                        logger.error(f"Report {name} failed to generate")
                        
                except Exception as e:
                    logger.error(f"Error running report {name}: {str(e)}")
                    continue
            
            if not results:
                logger.error("No reports ran successfully")
                return {}
                
            logger.info(f"Successfully generated {len(results)} reports")
            return results
            
        except Exception as e:
            logger.error(f"Error running all reports: {str(e)}")
            return {}
    
    def run_specific_model(self, model_name, data):
        """
        Run a specific model.
        
        Args:
            model_name (str): Name of the model to run
            data (pd.DataFrame): Input data for the model
            
        Returns:
            dict: Model results
        """
        try:
            # Ensure models are loaded
            if not self.models:
                self.load_all()
            
            # Check if model exists
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return None
            
            # Initialize model
            model_class = self.models[model_name]
            model = model_class()
            
            # Run analysis
            if model.run_analysis(data):
                return model.get_results()
            else:
                logger.error(f"Model {model_name} failed to run")
                return None
                
        except Exception as e:
            logger.error(f"Error running model {model_name}: {str(e)}")
            return None
    
    def run_specific_report(self, report_name, data):
        """
        Run a specific report.
        
        Args:
            report_name (str): Name of the report to run
            data (pd.DataFrame): Input data for the report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure reports are loaded
            if not self.reports:
                self.load_all()
            
            # Check if report exists
            if report_name not in self.reports:
                logger.error(f"Report {report_name} not found")
                return False
            
            # Initialize report
            report_class = self.reports[report_name]
            report = report_class()
            
            # Load and prepare data
            prepared_data = report.load_and_prepare_data(data)
            if prepared_data is None:
                logger.error(f"Report {report_name} failed to prepare data")
                return False
            
            # Generate report
            if report.generate_report():
                return True
            else:
                logger.error(f"Report {report_name} failed to generate")
                return False
                
        except Exception as e:
            logger.error(f"Error running report {report_name}: {str(e)}")
            return False
