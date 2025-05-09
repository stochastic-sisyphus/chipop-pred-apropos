import importlib
import inspect
import logging
from pathlib import Path
from types import ModuleType
from typing import List, Type, Optional

logger = logging.getLogger(__name__)


class DynamicRunner:
    """
    Dynamically discovers and runs models and reports in the pipeline.
    It loads all classes that follow naming conventions and provides methods
    to execute them without manual imports.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize the DynamicRunner.

        Args:
            base_dir (Path): The base path where 'models' and 'reporting' folders live.
        """
        self.base_dir = base_dir
        self.model_classes: List[Type] = []
        self.report_classes: List[Type] = []

    def discover_modules(self, subpackage: str) -> List[ModuleType]:
        """
        Discover all modules within a given subpackage (e.g., 'models', 'reporting').

        Args:
            subpackage (str): The subpackage to search.

        Returns:
            List of imported modules.
        """
        modules = []
        package_dir = self.base_dir / subpackage

        if not package_dir.exists() or not package_dir.is_dir():
            logger.warning(f"Subpackage directory '{subpackage}' not found at {package_dir}")
            return modules

        for file in package_dir.glob("*.py"):
            if file.name.startswith("_") or not file.name.endswith(".py"):
                continue  # skip __init__.py and non-py files

            module_name = f"{self.base_dir.name}.{subpackage}.{file.stem}"
            try:
                module = importlib.import_module(module_name)
                modules.append(module)
                logger.info(f"Imported module: {module_name}")
            except Exception as e:
                logger.error(f"Failed to import module '{module_name}': {e}")

        return modules

    def discover_classes(self, modules: List[ModuleType], class_type: str) -> List[Type]:
        """
        Discover all classes in the given modules that match the type ('Model' or 'Report').

        Args:
            modules (List[ModuleType]): List of modules to search.
            class_type (str): Either 'Model' or 'Report'.

        Returns:
            List of classes that match.
        """
        classes = []
        for module in modules:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__ and class_type in name:
                    classes.append(obj)
                    logger.info(f"Discovered {class_type}: {obj.__name__} in {module.__name__}")
        return classes

    def load_all(self):
        """
        Load all model and report classes dynamically.
        """
        self.model_classes = self._extracted_from_load_all_5(
            "Discovering models...", "models", "Model"
        )
        self.report_classes = self._extracted_from_load_all_5(
            "Discovering reports...", "reporting", "Report"
        )

    # TODO Rename this here and in `load_all`
    def _extracted_from_load_all_5(self, arg0, arg1, arg2):
        logger.info(arg0)
        model_modules = self.discover_modules(arg1)
        return self.discover_classes(model_modules, arg2)

    def run_all_models(self, *args, **kwargs):
        """
        Instantiate and run all discovered model classes.

        Args:
            *args, **kwargs: Passed to each model's `train()` method.
        """
        if not self.model_classes:
            logger.warning("No models found to run.")
            return

        for model_cls in self.model_classes:
            try:
                logger.info(f"Running model: {model_cls.__name__}")
                model_instance = model_cls()
                if hasattr(model_instance, "train"):
                    model_instance.train(*args, **kwargs)
                else:
                    logger.warning(f"{model_cls.__name__} has no 'train()' method.")
            except Exception as e:
                logger.error(f"Error running model {model_cls.__name__}: {e}")

    def run_all_reports(self, *args, **kwargs):
        """
        Instantiate and run all discovered report classes.

        Args:
            *args, **kwargs: Passed to each report's `generate_report()` method.
        """
        if not self.report_classes:
            logger.warning("No reports found to run.")
            return

        for report_cls in self.report_classes:
            try:
                logger.info(f"Running report: {report_cls.__name__}")
                report_instance = report_cls()
                if hasattr(report_instance, "generate_report"):
                    report_instance.generate_report(*args, **kwargs)
                else:
                    logger.warning(f"{report_cls.__name__} has no 'generate_report()' method.")
            except Exception as e:
                logger.error(f"Error running report {report_cls.__name__}: {e}")

    def run_specific_model(self, model_name: str, *args, **kwargs) -> Optional[Type]:
        """
        Run a specific model by class name.

        Args:
            model_name (str): Name of the model class to run.
            *args, **kwargs: Passed to the model's `train()` method.

        Returns:
            The model class if found and run, else None.
        """
        for model_cls in self.model_classes:
            if model_cls.__name__ == model_name:
                try:
                    logger.info(f"Running specific model: {model_name}")
                    model_instance = model_cls()
                    if hasattr(model_instance, "train"):
                        model_instance.train(*args, **kwargs)
                    else:
                        logger.warning(f"{model_name} has no 'train()' method.")
                    return model_cls
                except Exception as e:
                    logger.error(f"Error running model {model_name}: {e}")
                    return None
        logger.warning(f"Model '{model_name}' not found.")
        return None

    def run_specific_report(self, report_name: str, *args, **kwargs) -> Optional[Type]:
        """
        Run a specific report by class name.

        Args:
            report_name (str): Name of the report class to run.
            *args, **kwargs: Passed to the report's `generate_report()` method.

        Returns:
            The report class if found and run, else None.
        """
        for report_cls in self.report_classes:
            if report_cls.__name__ == report_name:
                try:
                    logger.info(f"Running specific report: {report_name}")
                    report_instance = report_cls()
                    if hasattr(report_instance, "generate_report"):
                        report_instance.generate_report(*args, **kwargs)
                    else:
                        logger.warning(f"{report_name} has no 'generate_report()' method.")
                    return report_cls
                except Exception as e:
                    logger.error(f"Error running report {report_name}: {e}")
                    return None
        logger.warning(f"Report '{report_name}' not found.")
        return None