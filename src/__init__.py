"""
Chicago Population Analysis Pipeline
A comprehensive tool for analyzing population shifts in Chicago.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.config import settings
from src.data_collection.collector import DataCollector
from src.data_processing.processor import DataProcessor
from src.models.population_model import PopulationModel
from src.visualization.visualizer import Visualizer

__all__ = ["settings", "DataCollector", "DataProcessor", "PopulationModel", "Visualizer"]
