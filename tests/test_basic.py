"""
Basic tests to verify project structure and imports.
"""

import os
import pytest
from pathlib import Path

def test_project_structure():
    """Test that all required directories exist."""
    required_dirs = [
        'data/raw',
        'data/interim',
        'data/processed',
        'output/models',
        'output/visualizations',
        'output/reports',
        'logs'
    ]
    
    base_dir = Path(__file__).resolve().parent.parent
    for dir_path in required_dirs:
        assert (base_dir / dir_path).exists(), f"Directory {dir_path} does not exist"

def test_config_imports():
    """Test that configuration can be imported."""
    try:
        from src.config import settings
        assert hasattr(settings, 'BASE_DIR')
        assert hasattr(settings, 'DATA_DIR')
        assert hasattr(settings, 'OUTPUT_DIR')
    except ImportError as e:
        pytest.fail(f"Failed to import settings: {str(e)}")

def test_package_imports():
    """Test that main package components can be imported."""
    try:
        from src.data_collection.collector import DataCollector
        from src.data_processing.processor import DataProcessor
        from src.models.population_model import PopulationModel
        from src.visualization.visualizer import Visualizer
    except ImportError as e:
        pytest.fail(f"Failed to import package components: {str(e)}")

def test_environment_variables():
    """Test that required environment variables are defined."""
    required_vars = [
        'CENSUS_API_KEY',
        'FRED_API_KEY'
    ]
    
    # Load .env file if it exists
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    for var in required_vars:
        assert os.getenv(var) is not None, f"Environment variable {var} is not set" 