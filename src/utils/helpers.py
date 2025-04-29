"""
Utility functions for Chicago population analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

import pandas as pd
import numpy as np

from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)

def ensure_directory(path: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {str(e)}")
        return False

def load_json(path: Union[str, Path]) -> Optional[Dict]:
    """
    Load JSON file safely.
    
    Args:
        path: Path to JSON file
        
    Returns:
        dict: Loaded JSON data or None if error
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {path}: {str(e)}")
        return None

def save_json(data: Dict, path: Union[str, Path]) -> bool:
    """
    Save data to JSON file safely.
    
    Args:
        data: Data to save
        path: Output path
        
    Returns:
        bool: True if saved successfully
    """
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {path}: {str(e)}")
        return False

def calculate_growth_rate(
    current: float,
    previous: float,
    annualize: bool = False,
    periods: int = 1
) -> float:
    """
    Calculate growth rate between two values.
    
    Args:
        current: Current value
        previous: Previous value
        annualize: Whether to annualize the rate
        periods: Number of periods if annualizing
        
    Returns:
        float: Growth rate
    """
    try:
        rate = (current - previous) / previous
        if annualize:
            rate = (1 + rate) ** (1/periods) - 1
        return rate
    except ZeroDivisionError:
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating growth rate: {str(e)}")
        return 0.0

def calculate_summary_stats(data: pd.Series) -> Dict[str, float]:
    """
    Calculate summary statistics for a series.
    
    Args:
        data: Data series to summarize
        
    Returns:
        dict: Summary statistics
    """
    try:
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'count': len(data)
        }
    except Exception as e:
        logger.error(f"Error calculating summary stats: {str(e)}")
        return {}

def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency string.
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        str: Formatted currency string
    """
    try:
        return f"${value:,.{decimals}f}"
    except Exception as e:
        logger.error(f"Error formatting currency: {str(e)}")
        return str(value)

def format_percent(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    try:
        return f"{value*100:.{decimals}f}%"
    except Exception as e:
        logger.error(f"Error formatting percentage: {str(e)}")
        return str(value)

def validate_zip_code(zip_code: str) -> bool:
    """
    Validate if a ZIP code is in Chicago.
    
    Args:
        zip_code: ZIP code to validate
        
    Returns:
        bool: True if valid Chicago ZIP code
    """
    try:
        return zip_code in settings.CHICAGO_ZIP_CODES
    except Exception as e:
        logger.error(f"Error validating ZIP code: {str(e)}")
        return False

def calculate_confidence_interval(
    data: pd.Series,
    confidence: float = 0.95
) -> tuple:
    """
    Calculate confidence interval for a series.
    
    Args:
        data: Data series
        confidence: Confidence level (0-1)
        
    Returns:
        tuple: (lower bound, upper bound)
    """
    try:
        mean = data.mean()
        std = data.std()
        z_score = abs(np.percentile(np.random.standard_normal(10000), (1-confidence)/2))
        margin = z_score * (std / np.sqrt(len(data)))
        return (mean - margin, mean + margin)
    except Exception as e:
        logger.error(f"Error calculating confidence interval: {str(e)}")
        return (None, None) 