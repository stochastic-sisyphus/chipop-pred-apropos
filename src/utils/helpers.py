"""
Utility functions for Chicago population analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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

def resolve_column(df, possible_names, required=True):
    """Return the first matching column name from possible_names in df, or raise if required and not found."""
    for name in possible_names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"None of the expected columns found: {possible_names}")
    return None 

def sanitize_features(df: pd.DataFrame, feature_cols: list, target_col: str = None) -> pd.DataFrame:
    """
    Coerce features (and optionally the target) to numeric, drop rows with NaN/infs.

    Args:
        df (pd.DataFrame): Input dataframe
        feature_cols (list): List of columns to use as features
        target_col (str): Optional target column to include in cleaning

    Returns:
        pd.DataFrame: Cleaned subset of the original dataframe
    """
    cols_to_check = feature_cols.copy()
    if target_col:
        cols_to_check.append(target_col)

    if missing := [col for col in feature_cols if col not in df.columns]:
        logger.warning(f"sanitize_features(): Missing expected columns: {missing}")

    df_clean = df[cols_to_check].copy()
    for col in cols_to_check:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df_clean)
    df_clean.dropna(inplace=True)
    after = len(df_clean)

    logger.info(f"sanitize_features(): Dropped {before - after} rows with NaN or inf values.")
    return df_clean


def match_features(df: pd.DataFrame, expected: List[str], aliases: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Match features in the DataFrame to an expected list using direct or alias-based logic.

    Args:
        df: The dataframe to match against
        expected: List of canonical feature names
        aliases: Optional dict mapping expected names to list of possible alternatives

    Returns:
        List of resolved column names present in df, in the order of `expected`
    """
    resolved = []
    columns_lower = {col.lower(): col for col in df.columns}

    for feat in expected:
        candidates = [feat]
        if aliases and feat in aliases:
            candidates.extend(aliases[feat])

        if match := next(
            (
                columns_lower[cand.lower()]
                for cand in candidates
                if cand.lower() in columns_lower
            ),
            None,
        ):
            resolved.append(match)
        else:
            logger.warning(f"Feature '{feat}' not found and no alias match.")
    return resolved

def resolve_column_name(df: pd.DataFrame, name: str, aliases: Dict[str, List[str]]) -> Optional[str]:
    """
    Resolve a single column name using alias mapping.
    
    Args:
        df: DataFrame to check
        name: Canonical column name
        aliases: Alias dictionary

    Returns:
        Optional[str]: Resolved column name if found, else None
    """
    candidates = [name] + aliases.get(name, [])
    columns_lower = {col.lower(): col for col in df.columns}
    
    for cand in candidates:
        if cand.lower() in columns_lower:
            return columns_lower[cand.lower()]
            
    logger.warning(f"Could not resolve column: {name}")
    return None

def safe_train_model(model, X: pd.DataFrame, y: pd.Series, model_name: str = "model") -> bool:
    """
    Safely trains model with aligned, clean data.
    
    - Drops any row where X or y is NaN or inf
    - Ensures indices are aligned
    - Logs row counts
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Drop rows where y is NaN or inf
        combined = X.copy()
        combined['__target__'] = y
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined.dropna(inplace=True)

        if combined.empty:
            logger.error(f"{model_name}: No valid data to train on.")
            return False

        X_clean = combined.drop(columns='__target__')
        y_clean = combined['__target__']

        logger.info(f"{model_name}: Final training set size: {len(X_clean)} rows")
        model.fit(X_clean, y_clean)
        logger.info(f"{model_name}: Training complete.")
        return True
    except Exception as e:
        logger.error(f"{model_name}: Training failed - {str(e)}")
        return False
