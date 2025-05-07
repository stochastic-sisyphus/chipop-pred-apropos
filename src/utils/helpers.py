"""
Utility functions for Chicago population analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import requests
import time
from functools import lru_cache
import os
import re

import pandas as pd
import numpy as np

from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Example: community_area to ZIP mapping (should be loaded from a config or data file in production)
COMMUNITY_AREA_ZIP_MAP = {
    # 'community_area_name': 'zip_code',
    'Austin': '60644',
    'Englewood': '60621',
    # ... add all mappings ...
}

# In-memory cache for geocoding results
_GEOCODE_CACHE = {}

# Suspicious city names for Chicago context
CHICAGO_CITIES = {'CHICAGO', 'CICERO', 'EVANSTON', 'OAK PARK', 'BERWYN', 'SKOKIE', 'MT PROSPECT', 'MELROSE PARK', 'FOREST PARK', 'MAYWOOD', 'BELLWOOD', 'RIVER FOREST', 'NILES', 'PARK RIDGE', 'LINCOLNWOOD', 'ROSEMONT', 'SUMMIT', 'ELMWOOD PARK', 'RIVER GROVE', 'NORRIDGE', 'HARWOOD HEIGHTS'}


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
        with open(path, "r") as f:
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
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {path}: {str(e)}")
        return False


def calculate_growth_rate(
    current: float, previous: float, annualize: bool = False, periods: int = 1
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
            rate = (1 + rate) ** (1 / periods) - 1
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
            "mean": data.mean(),
            "median": data.median(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "count": len(data),
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


def calculate_confidence_interval(data: pd.Series, confidence: float = 0.95) -> tuple:
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
        z_score = abs(np.percentile(np.random.standard_normal(10000), (1 - confidence) / 2))
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
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df_clean)
    df_clean.dropna(inplace=True)
    after = len(df_clean)

    logger.info(f"sanitize_features(): Dropped {before - after} rows with NaN or inf values.")
    return df_clean


def match_features(
    df: pd.DataFrame, expected: List[str], aliases: Optional[Dict[str, List[str]]] = None
) -> List[str]:
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
            (columns_lower[cand.lower()] for cand in candidates if cand.lower() in columns_lower),
            None,
        ):
            resolved.append(match)
        else:
            logger.warning(f"Feature '{feat}' not found and no alias match.")
    return resolved


def resolve_column_name(
    df: pd.DataFrame, name: str, aliases: Dict[str, List[str]]
) -> Optional[str]:
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
        combined["__target__"] = y
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined.dropna(inplace=True)

        if combined.empty:
            logger.error(f"{model_name}: No valid data to train on.")
            return False

        X_clean = combined.drop(columns="__target__")
        y_clean = combined["__target__"]

        logger.info(f"{model_name}: Final training set size: {len(X_clean)} rows")
        model.fit(X_clean, y_clean)
        logger.info(f"{model_name}: Training complete.")
        return True
    except Exception as e:
        logger.error(f"{model_name}: Training failed - {str(e)}")
        return False


def ensure_output_structure() -> bool:
    """
    Ensure all required directories and files exist.
    Creates directories if they don't exist and validates required files.

    Returns:
        bool: True if successful, False if there were any errors
    """
    try:
        # Create all required directories
        for directory in [
            settings.DATA_DIR,
            settings.RAW_DATA_DIR,
            settings.INTERIM_DATA_DIR,
            settings.PROCESSED_DATA_DIR,
            settings.OUTPUT_DIR,
            settings.PREDICTIONS_DIR,
            settings.VISUALIZATIONS_DIR,
            settings.REPORTS_DIR,
            settings.MODEL_METRICS_DIR,
            settings.TRAINED_MODELS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

        # Create empty files for required outputs if they don't exist
        for filename, directory in settings.REQUIRED_OUTPUTS.items():
            filepath = directory / filename
            if not filepath.exists():
                filepath.touch()
                logger.info(f"Created empty file: {filepath}")

        # Create empty files for required reports
        for filename, directory in settings.REQUIRED_REPORTS.items():
            filepath = directory / filename
            if not filepath.exists():
                filepath.write_text(
                    "# " + filename.replace(".md", "").replace("_", " ").title() + "\n\n"
                )
                logger.info(f"Created report template: {filepath}")

        logger.info("Successfully ensured output structure")
        return True

    except Exception as e:
        logger.error(f"Error ensuring output structure: {str(e)}")
        return False


def validate_outputs() -> bool:
    """
    Validate that all required outputs exist and have content.

    Returns:
        bool: True if all outputs are valid, False otherwise
    """
    try:
        # Check required CSV outputs
        for filename, directory in settings.REQUIRED_OUTPUTS.items():
            filepath = directory / filename
            if not filepath.exists():
                logger.error(f"Missing required output: {filepath}")
                return False
            if filepath.stat().st_size == 0:
                logger.error(f"Empty output file: {filepath}")
                return False

        # Check required reports
        for filename, directory in settings.REQUIRED_REPORTS.items():
            filepath = directory / filename
            if not filepath.exists():
                logger.error(f"Missing required report: {filepath}")
                return False
            if filepath.stat().st_size == 0:
                logger.error(f"Empty report file: {filepath}")
                return False

        # Check required visualizations
        for filename, directory in settings.REQUIRED_VISUALIZATIONS.items():
            filepath = directory / filename
            if not filepath.exists():
                logger.error(f"Missing required visualization: {filepath}")
                return False
            if filepath.stat().st_size == 0:
                logger.error(f"Empty visualization file: {filepath}")
                return False

        logger.info("All required outputs validated successfully")
        return True

    except Exception as e:
        logger.error(f"Error validating outputs: {str(e)}")
        return False

def resolve_zip_for_addresses(
    df: pd.DataFrame,
    address_col: str = 'address',
    community_area_col: str = 'community_area',
    city_col: str = 'city',
    state_col: str = 'state',
    zip_col: str = 'zip_code'
) -> pd.DataFrame:
    """
    For each row, resolve ZIP code using original, community_area fallback, or geocoding.
    Adds columns: parsed_street, parsed_city, parsed_state, resolved_zip, zip_source
    """
    logger = logging.getLogger(__name__)
    
    def process_row(row: pd.Series) -> dict:
        """Process a single row to resolve its ZIP code."""
        orig_zip = str(row.get(zip_col, '')).strip()
        address = str(row.get(address_col, '')).strip()
        community_area = str(row.get(community_area_col, '')).strip()
        city = str(row.get(city_col, '')).strip()
        state = str(row.get(state_col, '')).strip() or 'IL'
        
        suspicious = any(city_name in address.upper() for city_name in CHICAGO_CITIES)
        
        if orig_zip and orig_zip.isdigit() and len(orig_zip) == 5:
            resolved_zip = orig_zip
            zip_source = 'original'
        elif community_area and community_area in COMMUNITY_AREA_ZIP_MAP:
            resolved_zip = COMMUNITY_AREA_ZIP_MAP[community_area]
            zip_source = 'community_area_fallback'
            logger.info(f"Fallback ZIP from community_area: {resolved_zip} for address: {address}")
        elif address:
            resolved_zip = geocode_address_zip(address, city, state)
            zip_source = 'geocoded' if resolved_zip else 'unresolved'
            if resolved_zip:
                logger.info(f"Geocoded ZIP: {resolved_zip} for address: {address}")
            else:
                logger.warning(f"Failed to geocode ZIP for address: {address}")
        else:
            resolved_zip = ''
            zip_source = 'unresolved'
            logger.warning(f"No ZIP found, no address to geocode for row {row.name}")
            
        return {
            'original_address': address,
            'parsed_street': address,
            'parsed_city': city,
            'parsed_state': state,
            'resolved_zip': resolved_zip,
            'zip_source': zip_source,
            'suspicious_address': suspicious
        }

    # Process all rows
    results = [process_row(row) for _, row in df.iterrows()]
    clean_df = pd.DataFrame(results)
    
    # Check for duplicate coordinates if available
    if 'lat' in df.columns and 'lon' in df.columns:
        latlon_dupes = df.groupby(['lat', 'lon']).size().reset_index(name='count')
        latlon_dupes = latlon_dupes[latlon_dupes['count'] > 1]
        if not latlon_dupes.empty:
            logger.warning(f"Duplicate lat/lon found: {latlon_dupes}")
    
    # Merge results back to original DataFrame
    result_df = df.copy()
    result_df.update(clean_df)
    return result_df


def load_chicago_zip_crosswalk(path: str) -> dict:
    """Load a local CSV mapping address/community area to ZIP code."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        # Expect columns: 'address', 'community_area', 'zip_code'
        crosswalk = {}
        for _, row in df.iterrows():
            if 'address' in row and pd.notna(row['address']):
                crosswalk[row['address'].strip().lower()] = row['zip_code']
            if 'community_area' in row and pd.notna(row['community_area']):
                crosswalk[row['community_area'].strip().lower()] = row['zip_code']
        return crosswalk
    except Exception as e:
        logging.warning(f"Failed to load ZIP crosswalk: {e}")
        return {}


def geocode_address_zip(address: str, city: str = '', state: str = 'IL', sleep: float = 1.0) -> Optional[str]:
    """Geocode an address to get ZIP code using Nominatim, with throttling and error handling."""
    try:
        if not address or not isinstance(address, str):
            return None
        query = f"{address}, {city}, {state}"
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": query, "format": "json", "addressdetails": 1, "limit": 1}
        headers = {"User-Agent": "chipop-pipeline/1.0 (contact: youremail@example.com)"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and 'address' in data[0] and 'postcode' in data[0]['address']:
                return data[0]['address']['postcode']
        time.sleep(sleep)  # Throttle
    except Exception as e:
        logging.warning(f"Nominatim geocode failed for: {address}, {city}, {state}: {e}")
    return None


def clean_zip(zip_val):
    """
    Clean and validate a ZIP code string. Returns a valid 5-digit ZIP or None.
    Handles trailing dashes, non-digit characters, and malformed values.
    Adds debug logging for invalid/cleaned ZIPs.
    """
    if pd.isnull(zip_val):
        return None
    zip_str = str(zip_val).strip()
    # Remove trailing dashes or anything after a dash
    zip_str = zip_str.split('-')[0]
    # Remove all non-digit characters
    zip_str = re.sub(r'\D', '', zip_str)
    if len(zip_str) == 5 and zip_str.isdigit():
        return zip_str
    logger.debug(f"Invalid or cleaned ZIP: original='{zip_val}', cleaned='{zip_str}'")
    return None
