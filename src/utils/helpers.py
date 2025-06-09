"""
Helper utilities for the Chicago Population Analysis project.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import re

logger = logging.getLogger(__name__)

def ensure_directory(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path (str or Path): Directory path to ensure exists
        
    Returns:
        Path: Path to the directory
    """
    try:
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(f"Error ensuring directory {directory_path}: {str(e)}")
        return None

def load_json(file_path):
    """
    Load JSON data from file.
    
    Args:
        file_path (str or Path): Path to JSON file
        
    Returns:
        dict: Loaded JSON data or None if error
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"JSON file not found: {path}")
            return None
            
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return None

def save_json(data, file_path):
    """
    Save data to JSON file.
    
    Args:
        data (dict): Data to save
        file_path (str or Path): Path to save JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        ensure_directory(path.parent)
            
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False

def calculate_growth_rate(start_value, end_value, periods=1):
    """
    Calculate compound annual growth rate.
    
    Args:
        start_value (float): Starting value
        end_value (float): Ending value
        periods (int): Number of periods
        
    Returns:
        float: Growth rate as decimal
    """
    try:
        if start_value <= 0 or end_value <= 0:
            return 0
            
        return (end_value / start_value) ** (1 / periods) - 1
    except Exception as e:
        logger.error(f"Error calculating growth rate: {str(e)}")
        return 0

def calculate_summary_stats(data):
    """
    Calculate summary statistics for a series.
    
    Args:
        data (pd.Series): Data series
        
    Returns:
        dict: Summary statistics
    """
    try:
        if data is None or len(data) == 0:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'std': None
            }
            
        return {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'min': data.min(),
            'max': data.max(),
            'std': data.std()
        }
    except Exception as e:
        logger.error(f"Error calculating summary stats: {str(e)}")
        return None

def format_currency(value, decimals=0):
    """
    Format value as currency.
    
    Args:
        value (float): Value to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted currency string
    """
    try:
        if pd.isna(value):
            return "N/A"
            
        return f"${value:,.{decimals}f}"
    except Exception as e:
        logger.error(f"Error formatting currency: {str(e)}")
        return str(value)

def format_percent(value, decimals=1):
    """
    Format value as percentage.
    
    Args:
        value (float): Value to format (as decimal)
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    try:
        if pd.isna(value):
            return "N/A"
            
        return f"{value * 100:.{decimals}f}%"
    except Exception as e:
        logger.error(f"Error formatting percentage: {str(e)}")
        return str(value)

def validate_zip_code(zip_code):
    """
    Validate a ZIP code.
    
    Args:
        zip_code (str): ZIP code to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Ensure it's a string
        zip_str = str(zip_code).strip()
        
        # Check length
        if len(zip_str) != 5:
            return False
            
        # Check if numeric
        if not zip_str.isdigit():
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating ZIP code: {str(e)}")
        return False

def clean_zip(zip_code):
    """
    Clean and standardize a ZIP code.
    
    Args:
        zip_code (str or int): ZIP code to clean
        
    Returns:
        str: Cleaned ZIP code or None if invalid
    """
    try:
        if pd.isna(zip_code):
            return None
            
        # Convert to string
        zip_str = str(zip_code).strip()
        
        # Remove non-numeric characters
        zip_str = re.sub(r'[^0-9]', '', zip_str)
        
        # Check if empty
        if not zip_str:
            return None
            
        # Take first 5 digits
        zip_str = zip_str[:5]
        
        # Pad with leading zeros if needed
        zip_str = zip_str.zfill(5)
        
        # Validate
        if validate_zip_code(zip_str):
            return zip_str
        else:
            return None
    except Exception as e:
        logger.error(f"Error cleaning ZIP code: {str(e)}")
        return None

def geocode_address_zip(address, city, state, sleep=0.0):
    """
    Geocode an address to get its ZIP code using Nominatim.
    
    Args:
        address (str): Street address
        city (str): City
        state (str): State
        sleep (float): Sleep time between requests to avoid rate limiting
        
    Returns:
        str: ZIP code or None if geocoding failed
    """
    try:
        # Import here to avoid dependency issues
        import requests
        
        # Sleep to avoid rate limiting
        if sleep > 0:
            time.sleep(sleep)
            
        # Format address
        formatted_address = f"{address}, {city}, {state}"
        
        # Call Nominatim API
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": formatted_address,
            "format": "json",
            "addressdetails": 1,
            "limit": 1
        }
        headers = {
            "User-Agent": "ChicagoPopulationAnalysis/1.0"
        }
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"Geocoding failed with status {response.status_code}: {formatted_address}")
            return None
            
        results = response.json()
        
        if not results:
            logger.warning(f"No geocoding results for: {formatted_address}")
            return None
            
        # Extract ZIP code from address details
        address_details = results[0].get("address", {})
        
        # Try different possible field names for postal code
        for field in ["postcode", "postal_code", "postalcode", "zip", "zipcode"]:
            if field in address_details:
                zip_code = address_details[field]
                return clean_zip(zip_code)
                
        logger.warning(f"No ZIP code found in geocoding results for: {formatted_address}")
        return None
        
    except Exception as e:
        logger.error(f"Error geocoding address: {str(e)}")
        return None

def ensure_output_structure():
    """
    Ensure all required output directories exist.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create required directories
        directories = [
            "data/raw",
            "data/interim",
            "data/processed",
            "output/models",
            "output/visualizations",
            "output/reports",
            "output/reports/figures",
            "output/reports/tables",
            "logs"
        ]
        
        for directory in directories:
            ensure_directory(directory)
        
        return True
    except Exception as e:
        logger.error(f"Error ensuring output structure: {str(e)}")
        return False

def validate_outputs():
    """
    Validate that all required outputs exist.
    
    Returns:
        bool: True if all outputs exist, False otherwise
    """
    try:
        # Required output files
        required_files = [
            "data/processed/merged_dataset.csv",
            "data/processed/census_processed.csv",
            "data/processed/permits_processed.csv",
            "data/processed/business_licenses_processed.csv"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing required output files: {', '.join(missing_files)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating outputs: {str(e)}")
        return False
