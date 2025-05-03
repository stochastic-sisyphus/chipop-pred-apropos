"""Data validation utilities."""
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)

def validate_data_files() -> Dict[str, bool]:
    """Validate existence and contents of all required data files."""
    required_files = {
        'census': settings.PROCESSED_DATA_DIR / 'census_processed.csv',
        'permits': settings.PROCESSED_DATA_DIR / 'permits_processed.csv',
        'economic': settings.PROCESSED_DATA_DIR / 'economic_processed.csv',
        'retail': settings.PROCESSED_DATA_DIR / 'retail_metrics.csv',
        'business': settings.PROCESSED_DATA_DIR / 'business_licenses.csv',
        'predictions': settings.PREDICTIONS_DIR / 'scenario_predictions.csv'
    }
    
    results = {}
    for name, path in required_files.items():
        if not path.exists():
            logger.error(f"{name} data file missing: {path}")
            results[name] = False
            continue
            
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.error(f"{name} data file is empty: {path}")
                results[name] = False
                continue
                
            logger.info(f"Validated {name} data: {df.shape[0]} rows, {df.shape[1]} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Check for required columns
            missing_cols = check_required_columns(name, df)
            if missing_cols:
                logger.error(f"Missing required columns in {name}: {missing_cols}")
                results[name] = False
                continue
                
            results[name] = True
            
        except Exception as e:
            logger.error(f"Error validating {name} data: {str(e)}")
            results[name] = False
            
    return results

def check_required_columns(dataset: str, df: pd.DataFrame) -> List[str]:
    """Check if required columns are present in dataset."""
    required_columns = {
        'census': ['zip_code', 'year', 'total_population', 'median_household_income'],
        'permits': ['zip_code', 'year', 'residential_permits', 'commercial_permits', 'retail_permits'],
        'economic': ['year', 'unemployment_rate', 'real_gdp', 'per_capita_income'],
        'retail': ['zip_code', 'year', 'retail_deficit', 'retail_leakage'],
        'business': ['zip_code', 'year', 'license_count'],
        'predictions': ['zip_code', 'year', 'predicted_population', 'scenario']
    }
    
    if dataset not in required_columns:
        return []
        
    return [col for col in required_columns[dataset] if col not in df.columns]

def validate_merged_dataset(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate the merged dataset for completeness."""
    issues = {
        'missing_columns': [],
        'null_columns': [],
        'dtype_issues': []
    }
    
    # Check required columns
    required_columns = [
        'zip_code', 'year', 'total_population', 'median_household_income',
        'residential_permits', 'commercial_permits', 'retail_permits',
        'retail_deficit', 'retail_leakage'
    ]
    
    issues['missing_columns'] = [col for col in required_columns if col not in df.columns]
    
    # Check for null values
    null_cols = df[df.columns[df.isnull().any()]].columns.tolist()
    if null_cols:
        issues['null_columns'] = null_cols
        
    # Check data types
    if 'zip_code' in df.columns and df['zip_code'].dtype != 'object':
        issues['dtype_issues'].append('zip_code should be string type')
    if 'year' in df.columns and df['year'].dtype != 'int64':
        issues['dtype_issues'].append('year should be integer type')
        
    return issues

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Validate all data files
    results = validate_data_files()
    
    # Print summary
    print("\nValidation Results:")
    for dataset, is_valid in results.items():
        status = "✅" if is_valid else "❌"
        print(f"{status} {dataset}")
        
    # Exit with error if any validation failed
    if not all(results.values()):
        exit(1)
    exit(0) 