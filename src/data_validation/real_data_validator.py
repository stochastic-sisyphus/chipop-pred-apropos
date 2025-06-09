"""
Real Data Validator - Ensures only authentic, real data is used in the pipeline.
Rejects any simulated, calculated, or default data.
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
import re

logger = logging.getLogger(__name__)

class RealDataValidator:
    """Strict validator to ensure only real, authentic data is used."""
    
    def __init__(self):
        """Initialize the real data validator."""
        self.validation_failures = []
        self.required_real_fields = {
            'census': ['population', 'median_income', 'housing_units', 'occupied_housing_units'],
            'fred': ['date', 'value', 'series_id'],
            'chicago_permits': ['permit_', 'issue_date', 'work_description', 'reported_cost'],
            'chicago_licenses': ['license_number', 'license_start_date', 'business_activity'],
            'chicago_zoning': ['zone_class', 'zone_type'],
            'permits': ['permit_number', 'issue_date', 'reported_cost'],
            'housing': ['permit_number', 'issue_date', 'reported_cost'],
            'retail': ['business_activity', 'license_start_date'],
            'licenses': ['license_number', 'business_activity'],
            'zoning': ['zone_class']
        }
        
        self.field_alternatives = {
            'permit_number': ['permit_', 'id', 'permit_id'],
            'case_number': ['ordinance_number', 'case_id', 'id'],
            'address': ['work_location', 'site_location', 'location'],
            'reported_cost': ['estimated_cost', 'total_fee', 'permit_cost'],
            'business_activity': ['business_type', 'license_description', 'business_category']
        }
        
        self.forbidden_calculated_fields = [
            'permit_number', 'license_number', 'case_number', 'address',
            'work_description', 'business_activity', 'reported_cost'
        ]
        
        self.calculated_patterns = {
            'sequential_numbers': r'^\d+$',
            'default_values': [0, 1, -1, 999, 9999, 99999],
            'round_numbers': lambda x: x % 1000 == 0 if pd.notnull(x) and isinstance(x, (int, float)) else False
        }
    
    def validate_dataset(self, data: pd.DataFrame, dataset_name: str, source_type: str) -> Tuple[bool, List[str]]:
        """
        Validate that a dataset contains only real, authentic data.
        
        Args:
            data: Dataset to validate
            dataset_name: Name of the dataset
            source_type: Type of data source (census, fred, chicago_permits, etc.)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if data is None or (isinstance(data, pd.DataFrame) and len(data) == 0):
            issues.append(f"Dataset {dataset_name} is empty or None")
            return False, issues
        
        if isinstance(data, pd.DataFrame):
            issues.extend(self._validate_dataframe(data, dataset_name, source_type))
        elif isinstance(data, dict):
            for sub_name, sub_data in data.items():
                if isinstance(sub_data, pd.DataFrame):
                    sub_source_type = self._determine_source_type(sub_name)
                    sub_issues = self._validate_dataframe(sub_data, sub_name, sub_source_type)
                    issues.extend(sub_issues)
        else:
            issues.append(f"Dataset {dataset_name} has unsupported data type: {type(data)}")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Dataset {dataset_name} validation failed with {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def _validate_dataframe(self, data: pd.DataFrame, dataset_name: str, source_type: str) -> List[str]:
        """Validate a DataFrame for real data quality."""
        issues = []
        
        required_fields = self._get_required_real_fields(source_type)
        
        if required_fields:
            missing_fields = []
            
            for field in required_fields:
                if field in data.columns:
                    continue
                
                found_alternative = False
                if field in self.field_alternatives:
                    for alt_field in self.field_alternatives[field]:
                        if alt_field in data.columns:
                            found_alternative = True
                            break
                
                if not found_alternative:
                    missing_fields.append(field)
            
            if missing_fields:
                issues.append(f"Dataset {dataset_name} missing required real fields: {missing_fields}")
        
        if source_type not in ['retail_sales_collected', 'consumer_spending_collected']:
            for col in data.columns:
                if col in self.forbidden_calculated_fields:
                    if self._is_calculated_field(data[col], col):
                        issues.append(f"Field '{col}' in {dataset_name} appears to contain calculated/default values")
        
        quality_issues = self._validate_data_quality(data, dataset_name)
        issues.extend(quality_issues)
        
        authenticity_issues = self._validate_data_authenticity(data, dataset_name, source_type)
        issues.extend(authenticity_issues)
        
        return issues
    
    def _is_calculated_field(self, series: pd.Series, field_name: str) -> bool:
        """Check if a field contains calculated/default values."""
        if series.empty:
            return True
        
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return True
        
        if clean_series.nunique() == 1:
            single_value = clean_series.iloc[0]
            if single_value in self.calculated_patterns['default_values']:
                return True
            if isinstance(single_value, str) and single_value in ['', 'N/A', 'Unknown', 'Default']:
                return True
        
        if field_name in ['permit_number', 'license_number', 'case_number']:
            if self._appears_sequential(clean_series):
                return True
        
        if field_name in ['reported_cost', 'estimated_cost']:
            if self._appears_calculated_cost(clean_series):
                return True
        
        if field_name in ['unit_count', 'retail_sales', 'consumer_spending']:
            if self._appears_calculated_from_demographics(clean_series):
                return True
        
        return False
    
    def _appears_sequential(self, series: pd.Series) -> bool:
        """Check if values appear to be sequential (auto-generated)."""
        try:
            if series.dtype == 'object':
                numeric_series = series.str.extract(r'(\d+)')[0].astype(float)
            else:
                numeric_series = pd.to_numeric(series, errors='coerce')
            
            numeric_series = numeric_series.dropna()
            if len(numeric_series) < 3:
                return False
            
            sorted_values = sorted(numeric_series.unique())
            if len(sorted_values) < 3:
                return False
            
            consecutive_count = 0
            for i in range(1, len(sorted_values)):
                if sorted_values[i] - sorted_values[i-1] == 1:
                    consecutive_count += 1
            
            consecutive_ratio = consecutive_count / (len(sorted_values) - 1)
            return consecutive_ratio > 0.7
            
        except Exception:
            return False
    
    def _appears_calculated_cost(self, series: pd.Series) -> bool:
        """Check if cost values appear to be calculated rather than real reported costs."""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) == 0:
                return True
            
            round_thousands = sum(self.calculated_patterns['round_numbers'](x) for x in numeric_series)
            round_percentage = round_thousands / len(numeric_series)
            
            if round_percentage > 0.6:
                return True
            
            if numeric_series.nunique() == 1:
                return True
            
            coefficient_of_variation = numeric_series.std() / numeric_series.mean()
            if coefficient_of_variation < 0.1:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _appears_calculated_from_demographics(self, series: pd.Series) -> bool:
        """Check if values appear to be calculated from demographics."""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_series) == 0:
                return True
            
            round_percentage = sum(self.calculated_patterns['round_numbers'](x) for x in numeric_series) / len(numeric_series)
            if round_percentage > 0.7:
                return True
            
            if len(numeric_series) > 1 and numeric_series.nunique() == 1:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _validate_data_authenticity(self, data: pd.DataFrame, dataset_name: str, source_type: str) -> List[str]:
        """Validate that data appears to be authentic (not simulated)."""
        issues = []
        
        if 'zip_code' in data.columns:
            zip_codes = data['zip_code'].unique()
            if source_type.startswith('chicago') and len(zip_codes) > 0:
                chicago_zip_pattern = r'^60[0-9]{3}$'
                valid_chicago_zips = sum(1 for zip_code in zip_codes if re.match(chicago_zip_pattern, str(zip_code)))
                
                if valid_chicago_zips == 0:
                    issues.append(f"Dataset {dataset_name} has no valid Chicago ZIP codes")
                elif valid_chicago_zips / len(zip_codes) < 0.8:
                    issues.append(f"Dataset {dataset_name} has unusually few Chicago ZIP codes")
        
        if source_type == 'chicago_zoning' and len(data) == 0:
            issues.append(f"Dataset {dataset_name} has only {len(data)} ZIP codes (unrealistic for Chicago)")
        
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                date_series = pd.to_datetime(data[col], errors='coerce').dropna()
                if len(date_series) > 0:
                    unique_dates = date_series.nunique()
                    if unique_dates == 1 and len(date_series) > 10:
                        issues.append(f"Date column '{col}' in {dataset_name} has identical dates (suggests default values)")
            except Exception:
                pass
        
        return issues
    
    def _validate_data_quality(self, data: pd.DataFrame, dataset_name: str) -> List[str]:
        """Validate data quality and realistic ranges."""
        issues = []
        
        if 'population' in data.columns:
            pop_series = pd.to_numeric(data['population'], errors='coerce').dropna()
            if len(pop_series) > 0:
                if (pop_series < 100).any() or (pop_series > 200000).any():
                    issues.append(f"Population values in {dataset_name} outside realistic range for Chicago ZIP codes")
        
        if 'median_income' in data.columns:
            income_series = pd.to_numeric(data['median_income'], errors='coerce').dropna()
            if len(income_series) > 0:
                if (income_series < 10000).any() or (income_series > 500000).any():
                    issues.append(f"Income values in {dataset_name} outside realistic range")
        
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                date_series = pd.to_datetime(data[col], errors='coerce')
                valid_dates = date_series.dropna()
                
                if len(valid_dates) > 0:
                    current_year = pd.Timestamp.now().year
                    years = valid_dates.dt.year
                    
                    unrealistic_years = years[(years > current_year + 10) | (years < 2000)]
                    if len(unrealistic_years) > len(years) * 0.1:
                        issues.append(f"Date column '{col}' in {dataset_name} contains unrealistic dates")
            except Exception:
                issues.append(f"Date column '{col}' in {dataset_name} contains invalid date formats")
        
        return issues
    
    def enforce_real_data_only(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Enforce real data only policy - remove any datasets that fail validation.
        
        Args:
            data_dict (dict): Dictionary of datasets
            
        Returns:
            dict: Cleaned dictionary with only valid real data
        """
        validated_data = {}
        
        for dataset_name, dataset in data_dict.items():
            source_type = self._determine_source_type(dataset_name)
            
            is_valid, issues = self.validate_dataset(dataset, dataset_name, source_type)
            
            if is_valid:
                validated_data[dataset_name] = dataset
                logger.info(f"✅ Dataset '{dataset_name}' validated as real data")
            else:
                logger.error(f"❌ Dataset '{dataset_name}' REJECTED - contains simulated/calculated data:")
                for issue in issues:
                    logger.error(f"  - {issue}")
                
                self.validation_failures.append({
                    'dataset': dataset_name,
                    'issues': issues
                })
        
        return validated_data
    
    def _get_required_real_fields(self, source_type: str) -> List[str]:
        """Get required real fields for different data source types."""
        field_requirements = {
            'government_economic': [
                'date', 'series_id', 'value', 'data_source'
            ],
            'business_permit': [
                'permit_number', 'permit_type', 'issue_date', 'zip_code'
            ],
            'business_license': [
                'license_number', 'business_activity', 'license_start_date', 'zip_code'
            ],
            'census_demographic': [
                'zip_code', 'population', 'median_income', 'housing_units'
            ],
            'retail_data': [
                'zip_code', 'retail_sales', 'year', 'data_source'
            ],
            'retail_sales_collected': [
                'zip_code', 'retail_sales', 'year', 'data_source'
            ],
            'consumer_spending_collected': [
                'zip_code', 'consumer_spending', 'year', 'data_source'
            ],
            'chicago_permits': [
                'permit_number', 'permit_type', 'issue_date', 'zip_code'
            ],
            'chicago_licenses': [
                'license_number', 'business_activity', 'license_start_date', 'zip_code'
            ],
            'chicago_zoning': [
                'zone_type', 'zip_code'
            ]
        }
        
        return field_requirements.get(source_type, [])
    
    def _determine_source_type(self, dataset_name: str) -> str:
        """Determine the source type from dataset name."""
        dataset_name_lower = dataset_name.lower()
        
        # **ENHANCED: Better retail data recognition**
        if 'retail_sales_collected' in dataset_name_lower:
            return 'retail_sales_collected'
        elif 'consumer_spending_collected' in dataset_name_lower:
            return 'consumer_spending_collected'
        elif 'retail' in dataset_name_lower and 'sales' in dataset_name_lower:
            return 'retail_data'
        elif 'retail' in dataset_name_lower:
            return 'retail_data'
        elif 'fred' in dataset_name_lower or 'economic' in dataset_name_lower:
            return 'government_economic'
        elif 'permit' in dataset_name_lower:
            if 'chicago' in dataset_name_lower:
                return 'chicago_permits'
            return 'business_permit'
        elif 'license' in dataset_name_lower:
            if 'chicago' in dataset_name_lower:
                return 'chicago_licenses'
            return 'business_license'
        elif 'census' in dataset_name_lower:
            return 'census_demographic'
        elif 'zoning' in dataset_name_lower:
            return 'chicago_zoning'
        elif 'consumer' in dataset_name_lower and 'spending' in dataset_name_lower:
            return 'consumer_spending_collected'
        else:
            return 'unknown'
    
    def get_validation_report(self) -> Dict:
        """Get a summary report of validation results."""
        return {
            'total_validation_failures': len(self.validation_failures),
            'failed_datasets': [failure['dataset'] for failure in self.validation_failures],
            'failure_details': self.validation_failures
        }
    
    def require_minimum_data_quality(self, data_dict: Dict[str, pd.DataFrame], 
                                   min_datasets: int = 3) -> Tuple[bool, str]:
        """
        Ensure minimum data quality requirements are met.
        
        Args:
            data_dict (dict): Dictionary of validated datasets
            min_datasets (int): Minimum number of datasets required
            
        Returns:
            Tuple[bool, str]: (meets_requirements, reason_if_not)
        """
        if len(data_dict) < min_datasets:
            return False, f"Only {len(data_dict)} datasets passed validation, minimum {min_datasets} required"
        
        essential_datasets = ['census']
        missing_essential = [ds for ds in essential_datasets if not any(ds in name.lower() for name in data_dict.keys())]
        
        if missing_essential:
            return False, f"Missing essential real datasets: {missing_essential}"
        
        return True, "Data quality requirements met" 