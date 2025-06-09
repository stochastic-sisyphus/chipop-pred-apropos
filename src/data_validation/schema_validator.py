import pandas as pd
import logging
import traceback

logger = logging.getLogger(__name__)

class SchemaValidator:
    def validate_dataframe(self, df, schema_name):
        """
        Validate a DataFrame against a schema.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            schema_name (str): Name of the schema to validate against
            
        Returns:
            tuple: (bool, list) - (is_valid, list of validation errors)
        """
        try:
            # Check if DataFrame is empty
            if df is None or df.empty:
                logger.warning(f"Empty DataFrame provided for schema validation: {schema_name}")
                return False, ["Empty DataFrame provided"]
            
            # Get schema
            schema = self.get_schema(schema_name)
            if not schema:
                return False, [f"Schema not found: {schema_name}"]
            
            # Initialize validation errors list
            validation_errors = []
            
            # Validate required columns
            required_columns = schema.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_errors.append(f"Missing required columns: {missing_columns}")
            
            # Validate column types
            column_types = schema.get('column_types', {})
            for col, expected_type in column_types.items():
                if col in df.columns:
                    # Handle numeric types
                    if expected_type in ['int', 'float']:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if df[col].isna().any():
                                validation_errors.append(f"Column {col} contains non-numeric values")
                        except Exception as e:
                            validation_errors.append(f"Error converting column {col} to {expected_type}: {str(e)}")
                    # Handle string types
                    elif expected_type == 'str':
                        try:
                            df[col] = df[col].astype(str)
                        except Exception as e:
                            validation_errors.append(f"Error converting column {col} to string: {str(e)}")
                    # Handle datetime types
                    elif expected_type == 'datetime':
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            if df[col].isna().any():
                                validation_errors.append(f"Column {col} contains invalid datetime values")
                        except Exception as e:
                            validation_errors.append(f"Error converting column {col} to datetime: {str(e)}")
            
            # Validate value ranges
            value_ranges = schema.get('value_ranges', {})
            for col, range_info in value_ranges.items():
                if col in df.columns:
                    min_val = range_info.get('min')
                    max_val = range_info.get('max')
                    if min_val is not None and df[col].min() < min_val:
                        validation_errors.append(f"Column {col} contains values below minimum {min_val}")
                    if max_val is not None and df[col].max() > max_val:
                        validation_errors.append(f"Column {col} contains values above maximum {max_val}")
            
            # Validate ZIP codes if present
            if 'zip_code' in df.columns:
                invalid_zips = df[~df['zip_code'].astype(str).str.match(r'^\d{5}$')]['zip_code'].unique()
                if len(invalid_zips) > 0:
                    validation_errors.append(f"Invalid ZIP codes found: {invalid_zips}")
            
            # Validate required fields are not null
            for col in required_columns:
                if col in df.columns and df[col].isna().any():
                    null_count = df[col].isna().sum()
                    validation_errors.append(f"Column {col} contains {null_count} null values")
            
            # Check if validation passed
            is_valid = len(validation_errors) == 0
            
            if not is_valid:
                logger.warning(f"Schema validation failed for {schema_name}:")
                for error in validation_errors:
                    logger.warning(f"- {error}")
            
            return is_valid, validation_errors
            
        except Exception as e:
            error_msg = f"Error validating DataFrame against schema {schema_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, [error_msg] 