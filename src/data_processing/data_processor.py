def clean_data(self, data):
        """
        Clean and preprocess the data.
        
        Args:
            data (pd.DataFrame): Raw data to clean
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            logger.info(f"Starting data cleaning for {len(data)} records")
            
            if data.empty:
                logger.warning("Empty DataFrame received, returning empty DataFrame with correct schema")
                return pd.DataFrame(columns=self.required_columns)
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Handle missing values
            for col in df.columns:
                if col in self.numeric_columns:
                    # For numeric columns, fill with median if available, otherwise 0
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
                elif col in self.categorical_columns:
                    # For categorical columns, fill with mode if available, otherwise empty string
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else ''
                    df[col] = df[col].fillna(mode_val)
                else:
                    # For other columns, fill with empty string
                    df[col] = df[col].fillna('')
            
            # Clean numeric columns
            for col in self.numeric_columns:
                if col in df.columns:
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Replace negative values with 0 for non-negative columns
                    if col in self.non_negative_columns:
                        df[col] = df[col].clip(lower=0)
                    # Handle outliers using IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    # Replace outliers with bounds
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Clean categorical columns
            for col in self.categorical_columns:
                if col in df.columns:
                    # Convert to string and strip whitespace
                    df[col] = df[col].astype(str).str.strip()
                    # Convert to lowercase
                    df[col] = df[col].str.lower()
                    # Remove special characters
                    df[col] = df[col].str.replace(r'[^a-z0-9\s]', '', regex=True)
                    # Replace multiple spaces with single space
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Clean ZIP codes
            if 'zip_code' in df.columns:
                # Convert to string and strip
                df['zip_code'] = df['zip_code'].astype(str).str.strip()
                # Extract 5-digit ZIP code if embedded in longer string
                zip_extracted = df['zip_code'].str.extract(r'(\d{5})')
                if zip_extracted is not None and not zip_extracted.empty:
                    df['zip_code'] = zip_extracted.iloc[:, 0]
                # Ensure 5-digit format
                mask = df['zip_code'].str.len() > 0
                df.loc[mask, 'zip_code'] = df.loc[mask, 'zip_code'].str.zfill(5)
                # Drop invalid ZIP codes
                valid_zip_mask = df['zip_code'].str.match(r'^\d{5}$')
                df = df[valid_zip_mask]
            
            # Clean dates
            for col in self.date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Error converting {col} to datetime: {str(e)}")
            
            # Log cleaning statistics
            logger.info("\nCleaning Statistics:")
            for col in df.columns:
                non_null = df[col].count()
                null_pct = (df[col].isna().sum() / len(df)) * 100
                logger.info(f"{col}: {non_null} non-null values ({null_pct:.1f}% null)")
            
            logger.info(f"Data cleaning completed successfully for {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            logger.error(traceback.format_exc())
            # Return empty DataFrame with correct schema instead of raising error
            return pd.DataFrame(columns=self.required_columns) 