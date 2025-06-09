def collect_data(self, zip_code, data_type):
        """
        Collect data for a specific ZIP code and data type.
        
        Args:
            zip_code (str): ZIP code to collect data for
            data_type (str): Type of data to collect ('census', 'economic', 'vacancy')
            
        Returns:
            pd.DataFrame: Collected data
        """
        try:
            logger.info(f"Collecting {data_type} data for ZIP code {zip_code}")
            
            # Validate inputs
            if not zip_code or not isinstance(zip_code, str):
                raise ValueError("Invalid ZIP code provided")
            if data_type not in self.supported_data_types:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            # Format ZIP code
            zip_code = str(zip_code).strip().zfill(5)
            
            # Get API endpoint and parameters
            endpoint = self.get_endpoint(data_type)
            params = self.get_parameters(zip_code, data_type)
            
            if not endpoint:
                logger.error(f"No endpoint found for data type: {data_type}")
                return pd.DataFrame()
            
            # Implement retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Make API request
                    response = requests.get(
                        endpoint,
                        params=params,
                        headers=self.headers,
                        timeout=30
                    )
                    
                    # Check response status
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    # Validate response data
                    if not data:
                        logger.warning(f"Empty response received for {data_type} data")
                        return pd.DataFrame()
                    
                    # Transform data to DataFrame
                    df = self.transform_data(data, data_type, zip_code)
                    
                    # Validate DataFrame
                    if df.empty:
                        logger.warning(f"No valid data found for {data_type} in ZIP {zip_code}")
                        return pd.DataFrame()
                    
                    logger.info(f"Successfully collected {data_type} data for ZIP {zip_code}")
                    return df
                    
                except requests.exceptions.RequestException as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All retry attempts failed for {data_type} data in ZIP {zip_code}")
                        break
                        
                except Exception as e:
                    last_error = e
                    logger.error(f"Error processing {data_type} data: {str(e)}")
                    break
            
            # If we get here, all retries failed
            logger.error(f"Failed to collect {data_type} data for ZIP {zip_code}: {str(last_error)}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Unexpected error collecting {data_type} data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_endpoint(self, data_type):
        """Get API endpoint for data type."""
        endpoints = {
            'census': 'https://api.census.gov/data/2020/dec/pl',
            'economic': 'https://api.census.gov/data/2020/acs/acs5',
            'vacancy': 'https://api.census.gov/data/2020/acs/acs5'
        }
        return endpoints.get(data_type)
    
    def get_parameters(self, zip_code, data_type):
        """Get API parameters for data type."""
        base_params = {
            'key': self.api_key,
            'for': f'zip code tabulation area:{zip_code}'
        }
        
        if data_type == 'census':
            base_params.update({
                'get': 'P1_001N',  # Total population
                'vintage': '2020'
            })
        elif data_type == 'economic':
            base_params.update({
                'get': 'B19013_001E',  # Median household income
                'vintage': '2020'
            })
        elif data_type == 'vacancy':
            base_params.update({
                'get': 'B25002_003E',  # Vacant housing units
                'vintage': '2020'
            })
        
        return base_params
    
    def transform_data(self, data, data_type, zip_code):
        """Transform API response to DataFrame."""
        try:
            if not data or not isinstance(data, list) or len(data) < 2:
                return pd.DataFrame()
            
            # Extract headers and values
            headers = data[0]
            values = data[1]
            
            # Create DataFrame
            df = pd.DataFrame([values], columns=headers)
            
            # Add metadata
            df['zip_code'] = zip_code
            df['data_type'] = data_type
            df['collection_date'] = pd.Timestamp.now()
            
            return df
            
        except Exception as e:
            logger.error(f"Error transforming {data_type} data: {str(e)}")
            return pd.DataFrame() 