# Chicago Housing Pipeline & Population Shift Project - Data Integration Framework

## Overview

This document outlines the comprehensive data integration framework for the Chicago Housing Pipeline & Population Shift Project. The framework is designed to address the critical gaps identified in the audit and provide a robust foundation for advanced analytics, visualization, and reporting.

## Architecture

The data integration framework follows a modular, layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources Layer                      │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│  Census API │   FRED API  │ Chicago Data│    BEA API  │ Web │
│             │             │   Portal    │             │Scrape│
└─────┬───────┴──────┬──────┴──────┬──────┴──────┬──────┴──┬──┘
      │              │             │             │         │
      ▼              ▼             ▼             ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                     │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│ Demographic │  Economic   │   Housing   │   Retail    │ Geo- │
│ Collector   │  Collector  │  Collector  │  Collector  │ Data │
└─────┬───────┴──────┬──────┴──────┬──────┴──────┬──────┴──┬──┘
      │              │             │             │         │
      ▼              ▼             ▼             ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                     │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│ Validation  │  Cleaning   │ Integration │   Feature   │ Data │
│             │             │             │ Engineering │ Store│
└─────┬───────┴──────┬──────┴──────┬──────┴──────┬──────┴──┬──┘
      │              │             │             │         │
      ▼              ▼             ▼             ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Analytics Layer                          │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│ Forecasting │ Regression  │  Clustering │  Spatial    │ ML  │
│  Models     │  Analysis   │ Algorithms  │  Analysis   │Models│
└─────────────┴─────────────┴─────────────┴─────────────┴─────┘
```

## Data Sources

### Census API
- **Data Types**: Population, demographics, housing units, income
- **Granularity**: ZIP code level
- **Time Range**: Historical (10+ years) and current
- **Update Frequency**: Annual
- **Authentication**: API key required

### FRED API
- **Data Types**: Economic indicators, interest rates, market trends
- **Granularity**: National, state, and metro area
- **Time Range**: Historical (20+ years) and current
- **Update Frequency**: Monthly/quarterly
- **Authentication**: API key required

### Chicago Data Portal
- **Data Types**: Building permits, business licenses, zoning changes
- **Granularity**: Address and ZIP code level
- **Time Range**: Current and recent historical
- **Update Frequency**: Daily/weekly
- **Authentication**: Token required for high-volume access

### BEA API
- **Data Types**: Retail GDP, economic indicators, consumer spending
- **Granularity**: County and metro area
- **Time Range**: Historical (10+ years) and current
- **Update Frequency**: Quarterly/annual
- **Authentication**: API key required

### Web Scraping
- **Data Types**: Retail vacancies, property listings, development news
- **Granularity**: Address and neighborhood level
- **Time Range**: Current
- **Update Frequency**: On-demand
- **Authentication**: None (public data)

## Data Collection Components

### DemographicCollector
- Collects population and demographic data from Census API
- Handles historical trends and current estimates
- Supports ZIP code level granularity
- Implements caching for efficient API usage

### EconomicCollector
- Collects economic indicators from FRED API and BEA
- Handles national, state, and local economic data
- Supports time series data for trend analysis
- Implements rate limiting and error handling

### HousingCollector
- Collects building permits and housing data from Chicago Data Portal
- Handles property development and zoning information
- Supports address and ZIP code level granularity
- Implements geospatial data processing

### RetailCollector
- Collects business licenses and retail data from Chicago Data Portal
- Handles retail category classification and vacancy information
- Supports address and ZIP code level granularity
- Implements retail category mapping and standardization

### GeoDataCollector
- Collects geospatial data for mapping and spatial analysis
- Handles ZIP code boundaries and neighborhood definitions
- Supports various geographic levels (ZIP, neighborhood, community area)
- Implements spatial data processing and transformation

## Data Processing Components

### DataValidator
- Validates data against schema definitions
- Handles missing data detection and reporting
- Supports data quality metrics and thresholds
- Implements validation rules for each data type

### DataCleaner
- Cleans and standardizes data formats
- Handles outlier detection and treatment
- Supports data imputation for missing values
- Implements data normalization and standardization

### DataIntegrator
- Integrates data from multiple sources
- Handles entity resolution and deduplication
- Supports temporal alignment of different data sources
- Implements data merging and joining strategies

### FeatureEngineer
- Creates derived features for advanced analytics
- Handles feature selection and importance analysis
- Supports dimensionality reduction techniques
- Implements feature transformation and encoding

### DataStore
- Stores processed data in standardized formats
- Handles versioning and lineage tracking
- Supports efficient data retrieval for analytics
- Implements caching and persistence strategies

## Implementation Details

### Data Collection Implementation

```python
class BaseCollector:
    """Base class for all data collectors."""
    
    def __init__(self, api_key=None, cache_dir=None):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else Path(settings.DATA_DIR) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def collect(self):
        """
        Collect data from the source.
        
        Returns:
            pd.DataFrame: Collected data
        """
        raise NotImplementedError("Subclasses must implement collect()")
    
    def _cache_data(self, data, cache_name):
        """Cache data to file."""
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        data.to_pickle(cache_path)
        self.logger.info(f"Cached data to {cache_path}")
    
    def _load_cached_data(self, cache_name):
        """Load data from cache if available."""
        cache_path = self.cache_dir / f"{cache_name}.pkl"
        if cache_path.exists():
            self.logger.info(f"Loading cached data from {cache_path}")
            return pd.read_pickle(cache_path)
        return None
    
    def _handle_api_error(self, error, retry_count=3, retry_delay=5):
        """Handle API errors with retry logic."""
        for i in range(retry_count):
            self.logger.warning(f"API error: {error}. Retrying {i+1}/{retry_count}...")
            time.sleep(retry_delay)
            try:
                return self.collect()
            except Exception as e:
                error = e
        self.logger.error(f"Failed after {retry_count} retries: {error}")
        return None
```

### Census API Integration

```python
class CensusCollector(BaseCollector):
    """Collector for Census API data."""
    
    def __init__(self, api_key=None, cache_dir=None):
        super().__init__(api_key, cache_dir)
        self.api_key = api_key or os.environ.get('CENSUS_API_KEY')
        if not self.api_key:
            self.logger.warning("Census API key not set. Set CENSUS_API_KEY environment variable.")
    
    def collect(self, year=None, variables=None, geo_unit='zip code tabulation area'):
        """
        Collect demographic data from Census API.
        
        Args:
            year (int, optional): Census year. Defaults to latest available.
            variables (list, optional): Census variables to collect. Defaults to standard set.
            geo_unit (str, optional): Geographic unit. Defaults to 'zip code tabulation area'.
            
        Returns:
            pd.DataFrame: Census data
        """
        try:
            # Check cache first
            cache_name = f"census_{year}_{geo_unit.replace(' ', '_')}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Set default year if not provided
            if year is None:
                year = 2020  # Latest decennial census
            
            # Set default variables if not provided
            if variables is None:
                variables = [
                    'B01001_001E',  # Total population
                    'B19013_001E',  # Median household income
                    'B25001_001E',  # Total housing units
                    'B25003_001E',  # Occupied housing units
                    'B25003_003E',  # Renter-occupied housing units
                ]
            
            # Initialize Census API client
            from census import Census
            c = Census(self.api_key)
            
            # Collect data
            self.logger.info(f"Collecting Census data for year {year}")
            if year >= 2010:
                # Use ACS 5-year estimates for recent years
                data = c.acs5.get(variables, {'for': f'{geo_unit}:*'}, year=year)
            else:
                # Use decennial census for older years
                data = c.sf1.get(variables, {'for': f'{geo_unit}:*'}, year=year)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns
            column_map = {
                'B01001_001E': 'population',
                'B19013_001E': 'median_income',
                'B25001_001E': 'housing_units',
                'B25003_001E': 'occupied_housing_units',
                'B25003_003E': 'renter_occupied_units',
                f'{geo_unit}': 'geo_code'
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            
            # Add year column
            df['year'] = year
            
            # Convert numeric columns
            for col in df.columns:
                if col not in ['geo_code', 'year']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename ZIP code column and ensure string format
            if 'zip code tabulation area' in df.columns:
                df = df.rename(columns={'zip code tabulation area': 'zip_code'})
            if 'geo_code' in df.columns:
                df = df.rename(columns={'geo_code': 'zip_code'})
            
            if 'zip_code' in df.columns:
                df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            self.logger.info(f"Collected {len(df)} records from Census API")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting Census data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._handle_api_error(e)
    
    def collect_historical(self, start_year=2010, end_year=2020, variables=None):
        """
        Collect historical Census data for multiple years.
        
        Args:
            start_year (int): Start year
            end_year (int): End year
            variables (list, optional): Census variables to collect
            
        Returns:
            pd.DataFrame: Historical Census data
        """
        all_data = []
        for year in range(start_year, end_year + 1):
            df = self.collect(year=year, variables=variables)
            if df is not None:
                all_data.append(df)
        
        if not all_data:
            self.logger.error("Failed to collect any historical Census data")
            return None
        
        # Combine all years
        historical_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Collected historical Census data from {start_year} to {end_year}")
        return historical_df
```

### FRED API Integration

```python
class FREDCollector(BaseCollector):
    """Collector for FRED API data."""
    
    def __init__(self, api_key=None, cache_dir=None):
        super().__init__(api_key, cache_dir)
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        if not self.api_key:
            self.logger.warning("FRED API key not set. Set FRED_API_KEY environment variable.")
    
    def collect(self, series_ids=None, start_date=None, end_date=None):
        """
        Collect economic data from FRED API.
        
        Args:
            series_ids (list, optional): FRED series IDs to collect. Defaults to standard set.
            start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to 10 years ago.
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
            
        Returns:
            pd.DataFrame: FRED economic data
        """
        try:
            # Check cache first
            cache_name = f"fred_{start_date}_{end_date}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')  # ~10 years
            
            # Set default series IDs if not provided
            if series_ids is None:
                series_ids = [
                    'MORTGAGE30US',  # 30-Year Fixed Rate Mortgage Average
                    'CPIAUCSL',      # Consumer Price Index for All Urban Consumers
                    'HOUST',         # Housing Starts: Total: New Privately Owned Housing Units Started
                    'RRVRUSQ156N',   # Rental Vacancy Rate in the United States
                    'MSPUS',         # Median Sales Price of Houses Sold for the United States
                ]
            
            # Initialize FRED API client
            import fredapi
            fred = fredapi.Fred(api_key=self.api_key)
            
            # Collect data for each series
            all_series = []
            for series_id in series_ids:
                self.logger.info(f"Collecting FRED data for series {series_id}")
                series = fred.get_series(series_id, start_date, end_date)
                if series is not None:
                    # Convert to DataFrame
                    series_df = series.reset_index()
                    series_df.columns = ['date', 'value']
                    series_df['series_id'] = series_id
                    all_series.append(series_df)
            
            if not all_series:
                self.logger.error("Failed to collect any FRED data")
                return None
            
            # Combine all series
            df = pd.concat(all_series, ignore_index=True)
            
            # Add year column
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            self.logger.info(f"Collected {len(df)} records from FRED API")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting FRED data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._handle_api_error(e)
    
    def collect_local_indicators(self, metro_area='CHICAGO', indicators=None):
        """
        Collect local economic indicators for a specific metro area.
        
        Args:
            metro_area (str): Metro area code
            indicators (list, optional): Indicator series IDs
            
        Returns:
            pd.DataFrame: Local economic indicators
        """
        # Set default indicators if not provided
        if indicators is None:
            indicators = [
                f'{metro_area}UR',      # Unemployment Rate
                f'{metro_area}HOUS',    # Housing Price Index
                f'{metro_area}RETAIL',  # Retail Sales
                f'{metro_area}EMP',     # Total Employment
            ]
        
        return self.collect(series_ids=indicators)
```

### Chicago Data Portal Integration

```python
class ChicagoDataCollector(BaseCollector):
    """Collector for Chicago Data Portal."""
    
    def __init__(self, api_token=None, cache_dir=None):
        super().__init__(api_token, cache_dir)
        self.api_token = api_token or os.environ.get('CHICAGO_DATA_TOKEN')
        if not self.api_token:
            self.logger.warning("Chicago Data Portal token not set. Set CHICAGO_DATA_TOKEN environment variable.")
    
    def collect_building_permits(self, limit=10000, years=None):
        """
        Collect building permit data from Chicago Data Portal.
        
        Args:
            limit (int): Maximum number of records to collect
            years (list, optional): List of years to filter by
            
        Returns:
            pd.DataFrame: Building permit data
        """
        try:
            # Check cache first
            years_str = '_'.join(map(str, years)) if years else 'all'
            cache_name = f"chicago_permits_{years_str}_{limit}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Initialize Socrata client
            from sodapy import Socrata
            client = Socrata("data.cityofchicago.org", self.api_token)
            
            # Build query
            query = f"SELECT * LIMIT {limit}"
            if years:
                year_filters = " OR ".join([f"issue_date LIKE '%{year}%'" for year in years])
                query = f"SELECT * WHERE ({year_filters}) LIMIT {limit}"
            
            # Collect data
            self.logger.info(f"Collecting building permit data from Chicago Data Portal")
            results = client.get("ydr8-5enu", query=query)
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)
            
            # Process data
            if 'issue_date' in df.columns:
                df['issue_date'] = pd.to_datetime(df['issue_date'])
                df['permit_year'] = df['issue_date'].dt.year
            
            if 'zip_code' in df.columns:
                df['zip_code'] = df['zip_code'].astype(str).str.extract('(\d{5})').iloc[:, 0]
                df['zip_code'] = df['zip_code'].str.zfill(5)
            
            # Extract unit count from permit description
            if 'permit_type' in df.columns and 'unit_count' not in df.columns:
                df['unit_count'] = 1  # Default
                # Extract unit counts from descriptions
                multi_family_mask = df['permit_type'].str.contains('MULTI-FAMILY|APARTMENT', case=False, na=False)
                df.loc[multi_family_mask, 'is_multifamily'] = True
            else:
                df['is_multifamily'] = False
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            self.logger.info(f"Collected {len(df)} building permits from Chicago Data Portal")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting building permit data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._handle_api_error(e)
    
    def collect_business_licenses(self, limit=10000, years=None):
        """
        Collect business license data from Chicago Data Portal.
        
        Args:
            limit (int): Maximum number of records to collect
            years (list, optional): List of years to filter by
            
        Returns:
            pd.DataFrame: Business license data
        """
        try:
            # Check cache first
            years_str = '_'.join(map(str, years)) if years else 'all'
            cache_name = f"chicago_licenses_{years_str}_{limit}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Initialize Socrata client
            from sodapy import Socrata
            client = Socrata("data.cityofchicago.org", self.api_token)
            
            # Build query
            query = f"SELECT * LIMIT {limit}"
            if years:
                year_filters = " OR ".join([f"license_start_date LIKE '%{year}%'" for year in years])
                query = f"SELECT * WHERE ({year_filters}) LIMIT {limit}"
            
            # Collect data
            self.logger.info(f"Collecting business license data from Chicago Data Portal")
            results = client.get("r5kz-chrr", query=query)
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(results)
            
            # Process data
            if 'license_start_date' in df.columns:
                df['license_start_date'] = pd.to_datetime(df['license_start_date'])
                df['license_year'] = df['license_start_date'].dt.year
            
            if 'zip_code' in df.columns:
                df['zip_code'] = df['zip_code'].astype(str).str.extract('(\d{5})').iloc[:, 0]
                df['zip_code'] = df['zip_code'].str.zfill(5)
            
            # Categorize business types
            if 'business_activity' in df.columns and 'retail_category' not in df.columns:
                # Map business activities to retail categories
                retail_mapping = {
                    'RETAIL FOOD ESTABLISHMENT': 'food',
                    'RETAIL STORE': 'general',
                    'RESTAURANT': 'food',
                    'TAVERN': 'food',
                    'PACKAGE GOODS': 'food',
                    'TOBACCO': 'general',
                    'FILLING STATION': 'auto',
                    'MOTOR VEHICLE REPAIR': 'auto',
                    'RETAIL COMPUTER STORE': 'electronics',
                    'RETAIL CLOTHING STORE': 'clothing',
                    'RETAIL FURNITURE STORE': 'furniture',
                    'RETAIL DRUG STORE': 'health',
                    'RETAIL HEALTH CLUB': 'health',
                    'RETAIL GROCERY STORE': 'food',
                    'RETAIL BAKERY': 'food',
                    'RETAIL FLORIST': 'general',
                    'RETAIL HARDWARE STORE': 'general',
                    'RETAIL BOOK STORE': 'general',
                    'RETAIL DEPARTMENT STORE': 'general',
                    'RETAIL AUTO PARTS STORE': 'auto',
                    'RETAIL APPAREL STORE': 'clothing',
                    'RETAIL SHOE STORE': 'clothing',
                    'RETAIL SPORTING GOODS STORE': 'general',
                    'RETAIL JEWELRY STORE': 'general',
                    'RETAIL GARDEN SUPPLY STORE': 'general',
                    'RETAIL OFFICE SUPPLY STORE': 'general',
                    'RETAIL ELECTRONICS STORE': 'electronics',
                    'RETAIL HOME IMPROVEMENT STORE': 'general',
                    'RETAIL COSMETICS STORE': 'health',
                    'RETAIL PET STORE': 'general',
                    'RETAIL TOY STORE': 'general',
                    'RETAIL MUSIC STORE': 'general',
                    'RETAIL LIQUOR STORE': 'food',
                    'RETAIL OPTICAL STORE': 'health',
                    'RETAIL PHARMACY': 'health',
                    'RETAIL CONVENIENCE STORE': 'food',
                }
                
                df['retail_category'] = 'other'
                for activity, category in retail_mapping.items():
                    mask = df['business_activity'].str.contains(activity, case=False, na=False)
                    df.loc[mask, 'retail_category'] = category
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            self.logger.info(f"Collected {len(df)} business licenses from Chicago Data Portal")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting business license data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._handle_api_error(e)
```

### BEA API Integration

```python
class BEACollector(BaseCollector):
    """Collector for BEA API data."""
    
    def __init__(self, api_key=None, cache_dir=None):
        super().__init__(api_key, cache_dir)
        self.api_key = api_key or os.environ.get('BEA_API_KEY')
        if not self.api_key:
            self.logger.warning("BEA API key not set. Set BEA_API_KEY environment variable.")
    
    def collect_retail_gdp(self, years=None, area_type='MSA'):
        """
        Collect retail GDP data from BEA API.
        
        Args:
            years (list, optional): List of years to collect data for
            area_type (str): Geographic area type (MSA, county, state)
            
        Returns:
            pd.DataFrame: Retail GDP data
        """
        try:
            # Check cache first
            years_str = '_'.join(map(str, years)) if years else 'all'
            cache_name = f"bea_retail_{area_type}_{years_str}"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Set default years if not provided
            if years is None:
                current_year = datetime.now().year
                years = list(range(current_year - 10, current_year))
            
            # Build API request parameters
            params = {
                'UserID': self.api_key,
                'method': 'GetData',
                'datasetname': 'Regional',
                'TableName': 'CAGDP2',
                'LineCode': 29,  # Retail Trade
                'GeoFips': 'CHICAGO MSA' if area_type == 'MSA' else 'STATE',
                'Year': ','.join(map(str, years)),
                'ResultFormat': 'JSON'
            }
            
            # Make API request
            self.logger.info(f"Collecting retail GDP data from BEA API")
            import requests
            response = requests.get('https://apps.bea.gov/api/data', params=params)
            
            if response.status_code != 200:
                self.logger.error(f"BEA API error: {response.status_code} - {response.text}")
                return None
            
            # Parse response
            data = response.json()
            
            # Extract data from response
            if 'BEAAPI' in data and 'Results' in data['BEAAPI'] and 'Data' in data['BEAAPI']['Results']:
                results = data['BEAAPI']['Results']['Data']
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Process data
                if 'TimePeriod' in df.columns:
                    df['year'] = df['TimePeriod'].astype(int)
                
                if 'DataValue' in df.columns:
                    df['retail_gdp'] = pd.to_numeric(df['DataValue'], errors='coerce')
                
                if 'GeoName' in df.columns:
                    df['area_name'] = df['GeoName']
                
                # Cache the data
                self._cache_data(df, cache_name)
                
                self.logger.info(f"Collected {len(df)} retail GDP records from BEA API")
                return df
            else:
                self.logger.error("Invalid response format from BEA API")
                return None
            
        except Exception as e:
            self.logger.error(f"Error collecting retail GDP data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._handle_api_error(e)
```

### Web Scraping Integration

```python
class WebScraper(BaseCollector):
    """Web scraper for additional data sources."""
    
    def __init__(self, cache_dir=None):
        super().__init__(None, cache_dir)
    
    def scrape_retail_vacancies(self, urls=None):
        """
        Scrape retail vacancy data from specified URLs.
        
        Args:
            urls (list, optional): List of URLs to scrape
            
        Returns:
            pd.DataFrame: Retail vacancy data
        """
        try:
            # Check cache first
            cache_name = "scraped_retail_vacancies"
            cached_data = self._load_cached_data(cache_name)
            if cached_data is not None:
                return cached_data
            
            # Set default URLs if not provided
            if urls is None:
                urls = [
                    "https://www.loopnet.com/search/retail-space/chicago-il/for-lease/",
                    "https://www.showcase.com/retail/il/chicago/",
                    "https://www.crexi.com/lease/properties/retail/illinois/chicago"
                ]
            
            # Initialize web scraping tools
            import requests
            from bs4 import BeautifulSoup
            
            all_data = []
            
            for url in urls:
                self.logger.info(f"Scraping retail vacancy data from {url}")
                
                # Make request with headers to avoid blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    self.logger.warning(f"Failed to scrape {url}: {response.status_code}")
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data (implementation depends on specific website structure)
                # This is a simplified example
                listings = []
                
                # Example for LoopNet
                if "loopnet.com" in url:
                    for item in soup.select('.placard'):
                        address = item.select_one('.placard-address')
                        size = item.select_one('.placard-size')
                        price = item.select_one('.placard-price')
                        
                        if address:
                            listing = {
                                'address': address.text.strip(),
                                'size_sqft': size.text.strip() if size else None,
                                'price': price.text.strip() if price else None,
                                'source': 'loopnet',
                                'url': url
                            }
                            listings.append(listing)
                
                # Add more website-specific parsing logic here
                
                all_data.extend(listings)
            
            if not all_data:
                self.logger.warning("No retail vacancy data scraped")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Process data
            if 'address' in df.columns:
                # Extract ZIP code from address
                df['zip_code'] = df['address'].str.extract(r'Chicago,\s+IL\s+(\d{5})')
            
            if 'size_sqft' in df.columns:
                # Extract numeric value from size
                df['size_sqft'] = df['size_sqft'].str.extract(r'([\d,]+)').replace(',', '', regex=True)
                df['size_sqft'] = pd.to_numeric(df['size_sqft'], errors='coerce')
            
            # Add current date
            df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Cache the data
            self._cache_data(df, cache_name)
            
            self.logger.info(f"Scraped {len(df)} retail vacancy listings")
            return df
            
        except Exception as e:
            self.logger.error(f"Error scraping retail vacancy data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
```

### Data Processing Implementation

```python
class DataProcessor:
    """
    Data processor for Chicago Housing Pipeline.
    
    Handles data validation, cleaning, integration, and feature engineering.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the data processor.
        
        Args:
            output_dir (Path, optional): Directory to save processed data
        """
        self.output_dir = Path(output_dir) if output_dir else Path(settings.DATA_DIR) / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_all(self, data_dict=None, save=True):
        """
        Process all data sources.
        
        Args:
            data_dict (dict, optional): Dictionary of data sources
            save (bool): Whether to save processed data
            
        Returns:
            pd.DataFrame: Processed and integrated data
        """
        try:
            self.logger.info("Processing all data sources")
            
            # If data_dict not provided, collect data
            if data_dict is None:
                data_dict = self._collect_data()
            
            # Validate data
            validated_data = self._validate_data(data_dict)
            if validated_data is None:
                self.logger.error("Data validation failed")
                return None
            
            # Clean data
            cleaned_data = self._clean_data(validated_data)
            if cleaned_data is None:
                self.logger.error("Data cleaning failed")
                return None
            
            # Integrate data
            integrated_data = self._integrate_data(cleaned_data)
            if integrated_data is None:
                self.logger.error("Data integration failed")
                return None
            
            # Engineer features
            final_data = self._engineer_features(integrated_data)
            if final_data is None:
                self.logger.error("Feature engineering failed")
                return None
            
            # Save processed data
            if save:
                self._save_data(final_data)
            
            self.logger.info(f"Processed {len(final_data)} records")
            return final_data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _collect_data(self):
        """
        Collect data from all sources.
        
        Returns:
            dict: Dictionary of data sources
        """
        self.logger.info("Collecting data from all sources")
        
        # Initialize collectors
        census_collector = CensusCollector()
        fred_collector = FREDCollector()
        chicago_collector = ChicagoDataCollector()
        bea_collector = BEACollector()
        web_scraper = WebScraper()
        
        # Collect data
        census_data = census_collector.collect()
        economic_data = fred_collector.collect()
        building_permits = chicago_collector.collect_building_permits()
        business_licenses = chicago_collector.collect_business_licenses()
        retail_gdp = bea_collector.collect_retail_gdp()
        retail_vacancies = web_scraper.scrape_retail_vacancies()
        
        # Return data dictionary
        return {
            'census': census_data,
            'economic': economic_data,
            'building_permits': building_permits,
            'business_licenses': business_licenses,
            'retail_gdp': retail_gdp,
            'retail_vacancies': retail_vacancies
        }
    
    def _validate_data(self, data_dict):
        """
        Validate data from all sources.
        
        Args:
            data_dict (dict): Dictionary of data sources
            
        Returns:
            dict: Validated data dictionary
        """
        self.logger.info("Validating data")
        
        validated_data = {}
        
        for source, df in data_dict.items():
            if df is None or df.empty:
                self.logger.warning(f"No data for source: {source}")
                continue
            
            # Check for required columns based on source
            if source == 'census':
                required_cols = ['zip_code', 'population', 'housing_units']
            elif source == 'economic':
                required_cols = ['date', 'value', 'series_id']
            elif source == 'building_permits':
                required_cols = ['zip_code', 'issue_date']
            elif source == 'business_licenses':
                required_cols = ['zip_code', 'license_start_date']
            elif source == 'retail_gdp':
                required_cols = ['year', 'retail_gdp']
            elif source == 'retail_vacancies':
                required_cols = ['address', 'scrape_date']
            else:
                required_cols = []
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Missing required columns for {source}: {missing_cols}")
                # Try to derive missing columns
                df = self._derive_missing_columns(df, missing_cols, source)
            
            # Check for data quality issues
            if 'zip_code' in df.columns:
                # Ensure ZIP code is string and 5 digits
                df['zip_code'] = df['zip_code'].astype(str).str.extract('(\d{5})').iloc[:, 0]
                df['zip_code'] = df['zip_code'].str.zfill(5)
                
                # Check for invalid ZIP codes
                invalid_zips = df[~df['zip_code'].str.match(r'^\d{5}$')]['zip_code'].unique()
                if len(invalid_zips) > 0:
                    self.logger.warning(f"Invalid ZIP codes in {source}: {invalid_zips}")
                    # Filter out invalid ZIP codes
                    df = df[df['zip_code'].str.match(r'^\d{5}$')]
            
            # Add source column
            df['data_source'] = source
            
            validated_data[source] = df
        
        return validated_data
    
    def _derive_missing_columns(self, df, missing_cols, source):
        """
        Derive missing columns based on source.
        
        Args:
            df (pd.DataFrame): Data frame
            missing_cols (list): List of missing columns
            source (str): Data source
            
        Returns:
            pd.DataFrame: Data frame with derived columns
        """
        self.logger.info(f"Deriving missing columns for {source}: {missing_cols}")
        
        for col in missing_cols:
            if col == 'zip_code' and 'address' in df.columns:
                # Extract ZIP code from address
                df['zip_code'] = df['address'].str.extract(r'Chicago,\s+IL\s+(\d{5})')
            
            elif col == 'population' and 'housing_units' in df.columns:
                # Estimate population from housing units
                df['population'] = df['housing_units'] * 2.5  # Assuming 2.5 people per household
            
            elif col == 'housing_units' and 'population' in df.columns:
                # Estimate housing units from population
                df['housing_units'] = df['population'] / 2.5  # Assuming 2.5 people per household
            
            elif col == 'issue_date' and 'permit_year' in df.columns:
                # Create issue_date from permit_year
                df['issue_date'] = pd.to_datetime(df['permit_year'].astype(str) + '-01-01')
            
            elif col == 'license_start_date' and 'license_year' in df.columns:
                # Create license_start_date from license_year
                df['license_start_date'] = pd.to_datetime(df['license_year'].astype(str) + '-01-01')
            
            elif col == 'year' and 'date' in df.columns:
                # Extract year from date
                df['year'] = pd.to_datetime(df['date']).dt.year
            
            elif col == 'retail_gdp' and 'value' in df.columns and source == 'economic':
                # Use value as retail_gdp for economic data
                df['retail_gdp'] = df['value']
            
            elif col == 'scrape_date':
                # Add current date
                df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    def _clean_data(self, data_dict):
        """
        Clean data from all sources.
        
        Args:
            data_dict (dict): Dictionary of data sources
            
        Returns:
            dict: Cleaned data dictionary
        """
        self.logger.info("Cleaning data")
        
        cleaned_data = {}
        
        for source, df in data_dict.items():
            if df is None or df.empty:
                continue
            
            # Convert date columns to datetime
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Convert numeric columns to numeric
            numeric_columns = [
                'population', 'housing_units', 'median_income', 'unit_count',
                'value', 'retail_gdp', 'size_sqft'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            if source == 'census':
                # Fill missing population with median
                if 'population' in df.columns:
                    df['population'] = df['population'].fillna(df['population'].median())
                
                # Fill missing housing units with median
                if 'housing_units' in df.columns:
                    df['housing_units'] = df['housing_units'].fillna(df['housing_units'].median())
            
            elif source == 'economic':
                # Forward fill missing values in time series
                if 'date' in df.columns and 'value' in df.columns:
                    df = df.sort_values('date')
                    df['value'] = df.groupby('series_id')['value'].fillna(method='ffill')
            
            elif source == 'building_permits':
                # Fill missing unit count with 1
                if 'unit_count' in df.columns:
                    df['unit_count'] = df['unit_count'].fillna(1)
            
            # Remove duplicates
            if 'zip_code' in df.columns:
                if source == 'census' and 'year' in df.columns:
                    df = df.drop_duplicates(subset=['zip_code', 'year'])
                elif source == 'building_permits' and 'issue_date' in df.columns:
                    df = df.drop_duplicates(subset=['zip_code', 'issue_date'])
                elif source == 'business_licenses' and 'license_start_date' in df.columns:
                    df = df.drop_duplicates(subset=['zip_code', 'license_start_date'])
            
            cleaned_data[source] = df
        
        return cleaned_data
    
    def _integrate_data(self, data_dict):
        """
        Integrate data from all sources.
        
        Args:
            data_dict (dict): Dictionary of data sources
            
        Returns:
            pd.DataFrame: Integrated data
        """
        self.logger.info("Integrating data")
        
        # Start with census data as the base
        if 'census' not in data_dict or data_dict['census'] is None or data_dict['census'].empty:
            self.logger.error("No census data available for integration")
            return None
        
        base_df = data_dict['census'].copy()
        
        # Ensure ZIP code is string
        if 'zip_code' in base_df.columns:
            base_df['zip_code'] = base_df['zip_code'].astype(str).str.zfill(5)
        
        # Add year column if not present
        if 'year' not in base_df.columns and 'date' in base_df.columns:
            base_df['year'] = pd.to_datetime(base_df['date']).dt.year
        
        # Integrate building permits
        if 'building_permits' in data_dict and data_dict['building_permits'] is not None:
            permits_df = data_dict['building_permits']
            
            if 'zip_code' in permits_df.columns and 'issue_date' in permits_df.columns:
                # Extract year from issue_date
                permits_df['permit_year'] = pd.to_datetime(permits_df['issue_date']).dt.year
                
                # Aggregate permits by ZIP code and year
                permits_agg = permits_df.groupby(['zip_code', 'permit_year']).agg({
                    'unit_count': 'sum',
                    'id': 'count'
                }).reset_index()
                
                permits_agg.rename(columns={'id': 'permit_count'}, inplace=True)
                
                # Merge with base data
                base_df = pd.merge(
                    base_df,
                    permits_agg,
                    left_on=['zip_code', 'year'],
                    right_on=['zip_code', 'permit_year'],
                    how='left'
                )
                
                # Fill missing values
                base_df['unit_count'] = base_df['unit_count'].fillna(0)
                base_df['permit_count'] = base_df['permit_count'].fillna(0)
                
                # Drop redundant column
                if 'permit_year' in base_df.columns:
                    base_df = base_df.drop(columns=['permit_year'])
        
        # Integrate business licenses
        if 'business_licenses' in data_dict and data_dict['business_licenses'] is not None:
            licenses_df = data_dict['business_licenses']
            
            if 'zip_code' in licenses_df.columns and 'license_start_date' in licenses_df.columns:
                # Extract year from license_start_date
                licenses_df['license_year'] = pd.to_datetime(licenses_df['license_start_date']).dt.year
                
                # Aggregate licenses by ZIP code and year
                licenses_agg = licenses_df.groupby(['zip_code', 'license_year']).agg({
                    'id': 'count'
                }).reset_index()
                
                licenses_agg.rename(columns={'id': 'business_count'}, inplace=True)
                
                # Merge with base data
                base_df = pd.merge(
                    base_df,
                    licenses_agg,
                    left_on=['zip_code', 'year'],
                    right_on=['zip_code', 'license_year'],
                    how='left'
                )
                
                # Fill missing values
                base_df['business_count'] = base_df['business_count'].fillna(0)
                
                # Drop redundant column
                if 'license_year' in base_df.columns:
                    base_df = base_df.drop(columns=['license_year'])
                
                # Add retail category counts if available
                if 'retail_category' in licenses_df.columns:
                    # Get retail categories
                    retail_categories = licenses_df['retail_category'].unique()
                    
                    for category in retail_categories:
                        # Filter licenses by category
                        category_df = licenses_df[licenses_df['retail_category'] == category]
                        
                        # Aggregate by ZIP code and year
                        category_agg = category_df.groupby(['zip_code', 'license_year']).agg({
                            'id': 'count'
                        }).reset_index()
                        
                        category_agg.rename(columns={'id': f'{category}_count'}, inplace=True)
                        
                        # Merge with base data
                        base_df = pd.merge(
                            base_df,
                            category_agg,
                            left_on=['zip_code', 'year'],
                            right_on=['zip_code', 'license_year'],
                            how='left'
                        )
                        
                        # Fill missing values
                        base_df[f'{category}_count'] = base_df[f'{category}_count'].fillna(0)
                        
                        # Drop redundant column
                        if 'license_year' in base_df.columns:
                            base_df = base_df.drop(columns=['license_year'])
        
        # Integrate economic data
        if 'economic' in data_dict and data_dict['economic'] is not None:
            econ_df = data_dict['economic']
            
            if 'date' in econ_df.columns and 'series_id' in econ_df.columns and 'value' in econ_df.columns:
                # Extract year from date
                econ_df['year'] = pd.to_datetime(econ_df['date']).dt.year
                
                # Pivot to get series as columns
                econ_pivot = econ_df.pivot_table(
                    index='year',
                    columns='series_id',
                    values='value',
                    aggfunc='mean'
                ).reset_index()
                
                # Rename columns to be more descriptive
                econ_pivot.columns = [col if col == 'year' else f'econ_{col.lower()}' for col in econ_pivot.columns]
                
                # Merge with base data
                base_df = pd.merge(
                    base_df,
                    econ_pivot,
                    on='year',
                    how='left'
                )
                
                # Forward fill missing economic indicators
                econ_cols = [col for col in base_df.columns if col.startswith('econ_')]
                for col in econ_cols:
                    base_df[col] = base_df.groupby('zip_code')[col].fillna(method='ffill')
                    base_df[col] = base_df.groupby('zip_code')[col].fillna(method='bfill')
        
        # Integrate retail GDP data
        if 'retail_gdp' in data_dict and data_dict['retail_gdp'] is not None:
            gdp_df = data_dict['retail_gdp']
            
            if 'year' in gdp_df.columns and 'retail_gdp' in gdp_df.columns:
                # Merge with base data
                base_df = pd.merge(
                    base_df,
                    gdp_df[['year', 'retail_gdp']],
                    on='year',
                    how='left'
                )
                
                # Forward fill missing GDP values
                base_df['retail_gdp'] = base_df.groupby('zip_code')['retail_gdp'].fillna(method='ffill')
                base_df['retail_gdp'] = base_df.groupby('zip_code')['retail_gdp'].fillna(method='bfill')
        
        # Integrate retail vacancy data
        if 'retail_vacancies' in data_dict and data_dict['retail_vacancies'] is not None:
            vacancy_df = data_dict['retail_vacancies']
            
            if 'zip_code' in vacancy_df.columns and 'scrape_date' in vacancy_df.columns:
                # Extract year from scrape_date
                vacancy_df['year'] = pd.to_datetime(vacancy_df['scrape_date']).dt.year
                
                # Aggregate vacancies by ZIP code and year
                vacancy_agg = vacancy_df.groupby(['zip_code', 'year']).agg({
                    'size_sqft': 'sum',
                    'address': 'count'
                }).reset_index()
                
                vacancy_agg.rename(columns={
                    'size_sqft': 'vacant_sqft',
                    'address': 'vacancy_count'
                }, inplace=True)
                
                # Merge with base data
                base_df = pd.merge(
                    base_df,
                    vacancy_agg,
                    on=['zip_code', 'year'],
                    how='left'
                )
                
                # Fill missing values
                base_df['vacant_sqft'] = base_df['vacant_sqft'].fillna(0)
                base_df['vacancy_count'] = base_df['vacancy_count'].fillna(0)
        
        # Fill any remaining missing values
        base_df = base_df.fillna(0)
        
        return base_df
    
    def _engineer_features(self, df):
        """
        Engineer features for analysis.
        
        Args:
            df (pd.DataFrame): Integrated data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        self.logger.info("Engineering features")
        
        # Ensure we have the necessary columns
        required_cols = ['zip_code', 'year', 'population']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns for feature engineering: {missing_cols}")
            return df
        
        # Calculate population growth
        df = df.sort_values(['zip_code', 'year'])
        df['population_prev'] = df.groupby('zip_code')['population'].shift(1)
        df['population_growth'] = (df['population'] - df['population_prev']) / df['population_prev'].replace(0, 1) * 100
        
        # Calculate housing growth if unit_count is available
        if 'unit_count' in df.columns:
            df['unit_count_prev'] = df.groupby('zip_code')['unit_count'].shift(1)
            df['unit_growth'] = (df['unit_count'] - df['unit_count_prev']) / df['unit_count_prev'].replace(0, 1) * 100
        
        # Calculate business growth if business_count is available
        if 'business_count' in df.columns:
            df['business_count_prev'] = df.groupby('zip_code')['business_count'].shift(1)
            df['business_growth'] = (df['business_count'] - df['business_count_prev']) / df['business_count_prev'].replace(0, 1) * 100
        
        # Calculate retail density (businesses per 1000 people)
        if 'business_count' in df.columns and 'population' in df.columns:
            df['retail_density'] = df['business_count'] / df['population'] * 1000
        
        # Calculate housing density (units per 1000 people)
        if 'housing_units' in df.columns and 'population' in df.columns:
            df['housing_density'] = df['housing_units'] / df['population'] * 1000
        
        # Calculate permit density (permits per 1000 people)
        if 'permit_count' in df.columns and 'population' in df.columns:
            df['permit_density'] = df['permit_count'] / df['population'] * 1000
        
        # Calculate vacancy rate if vacancy data is available
        if 'vacancy_count' in df.columns and 'business_count' in df.columns:
            df['vacancy_rate'] = df['vacancy_count'] / (df['business_count'] + df['vacancy_count']) * 100
        
        # Calculate retail gap (difference between expected and actual retail density)
        if 'retail_density' in df.columns:
            # Calculate city-wide average retail density
            city_avg_density = df.groupby('year')['retail_density'].transform('mean')
            
            # Calculate retail gap
            df['retail_gap'] = city_avg_density - df['retail_density']
            
            # Calculate retail gap score (positive = opportunity, negative = oversupply)
            df['retail_gap_score'] = df['retail_gap'] * df['population'] / 1000
        
        # Calculate housing-retail balance
        if 'housing_density' in df.columns and 'retail_density' in df.columns:
            df['housing_retail_ratio'] = df['housing_density'] / df['retail_density'].replace(0, 0.1)
        
        # Flag high growth areas
        if 'population_growth' in df.columns:
            df['high_population_growth'] = df['population_growth'] >= 20
        
        if 'unit_growth' in df.columns:
            df['high_unit_growth'] = df['unit_growth'] >= 20
        
        if 'business_growth' in df.columns:
            df['high_business_growth'] = df['business_growth'] >= 20
        
        # Flag retail gap areas
        if 'retail_gap_score' in df.columns and 'high_population_growth' in df.columns:
            df['retail_gap_area'] = (df['retail_gap_score'] > 0) & df['high_population_growth']
        
        # Flag South/West side ZIP codes
        if 'zip_code' in df.columns:
            south_west_zips = settings.SOUTH_WEST_ZIP_CODES
            df['is_south_west'] = df['zip_code'].isin(south_west_zips)
        
        # Flag downtown/Loop ZIP codes
        if 'zip_code' in df.columns:
            downtown_zips = settings.DOWNTOWN_ZIP_CODES
            df['is_downtown'] = df['zip_code'].isin(downtown_zips)
        
        # Calculate 5-year moving averages for trend analysis
        for col in ['population', 'housing_units', 'business_count', 'unit_count']:
            if col in df.columns:
                df[f'{col}_5yr_avg'] = df.groupby('zip_code')[col].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
        
        # Calculate compound annual growth rate (CAGR) for key metrics
        for col in ['population', 'housing_units', 'business_count']:
            if col in df.columns:
                # Get min and max years for each ZIP code
                zip_years = df.groupby('zip_code')['year'].agg(['min', 'max']).reset_index()
                
                for _, row in zip_years.iterrows():
                    zip_code = row['zip_code']
                    min_year = row['min']
                    max_year = row['max']
                    
                    if max_year > min_year:
                        # Get values for min and max years
                        min_val = df[(df['zip_code'] == zip_code) & (df['year'] == min_year)][col].values
                        max_val = df[(df['zip_code'] == zip_code) & (df['year'] == max_year)][col].values
                        
                        if len(min_val) > 0 and len(max_val) > 0:
                            min_val = min_val[0]
                            max_val = max_val[0]
                            
                            if min_val > 0:
                                # Calculate CAGR
                                years = max_year - min_year
                                cagr = (max_val / min_val) ** (1 / years) - 1
                                
                                # Add CAGR to all rows for this ZIP code
                                df.loc[df['zip_code'] == zip_code, f'{col}_cagr'] = cagr * 100
        
        return df
    
    def _save_data(self, df):
        """
        Save processed data to file.
        
        Args:
            df (pd.DataFrame): Processed data
        """
        self.logger.info("Saving processed data")
        
        # Save to CSV
        csv_path = self.output_dir / "merged_data.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved processed data to {csv_path}")
        
        # Save to pickle for faster loading
        pkl_path = self.output_dir / "merged_data.pkl"
        df.to_pickle(pkl_path)
        self.logger.info(f"Saved processed data to {pkl_path}")
```

## Data Integration Manager

```python
class DataIntegrationManager:
    """
    Manager for data integration process.
    
    Coordinates data collection, processing, and storage.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the data integration manager.
        
        Args:
            output_dir (Path, optional): Directory to save outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path(settings.DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.collectors = {
            'census': CensusCollector(),
            'fred': FREDCollector(),
            'chicago': ChicagoDataCollector(),
            'bea': BEACollector(),
            'web': WebScraper()
        }
        
        self.processor = DataProcessor(output_dir=self.output_dir / "processed")
    
    def run(self, use_cache=True, save=True):
        """
        Run the data integration process.
        
        Args:
            use_cache (bool): Whether to use cached data
            save (bool): Whether to save processed data
            
        Returns:
            pd.DataFrame: Integrated and processed data
        """
        try:
            self.logger.info("Running data integration process")
            
            # Check if processed data exists and use_cache is True
            if use_cache:
                processed_path = self.output_dir / "processed" / "merged_data.pkl"
                if processed_path.exists():
                    self.logger.info(f"Loading processed data from {processed_path}")
                    return pd.read_pickle(processed_path)
            
            # Collect data from all sources
            data_dict = self._collect_data()
            
            # Process data
            processed_data = self.processor.process_all(data_dict, save=save)
            
            self.logger.info("Data integration process completed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in data integration process: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _collect_data(self):
        """
        Collect data from all sources.
        
        Returns:
            dict: Dictionary of data sources
        """
        self.logger.info("Collecting data from all sources")
        
        data_dict = {}
        
        # Collect Census data
        self.logger.info("Collecting Census data")
        census_collector = self.collectors['census']
        data_dict['census'] = census_collector.collect_historical(start_year=2010, end_year=2020)
        
        # Collect FRED data
        self.logger.info("Collecting FRED data")
        fred_collector = self.collectors['fred']
        data_dict['economic'] = fred_collector.collect()
        data_dict['local_economic'] = fred_collector.collect_local_indicators()
        
        # Collect Chicago Data Portal data
        self.logger.info("Collecting Chicago Data Portal data")
        chicago_collector = self.collectors['chicago']
        data_dict['building_permits'] = chicago_collector.collect_building_permits(years=list(range(2010, 2023)))
        data_dict['business_licenses'] = chicago_collector.collect_business_licenses(years=list(range(2010, 2023)))
        
        # Collect BEA data
        self.logger.info("Collecting BEA data")
        bea_collector = self.collectors['bea']
        data_dict['retail_gdp'] = bea_collector.collect_retail_gdp(years=list(range(2010, 2023)))
        
        # Collect web scraped data
        self.logger.info("Collecting web scraped data")
        web_scraper = self.collectors['web']
        data_dict['retail_vacancies'] = web_scraper.scrape_retail_vacancies()
        
        # Check for missing data
        missing_sources = [source for source, df in data_dict.items() if df is None or df.empty]
        if missing_sources:
            self.logger.warning(f"Missing data for sources: {missing_sources}")
            
            # Try to use sample data for missing sources
            for source in missing_sources:
                sample_path = self.output_dir / "sample" / f"{source}.csv"
                if sample_path.exists():
                    self.logger.info(f"Using sample data for {source} from {sample_path}")
                    data_dict[source] = pd.read_csv(sample_path)
        
        return data_dict
```

## Conclusion

This comprehensive data integration framework provides a robust foundation for the Chicago Housing Pipeline & Population Shift Project. It addresses the critical gaps identified in the audit and enables advanced analytics, visualization, and reporting capabilities.

Key features of the framework include:

1. **Modular Architecture**: Clear separation of concerns with specialized components for collection, processing, and integration.

2. **Robust API Integrations**: Comprehensive integration with Census API, FRED API, Chicago Data Portal, and BEA API.

3. **Advanced Data Processing**: Sophisticated validation, cleaning, integration, and feature engineering capabilities.

4. **Flexible Caching**: Efficient caching mechanisms to minimize API calls and improve performance.

5. **Error Handling**: Comprehensive error handling and logging throughout the framework.

6. **Feature Engineering**: Advanced feature creation for time series analysis, growth metrics, and spatial patterns.

7. **Data Quality**: Robust validation and cleaning to ensure high-quality data for analytics.

The framework is designed to be extensible, allowing for easy addition of new data sources and processing capabilities as needed. It provides a solid foundation for the advanced analytics, visualization, and reporting capabilities required by the project objectives.
