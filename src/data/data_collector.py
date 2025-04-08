import os
import pandas as pd
import numpy as np
from census import Census
import logging
from datetime import datetime
from pathlib import Path
import requests
from sodapy import Socrata
from fredapi import Fred
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataCollector:
    def __init__(self):
        """Initialize data collector with API keys and data paths"""
        self.census_api_key = os.getenv('CENSUS_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.chicago_data_portal = os.getenv('CHICAGO_DATA_PORTAL')
        self.building_permits_dataset = os.getenv('BUILDING_PERMITS_DATASET')
        self.zoning_dataset = os.getenv('ZONING_DATASET')
        
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.data_dir.mkdir(exist_ok=True)
        
        # Validate required API keys
        if not self.census_api_key:
            logger.warning("Census API key not found in environment variables")
        if not self.fred_api_key:
            logger.warning("FRED API key not found in environment variables")

    def get_census_data(self, start_year=2005):
        """
        Collect historical population and income data for Chicago zip codes.
        
        Args:
            start_year (int): Starting year for historical data collection
        """
        if not self.census_api_key:
            logger.error("Cannot fetch Census data: API key missing")
            return None
            
        c = Census(self.census_api_key)
        current_year = datetime.now().year
        
        all_data = []
        # B01003_001E is total population
        # B19013_001E is median household income
        # Income distribution variables (multiple brackets)
        income_vars = [
            'B19001_002E',  # Less than $10,000
            'B19001_003E',  # $10,000 to $14,999
            'B19001_004E',  # $15,000 to $19,999
            'B19001_005E',  # $20,000 to $24,999
            'B19001_006E',  # $25,000 to $29,999
            'B19001_007E',  # $30,000 to $34,999
            'B19001_008E',  # $35,000 to $39,999
            'B19001_009E',  # $40,000 to $44,999
            'B19001_010E',  # $45,000 to $49,999
            'B19001_011E',  # $50,000 to $59,999
            'B19001_012E',  # $60,000 to $74,999
            'B19001_013E',  # $75,000 to $99,999
            'B19001_014E',  # $100,000 to $124,999
            'B19001_015E',  # $125,000 to $149,999
            'B19001_016E',  # $150,000 to $199,999
            'B19001_017E',  # $200,000 or more
            'B19001_001E',  # Total households (for calculating percentages)
        ]
        
        variables = ['B01003_001E', 'B19013_001E'] + income_vars
        
        for year in range(start_year, current_year):
            try:
                logger.info(f"Fetching census data for {year}")
                if data := c.acs5.get(variables, {'for': 'zip code tabulation area:*'}, year=year):
                    df = pd.DataFrame(data)
                    df['year'] = year
                    all_data.append(df)
                    logger.info(f"Successfully collected data for {year}")
                else:
                    logger.warning(f"No data available for {year}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {year}: {str(e)}")
        
        if all_data:
            # Combine all years
            historical_df = pd.concat(all_data, ignore_index=True)
            
            # Get column names for clear mapping
            col_names = ['population', 'median_household_income']
            income_brackets = [
                'income_less_10k', 'income_10k_15k', 'income_15k_20k', 
                'income_20k_25k', 'income_25k_30k', 'income_30k_35k',
                'income_35k_40k', 'income_40k_45k', 'income_45k_50k',
                'income_50k_60k', 'income_60k_75k', 'income_75k_100k',
                'income_100k_125k', 'income_125k_150k', 'income_150k_200k',
                'income_200k_plus', 'total_households'
            ]
            
            all_cols = col_names + income_brackets + ['state', 'zip_code', 'year']
            historical_df.columns = all_cols
            
            # Convert numeric columns
            numeric_cols = col_names + income_brackets
            historical_df[numeric_cols] = historical_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Calculate income distribution percentages
            for bracket in income_brackets[:-1]:  # Exclude total_households
                historical_df[f'{bracket}_pct'] = historical_df[bracket] / historical_df['total_households'] * 100
            
            # Save to CSV
            output_file = self.data_dir / 'historical_population.csv'
            historical_df.to_csv(output_file, index=False)
            logger.info(f"Saved historical data to {output_file}")
            
            return historical_df
        else:
            logger.error("No census data was collected")
            return None

    def get_building_permits(self):
        """Collect building permit data from Chicago Data Portal"""
        try:
            logger.info("Fetching building permits data")
            
            # Initialize Socrata client
            client = Socrata(self.chicago_data_portal, None)
            
            # Get building permits with a limit high enough to get all records
            # Filter for residential/mixed-use construction
            results = client.get(
                self.building_permits_dataset,
                limit=200000,
                where="permit_type in ('PERMIT - NEW CONSTRUCTION', 'PERMIT - RENOVATION/ALTERATION')"
            )
            
            if not results:
                logger.error("No building permit data retrieved")
                return None
                
            permits_df = pd.DataFrame.from_records(results)
            
            # Clean and transform data
            # Convert dates
            if 'issue_date' in permits_df.columns:
                permits_df['issue_date'] = pd.to_datetime(permits_df['issue_date'])
            
            # Extract zip code
            if 'zip_code' in permits_df.columns:
                permits_df['zip_code'] = permits_df['zip_code'].astype(str).str[:5]
            
            # Convert numeric fields
            numeric_cols = ['reported_cost', 'stories', 'total_land_square_feet']
            for col in numeric_cols:
                if col in permits_df.columns:
                    permits_df[col] = pd.to_numeric(permits_df[col], errors='coerce')
            
            # Save to file
            output_file = self.data_dir / 'building_permits.csv'
            permits_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(permits_df)} building permits to {output_file}")
            
            return permits_df
            
        except Exception as e:
            logger.error(f"Error fetching building permits: {str(e)}")
            return None

    def get_zoning_data(self):
        """Collect zoning data from Chicago Data Portal"""
        try:
            logger.info("Fetching zoning data")
            
            # Get GeoJSON data
            url = f"https://{self.chicago_data_portal}/resource/{self.zoning_dataset}.geojson"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch zoning data: {response.status_code}")
                return None
                
            # Save raw GeoJSON
            output_geojson = self.data_dir / 'zoning.geojson'
            with open(output_geojson, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Saved zoning GeoJSON data to {output_geojson}")
            
            # Also convert to CSV for easier processing
            # Initialize Socrata client
            client = Socrata(self.chicago_data_portal, None)
            
            # Get the data without geometry for CSV
            results = client.get(self.zoning_dataset, limit=200000)
            
            if not results:
                logger.error("No zoning data retrieved")
                return None
                
            zoning_df = pd.DataFrame.from_records(results)
            
            # Save CSV
            output_csv = self.data_dir / 'zoning.csv'
            zoning_df.to_csv(output_csv, index=False)
            logger.info(f"Saved zoning data to {output_csv}")
            
            return zoning_df
            
        except Exception as e:
            logger.error(f"Error fetching zoning data: {str(e)}")
            return None

    def get_economic_indicators(self):
        """Collect economic indicators from FRED"""
        try:
            if not self.fred_api_key:
                logger.error("Cannot fetch economic data: FRED API key missing")
                return None
                
            logger.info("Fetching economic indicators from FRED")
            
            fred = Fred(api_key=self.fred_api_key)
            
            # Define economic indicators to fetch
            indicators = {
                'DGS10': 'treasury_10y',          # 10-Year Treasury Constant Maturity Rate
                'MORTGAGE30US': 'mortgage_30y',   # 30-Year Fixed Rate Mortgage Average
                'UMCSENT': 'consumer_sentiment',  # University of Michigan Consumer Sentiment
                'USREC': 'recession_indicator',   # US Recession Indicator
                'HOUST': 'housing_starts',        # Housing Starts
                'CSUSHPISA': 'house_price_index'  # Case-Shiller Home Price Index
            }
            
            # Start from 2000 to ensure we have enough historical data
            start_date = '2000-01-01'
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch data for each indicator
            data = {}
            for series_id, name in indicators.items():
                try:
                    logger.info(f"Fetching {name} data")
                    series = fred.get_series(series_id, start_date, end_date)
                    data[name] = series
                except Exception as e:
                    logger.error(f"Error fetching {name}: {str(e)}")
            
            # Combine all indicators into a single DataFrame
            econ_df = pd.DataFrame(data)
            
            # Forward-fill missing values
            econ_df = econ_df.fillna(method='ffill')
            
            # Add date information
            econ_df = econ_df.reset_index()
            econ_df.columns.name = None
            econ_df.rename(columns={'index': 'date'}, inplace=True)
            econ_df['year'] = econ_df['date'].dt.year
            econ_df['month'] = econ_df['date'].dt.month
            
            # Save to file
            output_file = self.data_dir / 'economic_indicators.csv'
            econ_df.to_csv(output_file, index=False)
            logger.info(f"Saved economic indicators to {output_file}")
            
            return econ_df
            
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {str(e)}")
            return None


if __name__ == "__main__":
    collector = DataCollector()
    
    # Collect all data
    collector.get_census_data()
    collector.get_building_permits()
    collector.get_zoning_data()
    collector.get_economic_indicators()