import pandas as pd
import geopandas as gpd
import numpy as np
from census import Census
from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChicagoPopulationAnalyzer:
    def __init__(self, census_api_key):
        self.census = Census(census_api_key)
        self.data_dir = Path('data')
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Load existing datasets
        self.load_data()
        
    def load_data(self):
        """Load all necessary datasets"""
        logger.info("Loading datasets...")
        
        # Load building permits
        try:
            self.permits_df = pd.read_csv(self.data_dir / 'building_permits.csv')
            logger.info(f"Loaded {len(self.permits_df)} building permits")
        except FileNotFoundError:
            logger.error("Building permits data not found")
            self.permits_df = None
            
        # Load economic indicators
        try:
            self.econ_df = pd.read_csv(self.data_dir / 'economic_indicators.csv')
            logger.info(f"Loaded {len(self.econ_df)} economic indicators")
        except FileNotFoundError:
            logger.error("Economic indicators data not found")
            self.econ_df = None
    
    def get_census_data(self, year):
        """Get population and income data for Chicago zip codes"""
        # B01003_001E is total population
        # B19013_001E is median household income
        # B19001 series for income distribution
        variables = ['B01003_001E', 'B19013_001E']
        
        try:
            data = self.census.acs5.get(
                variables,
                {'for': 'zip code tabulation area:*',
                 'in': 'state:17'},  # Illinois state code
                year=year
            )
            
            df = pd.DataFrame(data)
            # Rename columns
            df.columns = ['population', 'median_income', 'state', 'zip']
            # Convert numeric columns
            df[['population', 'median_income']] = df[['population', 'median_income']].apply(pd.to_numeric, errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching census data for year {year}: {str(e)}")
            return None
    
    def get_historical_population(self, start_year=2005):
        """Get historical population data for the past years"""
        current_year = datetime.now().year
        years = range(start_year, current_year)
        
        population_dfs = []
        for year in years:
            logger.info(f"Fetching census data for {year}")
            df = self.get_census_data(year)
            if df is not None:
                df['year'] = year
                population_dfs.append(df)
        
        if population_dfs:
            historical_df = pd.concat(population_dfs, ignore_index=True)
            historical_df.to_csv(self.data_dir / 'historical_population.csv', index=False)
            return historical_df
        return None
    
    def analyze_population_trends(self):
        """Analyze population trends by zip code"""
        if not hasattr(self, 'historical_df'):
            try:
                self.historical_df = pd.read_csv(self.data_dir / 'historical_population.csv')
            except FileNotFoundError:
                logger.info("Historical population data not found, fetching from Census...")
                self.historical_df = self.get_historical_population()
        
        if self.historical_df is None:
            logger.error("No historical population data available")
            return
        
        # Calculate year-over-year changes
        # Ensure we have a consistent zip_code column
        if 'zip' in self.historical_df.columns and 'zip_code' not in self.historical_df.columns:
            self.historical_df = self.historical_df.rename(columns={'zip': 'zip_code'})
            
        pivot_df = self.historical_df.pivot(index='zip_code', columns='year', values='population')
        yoy_changes = pivot_df.pct_change(axis=1)
        
        # Plot overall population trend
        plt.figure(figsize=(12, 6))
        total_pop = self.historical_df.groupby('year')['population'].sum()
        total_pop.plot(kind='line', marker='o')
        plt.title('Chicago Total Population Trend')
        plt.xlabel('Year')
        plt.ylabel('Total Population')
        plt.grid(True)
        plt.savefig(self.output_dir / 'total_population_trend.png')
        plt.close()
        
        # Plot population changes by zip code
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=yoy_changes)
        plt.title('Distribution of Population Changes by Year')
        plt.xlabel('Year')
        plt.ylabel('Year-over-Year Change (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'population_changes_distribution.png')
        plt.close()
        
        return total_pop, yoy_changes
    
    def analyze_income_trends(self):
        """Analyze income trends by zip code"""
        if not hasattr(self, 'historical_df'):
            try:
                self.historical_df = pd.read_csv(self.data_dir / 'historical_population.csv')
            except FileNotFoundError:
                logger.error("Historical population data not found")
                return
        
        # Check which income column exists in the dataframe
        income_column = None
        if 'median_income' in self.historical_df.columns:
            income_column = 'median_income'
        elif 'median_household_income' in self.historical_df.columns:
            income_column = 'median_household_income'
        else:
            logger.error("No income column found in historical data")
            return None
            
        logger.info(f"Using income column: {income_column}")
            
        # Calculate median income trends
        median_income_by_year = self.historical_df.groupby('year')[income_column].agg(['mean', 'median', 'std'])
        
        # Plot median income trends
        plt.figure(figsize=(12, 6))
        median_income_by_year['median'].plot(kind='line', marker='o')
        plt.fill_between(
            median_income_by_year.index,
            median_income_by_year['median'] - median_income_by_year['std'],
            median_income_by_year['median'] + median_income_by_year['std'],
            alpha=0.2
        )
        plt.title('Chicago Median Income Trend with Standard Deviation')
        plt.xlabel('Year')
        plt.ylabel('Median Income ($)')
        plt.grid(True)
        plt.savefig(self.output_dir / 'median_income_trend.png')
        plt.close()
        
        return median_income_by_year
    
    def analyze_permit_impact(self):
        """Analyze the impact of building permits on population changes"""
        if self.permits_df is None:
            logger.error("Building permits data not available")
            return
        
        # Convert permit dates and aggregate by zip and year
        self.permits_df['issue_date'] = pd.to_datetime(self.permits_df['issue_date'])
        self.permits_df['year'] = self.permits_df['issue_date'].dt.year
        
        # Check which zip code column exists in the permits dataframe
        zip_column = None
        if 'zip_code' in self.permits_df.columns:
            zip_column = 'zip_code'
        elif 'zip' in self.permits_df.columns:
            self.permits_df = self.permits_df.rename(columns={'zip': 'zip_code'})
            zip_column = 'zip_code'
        else:
            logger.error("No zip code column found in permits data")
            return None
            
        logger.info(f"Using zip code column: {zip_column}")
        
        permits_by_zip_year = self.permits_df.groupby(['zip_code', 'year']).size().reset_index(name='permit_count')
        
        # Merge with population data
        if hasattr(self, 'historical_df'):
            # Ensure consistent column naming
            if 'zip' in self.historical_df.columns and 'zip_code' not in self.historical_df.columns:
                self.historical_df = self.historical_df.rename(columns={'zip': 'zip_code'})
            
            merged_df = permits_by_zip_year.merge(
                self.historical_df[['zip_code', 'year', 'population']],
                on=['zip_code', 'year'],
                how='inner'
            )
            
            # Calculate correlation between permits and population
            correlation = merged_df.groupby('year').apply(
                lambda x: x['permit_count'].corr(x['population'])
            )
            
            # Plot correlation over time
            plt.figure(figsize=(12, 6))
            correlation.plot(kind='line', marker='o')
            plt.title('Correlation between Building Permits and Population by Year')
            plt.xlabel('Year')
            plt.ylabel('Correlation Coefficient')
            plt.grid(True)
            plt.savefig(self.output_dir / 'permit_population_correlation.png')
            plt.close()
            
            return correlation
        
        return None

def main():
    # Get Census API key from environment variable
    census_api_key = os.getenv('CENSUS_API_KEY')
    if not census_api_key:
        logger.error("Census API key not found in environment variables. Please set CENSUS_API_KEY.")
        return
    
    # Initialize analyzer with Census API key
    analyzer = ChicagoPopulationAnalyzer(census_api_key=census_api_key)
    
    # Analyze population trends
    logger.info("Analyzing population trends...")
    total_pop, yoy_changes = analyzer.analyze_population_trends()
    if total_pop is not None:
        print("\nPopulation Trends Summary:")
        print(total_pop.describe())
    
    # Analyze income trends
    logger.info("Analyzing income trends...")
    income_trends = analyzer.analyze_income_trends()
    if income_trends is not None:
        print("\nIncome Trends Summary:")
        print(income_trends)
    
    # Analyze permit impact
    logger.info("Analyzing permit impact on population...")
    correlation = analyzer.analyze_permit_impact()
    if correlation is not None:
        print("\nPermit-Population Correlation Summary:")
        print(correlation.describe())

if __name__ == "__main__":
    main()