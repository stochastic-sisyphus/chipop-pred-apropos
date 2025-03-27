"""
Chicago Data Processor
This module processes data collected from various sources for the Chicago population analysis project.
It cleans, transforms, and merges datasets to prepare for modeling.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChicagoDataProcessor:
    """
    Processor for Chicago data sources.
    Handles cleaning, transformation, and merging of datasets.
    """
    def __init__(self):
        """Initialize the data processor with file paths from environment variables."""
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.output_dir = Path(os.getenv('OUTPUT_DIR', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Check if data files exist
        self.permit_file = self.data_dir / 'building_permits.csv'
        self.zoning_file = self.data_dir / 'zoning.geojson'
        self.population_file = self.data_dir / 'historical_population.csv'
        self.economic_file = self.data_dir / 'economic_indicators.csv'
        
        # Validate data files
        required_files = {
            'Building Permits': self.permit_file,
            'Zoning Data': self.zoning_file,
            'Population Data': self.population_file,
            'Economic Indicators': self.economic_file
        }
        
        missing_files = [name for name, path in required_files.items() if not path.exists()]
        if missing_files:
            logger.warning(f"Missing data files: {', '.join(missing_files)}")
            logger.warning("Some data processing steps may fail. Run data_collection.py first.")

    def load_permits(self):
        """
        Load and clean building permits data.
        
        Returns:
            pd.DataFrame: Cleaned permits dataframe or empty dataframe if file not found
        """
        if not self.permit_file.exists():
            logger.error(f"Building permits file not found: {self.permit_file}")
            return pd.DataFrame()
            
        logger.info("Loading building permits data...")
        df = pd.read_csv(self.permit_file)
        
        # Clean and transform
        if 'issue_date' in df.columns:
            df['issue_date'] = pd.to_datetime(df['issue_date'])
            # Create year and month columns for merging
            df['year'] = df['issue_date'].dt.year
            df['month'] = df['issue_date'].dt.month
        
        if 'zip_code' in df.columns:
            df['zip_code'] = df['zip_code'].astype(str).str[:5]
        
        return df

    def load_zoning(self):
        """
        Load and process zoning data.
        
        Returns:
            gpd.GeoDataFrame: Zoning geodataframe or empty geodataframe if file not found
        """
        if not self.zoning_file.exists():
            logger.error(f"Zoning file not found: {self.zoning_file}")
            return gpd.GeoDataFrame()
            
        logger.info("Loading zoning data...")
        try:
            gdf = gpd.read_file(self.zoning_file)
            return gdf
        except Exception as e:
            logger.error(f"Error loading zoning data: {str(e)}")
            return gpd.GeoDataFrame()

    def load_census_data(self):
        """
        Load and combine census data across years.
        
        Returns:
            pd.DataFrame: Census population dataframe or empty dataframe if file not found
        """
        if not self.population_file.exists():
            logger.error(f"Population data file not found: {self.population_file}")
            return pd.DataFrame()
            
        logger.info("Loading census data...")
        df = pd.read_csv(self.population_file)
        
        return df

    def load_economic_indicators(self):
        """
        Load economic indicators data.
        
        Returns:
            pd.DataFrame: Economic indicators dataframe or empty dataframe if file not found
        """
        if not self.economic_file.exists():
            logger.error(f"Economic indicators file not found: {self.economic_file}")
            return pd.DataFrame()
            
        logger.info("Loading economic indicators...")
        df = pd.read_csv(self.economic_file)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
        return df

    def calculate_permit_metrics(self, permits_df):
        """
        Calculate metrics from permit data.
        
        Args:
            permits_df (pd.DataFrame): Building permits dataframe
            
        Returns:
            pd.DataFrame: Aggregated permit metrics by zip code and year
        """
        if permits_df.empty:
            logger.warning("Cannot calculate permit metrics: empty dataframe")
            return pd.DataFrame()
            
        # Ensure required columns exist
        required_cols = ['zip_code', 'year']
        if not all(col in permits_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in permits_df.columns]
            logger.error(f"Cannot calculate permit metrics: missing columns {missing}")
            return pd.DataFrame()
            
        metrics = permits_df.groupby(['zip_code', 'year']).agg({
            'id': 'count',  # number of permits
            'reported_cost': ['sum', 'mean'],  # total and average cost
            'stories': 'mean',  # average number of stories
            'total_land_square_feet': 'sum'  # total land area
        }).reset_index()
        
        metrics.columns = ['zip_code', 'year', 'permit_count', 'total_cost', 
                         'avg_cost', 'avg_stories', 'total_land_area']
        return metrics

    def calculate_income_distribution(self, census_df):
        """
        Calculate income distribution metrics from census data.
        
        Args:
            census_df (pd.DataFrame): Census data dataframe
            
        Returns:
            pd.DataFrame: Census data with added income distribution metrics
        """
        if census_df.empty:
            logger.warning("Cannot calculate income distribution: empty dataframe")
            return census_df
            
        # Check for required percentage columns
        pct_cols = [col for col in census_df.columns if col.endswith('_pct')]
        if not pct_cols:
            logger.warning("No income percentage columns found in census data")
            return census_df
        
        # Calculate middle class percentage (adjust income ranges as needed)
        middle_class_cols = [col for col in pct_cols if any(bracket in col for bracket in 
                           ['50k_60k', '60k_75k', '75k_100k', '100k_125k'])]
        
        if middle_class_cols:
            census_df['middle_class_pct'] = census_df[middle_class_cols].sum(axis=1)
        
        # Calculate lower income percentage
        lower_income_cols = [col for col in pct_cols if any(bracket in col for bracket in 
                            ['less_10k', '10k_15k', '15k_20k', '20k_25k',
                             '25k_30k', '30k_35k', '35k_40k', '40k_45k', '45k_50k'])]
        
        if lower_income_cols:
            census_df['lower_income_pct'] = census_df[lower_income_cols].sum(axis=1)
        
        return census_df

    def merge_datasets(self):
        """
        Merge all datasets into a single analytical dataset.
        
        Returns:
            pd.DataFrame: Merged dataset for analysis or empty dataframe if merge fails
        """
        logger.info("Merging datasets...")
        
        # Load all datasets
        permits = self.load_permits()
        census = self.load_census_data()
        economic = self.load_economic_indicators()
        
        if permits.empty or census.empty or economic.empty:
            logger.error("Cannot merge datasets: one or more datasets are empty")
            return pd.DataFrame()
        
        # Calculate permit metrics
        permit_metrics = self.calculate_permit_metrics(permits)
        
        # Calculate income distribution
        census = self.calculate_income_distribution(census)
        
        # Merge permit metrics with census data
        merged = None
        try:
            # Ensure we have zip_code in census data (might be named differently)
            if 'zip_code' not in census.columns and 'zip' in census.columns:
                census = census.rename(columns={'zip': 'zip_code'})
                
            merged = permit_metrics.merge(
                census[['zip_code', 'year', 'population', 'median_household_income',
                       'middle_class_pct', 'lower_income_pct']],
                on=['zip_code', 'year'],
                how='outer'
            )
            
            # Merge with economic indicators
            merged = merged.merge(
                economic[[
                    'year', 'treasury_10y', 'mortgage_30y', 'consumer_sentiment',
                    'recession_indicator', 'housing_starts', 'house_price_index'
                ]],
                on='year',
                how='left'
            )
            
            # Fill missing values
            merged = merged.fillna({
                'permit_count': 0,
                'total_cost': 0,
                'total_land_area': 0
            })
            
            # Calculate year-over-year changes
            merged['population_change'] = merged.groupby('zip_code')['population'].pct_change()
            merged['income_change'] = merged.groupby('zip_code')['median_household_income'].pct_change()
            
            # Create future population change target (t+1 period)
            merged['population_change_next_year'] = merged.groupby('zip_code')['population_change'].shift(-1)
            
            # Sort by zip_code and year for better organization
            merged = merged.sort_values(['zip_code', 'year'])
            
            # Save merged dataset
            output_file = self.output_dir / 'merged_dataset.csv'
            merged.to_csv(output_file, index=False)
            logger.info(f"Successfully created merged dataset: {output_file}")
            logger.info(f"Added future population change target for next-year prediction")
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            if merged is not None:
                logger.error(f"Merged columns: {merged.columns.tolist()}")
            return pd.DataFrame()
        
        return merged


def main():
    """Main function to run the data processing pipeline."""
    processor = ChicagoDataProcessor()
    processor.merge_datasets()


if __name__ == "__main__":
    main()