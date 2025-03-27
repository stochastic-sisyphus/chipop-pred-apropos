import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChicagoAnalyzer:
    def __init__(self):
        self.data_dir = Path('data')
        
        # Load datasets
        logger.info("Loading datasets...")
        self.load_datasets()
        
    def load_datasets(self):
        """Load all datasets"""
        # Load building permits
        self.permits_df = pd.read_csv(self.data_dir / 'building_permits.csv')
        logger.info(f"Loaded {len(self.permits_df)} building permits")
        
        # Load zoning data
        self.zoning_df = pd.read_csv(self.data_dir / 'zoning.csv')
        logger.info(f"Loaded {len(self.zoning_df)} zoning records")
        
        # Load economic indicators
        self.econ_df = pd.read_csv(self.data_dir / 'economic_indicators.csv')
        logger.info(f"Loaded {len(self.econ_df)} economic indicator records")
        
        # Basic data cleaning
        self.clean_data()
    
    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Convert dates
        self.permits_df['issue_date'] = pd.to_datetime(self.permits_df['issue_date'])
        self.econ_df.index = pd.to_datetime(self.econ_df.index)
        
        # Clean numeric columns
        numeric_cols = ['reported_cost', 'community_area', 'ward']
        for col in numeric_cols:
            if col in self.permits_df.columns:
                self.permits_df[col] = pd.to_numeric(self.permits_df[col], errors='coerce')
    
    def analyze_permits_over_time(self):
        """Analyze building permits trends over time"""
        # Group by year and count permits
        yearly_permits = self.permits_df.groupby(
            self.permits_df['issue_date'].dt.year
        ).size()
        
        # Plot
        plt.figure(figsize=(12, 6))
        yearly_permits.plot(kind='bar')
        plt.title('Building Permits by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Permits')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('output/permits_by_year.png')
        plt.close()
        
        return yearly_permits
    
    def analyze_permit_costs(self):
        """Analyze reported costs of building permits"""
        # Remove outliers for visualization (above 99th percentile)
        cost_threshold = self.permits_df['reported_cost'].quantile(0.99)
        filtered_costs = self.permits_df[
            self.permits_df['reported_cost'] <= cost_threshold
        ]['reported_cost']
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(filtered_costs, bins=50)
        plt.title('Distribution of Building Permit Costs (excluding top 1%)')
        plt.xlabel('Reported Cost ($)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('output/permit_costs_distribution.png')
        plt.close()
        
        # Calculate summary statistics
        cost_stats = self.permits_df['reported_cost'].describe()
        return cost_stats
    
    def analyze_geographic_distribution(self):
        """Analyze geographic distribution of permits"""
        # Group by community area and count permits
        community_permits = self.permits_df.groupby('community_area').size()
        community_permits = community_permits.sort_values(ascending=False)
        
        # Plot top 20 communities
        plt.figure(figsize=(15, 6))
        community_permits.head(20).plot(kind='bar')
        plt.title('Top 20 Community Areas by Number of Building Permits')
        plt.xlabel('Community Area')
        plt.ylabel('Number of Permits')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('output/permits_by_community.png')
        plt.close()
        
        return community_permits
    
    def analyze_economic_indicators(self):
        """Analyze trends in economic indicators"""
        # Plot economic indicators over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot treasury rates
        self.econ_df['treasury_10y'].plot(ax=ax1)
        ax1.set_title('10-Year Treasury Rate Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Rate (%)')
        
        # Plot consumer sentiment
        self.econ_df['consumer_sentiment'].plot(ax=ax2)
        ax2.set_title('Consumer Sentiment Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Index')
        
        plt.tight_layout()
        plt.savefig('output/economic_indicators.png')
        plt.close()
        
        return self.econ_df.describe()

def main():
    # Create output directory if it doesn't exist
    Path('output').mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = ChicagoAnalyzer()
    
    # Run analyses
    logger.info("Analyzing permits over time...")
    yearly_permits = analyzer.analyze_permits_over_time()
    print("\nYearly Permits:")
    print(yearly_permits)
    
    logger.info("Analyzing permit costs...")
    cost_stats = analyzer.analyze_permit_costs()
    print("\nPermit Cost Statistics:")
    print(cost_stats)
    
    logger.info("Analyzing geographic distribution...")
    community_permits = analyzer.analyze_geographic_distribution()
    print("\nTop 10 Communities by Number of Permits:")
    print(community_permits.head(10))
    
    logger.info("Analyzing economic indicators...")
    econ_stats = analyzer.analyze_economic_indicators()
    print("\nEconomic Indicators Summary:")
    print(econ_stats)

if __name__ == "__main__":
    main()