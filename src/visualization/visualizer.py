import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PopulationVisualizer:
    def __init__(self):
        """Initialize the visualizer"""
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.output_dir = Path(os.getenv('OUTPUT_DIR', 'output'))
        self.viz_dir = self.output_dir / 'visualizations'
        
        # Create visualization directory
        self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style for static plots
        plt.style.use('seaborn')
        
    def load_data(self):
        """Load all necessary data for visualization"""
        try:
            # Load predictions
            predictions_df = pd.read_csv(self.output_dir / 'population_predictions.csv')
            
            # Load historical data
            historical_df = pd.read_csv(self.data_dir / 'historical_population.csv')
            
            # Load permits data
            permits_df = pd.read_csv(self.data_dir / 'building_permits.csv')
            
            return predictions_df, historical_df, permits_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None, None
            
    def plot_population_trends(self, historical_df):
        """Plot historical population trends by ZIP code"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Get top 10 ZIP codes by population
            top_zips = historical_df.groupby('zip_code')['population'].mean().nlargest(10).index
            
            # Plot trends for top ZIP codes
            for zip_code in top_zips:
                data = historical_df[historical_df['zip_code'] == zip_code]
                plt.plot(data['year'], data['population'], label=f'ZIP {zip_code}')
            
            plt.title('Population Trends in Top 10 Chicago ZIP Codes')
            plt.xlabel('Year')
            plt.ylabel('Population')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.viz_dir / 'population_trends.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting population trends: {str(e)}")
            
    def plot_scenario_predictions(self, predictions_df):
        """Create interactive plot of scenario predictions"""
        try:
            fig = go.Figure()
            
            # Add traces for each scenario
            fig.add_trace(go.Scatter(
                y=predictions_df['Actual'],
                name='Actual',
                mode='markers',
                marker=dict(size=8)
            ))
            
            for scenario in ['Optimistic', 'Neutral', 'Pessimistic']:
                fig.add_trace(go.Scatter(
                    y=predictions_df[scenario],
                    name=scenario,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title='Population Predictions by Scenario',
                xaxis_title='Sample',
                yaxis_title='Population',
                hovermode='x unified'
            )
            
            # Save as HTML
            fig.write_html(self.viz_dir / 'scenario_predictions.html')
            
        except Exception as e:
            logger.error(f"Error plotting scenario predictions: {str(e)}")
            
    def create_choropleth(self, predictions_df, historical_df):
        """Create choropleth map of population changes"""
        try:
            # Calculate population change
            latest_year = historical_df['year'].max()
            baseline = historical_df[historical_df['year'] == latest_year]
            
            # Merge with predictions
            map_data = baseline.merge(
                predictions_df[['zip_code', 'Neutral']],
                on='zip_code',
                how='left'
            )
            
            map_data['change_pct'] = (
                (map_data['Neutral'] - map_data['population']) / 
                map_data['population'] * 100
            )
            
            # Create choropleth
            fig = px.choropleth(
                map_data,
                locations='zip_code',
                color='change_pct',
                scope="usa",
                locationmode='USA-ZIP',
                color_continuous_scale="RdYlBu",
                range_color=[-20, 20],
                title='Projected Population Change by ZIP Code (%)'
            )
            
            fig.update_layout(
                margin={"r":0,"t":30,"l":0,"b":0},
                coloraxis_colorbar_title='% Change'
            )
            
            # Save as HTML
            fig.write_html(self.viz_dir / 'population_change_map.html')
            
        except Exception as e:
            logger.error(f"Error creating choropleth: {str(e)}")
            
    def plot_permit_activity(self, permits_df):
        """Plot building permit activity over time"""
        try:
            # Aggregate permits by year
            yearly_permits = permits_df.groupby(
                pd.to_datetime(permits_df['issue_date']).dt.year
            ).size().reset_index()
            yearly_permits.columns = ['year', 'permit_count']
            
            plt.figure(figsize=(10, 6))
            plt.bar(yearly_permits['year'], yearly_permits['permit_count'])
            plt.title('Building Permit Activity in Chicago')
            plt.xlabel('Year')
            plt.ylabel('Number of Permits')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.viz_dir / 'permit_activity.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting permit activity: {str(e)}")
            
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        try:
            logger.info("Loading data for visualization...")
            predictions_df, historical_df, permits_df = self.load_data()
            
            if all(df is not None for df in [predictions_df, historical_df, permits_df]):
                logger.info("Generating population trend plots...")
                self.plot_population_trends(historical_df)
                
                logger.info("Generating scenario prediction plots...")
                self.plot_scenario_predictions(predictions_df)
                
                logger.info("Generating choropleth map...")
                self.create_choropleth(predictions_df, historical_df)
                
                logger.info("Generating permit activity plot...")
                self.plot_permit_activity(permits_df)
                
                logger.info("All visualizations generated successfully.")
                return True
            else:
                logger.error("Could not generate visualizations due to missing data.")
                return False
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return False


if __name__ == "__main__":
    visualizer = PopulationVisualizer()
    visualizer.generate_all_visualizations()