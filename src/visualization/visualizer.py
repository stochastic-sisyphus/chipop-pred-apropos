"""
Visualization module for Chicago Population Analysis
Handles creation of all required visualizations and plots
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import settings
import geopandas as gpd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles creation of all visualizations for the Chicago Population Analysis project"""
    
    def __init__(self, data_dir: Path = None):
        """Initialize the visualizer with data paths."""
        self.data_dir = data_dir or settings.DATA_PROCESSED_DIR
        
        # Initialize data attributes as None
        self.population_data = None
        self.retail_deficit_data = None
        self.permit_data = None
        self.economic_data = None
        self.scenario_data = None
        self.business_data = None
        
    def load_data(self) -> bool:
        """Load all required datasets."""
        try:
            # Load population data
            try:
                self.population_data = pd.read_csv(self.data_dir / "census_processed.csv")
                logger.info("Population data loaded successfully")
            except Exception as e:
                logger.error(f"Error loading population data: {str(e)}")
                self.population_data = None
            
            # Load retail deficit data
            try:
                self.retail_deficit_data = pd.read_csv(self.data_dir / "retail_deficit_processed.csv")
                logger.info("Retail deficit data loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading retail deficit data: {str(e)}")
                self.retail_deficit_data = None
            
            # Load permit data
            try:
                self.permit_data = pd.read_csv(self.data_dir / "permits_processed.csv")
                logger.info("Permit data loaded successfully")
            except Exception as e:
                logger.error(f"Error loading permit data: {str(e)}")
                self.permit_data = None
            
            # Load economic data
            try:
                self.economic_data = pd.read_csv(self.data_dir / "economic_processed.csv")
                logger.info("Economic data loaded successfully")
            except Exception as e:
                logger.error(f"Error loading economic data: {str(e)}")
                self.economic_data = None
            
            # Load scenario data
            try:
                self.scenario_data = pd.read_csv(settings.PREDICTIONS_DIR / "scenario_predictions.csv")
                logger.info("Scenario data loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading scenario data: {str(e)}")
                self.scenario_data = None
            
            # Load business license data
            try:
                self.business_data = pd.read_csv(self.data_dir / "business_licenses_processed.csv")
                logger.info("Business license data loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading business license data: {str(e)}")
                self.business_data = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            return False
    
    def create_population_trend_chart(self, output_path: Optional[Path] = None) -> bool:
        """Create population trend chart."""
        try:
            if self.population_data is None:
                logger.warning("Population data not available for trend chart")
                return False
            
            # Create population trend chart
            fig = px.line(
                self.population_data,
                x='year',
                y='total_population',
                color='zip_code',
                title='Population Trends by ZIP Code'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Total Population",
                showlegend=True,
                template='plotly_white'
            )
            
            # Save chart
            output_path = output_path or settings.VIZ_DIR / "population_trends.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            
            logger.info(f"Population trend chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating population trend chart: {str(e)}")
            return False
    
    def create_retail_deficit_map(self, output_path: Optional[Path] = None) -> bool:
        """Create retail deficit map."""
        try:
            if self.retail_deficit_data is None:
                logger.warning("Retail deficit data not available for map")
                return False
            
            # Create retail deficit map
            fig = px.choropleth(
                self.retail_deficit_data,
                geojson=settings.GEOJSON_PATH,
                locations='zip_code',
                color='retail_deficit',
                color_continuous_scale="RdYlBu",
                scope="usa",
                title='Retail Deficit by ZIP Code',
                hover_data=['zip_code', 'retail_deficit', 'total_population']
            )
            
            # Update layout
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"r":0,"t":30,"l":0,"b":0},
                coloraxis_colorbar_title="Retail Deficit ($)"
            )
            
            # Save map
            output_path = output_path or settings.VIZ_DIR / "retail_deficit_map.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            
            logger.info(f"Retail deficit map saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating retail deficit map: {str(e)}")
            return False
    
    def create_permit_analysis_charts(self, output_path: Optional[Path] = None) -> bool:
        """Create permit analysis charts."""
        try:
            if self.permit_data is None:
                logger.warning("Permit data not available for analysis charts")
                return False
            
            # Create permit trend chart
            fig1 = px.line(
                self.permit_data,
                x='year',
                y='total_permits',
                color='zip_code',
                title='Building Permits by ZIP Code'
            )
            
            # Create permit type distribution chart
            permit_types = self.permit_data.groupby('permit_type')['total_permits'].sum()
            fig2 = px.pie(
                values=permit_types.values,
                names=permit_types.index,
                title='Distribution of Permit Types'
            )
            
            # Save charts
            output_path = output_path or settings.VIZ_DIR
            output_path.mkdir(parents=True, exist_ok=True)
            
            fig1.write_html(str(output_path / "permit_trends.html"))
            fig2.write_html(str(output_path / "permit_distribution.html"))
            
            logger.info(f"Permit analysis charts saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating permit analysis charts: {str(e)}")
            return False
    
    def create_economic_indicators_chart(self, output_path: Optional[Path] = None) -> bool:
        """Create economic indicators chart."""
        try:
            if self.economic_data is None:
                logger.warning("Economic data not available for indicators chart")
                return False
            
            # Create economic indicators chart
            fig = go.Figure()
            
            # Add GDP trend
            fig.add_trace(go.Scatter(
                x=self.economic_data['year'],
                y=self.economic_data['gdp'],
                name='GDP',
                mode='lines+markers'
            ))
            
            # Add employment trend
            fig.add_trace(go.Scatter(
                x=self.economic_data['year'],
                y=self.economic_data['employment'],
                name='Employment',
                mode='lines+markers'
            ))
            
            # Update layout
            fig.update_layout(
                title='Economic Indicators Over Time',
                xaxis_title="Year",
                yaxis_title="Value",
                template='plotly_white'
            )
            
            # Save chart
            output_path = output_path or settings.VIZ_DIR / "economic_indicators.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            
            logger.info(f"Economic indicators chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating economic indicators chart: {str(e)}")
            return False
    
    def create_scenario_impact_chart(self, output_path: Optional[Path] = None) -> bool:
        """Create scenario impact chart."""
        try:
            if self.scenario_data is None:
                logger.warning("Scenario data not available for impact chart")
                return False
            
            # Create scenario impact chart
            fig = px.line(
                self.scenario_data,
                x='year',
                y='population',
                color='scenario',
                title='Population Scenarios',
                line_dash='scenario'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Population",
                showlegend=True,
                template='plotly_white'
            )
            
            # Save chart
            output_path = output_path or settings.VIZ_DIR / "scenario_impact.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            
            logger.info(f"Scenario impact chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating scenario impact chart: {str(e)}")
            return False
    
    def create_business_activity_chart(self, output_path: Optional[Path] = None) -> bool:
        """Create business activity chart."""
        try:
            if self.business_data is None:
                logger.warning("Business license data not available for activity chart")
                return False
            
            # Create business activity chart
            fig = go.Figure()
            
            # Add active licenses trend
            fig.add_trace(go.Scatter(
                x=self.business_data['year'],
                y=self.business_data['active_licenses'],
                name='Active Licenses',
                mode='lines+markers'
            ))
            
            # Add retail licenses trend
            if 'retail_licenses' in self.business_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.business_data['year'],
                    y=self.business_data['retail_licenses'],
                    name='Retail Licenses',
                    mode='lines+markers'
                ))
            
            # Update layout
            fig.update_layout(
                title='Business License Activity Over Time',
                xaxis_title="Year",
                yaxis_title="Number of Licenses",
                template='plotly_white'
            )
            
            # Save chart
            output_path = output_path or settings.VIZ_DIR / "business_activity.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            
            logger.info(f"Business activity chart saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating business activity chart: {str(e)}")
            return False
    
    def create_dashboard(self, output_path: Optional[Path] = None) -> bool:
        """Create HTML dashboard with all visualizations."""
        try:
            # Generate all charts
            charts = {
                'population': self.create_population_trend_chart(),
                'retail': self.create_retail_deficit_map(),
                'permits': self.create_permit_analysis_charts(),
                'economic': self.create_economic_indicators_chart(),
                'scenarios': self.create_scenario_impact_chart(),
                'business': self.create_business_activity_chart()
            }
            
            # Create dashboard HTML
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chicago Population Analysis Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                    .dashboard { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
                    .chart { width: 100%; height: 500px; border: none; }
                    h1 { text-align: center; color: #333; }
                </style>
            </head>
            <body>
                <h1>Chicago Population Analysis Dashboard</h1>
                <div class="dashboard">
                    <iframe class="chart" src="population_trends.html"></iframe>
                    <iframe class="chart" src="retail_deficit_map.html"></iframe>
                    <iframe class="chart" src="permit_trends.html"></iframe>
                    <iframe class="chart" src="permit_distribution.html"></iframe>
                    <iframe class="chart" src="economic_indicators.html"></iframe>
                    <iframe class="chart" src="scenario_impact.html"></iframe>
                    <iframe class="chart" src="business_activity.html"></iframe>
                </div>
            </body>
            </html>
            """
            
            # Save dashboard
            output_path = output_path or settings.VIZ_DIR / "dashboard.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Dashboard created successfully at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return False

    def create_trend_plot(self, data, x_label, y_label, title, output_path):
        """
        Create a generic trend line plot.
        Args:
            data: pd.Series or pd.DataFrame (index as x, values as y)
            x_label: str
            y_label: str
            title: str
            output_path: Path or str
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            plt.figure(figsize=(12, 7))
            if hasattr(data, 'plot'):
                data.plot()
            else:
                plt.plot(data)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Created trend plot: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating trend plot: {str(e)}")
            return False

    def create_map_plot(self, data, column: str, title: str, output_path: str,
                       cmap: str = 'viridis', add_scalebar: bool = True,
                       legend_title: str = None) -> bool:
        """Create a geographic visualization from a GeoDataFrame.

        Args:
            data: GeoDataFrame or DataFrame with geometry column
            column: Column name to plot/visualize
            title: Title for the plot
            output_path: Path to save the visualization
            cmap: Colormap to use for visualization (default: 'viridis')
            add_scalebar: Whether to add a scale bar (default: True)
            legend_title: Custom title for the legend (default: None)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to GeoDataFrame if necessary
            if not isinstance(data, gpd.GeoDataFrame):
                try:
                    data = gpd.GeoDataFrame(data)
                except Exception as e:
                    logger.error(f"Failed to convert data to GeoDataFrame: {str(e)}")
                    return False

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot the data
            data.plot(column=column,
                     cmap=cmap,
                     legend=True,
                     ax=ax,
                     legend_kwds={'label': legend_title or column})

            # Add title
            ax.set_title(title, pad=20)

            # Add scale bar if requested
            if add_scalebar:
                scalebar = ScaleBar(1, location='lower right')  # 1 degree = 111km at equator
                ax.add_artist(scalebar)

            # Remove axis labels as they're not typically needed for maps
            ax.set_axis_off()

            # Add north arrow
            self._add_north_arrow(ax)

            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save the plot
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Successfully created map plot and saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating map plot: {str(e)}")
            return False

    def _add_north_arrow(self, ax):
        """Add a north arrow to the map."""
        try:
            # Get the current axis limits
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            
            # Calculate arrow position (upper right corner)
            arrow_x = x_lim[1] - (x_lim[1] - x_lim[0]) * 0.1
            arrow_y = y_lim[1] - (y_lim[1] - y_lim[0]) * 0.1
            arrow_length = (y_lim[1] - y_lim[0]) * 0.05

            # Draw the arrow
            ax.arrow(arrow_x, arrow_y - arrow_length,
                    0, arrow_length,
                    head_width=arrow_length * 0.3,
                    head_length=arrow_length * 0.3,
                    fc='k', ec='k')
            
            # Add 'N' label
            ax.text(arrow_x, arrow_y + arrow_length * 0.2,
                   'N', ha='center', va='bottom')

        except Exception as e:
            logger.warning(f"Failed to add north arrow: {str(e)}")
            # Don't raise the error as this is a non-critical feature 

    def create_balance_analysis_charts(self, balance_data: pd.DataFrame) -> bool:
        """Create visualizations for housing-retail balance analysis."""
        try:
            logger.info("Creating housing-retail balance analysis charts...")
            # Ensure output directory exists
            self.viz_dir.mkdir(parents=True, exist_ok=True)

            # 1. Bar plot: Balance Score by ZIP Code
            plt.figure(figsize=(16, 8))
            sorted_data = balance_data.sort_values('balance_score', ascending=False)
            plt.bar(sorted_data['zip_code'].astype(str), sorted_data['balance_score'], color='skyblue')
            plt.title('Housing-Retail Balance Score by ZIP Code')
            plt.xlabel('ZIP Code')
            plt.ylabel('Balance Score')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'balance_score_by_zip.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Scatter plot: Housing Units vs Retail Space colored by Balance Score
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                balance_data['housing_units'],
                balance_data['retail_space'],
                c=balance_data['balance_score'],
                cmap='coolwarm',
                alpha=0.7,
                edgecolor='k'
            )
            plt.colorbar(scatter, label='Balance Score')
            plt.title('Housing Units vs Retail Space by ZIP Code')
            plt.xlabel('Housing Units')
            plt.ylabel('Retail Space (sq ft)')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'housing_vs_retail_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Pie chart: Balance Category Distribution
            if 'balance_category' in balance_data.columns:
                plt.figure(figsize=(8, 8))
                category_counts = balance_data['balance_category'].value_counts()
                plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
                plt.title('Distribution of Balance Categories')
                plt.tight_layout()
                plt.savefig(self.viz_dir / 'balance_category_pie.png', dpi=300, bbox_inches='tight')
                plt.close()

            logger.info("Created housing-retail balance analysis charts")
            return True
        except Exception as e:
            logger.error(f"Error creating balance analysis charts: {str(e)}")
            return False 