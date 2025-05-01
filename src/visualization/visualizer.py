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
from src.utils.helpers import resolve_column_name
from src.config.column_alias_map import column_aliases

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
        """Load all required datasets for visualization."""
        try:
            self.population_data = pd.read_csv(settings.CENSUS_PROCESSED_PATH)
            self.permit_data = pd.read_csv(settings.PERMITS_PROCESSED_PATH)
            self.economic_data = pd.read_csv(settings.ECONOMIC_PROCESSED_PATH)
            self.scenario_data = pd.read_csv(settings.PREDICTIONS_DIR / 'scenario_predictions.csv')
            self.business_data = pd.read_csv(settings.BUSINESS_LICENSES_PROCESSED_PATH)
            # Try to load retail deficit processed, fallback to analysis_results
            try:
                self.retail_deficit_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'retail_deficit_processed.csv')
            except FileNotFoundError:
                try:
                    self.retail_deficit_data = pd.read_csv(settings.ANALYSIS_RESULTS_DIR / 'retail_deficit_areas.csv')
                    logger.warning("Loaded retail deficit data from analysis_results as fallback.")
                except FileNotFoundError:
                    self.retail_deficit_data = None
                    logger.warning("Retail deficit data not found in processed or analysis_results.")
            logger.info("All data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
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

            return self._extracted_from_create_business_activity_chart_26(
                output_path,
                "population_trends.html",
                fig,
                'Population trend chart saved to ',
            )
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

            return self._extracted_from_create_business_activity_chart_26(
                output_path,
                "retail_deficit_map.html",
                fig,
                'Retail deficit map saved to ',
            )
        except Exception as e:
            logger.error(f"Error creating retail deficit map: {str(e)}")
            return False
    
    def create_permit_analysis_charts(self) -> bool:
        """Create permit analysis visualizations."""
        try:
            if self.permit_data is None:
                logger.warning("Permit data not available for analysis")
                return False

            # Resolve column names
            year_col = resolve_column_name(self.permit_data, 'year', column_aliases)
            zip_col = resolve_column_name(self.permit_data, 'zip_code', column_aliases)
            permits_col = resolve_column_name(self.permit_data, 'total_permits', column_aliases)
            res_permits_col = resolve_column_name(self.permit_data, 'residential_permits', column_aliases)
            comm_permits_col = resolve_column_name(self.permit_data, 'commercial_permits', column_aliases)
            retail_permits_col = resolve_column_name(self.permit_data, 'retail_permits', column_aliases)
            
            if not all([year_col, zip_col, permits_col, res_permits_col, comm_permits_col, retail_permits_col]):
                logger.error("Required columns not found for permit visualization")
                return False
            
            # Create permits by year chart
            yearly_permits = self.permit_data.groupby(year_col).agg({
                res_permits_col: 'sum',
                comm_permits_col: 'sum',
                retail_permits_col: 'sum'
            }).reset_index()
            
            fig = go.Figure()
            for ptype in [res_permits_col, comm_permits_col, retail_permits_col]:
                fig.add_trace(go.Scatter(
                    x=yearly_permits[year_col],
                    y=yearly_permits[ptype],
                    name=ptype.replace('_permits', '').title(),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title='Building Permits by Year and Type',
                xaxis_title='Year',
                yaxis_title='Number of Permits',
                template='plotly_white'
            )
            
            fig.write_html(str(settings.VISUALIZATIONS_DIR / 'permits_by_year.html'))
            
            # Create permits by ZIP code chart
            zip_permits = self.permit_data.groupby(zip_col).agg({
                permits_col: 'sum'
            }).reset_index()
            
            fig = px.choropleth(
                zip_permits,
                geojson=settings.ZIP_GEOJSON_PATH,
                locations=zip_col,
                color=permits_col,
                color_continuous_scale='Viridis',
                title='Total Permits by ZIP Code'
            )
            
            fig.update_geos(fitbounds='locations', visible=False)
            fig.write_html(str(settings.VISUALIZATIONS_DIR / 'permits_by_zip.html'))
            
            logger.info("Created permit analysis visualizations")
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
            fig = go.Figure()
            # Add GDP trend
            if 'gdp' in self.economic_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.economic_data['year'],
                    y=self.economic_data['gdp'],
                    name='GDP',
                    mode='lines+markers'
                ))
            # Add employment trend if available
            if 'employment' in self.economic_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.economic_data['year'],
                    y=self.economic_data['employment'],
                    name='Employment',
                    mode='lines+markers'
                ))
            else:
                logger.warning(f"Employment column not found in economic data. Columns: {self.economic_data.columns.tolist()}")
            # Update layout
            fig.update_layout(
                title='Economic Indicators Over Time',
                xaxis_title="Year",
                yaxis_title="Value",
                template='plotly_white'
            )
            return self._extracted_from_create_business_activity_chart_26(
                output_path,
                "economic_indicators.html",
                fig,
                'Economic indicators chart saved to ',
            )
        except Exception as e:
            logger.error(f"Error creating economic indicators chart: {str(e)}")
            return False
    
    def create_scenario_impact_chart(self, output_path: Optional[Path] = None) -> bool:
        try:
            if self.scenario_data is None:
                logger.warning("Scenario data not available for impact chart")
                return False
            # Assert year column
            if 'year' not in self.scenario_data.columns:
                if self.scenario_data.index.name != 'year':
                    logger.warning("'year' column not found in scenario data; using index as fallback.")
                self.scenario_data['year'] = self.scenario_data.index
            # Assert zip_code column
            assert 'zip_code' in self.scenario_data.columns, "zip_code missing from scenario data!"
            y_col = next(
                (
                    col
                    for col in ['population', 'predicted_population']
                    if col in self.scenario_data.columns
                ),
                None,
            )
            if not y_col:
                # Fallback: pick first numeric column after 'year' and 'scenario'
                numeric_cols = [c for c in self.scenario_data.columns if c not in ['year', 'scenario'] and self.scenario_data[c].dtype.kind in 'fi']
                y_col = numeric_cols[0] if numeric_cols else self.scenario_data.columns[1]
            fig = px.line(
                self.scenario_data,
                x='year',
                y=y_col,
                color='scenario' if 'scenario' in self.scenario_data.columns else None,
                title='Population Scenarios',
                line_dash='scenario' if 'scenario' in self.scenario_data.columns else None
            )
            # Update layout
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title=y_col.replace('_', ' ').title(),
                showlegend=True,
                template='plotly_white'
            )
            # Save chart
            output_path = output_path or settings.VIZ_DIR / "scenario_impact.html"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            logger.info("Scenario impact chart created successfully")
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
            fig = go.Figure()
            # Add active licenses trend (fallback to total_licenses if needed)
            y_col = 'active_licenses' if 'active_licenses' in self.business_data.columns else (
                'total_licenses' if 'total_licenses' in self.business_data.columns else None)
            if y_col:
                fig.add_trace(go.Scatter(
                    x=self.business_data['year'],
                    y=self.business_data[y_col],
                    name='Active Licenses',
                    mode='lines+markers'
                ))
            # Add retail licenses trend if available
            if 'retail_licenses' in self.business_data.columns:
                fig.add_trace(go.Scatter(
                    x=self.business_data['year'],
                    y=self.business_data['retail_licenses'],
                    name='Retail Licenses',
                    mode='lines+markers'
                ))
            fig.update_layout(
                title='Business License Activity Over Time',
                xaxis_title="Year",
                yaxis_title="Number of Licenses",
                template='plotly_white'
            )
            return self._extracted_from_create_business_activity_chart_26(
                output_path,
                "business_activity.html",
                fig,
                'Business activity chart saved to ',
            )
        except Exception as e:
            logger.error(f"Error creating business activity chart: {str(e)}")
            return False

    # TODO Rename this here and in `create_population_trend_chart`, `create_retail_deficit_map`, `create_economic_indicators_chart` and `create_business_activity_chart`
    def _extracted_from_create_business_activity_chart_26(self, output_path, arg1, fig, arg3):
        output_path = output_path or settings.VIZ_DIR / arg1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"{arg3}{output_path}")
        return True
    
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
            self._extracted_from_create_balance_analysis_charts_19(title, x_label, y_label)
            plt.tight_layout()
            return self._extracted_from_create_map_plot_21(
                output_path, 'Created trend plot: '
            )
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
                scalebar = scalebar(1, location='lower right')  # 1 degree = 111km at equator
                ax.add_artist(scalebar)

            # Remove axis labels as they're not typically needed for maps
            ax.set_axis_off()

            # Add north arrow
            self._add_north_arrow(ax)

            # Adjust layout and save
            plt.tight_layout()

            if output_dir := os.path.dirname(output_path):
                os.makedirs(output_dir, exist_ok=True)

            return self._extracted_from_create_map_plot_21(
                output_path, 'Successfully created map plot and saved to '
            )
        except Exception as e:
            logger.error(f"Error creating map plot: {str(e)}")
            return False

    # TODO Rename this here and in `create_trend_plot` and `create_map_plot`
    def _extracted_from_create_map_plot_21(self, output_path, arg1):
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"{arg1}{output_path}")
        return True

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
            self._extracted_from_create_balance_analysis_charts_19(
                'Housing-Retail Balance Score by ZIP Code',
                'ZIP Code',
                'Balance Score',
            )
            plt.xticks(rotation=90)
            self._extracted_from_create_balance_analysis_charts_44(
                'balance_score_by_zip.png'
            )
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
            self._extracted_from_create_balance_analysis_charts_19(
                'Housing Units vs Retail Space by ZIP Code',
                'Housing Units',
                'Retail Space (sq ft)',
            )
            self._extracted_from_create_balance_analysis_charts_44(
                'housing_vs_retail_scatter.png'
            )
            # 3. Pie chart: Balance Category Distribution
            if 'balance_category' in balance_data.columns:
                plt.figure(figsize=(8, 8))
                category_counts = balance_data['balance_category'].value_counts()
                plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
                plt.title('Distribution of Balance Categories')
                self._extracted_from_create_balance_analysis_charts_44(
                    'balance_category_pie.png'
                )
            logger.info("Created housing-retail balance analysis charts")
            return True
        except Exception as e:
            logger.error(f"Error creating balance analysis charts: {str(e)}")
            return False 

    # TODO Rename this here and in `create_trend_plot` and `create_balance_analysis_charts`
    def _extracted_from_create_balance_analysis_charts_44(self, arg0):
        plt.tight_layout()
        plt.savefig(self.viz_dir / arg0, dpi=300, bbox_inches='tight')
        plt.close() 

    # TODO Rename this here and in `create_trend_plot` and `create_balance_analysis_charts`
    def _extracted_from_create_balance_analysis_charts_19(self, arg0, arg1, arg2):
        plt.title(arg0)
        plt.xlabel(arg1)
        plt.ylabel(arg2) 