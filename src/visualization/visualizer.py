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
import traceback

logger = logging.getLogger(__name__)

class Visualizer:
    """Class for creating visualizations."""
    
    def __init__(self):
        """Initialize the visualizer."""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = [12, 8]
        
        # Set font sizes
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Set color palette
        self.colors = sns.color_palette('husl', n_colors=59)  # Match exact number of Chicago ZIP codes
        
        # Set output directory
        self.output_dir = settings.VISUALIZATIONS_DIR
        
    def create_balance_analysis_charts(self, data: pd.DataFrame) -> None:
        """Create visualizations for housing-retail balance analysis."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_dir) / 'balance_analysis'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create scatter plot of housing vs retail
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=data,
                x='total_housing_units',
                y='retail_space',
                hue='balance_score' if 'balance_score' in data.columns else None,
                size='total_population',
                sizes=(50, 400),
                alpha=0.6
            )
            plt.title('Housing Units vs Retail Space by ZIP Code')
            plt.xlabel('Total Housing Units')
            plt.ylabel('Retail Space (sq ft)')
            plt.savefig(output_dir / 'housing_vs_retail_scatter.png')
            plt.close()
            
            # Balance score distribution by ZIP code
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=data,
                x='zip_code',
                y='balance_score',
                hue='zip_code',
                legend=False,
                palette=self.colors
            )
            plt.xticks(rotation=45)
            plt.title('Housing-Retail Balance Score by ZIP Code')
            plt.xlabel('ZIP Code')
            plt.ylabel('Balance Score')
            plt.tight_layout()
            plt.savefig(output_dir / 'balance_score_by_zip.png')
            plt.close()
            
            # Create pie chart of balance categories
            if 'balance_category' in data.columns:
                plt.figure(figsize=(10, 10))
                balance_counts = data['balance_category'].value_counts()
                plt.pie(
                    balance_counts,
                    labels=balance_counts.index,
                    autopct='%1.1f%%',
                    colors=self.colors
                )
                plt.title('Distribution of Balance Categories')
                plt.savefig(output_dir / 'balance_category_pie.png')
                plt.close()
            
            logger.info(f"Balance analysis charts saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating balance analysis charts: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def create_retail_deficit_charts(self, data: pd.DataFrame) -> None:
        """Create charts for retail deficit analysis."""
        try:
            # Create output directory
            charts_dir = self.output_dir / 'retail_deficit'
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Create bar plot of retail gaps
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=data.sort_values('retail_gap' if 'retail_gap' in data.columns else 'zip_code', ascending=False),
                x='zip_code',
                y='retail_gap' if 'retail_gap' in data.columns else 'retail_space',
                palette='viridis'
            )
            plt.title('Retail Gap by ZIP Code')
            plt.xlabel('ZIP Code')
            plt.ylabel('Retail Gap ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / 'retail_gap_by_zip.png')
            plt.close()
            
            # Create scatter plot of population vs retail gap
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=data,
                x='total_population',
                y='retail_gap' if 'retail_gap' in data.columns else 'retail_space',
                size='total_housing_units',
                sizes=(50, 400),
                alpha=0.6
            )
            plt.title('Population vs Retail Gap')
            plt.xlabel('Total Population')
            plt.ylabel('Retail Gap ($)')
            plt.savefig(charts_dir / 'population_vs_retail_gap.png')
            plt.close()
            
            # Create heatmap of retail metrics
            if all(col in data.columns for col in ['retail_demand', 'retail_supply', 'retail_gap']):
                plt.figure(figsize=(12, 8))
                metrics_df = data[['retail_demand', 'retail_supply', 'retail_gap']].corr()
                sns.heatmap(metrics_df, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation of Retail Metrics')
                plt.tight_layout()
                plt.savefig(charts_dir / 'retail_metrics_correlation.png')
                plt.close()
            
            logger.info(f"Retail deficit charts saved to {charts_dir}")
            
        except Exception as e:
            logger.error(f"Error creating retail deficit charts: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def create_development_charts(self, data: pd.DataFrame) -> None:
        """Create charts for development analysis."""
        try:
            # Create output directory
            charts_dir = self.output_dir / 'development'
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Create line plot of permits over time
            plt.figure(figsize=(12, 8))
            permits_by_year = data.groupby('year')['total_permits'].sum()
            plt.plot(permits_by_year.index, permits_by_year.values, marker='o')
            plt.title('Building Permits Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Permits')
            plt.grid(True)
            plt.savefig(charts_dir / 'permits_by_year.png')
            plt.close()
            
            # Create bar plot of permits by type
            if all(col in data.columns for col in ['residential_permits', 'commercial_permits', 'retail_permits']):
                plt.figure(figsize=(12, 8))
                permit_types = ['residential_permits', 'commercial_permits', 'retail_permits']
                permit_counts = [data[col].sum() for col in permit_types]
                plt.bar(
                    ['Residential', 'Commercial', 'Retail'],
                    permit_counts,
                    color=self.colors[:3]
                )
                plt.title('Permits by Type')
                plt.xlabel('Permit Type')
                plt.ylabel('Number of Permits')
                plt.savefig(charts_dir / 'permits_by_type.png')
                plt.close()
            
            # Create box plot of construction costs
            plt.figure(figsize=(12, 8))
            sns.boxplot(
                data=data,
                x='year',
                y='total_construction_cost',
                palette='viridis'
            )
            plt.title('Construction Costs Distribution by Year')
            plt.xlabel('Year')
            plt.ylabel('Construction Cost ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / 'construction_costs_distribution.png')
            plt.close()
            
            logger.info(f"Development charts saved to {charts_dir}")
            
        except Exception as e:
            logger.error(f"Error creating development charts: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def create_population_charts(self, data: pd.DataFrame) -> None:
        """Create charts for population analysis."""
        try:
            # Create output directory
            charts_dir = self.output_dir / 'population'
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Create line plot of total population over time
            plt.figure(figsize=(12, 8))
            pop_by_year = data.groupby('year')['total_population'].sum()
            plt.plot(pop_by_year.index, pop_by_year.values, marker='o')
            plt.title('Total Population Over Time')
            plt.xlabel('Year')
            plt.ylabel('Population')
            plt.grid(True)
            plt.savefig(charts_dir / 'total_population_trend.png')
            plt.close()
            
            # Create box plot of population distribution
            plt.figure(figsize=(12, 8))
            sns.boxplot(
                data=data,
                x='year',
                y='total_population',
                palette='viridis'
            )
            plt.title('Population Distribution by Year')
            plt.xlabel('Year')
            plt.ylabel('Population')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / 'population_distribution.png')
            plt.close()
            
            # Create line plot of median income over time
            plt.figure(figsize=(12, 8))
            income_by_year = data.groupby('year')['median_household_income'].mean()
            plt.plot(income_by_year.index, income_by_year.values, marker='o')
            plt.title('Median Household Income Over Time')
            plt.xlabel('Year')
            plt.ylabel('Income ($)')
            plt.grid(True)
            plt.savefig(charts_dir / 'median_income_trend.png')
            plt.close()
            
            logger.info(f"Population charts saved to {charts_dir}")
            
        except Exception as e:
            logger.error(f"Error creating population charts: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def create_economic_charts(self, data: pd.DataFrame) -> None:
        """Create charts for economic analysis."""
        try:
            # Create output directory
            charts_dir = self.output_dir / 'economic'
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Create line plot of GDP over time
            if 'real_gdp' in data.columns:
                plt.figure(figsize=(12, 8))
                gdp_by_year = data.groupby('year')['real_gdp'].mean()
                plt.plot(gdp_by_year.index, gdp_by_year.values, marker='o')
                plt.title('Real GDP Over Time')
                plt.xlabel('Year')
                plt.ylabel('GDP')
                plt.grid(True)
                plt.savefig(charts_dir / 'gdp_trend.png')
                plt.close()
            
            # Create line plot of unemployment rate
            if 'unemployment_rate' in data.columns:
                plt.figure(figsize=(12, 8))
                unemp_by_year = data.groupby('year')['unemployment_rate'].mean()
                plt.plot(unemp_by_year.index, unemp_by_year.values, marker='o')
                plt.title('Unemployment Rate Over Time')
                plt.xlabel('Year')
                plt.ylabel('Unemployment Rate (%)')
                plt.grid(True)
                plt.savefig(charts_dir / 'unemployment_trend.png')
                plt.close()
            
            # Create scatter plot of income vs home value
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=data,
                x='median_household_income',
                y='median_home_value',
                size='total_population',
                sizes=(50, 400),
                alpha=0.6
            )
            plt.title('Income vs Home Value')
            plt.xlabel('Median Household Income ($)')
            plt.ylabel('Median Home Value ($)')
            plt.savefig(charts_dir / 'income_vs_home_value.png')
            plt.close()
            
            logger.info(f"Economic charts saved to {charts_dir}")
            
        except Exception as e:
            logger.error(f"Error creating economic charts: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
    def create_all_charts(self, data: Dict[str, pd.DataFrame]) -> None:
        """Create all charts."""
        try:
            # Create each type of chart
            self.create_balance_analysis_charts(data['merged'])
            self.create_retail_deficit_charts(data['merged'])
            self.create_development_charts(data['merged'])
            self.create_population_charts(data['merged'])
            self.create_economic_charts(data['merged'])
            
            logger.info("All charts created successfully")
            
        except Exception as e:
            logger.error(f"Error creating charts: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

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

    def _extracted_from_create_balance_analysis_charts_44(self, arg0):
        plt.tight_layout()
        plt.savefig(self.output_dir / arg0, dpi=300, bbox_inches='tight')
        plt.close() 

    # TODO Rename this here and in `create_trend_plot` and `create_balance_analysis_charts`
    def _extracted_from_create_balance_analysis_charts_19(self, arg0, arg1, arg2):
        plt.title(arg0)
        plt.xlabel(arg1)
        plt.ylabel(arg2) 