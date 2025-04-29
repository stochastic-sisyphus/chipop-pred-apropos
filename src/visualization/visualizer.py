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

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles creation of all visualizations for the Chicago Population Analysis project"""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize visualizer with output directory"""
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for matplotlib and seaborn
        plt.style.use(settings.VIZ_SETTINGS['style'])  # Use default matplotlib style
        sns.set_theme(style="whitegrid")  # Apply seaborn theme with whitegrid
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def create_population_trend_chart(self, data: pd.DataFrame) -> bool:
        """Create population trend visualization"""
        try:
            fig = plt.figure(figsize=(14, 8))
            
            # Plot population trends by ZIP code
            sns.lineplot(data=data, x='year', y='total_population', hue='zip_code')
            
            plt.title('Population Trends by ZIP Code')
            plt.xlabel('Year')
            plt.ylabel('Total Population')
            plt.xticks(rotation=45)
            plt.legend(title='ZIP Code', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.viz_dir / 'population_trends.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Created population trend chart")
            return True
            
        except Exception as e:
            logger.error(f"Error creating population trend chart: {str(e)}")
            return False
            
    def create_permit_analysis_charts(self, permit_data: pd.DataFrame) -> bool:
        """Create permit analysis visualizations."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))

            # Permit distribution by type
            ax1 = plt.subplot(221)
            permit_types = ['residential_permits', 'commercial_permits', 'retail_permits']
            permit_counts = [permit_data[col].sum() for col in permit_types]
            ax1.pie(permit_counts, labels=['Residential', 'Commercial', 'Retail'], autopct='%1.1f%%')
            ax1.set_title('Permit Distribution by Type')

            # Construction cost by type
            ax2 = plt.subplot(222)
            cost_types = ['residential_construction_cost', 'commercial_construction_cost', 'retail_construction_cost']
            costs = [permit_data[col].sum() / 1e6 for col in cost_types]  # Convert to millions
            ax2.bar(['Residential', 'Commercial', 'Retail'], costs)
            ax2.set_title('Construction Cost by Type (Millions $)')
            ax2.set_ylabel('Cost (Millions $)')

            # Permit trends over time
            ax3 = plt.subplot(223)
            yearly_permits = permit_data.groupby('year')[permit_types].sum()
            self._extracted_from_create_permit_analysis_charts_25(
                yearly_permits, ax3, 'Permit Trends by Type', 'Number of Permits'
            )
            # Construction cost trends
            ax4 = plt.subplot(224)
            yearly_costs = permit_data.groupby('year')[cost_types].sum() / 1e6  # Convert to millions
            self._extracted_from_create_permit_analysis_charts_25(
                yearly_costs,
                ax4,
                'Construction Cost Trends by Type',
                'Cost (Millions $)',
            )
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / 'permit_analysis.png')
            plt.close()

            logger.info("Created permit analysis charts")
            return True

        except Exception as e:
            logger.error(f"Error creating permit analysis charts: {str(e)}")
            return False

    # TODO Rename this here and in `create_permit_analysis_charts`
    def _extracted_from_create_permit_analysis_charts_25(self, arg0, ax, arg2, arg3):
        arg0.plot(ax=ax)
        ax.set_title(arg2)
        ax.set_xlabel('Year')
        ax.set_ylabel(arg3)
            
    def create_retail_deficit_map(self, deficit_data: pd.DataFrame) -> bool:
        """Create retail deficit map visualization."""
        try:
            logger.info("Creating retail deficit map...")
            
            # Create figure
            fig = px.choropleth(
                deficit_data,
                locations='zip_code',
                color='leakage_rate',
                color_continuous_scale='RdYlBu_r',
                scope="usa",
                title='Retail Leakage by ZIP Code',
                labels={
                    'leakage_rate': 'Leakage Rate',
                    'zip_code': 'ZIP Code'
                }
            )
            
            # Update layout
            fig.update_layout(
                title_x=0.5,
                geo_scope='usa',
                width=1200,
                height=800
            )
            
            # Save figure
            output_path = self.viz_dir / 'retail_deficit_map.html'
            fig.write_html(str(output_path))
            logger.info(f"Saved retail deficit map to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating retail deficit map: {str(e)}")
            return False
            
    def create_economic_indicators_chart(self, economic_data: pd.DataFrame) -> bool:
        """Create economic indicators visualization"""
        try:
            # Filter out change columns
            base_indicators = [col for col in economic_data.columns 
                             if not col.endswith('_change') and col != 'year']
            
            if not base_indicators:
                logger.warning("No economic indicators available for visualization")
                return False
            
            # Create subplots based on available indicators
            rows = (len(base_indicators) + 1) // 2
            cols = min(2, len(base_indicators))
            
            fig = make_subplots(
                rows=rows, 
                cols=cols,
                subplot_titles=[col.replace('_', ' ').title() for col in base_indicators]
            )
            
            # Add traces for each available indicator
            for idx, col in enumerate(base_indicators):
                row = idx // 2 + 1
                col_num = idx % 2 + 1
                
                # Get change column if it exists
                change_col = f"{col}_change"
                
                # Create main indicator trace
                fig.add_trace(
                    go.Scatter(
                        x=economic_data['year'],
                        y=economic_data[col],
                        name=col.replace('_', ' ').title(),
                        mode='lines+markers'
                    ),
                    row=row,
                    col=col_num
                )
                
                # Add change trace if available
                if change_col in economic_data.columns:
                    fig.add_trace(
                        go.Bar(
                            x=economic_data['year'],
                            y=economic_data[change_col] * 100,  # Convert to percentage
                            name=f"{col.replace('_', ' ').title()} % Change",
                            yaxis=f"y{idx+1 if idx > 0 else ''}"
                        ),
                        row=row,
                        col=col_num
                    )
            
            # Update layout
            fig.update_layout(
                height=400 * rows,
                width=1200,
                title_text="Economic Indicators Over Time",
                showlegend=True,
                template="plotly_white"
            )
            
            # Update y-axes titles
            for i, indicator in enumerate(base_indicators):
                fig.update_yaxes(
                    title_text=indicator.replace('_', ' ').title(),
                    row=(i // 2) + 1,
                    col=(i % 2) + 1
                )
            
            fig.write_html(self.viz_dir / 'economic_indicators.html')
            logger.info("Created economic indicators chart")
            return True
            
        except Exception as e:
            logger.error(f"Error creating economic indicators chart: {str(e)}")
            return False
            
    def create_scenario_impact_chart(self, scenario_data: pd.DataFrame) -> bool:
        """Create scenario impact visualization"""
        try:
            fig = go.Figure()
            
            # Get scenario columns (excluding metadata columns)
            scenario_cols = [col for col in scenario_data.columns 
                           if col not in ['zip_code', 'current', 'year']]
            
            # Add traces for each scenario
            for scenario in scenario_cols:
                if not scenario.endswith('_change') and not scenario.endswith('_pct_change'):
                    fig.add_trace(go.Box(
                        y=scenario_data[scenario],
                        name=scenario.title(),
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
            
            fig.update_layout(
                title='Population Projections by Scenario',
                yaxis_title='Projected Population',
                boxmode='group'
            )
            
            fig.write_html(self.viz_dir / 'scenario_impact.html')
            logger.info("Created scenario impact chart")
            return True
            
        except Exception as e:
            logger.error(f"Error creating scenario impact chart: {str(e)}")
            return False
            
    def create_business_license_charts(self, license_data: pd.DataFrame) -> bool:
        """Create business license analysis visualizations."""
        try:
            logger.info("Creating business license analysis charts...")

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))

            # License trends over time
            ax1 = plt.subplot(221)
            yearly_licenses = license_data.groupby('start_year').size()
            yearly_licenses.plot(kind='line', marker='o', ax=ax1)
            self._extracted_from_create_business_license_charts_13(
                ax1, 'Business Licenses Over Time', 'Year'
            )
            # License type distribution
            ax2 = plt.subplot(222)
            license_types = license_data.groupby('license_description').size()
            license_types.nlargest(10).plot(kind='barh', ax=ax2)
            ax2.set_title('Top 10 License Types')
            ax2.set_xlabel('Number of Licenses')

            # License duration histogram
            ax3 = plt.subplot(223)
            license_data['license_duration'].hist(bins=50, ax=ax3)
            self._extracted_from_create_business_license_charts_13(
                ax3, 'License Duration Distribution', 'Duration (days)'
            )
            # Geographic distribution
            ax4 = plt.subplot(224)
            zip_distribution = license_data.groupby('zip_code').size()
            zip_distribution.plot(kind='bar', ax=ax4)
            self._extracted_from_create_business_license_charts_13(
                ax4, 'Licenses by ZIP Code', 'ZIP Code'
            )
            plt.xticks(rotation=45)

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'business_license_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create interactive choropleth map
            fig2 = px.choropleth(
                license_data.groupby('zip_code').agg({
                    'license_id': 'count',
                    'is_retail': 'sum',
                    'is_restaurant': 'sum'
                }).reset_index(),
                locations='zip_code',
                color='license_id',
                color_continuous_scale='Viridis',
                scope="usa",
                title='Business License Density by ZIP Code',
                labels={
                    'license_id': 'Number of Licenses',
                    'zip_code': 'ZIP Code'
                }
            )

            # Update layout
            fig2.update_layout(
                title_x=0.5,
                geo_scope='usa',
                width=1200,
                height=800
            )

            # Save interactive map
            fig2.write_html(str(self.viz_dir / 'business_license_map.html'))

            logger.info("Created business license analysis charts")
            return True

        except Exception as e:
            logger.error(f"Error creating business license charts: {str(e)}")
            return False

    # TODO Rename this here and in `create_business_license_charts`
    def _extracted_from_create_business_license_charts_13(self, arg0, arg1, arg2):
        arg0.set_title(arg1)
        arg0.set_xlabel(arg2)
        arg0.set_ylabel('Number of Licenses')
            
    def create_dashboard(self, data_dict: dict) -> bool:
        """Create an interactive dashboard combining all visualizations"""
        try:
            # Create dashboard layout
            dashboard = go.Figure()
            
            # Add all available plots
            for name, data in data_dict.items():
                if isinstance(data, pd.DataFrame) and 'year' in data.columns:
                    dashboard.add_trace(
                        go.Scatter(x=data['year'], 
                                 y=data[data.select_dtypes(include=[np.number]).columns[0]],
                                 name=name)
                    )
            
            dashboard.update_layout(
                title='Chicago Population Analysis Dashboard',
                height=800,
                showlegend=True
            )
            
            dashboard.write_html(self.output_dir / 'dashboard.html')
            logger.info("Created interactive dashboard")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return False
            
    def generate_all_visualizations(self, data_dict: dict) -> bool:
        """Generate all visualizations"""
        success = True
        
        # Run all visualization methods
        if 'population_data' in data_dict and not self.create_population_trend_chart(data_dict['population_data']):
            success = False
                
        if 'permit_data' in data_dict and not self.create_permit_analysis_charts(data_dict['permit_data']):
            success = False
                
        if 'retail_deficit' in data_dict and not self.create_retail_deficit_map(data_dict['retail_deficit']):
            success = False
                
        if 'economic_data' in data_dict and not self.create_economic_indicators_chart(data_dict['economic_data']):
            success = False
                
        if 'scenario_data' in data_dict and not self.create_scenario_impact_chart(data_dict['scenario_data']):
            success = False
                
        if 'business_licenses' in data_dict and not self.create_business_license_charts(data_dict['business_licenses']):
            success = False
                
        # Create dashboard last
        if not self.create_dashboard(data_dict):
            success = False
            
        return success
        
    def generate_all(self) -> bool:
        """Wrapper method to match interface expected by main.py"""
        try:
            # Load data from saved files
            data_dict = {}
            
            try:
                # Load base data
                data_dict['population_data'] = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')
                logger.info("Loaded population data")
            except Exception as e:
                logger.warning(f"Could not load population data: {str(e)}")
            
            try:
                # Load retail data
                data_dict['retail_deficit'] = pd.read_csv(settings.OUTPUT_DIR / 'retail_leakage_areas.csv')
                logger.info("Loaded retail deficit data")
            except Exception as e:
                logger.warning(f"Could not load retail deficit data: {str(e)}")
            
            try:
                # Load permit data
                data_dict['permit_data'] = pd.read_csv(settings.PROCESSED_DATA_DIR / 'permits_processed.csv')
                logger.info("Loaded permit data")
            except Exception as e:
                logger.warning(f"Could not load permit data: {str(e)}")
            
            try:
                # Load economic data
                data_dict['economic_data'] = pd.read_csv(settings.PROCESSED_DATA_DIR / 'economic_processed.csv')
                logger.info("Loaded economic data")
            except Exception as e:
                logger.warning(f"Could not load economic data: {str(e)}")
            
            try:
                # Load scenario data
                data_dict['scenario_data'] = pd.read_csv(settings.OUTPUT_DIR / 'scenario_predictions.csv')
                logger.info("Loaded scenario data")
            except Exception as e:
                logger.warning(f"Could not load scenario data: {str(e)}")
                
            try:
                # Load business license data
                data_dict['business_licenses'] = pd.read_csv(settings.PROCESSED_DATA_DIR / 'business_licenses.csv')
                logger.info("Loaded business license data")
            except Exception as e:
                logger.warning(f"Could not load business license data: {str(e)}")
            
            if not data_dict:
                logger.error("No data available for visualization")
                return False
            
            return self.generate_all_visualizations(data_dict)
            
        except Exception as e:
            logger.error(f"Error in generate_all: {str(e)}")
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
            import geopandas as gpd
            from matplotlib_scalebar.scalebar import ScaleBar

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