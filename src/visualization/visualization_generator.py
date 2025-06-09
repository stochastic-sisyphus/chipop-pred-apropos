"""
Visualization generator module for Chicago Housing Pipeline.

This module provides advanced visualization capabilities for all models and analyses.
"""

import os
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import settings

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """
    Advanced visualization generator for Chicago Housing Pipeline.
    
    Provides high-quality, interactive visualizations for all models and analyses.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir (Path, optional): Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path(settings.OUTPUT_DIR) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default styles
        self.set_visualization_style()
    
    def set_visualization_style(self):
        """
        Set default visualization styles.
        """
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        # Set Matplotlib params
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 20
        
        # Custom color palettes
        self.color_palette = sns.color_palette("viridis", 10)
        self.diverging_palette = sns.diverging_palette(240, 10, n=10)
        self.categorical_palette = sns.color_palette("Set2")
        
        # Custom colormaps
        self.growth_cmap = LinearSegmentedColormap.from_list(
            "growth_cmap", ["#FF5E5B", "#FFFFFF", "#00CECB"]
        )
        
        self.opportunity_cmap = LinearSegmentedColormap.from_list(
            "opportunity_cmap", ["#FFFFFF", "#FFED66", "#FF9505", "#FF5E5B"]
        )
    
    def generate_population_forecast_visualizations(self, model_results, forecast_data):
        """
        Generate visualizations for population forecast model.
        
        Args:
            model_results (dict): Model results
            forecast_data (pd.DataFrame): Forecast data
            
        Returns:
            dict: Paths to generated visualizations
        """
        try:
            logger.info("Generating population forecast visualizations")
            
            # Create output directory
            output_dir = self.output_dir / "population_forecast"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize visualization paths
            viz_paths = {}
            
            # Generate overall population trend
            trend_path = self._plot_overall_population_trend(forecast_data, output_dir)
            if trend_path:
                viz_paths['overall_trend'] = trend_path
            
            # Generate ZIP code comparison
            zip_path = self._plot_population_by_zip(forecast_data, output_dir)
            if zip_path:
                viz_paths['zip_comparison'] = zip_path
            
            # Generate growth rate heatmap
            heatmap_path = self._plot_growth_rate_heatmap(forecast_data, output_dir)
            if heatmap_path:
                viz_paths['growth_heatmap'] = heatmap_path
            
            # Generate top emerging ZIPs
            if 'top_growth_zips' in model_results:
                emerging_path = self._plot_top_emerging_zips(model_results['top_growth_zips'], output_dir)
                if emerging_path:
                    viz_paths['top_emerging_zips'] = emerging_path
            
            # Generate interactive forecast dashboard
            dashboard_path = self._create_interactive_forecast_dashboard(forecast_data, output_dir)
            if dashboard_path:
                viz_paths['interactive_dashboard'] = dashboard_path
            
            logger.info(f"Generated {len(viz_paths)} population forecast visualizations")
            return viz_paths
            
        except Exception as e:
            logger.error(f"Error generating population forecast visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _plot_overall_population_trend(self, forecast_data, output_dir):
        """
        Plot overall population trend.
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Group by year and forecast type
            trend_data = forecast_data.groupby(['year', 'forecast_type'])['population'].sum().reset_index()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot historical data
            historical = trend_data[trend_data['forecast_type'] == 'historical']
            ax.plot(historical['year'], historical['population'], '-o', 
                   color=self.color_palette[0], linewidth=3, markersize=8,
                   label="Historical Population")
            
            # Plot forecast data
            forecast = trend_data[trend_data['forecast_type'] == 'combined']
            ax.plot(forecast['year'], forecast['population'], '--o', 
                   color=self.color_palette[2], linewidth=3, markersize=8,
                   label="Forecast Population")
            
            # Add shaded confidence interval (simplified)
            if not forecast.empty:
                # Create simple confidence interval (±5%)
                lower_bound = forecast['population'] * 0.95
                upper_bound = forecast['population'] * 1.05
                
                ax.fill_between(forecast['year'], lower_bound, upper_bound, 
                               color=self.color_palette[2], alpha=0.2,
                               label="Forecast Uncertainty")
            
            # Calculate growth rate
            if not historical.empty and not forecast.empty:
                last_historical = historical['population'].iloc[-1]
                last_forecast = forecast['population'].iloc[-1]
                growth_pct = (last_forecast - last_historical) / last_historical * 100
                
                # Add growth rate annotation
                ax.annotate(f"Projected Growth: {growth_pct:.1f}%", 
                           xy=(0.75, 0.05), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                           fontsize=14)
            
            # Customize plot
            ax.set_title('Chicago Population Trend and Forecast', fontsize=20, pad=20)
            ax.set_xlabel('Year', fontsize=16)
            ax.set_ylabel('Total Population', fontsize=16)
            
            # Format y-axis with commas
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            ax.legend(fontsize=14, loc='upper left')
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "overall_population_trend.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved overall population trend plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting overall population trend: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_population_by_zip(self, forecast_data, output_dir):
        """
        Plot population by ZIP code.
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Get top 5 ZIP codes by population
            latest_year = forecast_data[forecast_data['forecast_type'] == 'historical']['year'].max()
            top_zips = forecast_data[
                (forecast_data['year'] == latest_year) & 
                (forecast_data['forecast_type'] == 'historical')
            ].nlargest(5, 'population')['zip_code'].unique()
            
            # Filter data to top ZIP codes
            plot_data = forecast_data[forecast_data['zip_code'].isin(top_zips)].copy()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot each ZIP code
            for i, zip_code in enumerate(top_zips):
                zip_data = plot_data[plot_data['zip_code'] == zip_code]
                
                # Split historical and forecast data
                historical = zip_data[zip_data['forecast_type'] == 'historical']
                forecast = zip_data[zip_data['forecast_type'] == 'combined']
                
                # Plot historical data as solid line
                ax.plot(historical['year'], historical['population'], '-', 
                       color=self.categorical_palette[i], linewidth=3,
                       label=f"ZIP {zip_code} (Historical)")
                
                # Plot forecast data as dashed line
                ax.plot(forecast['year'], forecast['population'], '--', 
                       color=self.categorical_palette[i], linewidth=3,
                       label=f"ZIP {zip_code} (Forecast)")
            
            # Add vertical line at the transition from historical to forecast
            if latest_year:
                ax.axvline(x=latest_year, color='black', linestyle=':', alpha=0.5,
                          label="Forecast Start")
            
            # Customize plot
            ax.set_title('Population Trends by ZIP Code', fontsize=20, pad=20)
            ax.set_xlabel('Year', fontsize=16)
            ax.set_ylabel('Population', fontsize=16)
            
            # Format y-axis with commas
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend with two columns
            ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=3, frameon=True, shadow=True)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "population_by_zip.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved population by ZIP plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting population by ZIP: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_growth_rate_heatmap(self, forecast_data, output_dir):
        """
        Plot growth rate heatmap by ZIP code and year.
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Filter to forecast data only
            forecast_df = forecast_data[forecast_data['forecast_type'] == 'combined'].copy()
            
            if forecast_df.empty:
                logger.warning("No forecast data available for growth rate heatmap")
                return None
            
            # Calculate growth rate
            forecast_df['growth_rate'] = forecast_df.groupby('zip_code')['population'].pct_change() * 100
            
            # Pivot data for heatmap
            pivot_data = forecast_df.pivot_table(
                index='zip_code', 
                columns='year', 
                values='growth_rate',
                aggfunc='mean'
            ).fillna(0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Create heatmap
            sns.heatmap(pivot_data, cmap=self.growth_cmap, center=0,
                       annot=True, fmt=".1f", linewidths=.5, ax=ax,
                       cbar_kws={'label': 'Annual Growth Rate (%)'})
            
            # Customize plot
            ax.set_title('Population Growth Rate by ZIP Code and Year', fontsize=20, pad=20)
            ax.set_xlabel('Year', fontsize=16)
            ax.set_ylabel('ZIP Code', fontsize=16)
            
            # Rotate x-axis labels
            plt.xticks(rotation=0)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "growth_rate_heatmap.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved growth rate heatmap to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting growth rate heatmap: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_top_emerging_zips(self, top_zips, output_dir):
        """
        Plot top emerging ZIP codes.
        
        Args:
            top_zips (list): List of top emerging ZIP codes with metrics
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not top_zips:
                logger.warning("No top emerging ZIP codes available")
                return None
            
            # Convert to DataFrame if needed
            if isinstance(top_zips, list):
                df = pd.DataFrame(top_zips)
            else:
                df = top_zips.copy()
            
            # Ensure we have required columns
            if 'zip_code' not in df.columns or 'population_growth_rate' not in df.columns:
                logger.warning("Missing required columns in top emerging ZIPs data")
                return None
            
            # Sort by growth rate
            df = df.sort_values('population_growth_rate', ascending=True)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar chart
            bars = ax.barh(df['zip_code'].astype(str), df['population_growth_rate'],
                          color=self.color_palette)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width if width >= 0 else 0
                ax.text(label_x_pos + 0.001, bar.get_y() + bar.get_height()/2,
                       f"{width:.2f}%", va='center', fontsize=12)
            
            # Customize plot
            ax.set_title('Top Emerging ZIP Codes by Growth Rate', fontsize=20, pad=20)
            ax.set_xlabel('Annual Population Growth Rate (%)', fontsize=16)
            ax.set_ylabel('ZIP Code', fontsize=16)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='x')
            
            # Format x-axis as percentage
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "top_emerging_zips.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved top emerging ZIP codes plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting top emerging ZIP codes: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_interactive_forecast_dashboard(self, forecast_data, output_dir):
        """
        Create interactive forecast dashboard.
        
        Args:
            forecast_data (pd.DataFrame): Forecast data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Create a copy of the data
            df = forecast_data.copy()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Overall Population Trend',
                    'Population by ZIP Code',
                    'Growth Rate Distribution',
                    'Population Density'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot 1: Overall Population Trend
            trend_data = df.groupby(['year', 'forecast_type'])['population'].sum().reset_index()
            
            # Historical data
            historical = trend_data[trend_data['forecast_type'] == 'historical']
            fig.add_trace(
                go.Scatter(
                    x=historical['year'], 
                    y=historical['population'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Forecast data
            forecast = trend_data[trend_data['forecast_type'] == 'combined']
            fig.add_trace(
                go.Scatter(
                    x=forecast['year'], 
                    y=forecast['population'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Plot 2: Population by ZIP Code
            # Get top 5 ZIP codes by population
            latest_year = df[df['forecast_type'] == 'historical']['year'].max()
            top_zips = df[
                (df['year'] == latest_year) & 
                (df['forecast_type'] == 'historical')
            ].nlargest(5, 'population')['zip_code'].unique()
            
            # Plot each ZIP code
            for zip_code in top_zips:
                zip_data = df[df['zip_code'] == zip_code]
                
                fig.add_trace(
                    go.Scatter(
                        x=zip_data['year'],
                        y=zip_data['population'],
                        mode='lines',
                        name=f'ZIP {zip_code}',
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
            
            # Plot 3: Growth Rate Distribution
            # Calculate average growth rate by ZIP code
            forecast_df = df[df['forecast_type'] == 'combined'].copy()
            forecast_df['growth_rate'] = forecast_df.groupby('zip_code')['population'].pct_change() * 100
            growth_by_zip = forecast_df.groupby('zip_code')['growth_rate'].mean().reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=growth_by_zip['zip_code'].astype(str),
                    y=growth_by_zip['growth_rate'],
                    name='Avg. Growth Rate',
                    marker_color='green'
                ),
                row=2, col=1
            )
            
            # Plot 4: Population Density (Bubble Chart)
            latest_forecast = df[
                (df['year'] == df[df['forecast_type'] == 'combined']['year'].max()) & 
                (df['forecast_type'] == 'combined')
            ]
            
            if 'housing_units' in latest_forecast.columns:
                latest_forecast['density'] = latest_forecast['population'] / latest_forecast['housing_units']
            else:
                latest_forecast['density'] = latest_forecast['population']
            
            fig.add_trace(
                go.Scatter(
                    x=latest_forecast['zip_code'].astype(str),
                    y=latest_forecast['population'],
                    mode='markers',
                    marker=dict(
                        size=latest_forecast['density'],
                        sizemode='area',
                        sizeref=2.*max(latest_forecast['density'])/(40.**2),
                        sizemin=4,
                        color=latest_forecast['density'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Density')
                    ),
                    text=latest_forecast['zip_code'],
                    name='Population Density'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text='Chicago Population Forecast Dashboard',
                height=800,
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text='Year', row=1, col=1)
            fig.update_yaxes(title_text='Population', row=1, col=1)
            
            fig.update_xaxes(title_text='Year', row=1, col=2)
            fig.update_yaxes(title_text='Population', row=1, col=2)
            
            fig.update_xaxes(title_text='ZIP Code', row=2, col=1)
            fig.update_yaxes(title_text='Growth Rate (%)', row=2, col=1)
            
            fig.update_xaxes(title_text='ZIP Code', row=2, col=2)
            fig.update_yaxes(title_text='Population', row=2, col=2)
            
            # Save as HTML
            output_path = output_dir / "interactive_forecast_dashboard.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Saved interactive forecast dashboard to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating interactive forecast dashboard: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_retail_gap_visualizations(self, model_results, gap_data):
        """
        Generate visualizations for retail gap model.
        
        Args:
            model_results (dict): Model results
            gap_data (pd.DataFrame): Retail gap data
            
        Returns:
            dict: Paths to generated visualizations
        """
        try:
            logger.info("Generating retail gap visualizations")
            
            # Create output directory
            output_dir = self.output_dir / "retail_gap"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize visualization paths
            viz_paths = {}
            
            # Generate retail gap distribution
            dist_path = self._plot_retail_gap_distribution(gap_data, output_dir)
            if dist_path:
                viz_paths['gap_distribution'] = dist_path
            
            # Generate opportunity zones map
            if 'opportunity_zones' in model_results:
                opp_path = self._plot_opportunity_zones(model_results['opportunity_zones'], gap_data, output_dir)
                if opp_path:
                    viz_paths['opportunity_zones'] = opp_path
            
            # Generate retail cluster analysis
            if 'retail_clusters' in model_results:
                cluster_path = self._plot_retail_clusters(model_results['retail_clusters'], gap_data, output_dir)
                if cluster_path:
                    viz_paths['retail_clusters'] = cluster_path
            
            # Generate category gap analysis
            if 'category_gaps' in model_results:
                category_path = self._plot_category_gaps(model_results['category_gaps'], output_dir)
                if category_path:
                    viz_paths['category_gaps'] = category_path
            
            # Generate interactive retail gap dashboard
            dashboard_path = self._create_interactive_retail_gap_dashboard(gap_data, model_results, output_dir)
            if dashboard_path:
                viz_paths['interactive_dashboard'] = dashboard_path
            
            logger.info(f"Generated {len(viz_paths)} retail gap visualizations")
            return viz_paths
            
        except Exception as e:
            logger.error(f"Error generating retail gap visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _plot_retail_gap_distribution(self, gap_data, output_dir):
        """
        Plot retail gap score distribution.
        
        Args:
            gap_data (pd.DataFrame): Retail gap data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if 'retail_gap_score' not in gap_data.columns:
                logger.warning("No retail gap score data available")
                return None
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Plot 1: Histogram with KDE
            sns.histplot(gap_data['retail_gap_score'], kde=True, ax=ax1, 
                        color=self.color_palette[0], bins=15)
            
            # Add vertical lines for thresholds
            ax1.axvline(x=-0.5, color='green', linestyle='--', linewidth=2, 
                       label='Opportunity Zone Threshold')
            ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                       label='Saturated Market Threshold')
            
            # Customize first plot
            ax1.set_title('Distribution of Retail Gap Scores', fontsize=16)
            ax1.set_xlabel('Retail Gap Score (Standardized)', fontsize=14)
            ax1.set_ylabel('Frequency', fontsize=14)
            ax1.legend(fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot 2: Box plot with swarm plot
            sns.boxplot(y=gap_data['retail_gap_score'], ax=ax2, color=self.color_palette[1])
            sns.swarmplot(y=gap_data['retail_gap_score'], ax=ax2, color='black', size=8, alpha=0.7)
            
            # Add horizontal lines for thresholds
            ax2.axhline(y=-0.5, color='green', linestyle='--', linewidth=2, 
                       label='Opportunity Zone Threshold')
            ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, 
                       label='Saturated Market Threshold')
            
            # Customize second plot
            ax2.set_title('Retail Gap Score Distribution', fontsize=16)
            ax2.set_ylabel('Retail Gap Score', fontsize=14)
            ax2.set_xlabel('', fontsize=14)
            ax2.legend(fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Add overall title
            fig.suptitle('Retail Gap Analysis', fontsize=20, y=1.05)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "retail_gap_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved retail gap distribution plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting retail gap distribution: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_opportunity_zones(self, opportunity_zones, gap_data, output_dir):
        """
        Plot retail opportunity zones.
        
        Args:
            opportunity_zones (list): List of opportunity zones
            gap_data (pd.DataFrame): Retail gap data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not opportunity_zones:
                logger.warning("No opportunity zones available")
                return None
            
            # Convert to DataFrame if needed
            if isinstance(opportunity_zones, list):
                opp_df = pd.DataFrame(opportunity_zones)
            else:
                opp_df = opportunity_zones.copy()
            
            # Ensure we have required columns
            if 'zip_code' not in opp_df.columns or 'retail_gap_score' not in opp_df.columns:
                logger.warning("Missing required columns in opportunity zones data")
                return None
            
            # Sort by gap score
            opp_df = opp_df.sort_values('retail_gap_score')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Create horizontal bar chart
            bars = ax.barh(opp_df['zip_code'].astype(str), opp_df['retail_gap_score'].abs(),
                          color='green')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                       f"{opp_df['retail_gap_score'].iloc[i]:.2f}", va='center', fontsize=12)
            
            # Add retail per capita and predicted retail per capita as text
            if 'retail_per_capita' in opp_df.columns and 'predicted_retail_per_capita' in opp_df.columns:
                for i, zip_code in enumerate(opp_df['zip_code']):
                    actual = opp_df['retail_per_capita'].iloc[i]
                    predicted = opp_df['predicted_retail_per_capita'].iloc[i]
                    ax.text(0.05, i + 0.25, f"Actual: {actual:.4f}", va='center', fontsize=10, color='blue')
                    ax.text(0.05, i - 0.25, f"Predicted: {predicted:.4f}", va='center', fontsize=10, color='red')
            
            # Customize plot
            ax.set_title('Top Retail Opportunity Zones', fontsize=20, pad=20)
            ax.set_xlabel('Retail Gap Score (Absolute Value)', fontsize=16)
            ax.set_ylabel('ZIP Code', fontsize=16)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='x')
            
            # Add annotation explaining opportunity zones
            ax.text(0.5, -0.15, 
                   "Opportunity Zones represent areas with significantly less retail than predicted based on population and economic factors.",
                   ha='center', va='center', transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "opportunity_zones.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved opportunity zones plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting opportunity zones: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_retail_clusters(self, retail_clusters, gap_data, output_dir):
        """
        Plot retail clusters.
        
        Args:
            retail_clusters (list): List of retail clusters
            gap_data (pd.DataFrame): Retail gap data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not retail_clusters:
                logger.warning("No retail clusters available")
                return None
            
            # Convert to DataFrame if needed
            if isinstance(retail_clusters, list):
                cluster_df = pd.DataFrame(retail_clusters)
            else:
                cluster_df = retail_clusters.copy()
            
            # Ensure we have required columns
            if 'retail_cluster' not in cluster_df.columns or 'retail_per_capita' not in cluster_df.columns:
                logger.warning("Missing required columns in retail clusters data")
                return None
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Plot 1: Cluster retail density
            sns.barplot(x='retail_cluster', y='retail_per_capita', data=cluster_df, 
                       palette=self.categorical_palette, ax=ax1)
            
            # Add value labels
            for i, p in enumerate(ax1.patches):
                height = p.get_height()
                ax1.text(p.get_x() + p.get_width()/2., height + 0.001,
                        f"{height:.4f}", ha="center", fontsize=12)
            
            # Customize first plot
            ax1.set_title('Retail Density by Cluster', fontsize=16)
            ax1.set_xlabel('Cluster', fontsize=14)
            ax1.set_ylabel('Retail Establishments per Capita', fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Plot 2: Cluster population and count
            ax2_twin = ax2.twinx()
            
            # Bar plot for population
            sns.barplot(x='retail_cluster', y='population', data=cluster_df, 
                       color=self.color_palette[3], ax=ax2, alpha=0.7,
                       label='Avg. Population')
            
            # Line plot for ZIP count
            sns.pointplot(x='retail_cluster', y='zip_code', data=cluster_df, 
                         color='red', ax=ax2_twin, label='ZIP Count')
            
            # Add value labels for population
            for i, p in enumerate(ax2.patches):
                height = p.get_height()
                ax2.text(p.get_x() + p.get_width()/2., height + 1000,
                        f"{int(height):,}", ha="center", fontsize=10)
            
            # Customize second plot
            ax2.set_title('Cluster Demographics', fontsize=16)
            ax2.set_xlabel('Cluster', fontsize=14)
            ax2.set_ylabel('Average Population', fontsize=14)
            ax2_twin.set_ylabel('Number of ZIP Codes', fontsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
            
            # Add overall title
            fig.suptitle('Retail Cluster Analysis', fontsize=20, y=1.05)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "retail_clusters.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved retail clusters plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting retail clusters: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_category_gaps(self, category_gaps, output_dir):
        """
        Plot retail category gaps.
        
        Args:
            category_gaps (dict): Category gaps by retail category
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not category_gaps:
                logger.warning("No category gaps available")
                return None
            
            # Count gaps by category
            gap_counts = {cat: len(gaps) for cat, gaps in category_gaps.items()}
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create bar chart
            categories = list(gap_counts.keys())
            counts = list(gap_counts.values())
            
            bars = ax.bar(categories, counts, color=self.categorical_palette)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f"{int(height)}", ha='center', va='bottom', fontsize=12)
            
            # Customize plot
            ax.set_title('Retail Category Gaps by Type', fontsize=20, pad=20)
            ax.set_xlabel('Retail Category', fontsize=16)
            ax.set_ylabel('Number of ZIP Codes with Gaps', fontsize=16)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add annotation explaining category gaps
            ax.text(0.5, -0.2, 
                   "Category gaps represent areas with significantly lower retail presence in specific categories compared to the average.",
                   ha='center', va='center', transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "category_gaps.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved category gaps plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting category gaps: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_interactive_retail_gap_dashboard(self, gap_data, model_results, output_dir):
        """
        Create interactive retail gap dashboard.
        
        Args:
            gap_data (pd.DataFrame): Retail gap data
            model_results (dict): Model results
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Create a copy of the data
            df = gap_data.copy()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Retail Gap Score by ZIP Code',
                    'Opportunity Zones',
                    'Retail per Capita vs. Predicted',
                    'Retail Category Distribution'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot 1: Retail Gap Score by ZIP Code
            if 'retail_gap_score' in df.columns:
                # Sort by gap score
                sorted_df = df.sort_values('retail_gap_score')
                
                # Create color array based on thresholds
                colors = ['green' if x < -0.5 else 'red' if x > 0.5 else 'blue' 
                         for x in sorted_df['retail_gap_score']]
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_df['zip_code'].astype(str),
                        y=sorted_df['retail_gap_score'],
                        marker_color=colors,
                        name='Retail Gap Score'
                    ),
                    row=1, col=1
                )
                
                # Add threshold lines
                fig.add_shape(
                    type="line",
                    x0=0, y0=0.5, x1=1, y1=0.5,
                    line=dict(color="red", width=2, dash="dash"),
                    xref="paper", yref="y",
                    row=1, col=1
                )
                
                fig.add_shape(
                    type="line",
                    x0=0, y0=-0.5, x1=1, y1=-0.5,
                    line=dict(color="green", width=2, dash="dash"),
                    xref="paper", yref="y",
                    row=1, col=1
                )
            
            # Plot 2: Opportunity Zones
            if 'opportunity_zones' in model_results:
                opp_zones = pd.DataFrame(model_results['opportunity_zones'])
                if not opp_zones.empty and 'zip_code' in opp_zones.columns and 'retail_gap_score' in opp_zones.columns:
                    # Sort by gap score
                    opp_zones = opp_zones.sort_values('retail_gap_score')
                    
                    fig.add_trace(
                        go.Bar(
                            x=opp_zones['zip_code'].astype(str),
                            y=opp_zones['retail_gap_score'].abs(),
                            marker_color='green',
                            name='Opportunity Score'
                        ),
                        row=1, col=2
                    )
            
            # Plot 3: Retail per Capita vs. Predicted
            if 'retail_per_capita' in df.columns and 'predicted_retail_per_capita' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['predicted_retail_per_capita'],
                        y=df['retail_per_capita'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=df['retail_gap_score'],
                            colorscale='RdBu_r',
                            showscale=True,
                            colorbar=dict(title='Gap Score')
                        ),
                        text=df['zip_code'],
                        name='Retail Comparison'
                    ),
                    row=2, col=1
                )
                
                # Add diagonal line (perfect prediction)
                max_val = max(df['retail_per_capita'].max(), df['predicted_retail_per_capita'].max())
                min_val = min(df['retail_per_capita'].min(), df['predicted_retail_per_capita'].min())
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='black', width=2, dash='dash'),
                        name='Perfect Prediction'
                    ),
                    row=2, col=1
                )
            
            # Plot 4: Retail Category Distribution
            # Check if we have retail categories
            retail_categories = [col for col in df.columns if col in [
                'food', 'general', 'clothing', 'electronics', 'furniture', 
                'health', 'auto', 'entertainment', 'specialty', 'other'
            ]]
            
            if retail_categories:
                # Calculate average per category
                category_avgs = {cat: df[cat].mean() for cat in retail_categories}
                
                fig.add_trace(
                    go.Bar(
                        x=list(category_avgs.keys()),
                        y=list(category_avgs.values()),
                        marker_color=self.categorical_palette[:len(category_avgs)],
                        name='Category Distribution'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text='Chicago Retail Gap Analysis Dashboard',
                height=800,
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text='ZIP Code', row=1, col=1)
            fig.update_yaxes(title_text='Retail Gap Score', row=1, col=1)
            
            fig.update_xaxes(title_text='ZIP Code', row=1, col=2)
            fig.update_yaxes(title_text='Opportunity Score (Abs)', row=1, col=2)
            
            fig.update_xaxes(title_text='Predicted Retail per Capita', row=2, col=1)
            fig.update_yaxes(title_text='Actual Retail per Capita', row=2, col=1)
            
            fig.update_xaxes(title_text='Retail Category', row=2, col=2)
            fig.update_yaxes(title_text='Average Establishments', row=2, col=2)
            
            # Save as HTML
            output_path = output_dir / "interactive_retail_gap_dashboard.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Saved interactive retail gap dashboard to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating interactive retail gap dashboard: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_retail_void_visualizations(self, model_results, void_data):
        """
        Generate visualizations for retail void model.
        
        Args:
            model_results (dict): Model results
            void_data (pd.DataFrame): Retail void data
            
        Returns:
            dict: Paths to generated visualizations
        """
        try:
            logger.info("Generating retail void visualizations")
            
            # Create output directory
            output_dir = self.output_dir / "retail_void"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize visualization paths
            viz_paths = {}
            
            # Generate leakage distribution
            if 'leakage_ratio' in void_data.columns:
                leakage_path = self._plot_leakage_distribution(void_data, output_dir)
                if leakage_path:
                    viz_paths['leakage_distribution'] = leakage_path
            
            # Generate void zones map
            if 'is_retail_void' in void_data.columns:
                void_path = self._plot_void_zones(void_data, output_dir)
                if void_path:
                    viz_paths['void_zones'] = void_path
            
            # Generate void zones by category
            if 'category_voids' in model_results:
                category_path = self._plot_void_zones_by_category(model_results['category_voids'], output_dir)
                if category_path:
                    viz_paths['category_voids'] = category_path
            
            # Generate spending leakage flow
            if 'leakage_patterns' in model_results:
                flow_path = self._plot_spending_leakage_flow(model_results['leakage_patterns'], void_data, output_dir)
                if flow_path:
                    viz_paths['leakage_flow'] = flow_path
            
            # Generate interactive retail void dashboard
            dashboard_path = self._create_interactive_retail_void_dashboard(void_data, model_results, output_dir)
            if dashboard_path:
                viz_paths['interactive_dashboard'] = dashboard_path
            
            logger.info(f"Generated {len(viz_paths)} retail void visualizations")
            return viz_paths
            
        except Exception as e:
            logger.error(f"Error generating retail void visualizations: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _plot_leakage_distribution(self, void_data, output_dir):
        """
        Plot spending leakage distribution.
        
        Args:
            void_data (pd.DataFrame): Retail void data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if 'leakage_ratio' not in void_data.columns:
                logger.warning("No leakage ratio data available")
                return None
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Plot 1: Histogram with KDE
            sns.histplot(void_data['leakage_ratio'], kde=True, ax=ax1, 
                        color=self.color_palette[4], bins=15)
            
            # Add vertical line for zero (no leakage)
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=2, 
                       label='No Leakage')
            
            # Add vertical line for mean
            mean_leakage = void_data['leakage_ratio'].mean()
            ax1.axvline(x=mean_leakage, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean Leakage ({mean_leakage:.2f})')
            
            # Customize first plot
            ax1.set_title('Distribution of Spending Leakage', fontsize=16)
            ax1.set_xlabel('Leakage Ratio (positive = spending leaving area)', fontsize=14)
            ax1.set_ylabel('Frequency', fontsize=14)
            ax1.legend(fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot 2: Leakage by ZIP code
            # Sort by leakage ratio
            sorted_data = void_data.sort_values('leakage_ratio', ascending=False)
            
            # Create color array based on leakage
            colors = ['red' if x > 0 else 'green' for x in sorted_data['leakage_ratio']]
            
            # Plot bars
            bars = ax2.bar(sorted_data['zip_code'].astype(str), sorted_data['leakage_ratio'],
                          color=colors)
            
            # Customize second plot
            ax2.set_title('Spending Leakage by ZIP Code', fontsize=16)
            ax2.set_xlabel('ZIP Code', fontsize=14)
            ax2.set_ylabel('Leakage Ratio', fontsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Rotate x-axis labels
            plt.sca(ax2)
            plt.xticks(rotation=90)
            
            # Add legend
            legend_elements = [
                Patch(facecolor='red', label='Spending Outflow'),
                Patch(facecolor='green', label='Spending Inflow')
            ]
            ax2.legend(handles=legend_elements, fontsize=12)
            
            # Add overall title
            fig.suptitle('Retail Spending Leakage Analysis', fontsize=20, y=1.05)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "leakage_distribution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved leakage distribution plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting leakage distribution: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_void_zones(self, void_data, output_dir):
        """
        Plot retail void zones.
        
        Args:
            void_data (pd.DataFrame): Retail void data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if 'is_retail_void' not in void_data.columns:
                logger.warning("No retail void data available")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Sort by ZIP code
            sorted_data = void_data.sort_values('zip_code')
            
            # Create color array based on void status
            colors = ['red' if x else 'blue' for x in sorted_data['is_retail_void']]
            
            # Plot bars
            bars = ax.bar(sorted_data['zip_code'].astype(str), sorted_data['is_retail_void'].astype(int),
                         color=colors)
            
            # Customize plot
            ax.set_title('Retail Void Zones by ZIP Code', fontsize=20, pad=20)
            ax.set_xlabel('ZIP Code', fontsize=16)
            ax.set_ylabel('Void Status (1 = Void Zone)', fontsize=16)
            
            # Rotate x-axis labels
            plt.xticks(rotation=90)
            
            # Set y-axis ticks
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['No Void', 'Void Zone'])
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add annotation explaining void zones
            ax.text(0.5, -0.15, 
                   "Void Zones represent areas with significant retail spending leakage and/or underdeveloped retail presence.",
                   ha='center', va='center', transform=ax.transAxes, fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "void_zones.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved void zones plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting void zones: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_void_zones_by_category(self, category_voids, output_dir):
        """
        Plot retail void zones by category.
        
        Args:
            category_voids (dict): Category voids by retail category
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not category_voids:
                logger.warning("No category voids available")
                return None
            
            # Count voids by category
            void_counts = {cat: len(voids) for cat, voids in category_voids.items()}
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Plot 1: Bar chart of void counts by category
            categories = list(void_counts.keys())
            counts = list(void_counts.values())
            
            bars = ax1.bar(categories, counts, color=self.categorical_palette[:len(categories)])
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f"{int(height)}", ha='center', va='bottom', fontsize=12)
            
            # Customize first plot
            ax1.set_title('Retail Category Voids by Type', fontsize=16)
            ax1.set_xlabel('Retail Category', fontsize=14)
            ax1.set_ylabel('Number of ZIP Codes with Voids', fontsize=14)
            ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Rotate x-axis labels
            plt.sca(ax1)
            plt.xticks(rotation=45, ha='right')
            
            # Plot 2: Heatmap of top void ZIP codes by category
            # Create a matrix of ZIP codes by category
            zip_category_matrix = {}
            
            # Get top 5 ZIP codes for each category
            for cat, voids in category_voids.items():
                if voids:
                    # Convert to DataFrame if needed
                    if isinstance(voids, list):
                        void_df = pd.DataFrame(voids)
                    else:
                        void_df = voids.copy()
                    
                    if 'zip_code' in void_df.columns:
                        # Get top 5 ZIP codes
                        top_zips = void_df['zip_code'].tolist()[:5]
                        
                        # Add to matrix
                        for zip_code in top_zips:
                            if zip_code not in zip_category_matrix:
                                zip_category_matrix[zip_code] = {c: 0 for c in categories}
                            zip_category_matrix[zip_code][cat] = 1
            
            # Convert matrix to DataFrame
            if zip_category_matrix:
                matrix_df = pd.DataFrame.from_dict(zip_category_matrix, orient='index')
                
                # Create heatmap
                sns.heatmap(matrix_df, cmap='YlOrRd', cbar=False, ax=ax2)
                
                # Customize second plot
                ax2.set_title('Top Void ZIP Codes by Category', fontsize=16)
                ax2.set_xlabel('Retail Category', fontsize=14)
                ax2.set_ylabel('ZIP Code', fontsize=14)
            else:
                ax2.text(0.5, 0.5, "No detailed void data available", 
                        ha='center', va='center', fontsize=14)
            
            # Add overall title
            fig.suptitle('Retail Category Void Analysis', fontsize=20, y=1.05)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            output_path = output_dir / "category_voids.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved category voids plot to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error plotting category voids: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _plot_spending_leakage_flow(self, leakage_patterns, void_data, output_dir):
        """
        Plot spending leakage flow.
        
        Args:
            leakage_patterns (dict): Leakage patterns
            void_data (pd.DataFrame): Retail void data
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            if not leakage_patterns:
                logger.warning("No leakage patterns available")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Check if we have high and low leakage ZIP codes
            high_leakage_zips = leakage_patterns.get('high_leakage_zips', [])
            low_leakage_zips = leakage_patterns.get('low_leakage_zips', [])
            
            if not high_leakage_zips and not low_leakage_zips:
                logger.warning("No high or low leakage ZIP codes available")
                return None
            
            # Filter void data to high and low leakage ZIP codes
            high_df = void_data[void_data['zip_code'].isin(high_leakage_zips)].copy() if high_leakage_zips else pd.DataFrame()
            low_df = void_data[void_data['zip_code'].isin(low_leakage_zips)].copy() if low_leakage_zips else pd.DataFrame()
            
            # Create a scatter plot of spending leakage
            if 'leakage_ratio' in void_data.columns and 'retail_per_capita' in void_data.columns:
                # Plot all ZIP codes
                ax.scatter(void_data['retail_per_capita'], void_data['leakage_ratio'],
                          color='gray', alpha=0.5, s=50, label='All ZIP Codes')
                
                # Plot high leakage ZIP codes
                if not high_df.empty:
                    ax.scatter(high_df['retail_per_capita'], high_df['leakage_ratio'],
                              color='red', s=100, label='High Leakage')
                
                # Plot low leakage ZIP codes
                if not low_df.empty:
                    ax.scatter(low_df['retail_per_capita'], low_df['leakage_ratio'],
                              color='green', s=100, label='Low Leakage (Attraction)')
                
                # Add ZIP code labels for high and low leakage
                for _, row in pd.concat([high_df, low_df]).iterrows():
                    ax.annotate(str(row['zip_code']),
                               (row['retail_per_capita'], row['leakage_ratio']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10)
                
                # Add horizontal line at zero (no leakage)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Add leakage statistics as text box
                stats_text = "\n".join([
                    f"Mean Leakage: {leakage_patterns.get('mean_leakage', 0):.3f}",
                    f"Median Leakage: {leakage_patterns.get('median_leakage', 0):.3f}",
                    f"Max Leakage: {leakage_patterns.get('max_leakage', 0):.3f}",
                    f"Min Leakage: {leakage_patterns.get('min_leakage', 0):.3f}",
                    f"Std Dev: {leakage_patterns.get('std_leakage', 0):.3f}"
                ])
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Customize plot
                ax.set_title('Retail Spending Leakage Flow', fontsize=20, pad=20)
                ax.set_xlabel('Retail Establishments per Capita', fontsize=16)
                ax.set_ylabel('Leakage Ratio (positive = spending leaving area)', fontsize=16)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend
                ax.legend(fontsize=12, loc='lower right')
                
                # Add annotation explaining the plot
                ax.text(0.5, -0.15, 
                       "This plot shows the relationship between retail density and spending leakage. " +
                       "ZIP codes with high leakage (red) represent areas where spending is flowing out, " +
                       "while areas with low leakage (green) are attracting spending from other areas.",
                       ha='center', va='center', transform=ax.transAxes, fontsize=14,
                       bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
                
                # Tight layout
                plt.tight_layout()
                
                # Save figure
                output_path = output_dir / "spending_leakage_flow.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved spending leakage flow plot to {output_path}")
                return str(output_path)
            else:
                logger.warning("Missing required columns for spending leakage flow plot")
                return None
            
        except Exception as e:
            logger.error(f"Error plotting spending leakage flow: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_interactive_retail_void_dashboard(self, void_data, model_results, output_dir):
        """
        Create interactive retail void dashboard.
        
        Args:
            void_data (pd.DataFrame): Retail void data
            model_results (dict): Model results
            output_dir (Path): Output directory
            
        Returns:
            str: Path to saved visualization
        """
        try:
            # Create a copy of the data
            df = void_data.copy()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Spending Leakage by ZIP Code',
                    'Retail Void Zones',
                    'Leakage vs. Retail Density',
                    'Category Void Analysis'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot 1: Spending Leakage by ZIP Code
            if 'leakage_ratio' in df.columns:
                # Sort by leakage ratio
                sorted_df = df.sort_values('leakage_ratio', ascending=False)
                
                # Create color array based on leakage
                colors = ['red' if x > 0 else 'green' for x in sorted_df['leakage_ratio']]
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_df['zip_code'].astype(str),
                        y=sorted_df['leakage_ratio'],
                        marker_color=colors,
                        name='Leakage Ratio'
                    ),
                    row=1, col=1
                )
                
                # Add zero line
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=0,
                    line=dict(color="black", width=2),
                    xref="paper", yref="y",
                    row=1, col=1
                )
            
            # Plot 2: Retail Void Zones
            if 'is_retail_void' in df.columns:
                # Sort by ZIP code
                sorted_df = df.sort_values('zip_code')
                
                # Create color array based on void status
                colors = ['red' if x else 'blue' for x in sorted_df['is_retail_void']]
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_df['zip_code'].astype(str),
                        y=sorted_df['is_retail_void'].astype(int),
                        marker_color=colors,
                        name='Void Status'
                    ),
                    row=1, col=2
                )
            
            # Plot 3: Leakage vs. Retail Density
            if 'leakage_ratio' in df.columns and 'retail_per_capita' in df.columns:
                # Check if we have void cluster information
                color_data = df['void_cluster'] if 'void_cluster' in df.columns else df['leakage_ratio']
                
                fig.add_trace(
                    go.Scatter(
                        x=df['retail_per_capita'],
                        y=df['leakage_ratio'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=color_data,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title='Cluster' if 'void_cluster' in df.columns else 'Leakage')
                        ),
                        text=df['zip_code'],
                        name='Leakage vs. Density'
                    ),
                    row=2, col=1
                )
                
                # Add zero line
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=0,
                    line=dict(color="black", width=2),
                    xref="paper", yref="y",
                    row=2, col=1
                )
            
            # Plot 4: Category Void Analysis
            if 'category_voids' in model_results:
                # Count voids by category
                void_counts = {cat: len(voids) for cat, voids in model_results['category_voids'].items()}
                
                if void_counts:
                    categories = list(void_counts.keys())
                    counts = list(void_counts.values())
                    
                    fig.add_trace(
                        go.Bar(
                            x=categories,
                            y=counts,
                            marker_color='purple',
                            name='Category Voids'
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title_text='Chicago Retail Void Analysis Dashboard',
                height=800,
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text='ZIP Code', row=1, col=1)
            fig.update_yaxes(title_text='Leakage Ratio', row=1, col=1)
            
            fig.update_xaxes(title_text='ZIP Code', row=1, col=2)
            fig.update_yaxes(title_text='Void Status (1=Void)', row=1, col=2)
            
            fig.update_xaxes(title_text='Retail per Capita', row=2, col=1)
            fig.update_yaxes(title_text='Leakage Ratio', row=2, col=1)
            
            fig.update_xaxes(title_text='Retail Category', row=2, col=2)
            fig.update_yaxes(title_text='Number of Void ZIP Codes', row=2, col=2)
            
            # Save as HTML
            output_path = output_dir / "interactive_retail_void_dashboard.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Saved interactive retail void dashboard to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating interactive retail void dashboard: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_combined_dashboard(self, population_data, retail_gap_data, retail_void_data, output_dir=None):
        """
        Generate a combined interactive dashboard for all models.
        
        Args:
            population_data (pd.DataFrame): Population forecast data
            retail_gap_data (pd.DataFrame): Retail gap data
            retail_void_data (pd.DataFrame): Retail void data
            output_dir (Path, optional): Output directory
            
        Returns:
            str: Path to saved dashboard
        """
        try:
            logger.info("Generating combined interactive dashboard")
            
            # Set output directory
            if output_dir is None:
                output_dir = self.output_dir
            
            # Create figure with tabs
            fig = go.Figure()
            
            # Tab 1: Population Forecast
            if population_data is not None:
                # Group by year and forecast type
                trend_data = population_data.groupby(['year', 'forecast_type'])['population'].sum().reset_index()
                
                # Historical data
                historical = trend_data[trend_data['forecast_type'] == 'historical']
                
                # Forecast data
                forecast = trend_data[trend_data['forecast_type'] == 'combined']
                
                # Create trace
                fig.add_trace(
                    go.Scatter(
                        x=historical['year'], 
                        y=historical['population'],
                        mode='lines+markers',
                        name='Historical Population',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8),
                        visible=True
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast['year'], 
                        y=forecast['population'],
                        mode='lines+markers',
                        name='Forecast Population',
                        line=dict(color='red', width=3, dash='dash'),
                        marker=dict(size=8),
                        visible=True
                    )
                )
            
            # Tab 2: Retail Gap Analysis
            if retail_gap_data is not None and 'retail_gap_score' in retail_gap_data.columns:
                # Sort by gap score
                sorted_gap = retail_gap_data.sort_values('retail_gap_score')
                
                # Create color array based on thresholds
                colors = ['green' if x < -0.5 else 'red' if x > 0.5 else 'blue' 
                         for x in sorted_gap['retail_gap_score']]
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_gap['zip_code'].astype(str),
                        y=sorted_gap['retail_gap_score'],
                        marker_color=colors,
                        name='Retail Gap Score',
                        visible=False
                    )
                )
            
            # Tab 3: Retail Void Analysis
            if retail_void_data is not None and 'leakage_ratio' in retail_void_data.columns:
                # Sort by leakage ratio
                sorted_void = retail_void_data.sort_values('leakage_ratio', ascending=False)
                
                # Create color array based on leakage
                colors = ['red' if x > 0 else 'green' for x in sorted_void['leakage_ratio']]
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_void['zip_code'].astype(str),
                        y=sorted_void['leakage_ratio'],
                        marker_color=colors,
                        name='Spending Leakage',
                        visible=False
                    )
                )
            
            # Create buttons for tabs
            buttons = []
            
            # Population Forecast tab
            buttons.append(
                dict(
                    label="Population Forecast",
                    method="update",
                    args=[
                        {"visible": [True, True, False, False]},
                        {"title": "Chicago Population Forecast",
                         "xaxis": {"title": "Year"},
                         "yaxis": {"title": "Population"}}
                    ]
                )
            )
            
            # Retail Gap tab
            buttons.append(
                dict(
                    label="Retail Gap Analysis",
                    method="update",
                    args=[
                        {"visible": [False, False, True, False]},
                        {"title": "Chicago Retail Gap Analysis",
                         "xaxis": {"title": "ZIP Code"},
                         "yaxis": {"title": "Retail Gap Score"}}
                    ]
                )
            )
            
            # Retail Void tab
            buttons.append(
                dict(
                    label="Retail Void Analysis",
                    method="update",
                    args=[
                        {"visible": [False, False, False, True]},
                        {"title": "Chicago Retail Void Analysis",
                         "xaxis": {"title": "ZIP Code"},
                         "yaxis": {"title": "Leakage Ratio"}}
                    ]
                )
            )
            
            # Update layout with buttons
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        active=0,
                        x=0.5,
                        y=1.15,
                        buttons=buttons
                    )
                ]
            )
            
            # Set initial title and layout
            fig.update_layout(
                title_text='Chicago Housing Pipeline & Population Shift Analysis',
                height=600,
                width=1000,
                xaxis_title="Year",
                yaxis_title="Population"
            )
            
            # Save as HTML
            output_path = output_dir / "combined_dashboard.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Saved combined interactive dashboard to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating combined dashboard: {str(e)}")
            logger.error(traceback.format_exc())
            return None
