import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import geopandas as gpd
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ChicagoDataVisualizer:
    def __init__(self):
        """Initialize the visualizer with data paths"""
        # Set up paths
        self.output_dir = Path(os.getenv('OUTPUT_DIR', 'output'))
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.model_dir = self.output_dir / 'models'
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Initialize data attributes
        self.merged_data = None
        self.predictions = None
        self.feature_importance = None
        self.zoning_data = None
        
        # Set the style for matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load all necessary datasets for visualization"""
        try:
            # Load the merged dataset
            merged_file = self.output_dir / 'merged_dataset.csv'
            if merged_file.exists():
                self.merged_data = pd.read_csv(merged_file)
                logger.info(f"Loaded merged dataset: {len(self.merged_data)} records")
            else:
                logger.warning(f"Merged dataset not found: {merged_file}")
                self.merged_data = pd.DataFrame()
            
            # Load predictions if available
            predictions_file = self.output_dir / 'scenario_predictions.csv'
            if predictions_file.exists():
                self.predictions = pd.read_csv(predictions_file)
                logger.info(f"Loaded predictions: {len(self.predictions)} records")
            else:
                logger.warning(f"Predictions not found: {predictions_file}")
                self.predictions = None
            
            # Load feature importance if available
            importance_file = self.output_dir / 'feature_importance.csv'
            if importance_file.exists():
                self.feature_importance = pd.read_csv(importance_file)
                logger.info(f"Loaded feature importance: {len(self.feature_importance)} records")
            else:
                logger.warning(f"Feature importance not found: {importance_file}")
                self.feature_importance = None
                
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.merged_data = pd.DataFrame()
            return False

    def plot_population_trends(self):
        """Plot historical population trends by ZIP code"""
        if self.merged_data is None:
            logger.error("Cannot create population trends plot: No data available")
            return False
            
        try:
            # Check if required columns exist
            required_cols = ['year', 'zip_code', 'population']
            for col in required_cols:
                if col not in self.merged_data.columns:
                    available_cols = self.merged_data.columns.tolist()
                    logger.error(f"Cannot create population trends: missing column '{col}'")
                    logger.error(f"Available columns: {available_cols}")
                    return False
            
            # Get top 10 ZIP codes by population
            top_zips = self.merged_data.groupby('zip_code')['population'].mean().nlargest(10).index
            
            # Filter data for these ZIP codes
            pop_data = self.merged_data[self.merged_data['zip_code'].isin(top_zips)]
            
            # Create a line plot of population over time for top ZIP codes
            plt.figure(figsize=(12, 8))
            for zip_code, group in pop_data.groupby('zip_code'):
                group = group.sort_values('year')
                plt.plot(group['year'], group['population'], marker='o', label=f'ZIP {zip_code}')
            
            plt.title('Population Trends for Top 10 Chicago ZIP Codes', fontsize=15)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Population', fontsize=12)
            plt.legend(title='ZIP Code')
            plt.grid(True)
            
            # Save the figure
            output_path = self.viz_dir / 'population_trends.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Population trends plot saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating population trends plot: {str(e)}")
            return False

    def plot_permit_analysis(self):
        """Create visualizations related to building permits"""
        if self.merged_data is None:
            logger.error("Cannot create permit analysis: No data available")
            return False
            
        try:
            # Check if required columns exist
            if 'permit_count' not in self.merged_data.columns:
                logger.error("Cannot create permit analysis: missing 'permit_count' column")
                return False
                
            if 'year' not in self.merged_data.columns:
                logger.error("Cannot create permit analysis: missing 'year' column")
                return False
            
            # 1. Permits by year
            plt.figure(figsize=(10, 6))
            annual_permits = self.merged_data.groupby('year')['permit_count'].sum().reset_index()
            sns.barplot(x='year', y='permit_count', data=annual_permits)
            plt.title('Building Permits Issued by Year', fontsize=15)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Number of Permits', fontsize=12)
            plt.xticks(rotation=45)
            
            # Save the figure
            output_path = self.viz_dir / 'permits_by_year.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Permits (t) vs. Population Change (t+1)
            if 'population_change_next_year' in self.merged_data.columns:
                plt.figure(figsize=(10, 8))
                
                # Remove outliers for better visualization
                filtered_data = self.merged_data.copy()
                filtered_data = filtered_data[
                    (filtered_data['population_change_next_year'] > -0.5) & 
                    (filtered_data['population_change_next_year'] < 0.5)
                ]
                
                sns.scatterplot(
                    x='permit_count', 
                    y='population_change_next_year',
                    hue='year',
                    palette='viridis',
                    data=filtered_data
                )
                
                plt.title('Relationship Between Building Permits (t) and Future Population Change (t+1)', fontsize=15)
                plt.xlabel('Number of Permits', fontsize=12)
                plt.ylabel('Future Population Change (t+1) (%)', fontsize=12)
                
                # Add trend line
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        filtered_data['permit_count'], 
                        filtered_data['population_change_next_year']
                    )
                    plt.plot(
                        filtered_data['permit_count'], 
                        intercept + slope * filtered_data['permit_count'], 
                        'r', 
                        label=f'Trend Line (R²: {r_value**2:.3f})'
                    )
                    plt.legend()
                except Exception as e:
                    logger.warning(f"Could not add trend line: {str(e)}")
                
                # Save the figure
                output_path = self.viz_dir / 'permits_vs_future_population_change.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. NEW: Add heatmap showing permit growth leading to population growth
                plt.figure(figsize=(12, 8))
                
                # Group by ZIP code and year, then calculate the correlation
                zip_correlations = []
                for zip_code, group in filtered_data.groupby('zip_code'):
                    if len(group) > 3:  # Need enough data points for correlation
                        corr = group['permit_count'].corr(group['population_change_next_year'])
                        zip_correlations.append({
                            'zip_code': zip_code,
                            'correlation': corr,
                            'avg_permits': group['permit_count'].mean(),
                            'avg_population_growth': group['population_change_next_year'].mean()
                        })
                
                if zip_correlations:
                    corr_df = pd.DataFrame(zip_correlations)
                    
                    # Create heatmap
                    pivot_df = corr_df.pivot_table(
                        index=pd.qcut(corr_df['avg_permits'], 4, labels=["Low", "Medium-Low", "Medium-High", "High"]),
                        columns=pd.qcut(corr_df['avg_population_growth'], 4, labels=["Low", "Medium-Low", "Medium-High", "High"]),
                        values='correlation',
                        aggfunc='mean'
                    )
                    
                    sns.heatmap(pivot_df, annot=True, cmap='RdYlBu', center=0)
                    plt.title('Correlation between Permits and Future Population Growth by ZIP Category', fontsize=15)
                    plt.xlabel('Average Population Growth Category', fontsize=12)
                    plt.ylabel('Average Permit Activity Category', fontsize=12)
                    
                    # Save the figure
                    output_path = self.viz_dir / 'permit_population_growth_heatmap.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"Permit analysis plots saved to {self.viz_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating permit analysis plots: {str(e)}")
            return False

    def plot_income_distribution(self):
        """Visualize income distribution and changes over time"""
        if self.merged_data is None:
            logger.error("Cannot create income distribution plot: No data available")
            return False
            
        try:
            # Check for income distribution columns
            income_cols = [col for col in self.merged_data.columns if 'income' in col.lower()]
            if not income_cols:
                logger.error("Cannot create income distribution plot: No income columns found")
                return False
                
            # Check specifically for the required columns
            required_cols = ['middle_class_pct', 'lower_income_pct', 'year']
            missing_cols = [col for col in required_cols if col not in self.merged_data.columns]
            if missing_cols:
                logger.error(f"Cannot create income distribution plot: Missing columns {missing_cols}")
                return False
            
            # 1. Plot changes in income distribution over time
            yearly_income = self.merged_data.groupby('year')[['middle_class_pct', 'lower_income_pct']].mean().reset_index()
            
            plt.figure(figsize=(12, 6))
            plt.plot(yearly_income['year'], yearly_income['middle_class_pct'], marker='o', label='Middle Class %')
            plt.plot(yearly_income['year'], yearly_income['lower_income_pct'], marker='s', label='Lower Income %')
            
            plt.title('Changes in Income Distribution Over Time', fontsize=15)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Percentage of Population', fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            output_path = self.viz_dir / 'income_distribution_trends.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. If median_household_income exists, plot it too
            if 'median_household_income' in self.merged_data.columns:
                yearly_median_income = self.merged_data.groupby('year')['median_household_income'].mean().reset_index()
                
                plt.figure(figsize=(12, 6))
                plt.plot(yearly_median_income['year'], yearly_median_income['median_household_income'], marker='o', color='darkgreen')
                
                plt.title('Changes in Median Household Income Over Time', fontsize=15)
                plt.xlabel('Year', fontsize=12)
                plt.ylabel('Median Household Income ($)', fontsize=12)
                plt.grid(True)
                
                # Add thousands separator to y-axis
                plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
                
                # Save the figure
                output_path = self.viz_dir / 'median_income_trends.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Income distribution plots saved to {self.viz_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating income distribution plots: {str(e)}")
            return False

    def plot_scenario_predictions(self):
        """Visualize predictions across different scenarios"""
        if self.predictions is None:
            logger.error("Cannot create scenario predictions plot: No predictions available")
            return False
            
        try:
            # Check required columns with updated format
            required_cols = ['scenario', 'predicted_population_change']
            
            # Additional columns in the enhanced format
            enhanced_format = all(col in self.predictions.columns 
                                for col in ['zip_code', 'prediction_year', 'year', 'prediction_window'])
            
            if enhanced_format:
                logger.info("Using enhanced scenario predictions format with prediction window")
            else:
                # Check for basic required columns
                missing_cols = [col for col in required_cols if col not in self.predictions.columns]
                if missing_cols:
                    logger.error(f"Cannot create scenario predictions plot: missing columns {missing_cols}")
                    logger.error(f"Available columns: {self.predictions.columns.tolist()}")
                    return False
                
                # Convert simple format to enhanced format if needed
                if 'zip_code' not in self.predictions.columns:
                    logger.warning("Converting simple format predictions to enhanced format")
                    scenarios = self.predictions.columns.tolist()
                    self.predictions = pd.melt(
                        self.predictions.reset_index(), 
                        id_vars=['index'], 
                        value_vars=scenarios,
                        var_name='scenario',
                        value_name='predicted_population_change'
                    )
                    self.predictions = self.predictions.drop(columns=['index'])
                    self.predictions['prediction_window'] = '1-Year'
            
            # 1. Distribution of predicted population changes by scenario
            plt.figure(figsize=(12, 8))
            
            # Create a violin plot for each scenario
            if enhanced_format and len(self.predictions['prediction_window'].unique()) > 1:
                # Create a separate violin plot for each prediction window
                prediction_windows = self.predictions['prediction_window'].unique()
                for i, window in enumerate(prediction_windows):
                    plt.subplot(1, len(prediction_windows), i+1)
                    
                    window_data = self.predictions[self.predictions['prediction_window'] == window]
                    sns.violinplot(
                        x='scenario', 
                        y='predicted_population_change',
                        data=window_data,
                        palette='Set2'
                    )
                    
                    plt.title(f'{window} Population Change', fontsize=15)
                    plt.xlabel('Scenario', fontsize=12)
                    plt.ylabel('Predicted Population Change (%)', fontsize=12)
                    plt.grid(True, axis='y')
                
                plt.suptitle('Distribution of Predicted Population Changes by Scenario and Time Horizon', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                # Create a single violin plot for all scenarios
                sns.violinplot(
                    x='scenario', 
                    y='predicted_population_change',
                    data=self.predictions,
                    palette='Set2'
                )
                
                if enhanced_format:
                    window = self.predictions['prediction_window'].iloc[0]
                    plt.title(f'Distribution of {window} Predicted Population Changes by Scenario', fontsize=15)
                else:
                    plt.title('Distribution of Predicted Population Changes by Scenario', fontsize=15)
                    
                plt.xlabel('Scenario', fontsize=12)
                plt.ylabel('Predicted Population Change (%)', fontsize=12)
                plt.grid(True, axis='y')
            
            # Save the figure
            output_path = self.viz_dir / 'scenario_predictions_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Top ZIP codes by growth for each scenario
            if enhanced_format:
                # For each prediction window, create a plot of top growing ZIPs
                prediction_windows = self.predictions['prediction_window'].unique()
                for window in prediction_windows:
                    window_data = self.predictions[self.predictions['prediction_window'] == window]
                    
                    # Create a separate figure for each scenario
                    for scenario in window_data['scenario'].unique():
                        scenario_data = window_data[window_data['scenario'] == scenario]
                        
                        # Get top 10 growing ZIPs for this scenario
                        top_growing_zips = scenario_data.groupby('zip_code')['predicted_population_change'].mean().nlargest(10)
                        
                        plt.figure(figsize=(12, 6))
                        top_growing_zips.plot(kind='bar', color='green')
                        
                        plt.title(f'Top 10 ZIP Codes with Highest {window} Predicted Population Growth ({scenario.title()} Scenario)', fontsize=15)
                        plt.xlabel('ZIP Code', fontsize=12)
                        plt.ylabel('Predicted Population Change (%)', fontsize=12)
                        plt.grid(True, axis='y')
                        
                        # Format y-axis as percentage
                        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
                        
                        # Save the figure
                        output_path = self.viz_dir / f'top_growing_zip_codes_{scenario}_{window.replace("-", "")}.png'
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        plt.close()
                
                # 3. Create a map or chart showing future population change by region for the optimistic scenario
                for window in prediction_windows:
                    window_data = self.predictions[self.predictions['prediction_window'] == window]
                    optimistic_data = window_data[window_data['scenario'] == 'optimistic']
                    
                    # Create a comparative bar chart for all scenarios for top 5 ZIPs
                    top_zips = optimistic_data.groupby('zip_code')['predicted_population_change'].mean().nlargest(5).index
                    top_zips_data = window_data[window_data['zip_code'].isin(top_zips)]
                    
                    plt.figure(figsize=(14, 8))
                    
                    # Create grouped bar chart
                    sns.barplot(
                        x='zip_code', 
                        y='predicted_population_change',
                        hue='scenario',
                        data=top_zips_data,
                        palette='Set2'
                    )
                    
                    plt.title(f'{window} Scenario Comparison for Top 5 Growing ZIP Codes', fontsize=15)
                    plt.xlabel('ZIP Code', fontsize=12)
                    plt.ylabel('Predicted Population Change (%)', fontsize=12)
                    plt.grid(True, axis='y')
                    plt.legend(title='Scenario')
                    
                    # Format y-axis as percentage
                    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
                    
                    # Save the figure
                    output_path = self.viz_dir / f'top_zips_scenario_comparison_{window.replace("-", "")}.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Create a heatmap of population changes
                    if 'year' in optimistic_data.columns:
                        prediction_year = optimistic_data['prediction_year'].iloc[0]
                        
                        plt.figure(figsize=(14, 10))
                        
                        # Pivot data for heatmap (top 20 ZIPs by absolute predicted change)
                        top20_zips = optimistic_data.groupby('zip_code')['predicted_population_change'].mean().abs().nlargest(20).index
                        filtered_data = window_data[window_data['zip_code'].isin(top20_zips)]
                        
                        pivot_data = filtered_data.pivot_table(
                            values='predicted_population_change',
                            index='zip_code',
                            columns='scenario',
                            aggfunc='mean'
                        )
                        
                        # Sort by optimistic scenario values
                        pivot_data = pivot_data.sort_values('optimistic', ascending=False)
                        
                        # Create heatmap
                        sns.heatmap(
                            pivot_data,
                            cmap='RdYlGn',
                            center=0,
                            annot=True,
                            fmt='.1%',
                            linewidths=.5
                        )
                        
                        plt.title(f'Predicted {window} Population Change by ZIP Code and Scenario (Year {prediction_year})', fontsize=15)
                        plt.ylabel('ZIP Code', fontsize=12)
                        plt.xlabel('Scenario', fontsize=12)
                        
                        # Save the figure
                        output_path = self.viz_dir / f'population_change_heatmap_{window.replace("-", "")}.png'
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        plt.close()
            else:
                # Legacy format handling - similar to original implementation
                if isinstance(self.predictions, pd.DataFrame) and 'scenario' in self.predictions.columns:
                    optimistic_scenario = self.predictions[self.predictions['scenario'] == 'optimistic']
                    top_growing_zips = optimistic_scenario.groupby('zip_code')['predicted_population_change'].mean().nlargest(10)
                    
                    plt.figure(figsize=(12, 6))
                    top_growing_zips.plot(kind='bar', color='green')
                    
                    plt.title('Top 10 ZIP Codes with Highest Predicted Population Growth (Optimistic Scenario)', fontsize=15)
                    plt.xlabel('ZIP Code', fontsize=12)
                    plt.ylabel('Predicted Population Change (%)', fontsize=12)
                    plt.grid(True, axis='y')
                    
                    # Format y-axis as percentage
                    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
                    
                    # Save the figure
                    output_path = self.viz_dir / 'top_growing_zip_codes.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # 3. Scenario comparison for selected zip codes
                    top_5_zips = top_growing_zips.index[:5]  # Top 5 growing zip codes
                    
                    # Filter predictions for these zip codes
                    selected_zips_predictions = self.predictions[self.predictions['zip_code'].isin(top_5_zips)]
                    
                    plt.figure(figsize=(14, 8))
                    
                    # Create grouped bar chart
                    sns.barplot(
                        x='zip_code', 
                        y='predicted_population_change',
                        hue='scenario',
                        data=selected_zips_predictions,
                        palette='Set2'
                    )
                    
                    plt.title('Scenario Comparison for Top 5 Growing ZIP Codes', fontsize=15)
                    plt.xlabel('ZIP Code', fontsize=12)
                    plt.ylabel('Predicted Population Change (%)', fontsize=12)
                    plt.grid(True, axis='y')
                    plt.legend(title='Scenario')
                    
                    # Format y-axis as percentage
                    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
                    
                    # Save the figure
                    output_path = self.viz_dir / 'top_zips_scenario_comparison.png'
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"Scenario prediction plots saved to {self.viz_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating scenario prediction plots: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def plot_feature_importance(self):
        """Visualize feature importance from the model"""
        if self.feature_importance is None:
            logger.error("Cannot create feature importance plot: No feature importance data available")
            return False
            
        try:
            # Check if required columns exist
            if 'feature' not in self.feature_importance.columns:
                logger.error("Cannot create feature importance plot: missing 'feature' column")
                return False
                
            if 'importance' not in self.feature_importance.columns:
                logger.error("Cannot create feature importance plot: missing 'importance' column")
                return False
            
            # Sort features by importance
            sorted_features = self.feature_importance.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar chart
            sns.barplot(
                x='importance', 
                y='feature',
                data=sorted_features,
                palette='viridis'
            )
            
            plt.title('Feature Importance in Population Change Prediction Model', fontsize=15)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            # Save the figure
            output_path = self.viz_dir / 'feature_importance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return False

    def create_interactive_map(self):
        """Create an interactive map of Chicago with population predictions"""
        try:
            # Check if we have predictions and zoning data
            if self.predictions is None:
                logger.error("Cannot create interactive map: No predictions available")
                return False
                
            if self.zoning_data is None:
                logger.error("Cannot create interactive map: No zoning data available")
                return False
            
            # For this example, we'll use a simpler approach with just a basic map
            # Create a dataframe with zip code level predictions
            zip_predictions = self.predictions.groupby(['zip_code', 'scenario'])['predicted_population_change'].mean().reset_index()
            
            # Convert to wide format for easier plotting
            zip_wide = zip_predictions.pivot(index='zip_code', columns='scenario', values='predicted_population_change').reset_index()
            zip_wide.columns.name = None  # Remove the columns name
            
            # Create an interactive bar chart with plotly
            fig = px.bar(
                zip_predictions,
                x='zip_code',
                y='predicted_population_change',
                color='scenario',
                barmode='group',
                title='Predicted Population Change by ZIP Code and Scenario',
                labels={
                    'zip_code': 'ZIP Code',
                    'predicted_population_change': 'Predicted Population Change (%)',
                    'scenario': 'Scenario'
                },
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='ZIP Code',
                yaxis_title='Predicted Population Change (%)',
                legend_title='Scenario',
                xaxis={'categoryorder': 'total descending'},
                yaxis_tickformat='.1%'
            )
            
            # Save as HTML file
            output_path = self.viz_dir / 'interactive_predictions.html'
            fig.write_html(str(output_path))
            
            # Also create a fixed PNG for reports
            output_path_png = self.viz_dir / 'predicted_changes_by_zipcode.png'
            fig.write_image(str(output_path_png), width=1200, height=800)
            
            logger.info(f"Interactive map saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating interactive map: {str(e)}")
            return False

    def plot_population_shift_analysis(self):
        """Create visualizations specifically focused on population shifts"""
        # Population change by ZIP code over time
        plt.figure(figsize=(15, 8))
        pivot_df = self.merged_data.pivot(index='year', columns='zip_code', values='population_change')
        sns.heatmap(pivot_df, cmap='RdYlBu', center=0)
        plt.title('Population Change Intensity by ZIP Code (2013-2023)')
        plt.ylabel('Year')
        plt.xlabel('ZIP Code')
        plt.savefig(self.viz_dir / 'population_shift_heatmap.png')
        plt.close()

        # Income distribution changes
        plt.figure(figsize=(15, 8))
        income_metrics = ['middle_class_pct', 'lower_income_pct']
        for metric in income_metrics:
            if metric in self.merged_data.columns:
                self.merged_data.groupby('year')[metric].mean().plot(label=metric.replace('_', ' ').title())
        plt.title('Income Distribution Changes Over Time')
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.legend()
        plt.savefig(self.viz_dir / 'income_distribution_changes.png')
        plt.close()

        # Building activity vs population change
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.merged_data, 
                       x='permit_count', 
                       y='population_change',
                       hue='zip_code',
                       alpha=0.6)
        plt.title('Building Permits Impact on Population Change')
        plt.xlabel('Number of Building Permits')
        plt.ylabel('Population Change (%)')
        plt.savefig(self.viz_dir / 'permits_population_impact.png')
        plt.close()

    def plot_economic_impact(self):
        """Visualize economic factors' impact on population shifts"""
        # Mortgage rates vs building activity
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Permits per ZIP', color='tab:blue')
        yearly_permits = self.merged_data.groupby('year')['permit_count'].mean()
        ax1.bar(yearly_permits.index, yearly_permits.values, color='tab:blue', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('30-Year Mortgage Rate (%)', color='tab:red')
        yearly_rates = self.merged_data.groupby('year')['mortgage_30y'].mean()
        ax2.plot(yearly_rates.index, yearly_rates.values, color='tab:red', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('Building Activity vs Mortgage Rates')
        plt.savefig(self.viz_dir / 'economic_impact_building.png')
        plt.close()

    def generate_all_visualizations(self):
        """Create all visualizations"""
        logger.info("Creating all visualizations...")
        
        # Original visualizations
        self.plot_population_trends()
        self.plot_permit_analysis()
        self.plot_income_distribution()
        self.plot_scenario_predictions()
        self.plot_feature_importance()
        self.create_interactive_map()
        
        # New targeted visualizations
        self.plot_population_shift_analysis()
        self.plot_economic_impact()
        
        logger.info(f"All visualizations saved to {self.viz_dir}")
        return True

def main():
    visualizer = ChicagoDataVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main() 