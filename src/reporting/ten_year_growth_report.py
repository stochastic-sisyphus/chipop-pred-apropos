"""
Ten Year Growth Report generator for the Chicago Population Analysis project.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.reporting.base_report import BaseReport
from src.utils.helpers import format_currency, format_percent, calculate_growth_rate

logger = logging.getLogger(__name__)

class TenYearGrowthReport(BaseReport):
    """Generates reports analyzing ten-year growth projections for Chicago ZIP codes."""
    
    def __init__(self, output_dir=None, template_dir=None):
        """
        Initialize the Ten Year Growth Report generator.
        
        Args:
            output_dir (Path, optional): Directory to save report outputs
            template_dir (Path, optional): Directory containing report templates
        """
        super().__init__("Ten Year Growth Projection", output_dir, template_dir)
    
    def _filter_valid_zips(self, df):
        """
        Filter dataframe to include only valid Chicago ZIP codes with sufficient data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        try:
            from src.config import settings
            
            # Ensure ZIP code is string type
            df['zip_code'] = df['zip_code'].astype(str)
            
            # Filter to Chicago ZIP codes
            chicago_zips = settings.CHICAGO_ZIP_CODES
            filtered_df = df[df['zip_code'].isin(chicago_zips)]
            
            if len(filtered_df) == 0:
                logger.error("No valid Chicago ZIP codes found in data")
                return None
            
            # Check for required columns
            required_cols = ['population', 'housing_units', 'retail_businesses']
            missing_cols = [col for col in required_cols if col not in filtered_df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None
            
            # Filter out rows with missing values in required columns
            for col in required_cols:
                filtered_df = filtered_df[filtered_df[col].notna()]
            
            if len(filtered_df) == 0:
                logger.error("No rows with complete data found")
                return None
            
            logger.info(f"Filtered to {len(filtered_df)} Chicago ZIP codes with complete data")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering valid ZIP codes: {str(e)}")
            return None
    
    def _flag_insufficient_data(self, df):
        """
        Flag rows with insufficient data for growth projection analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with data_status column
        """
        try:
            # Create a copy to avoid modifying the original
            result = df.copy()
            
            # Define minimum thresholds
            min_population = 1000  # Minimum population
            min_housing = 100      # Minimum housing units
            min_retail = 5         # Minimum retail businesses
            
            # Flag rows with insufficient data
            result['data_status'] = 'ok'
            
            # Check population
            if 'population' in result.columns:
                result.loc[result['population'] < min_population, 'data_status'] = 'insufficient'
            
            # Check housing units
            if 'housing_units' in result.columns:
                result.loc[result['housing_units'] < min_housing, 'data_status'] = 'insufficient'
            
            # Check retail businesses
            if 'retail_businesses' in result.columns:
                result.loc[result['retail_businesses'] < min_retail, 'data_status'] = 'insufficient'
            
            # Count flagged rows
            insufficient_count = (result['data_status'] == 'insufficient').sum()
            logger.info(f"Flagged {insufficient_count} rows with insufficient data")
            
            return result
            
        except Exception as e:
            logger.error(f"Error flagging insufficient data: {str(e)}")
            return df
    
    def _prepare_report_context(self):
        """
        Prepare context dictionary for report template.
        """
        try:
            # Call parent method to initialize context
            super()._prepare_report_context()
            
            # Calculate growth projections
            self._calculate_growth_projections()
            
            # Create visualizations
            self._create_visualizations()
            
            # Prepare ZIP code details
            self._prepare_zip_details()
            
            logger.info("Report context prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing report context: {str(e)}")
    
    def _calculate_growth_projections(self):
        """
        Calculate ten-year growth projections.
        """
        try:
            # Define growth rates based on current metrics
            # These are simplified models for demonstration
            
            # Population growth rate
            # Base rate plus adjustments for economic factors
            base_pop_growth = 0.005  # 0.5% annual growth
            
            if 'median_income' in self.data.columns:
                # Higher income areas grow faster
                income_factor = (self.data['median_income'] / self.data['median_income'].median() - 1) * 0.002
            else:
                income_factor = 0
                
            if 'employment_rate' in self.data.columns:
                # Higher employment areas grow faster
                employment_factor = (self.data['employment_rate'] / self.data['employment_rate'].median() - 1) * 0.003
            else:
                employment_factor = 0
            
            # Calculate population growth rate
            self.data['population_growth_rate'] = (
                base_pop_growth + income_factor + employment_factor
            ).clip(lower=-0.01, upper=0.03).round(4)
            
            # Housing growth rate
            # Base rate plus adjustments for population growth and retail
            base_housing_growth = 0.004  # 0.4% annual growth
            
            # Areas with higher population growth need more housing
            pop_growth_factor = self.data['population_growth_rate'] * 0.5
            
            # Areas with more retail per capita attract more housing
            if 'retail_businesses' in self.data.columns and 'population' in self.data.columns:
                self.data['retail_per_capita'] = (
                    self.data['retail_businesses'] / self.data['population'] * 1000
                ).round(2)
                retail_factor = (self.data['retail_per_capita'] / self.data['retail_per_capita'].median() - 1) * 0.002
            else:
                retail_factor = 0
            
            # Calculate housing growth rate
            self.data['housing_growth_rate'] = (
                base_housing_growth + pop_growth_factor + retail_factor
            ).clip(lower=-0.01, upper=0.04).round(4)
            
            # Retail growth rate
            # Base rate plus adjustments for population and income growth
            base_retail_growth = 0.003  # 0.3% annual growth
            
            # Areas with higher population growth need more retail
            pop_growth_factor = self.data['population_growth_rate'] * 0.7
            
            # Areas with higher income growth attract more retail
            if 'median_income' in self.data.columns:
                income_growth_factor = income_factor * 2
            else:
                income_growth_factor = 0
            
            # Calculate retail growth rate
            self.data['retail_growth_rate'] = (
                base_retail_growth + pop_growth_factor + income_growth_factor
            ).clip(lower=-0.02, upper=0.05).round(4)
            
            # Calculate 10-year projections
            years = 10
            
            # Population projection
            self.data['population_10yr'] = (
                self.data['population'] * (1 + self.data['population_growth_rate']) ** years
            ).round(0).astype(int)
            
            # Housing projection
            self.data['housing_10yr'] = (
                self.data['housing_units'] * (1 + self.data['housing_growth_rate']) ** years
            ).round(0).astype(int)
            
            # Retail projection
            self.data['retail_10yr'] = (
                self.data['retail_businesses'] * (1 + self.data['retail_growth_rate']) ** years
            ).round(0).astype(int)
            
            # Calculate growth percentages
            self.data['population_growth_pct'] = (
                (self.data['population_10yr'] / self.data['population'] - 1) * 100
            ).round(1)
            
            self.data['housing_growth_pct'] = (
                (self.data['housing_10yr'] / self.data['housing_units'] - 1) * 100
            ).round(1)
            
            self.data['retail_growth_pct'] = (
                (self.data['retail_10yr'] / self.data['retail_businesses'] - 1) * 100
            ).round(1)
            
            # Classify ZIP codes by growth potential
            self.data['growth_category'] = pd.cut(
                self.data['population_growth_pct'],
                bins=[-float('inf'), 0, 5, 10, 20, float('inf')],
                labels=['Declining', 'Stable', 'Moderate Growth', 'Strong Growth', 'Very Strong Growth']
            )
            
            # Calculate summary statistics
            self.context['total_population_current'] = self.data['population'].sum()
            self.context['total_population_10yr'] = self.data['population_10yr'].sum()
            self.context['total_population_growth_pct'] = (
                (self.context['total_population_10yr'] / self.context['total_population_current'] - 1) * 100
            ).round(1)
            
            self.context['total_housing_current'] = self.data['housing_units'].sum()
            self.context['total_housing_10yr'] = self.data['housing_10yr'].sum()
            self.context['total_housing_growth_pct'] = (
                (self.context['total_housing_10yr'] / self.context['total_housing_current'] - 1) * 100
            ).round(1)
            
            self.context['total_retail_current'] = self.data['retail_businesses'].sum()
            self.context['total_retail_10yr'] = self.data['retail_10yr'].sum()
            self.context['total_retail_growth_pct'] = (
                (self.context['total_retail_10yr'] / self.context['total_retail_current'] - 1) * 100
            ).round(1)
            
            # Find ZIP codes with highest and lowest growth
            max_growth_idx = self.data['population_growth_pct'].idxmax()
            min_growth_idx = self.data['population_growth_pct'].idxmin()
            
            self.context['max_growth_zip'] = self.data.loc[max_growth_idx, 'zip_code']
            self.context['max_growth_pct'] = self.data.loc[max_growth_idx, 'population_growth_pct']
            
            self.context['min_growth_zip'] = self.data.loc[min_growth_idx, 'zip_code']
            self.context['min_growth_pct'] = self.data.loc[min_growth_idx, 'population_growth_pct']
            
            # Count ZIP codes in each growth category
            growth_counts = self.data['growth_category'].value_counts().to_dict()
            self.context['growth_category_counts'] = growth_counts
            
            logger.info("Growth projections calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating growth projections: {str(e)}")
    
    def _create_visualizations(self):
        """
        Create visualizations for the report.
        """
        try:
            # Create figures directory
            figures_dir = self.output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            # List to store visualization paths
            self.context['visualizations'] = []
            
            # Population growth by ZIP code
            plt.figure(figsize=(12, 8))
            
            # Sort by population growth percentage
            plot_data = self.data.sort_values('population_growth_pct', ascending=False).head(20)
            
            # Create bar chart
            ax = sns.barplot(x='zip_code', y='population_growth_pct', data=plot_data)
            
            # Color bars based on positive/negative growth
            for i, bar in enumerate(ax.patches):
                if plot_data.iloc[i]['population_growth_pct'] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            plt.title('Projected 10-Year Population Growth by ZIP Code (Top 20)')
            plt.xlabel('ZIP Code')
            plt.ylabel('Population Growth (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'population_growth_by_zip.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created population growth by ZIP chart: {chart_path}")
            
            # Growth category distribution
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of growth categories
            growth_counts = self.data['growth_category'].value_counts().sort_index()
            growth_counts.plot(kind='bar')
            
            plt.title('Distribution of ZIP Codes by Growth Category')
            plt.xlabel('Growth Category')
            plt.ylabel('Number of ZIP Codes')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'growth_category_distribution.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created growth category distribution chart: {chart_path}")
            
            # Current vs projected population
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            plt.scatter(self.data['population'], 
                       self.data['population_10yr'], 
                       alpha=0.7)
            
            # Add diagonal line (current = projected)
            max_val = max(self.data['population'].max(), 
                         self.data['population_10yr'].max())
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            # Add labels for some points
            for i, row in self.data.iterrows():
                if abs(row['population_growth_pct']) > self.data['population_growth_pct'].abs().quantile(0.9):
                    plt.annotate(row['zip_code'], 
                                (row['population'], row['population_10yr']),
                                xytext=(5, 5), textcoords='offset points')
            
            plt.title('Current vs Projected Population by ZIP Code')
            plt.xlabel('Current Population')
            plt.ylabel('Projected Population (10 Years)')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'current_vs_projected_population.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created current vs projected population chart: {chart_path}")
            
            logger.info(f"Created {len(self.context['visualizations'])} visualizations")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _prepare_zip_details(self):
        """
        Prepare ZIP code details for the report.
        """
        try:
            # Create list of ZIP code details
            zip_details = []
            
            for _, row in self.data.iterrows():
                detail = {
                    'zip_code': row['zip_code'],
                    'population_current': row['population'],
                    'population_10yr': row['population_10yr'],
                    'population_growth_pct': row['population_growth_pct'],
                    'population_growth_rate': row['population_growth_rate'],
                    
                    'housing_current': row['housing_units'],
                    'housing_10yr': row['housing_10yr'],
                    'housing_growth_pct': row['housing_growth_pct'],
                    'housing_growth_rate': row['housing_growth_rate'],
                    
                    'retail_current': row['retail_businesses'],
                    'retail_10yr': row['retail_10yr'],
                    'retail_growth_pct': row['retail_growth_pct'],
                    'retail_growth_rate': row['retail_growth_rate'],
                    
                    'growth_category': row['growth_category']
                }
                
                # Add income data if available
                if 'median_income' in self.data.columns:
                    detail['median_income'] = row['median_income']
                    detail['median_income_fmt'] = format_currency(row['median_income'])
                
                zip_details.append(detail)
            
            # Sort by population growth percentage (descending)
            zip_details.sort(key=lambda x: x['population_growth_pct'], reverse=True)
            
            self.context['zip_details'] = zip_details
            logger.info(f"Prepared details for {len(zip_details)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error preparing ZIP code details: {str(e)}")
    
    def generate_report(self, output_filename=None):
        """
        Generate the Ten Year Growth Projection report.
        
        Args:
            output_filename (str, optional): Filename for the output report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use default filename if not provided
            if output_filename is None:
                output_filename = "ten_year_growth_report.html"
            
            # Generate report using parent method
            return super().generate_report(output_filename)
            
        except Exception as e:
            logger.error(f"Error generating Ten Year Growth Projection report: {str(e)}")
            return False
