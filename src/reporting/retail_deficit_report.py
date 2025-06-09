"""
Retail Deficit Report generator for the Chicago Population Analysis project.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.reporting.base_report import BaseReport
from src.utils.helpers import format_currency, format_percent

logger = logging.getLogger(__name__)

class RetailDeficitReport(BaseReport):
    """Generates reports analyzing retail deficits in Chicago ZIP codes."""
    
    def __init__(self, output_dir=None, template_dir=None):
        """
        Initialize the Retail Deficit Report generator.
        
        Args:
            output_dir (Path, optional): Directory to save report outputs
            template_dir (Path, optional): Directory containing report templates
        """
        super().__init__("Retail Deficit Analysis", output_dir, template_dir)
    
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
            required_cols = ['population', 'median_income', 'retail_businesses']
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
        Flag rows with insufficient data for retail deficit analysis.
        
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
            min_retail = 5         # Minimum retail businesses
            
            # Flag rows with insufficient data
            result['data_status'] = 'ok'
            
            # Check population
            if 'population' in result.columns:
                result.loc[result['population'] < min_population, 'data_status'] = 'insufficient'
            
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
            
            # Calculate retail metrics
            self._calculate_retail_metrics()
            
            # Create visualizations
            self._create_visualizations()
            
            # Prepare ZIP code details
            self._prepare_zip_details()
            
            logger.info("Report context prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing report context: {str(e)}")
    
    def _calculate_retail_metrics(self):
        """
        Calculate retail deficit metrics.
        """
        try:
            # Calculate retail per capita
            self.data['retail_per_capita'] = (
                self.data['retail_businesses'] / self.data['population'] * 1000
            ).round(2)
            
            # Calculate expected retail based on income and population
            # This is a simplified model: expected retail = population * income factor
            income_factor = self.data['median_income'] / 50000  # normalize to 50k income
            self.data['expected_retail'] = (
                self.data['population'] * income_factor * 0.002  # 0.002 businesses per person per normalized income
            ).round(0)
            
            # Calculate retail deficit/surplus
            self.data['retail_deficit'] = (
                self.data['expected_retail'] - self.data['retail_businesses']
            ).round(0)
            
            # Calculate retail deficit percentage
            self.data['retail_deficit_pct'] = (
                self.data['retail_deficit'] / self.data['expected_retail']
            ).round(2)
            
            # Classify ZIP codes by deficit/surplus
            self.data['deficit_category'] = pd.cut(
                self.data['retail_deficit_pct'],
                bins=[-float('inf'), -0.2, -0.05, 0.05, 0.2, float('inf')],
                labels=['Strong Surplus', 'Moderate Surplus', 'Balanced', 'Moderate Deficit', 'Strong Deficit']
            )
            
            # Calculate summary statistics
            self.context['avg_retail_per_capita'] = self.data['retail_per_capita'].mean().round(2)
            self.context['median_retail_per_capita'] = self.data['retail_per_capita'].median().round(2)
            
            self.context['total_retail_deficit'] = self.data['retail_deficit'].sum()
            self.context['avg_retail_deficit'] = self.data['retail_deficit'].mean().round(1)
            
            # Count ZIP codes in deficit vs surplus
            deficit_count = (self.data['retail_deficit'] > 0).sum()
            surplus_count = (self.data['retail_deficit'] <= 0).sum()
            
            self.context['deficit_count'] = deficit_count
            self.context['surplus_count'] = surplus_count
            
            # Find ZIP codes with highest deficit and surplus
            if deficit_count > 0:
                max_deficit_idx = self.data[self.data['retail_deficit'] > 0]['retail_deficit'].idxmax()
                self.context['max_deficit_zip'] = self.data.loc[max_deficit_idx, 'zip_code']
                self.context['max_deficit_value'] = self.data.loc[max_deficit_idx, 'retail_deficit']
            
            if surplus_count > 0:
                max_surplus_idx = self.data[self.data['retail_deficit'] <= 0]['retail_deficit'].idxmin()
                self.context['max_surplus_zip'] = self.data.loc[max_surplus_idx, 'zip_code']
                self.context['max_surplus_value'] = -self.data.loc[max_surplus_idx, 'retail_deficit']
            
            # Count ZIP codes in each category
            deficit_counts = self.data['deficit_category'].value_counts().to_dict()
            self.context['deficit_category_counts'] = deficit_counts
            
            logger.info("Retail metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating retail metrics: {str(e)}")
    
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
            
            # Retail deficit by ZIP code
            plt.figure(figsize=(12, 8))
            
            # Sort by retail deficit
            plot_data = self.data.sort_values('retail_deficit', ascending=False).head(20)
            
            # Create bar chart
            ax = sns.barplot(x='zip_code', y='retail_deficit', data=plot_data)
            
            # Color bars based on positive/negative deficit
            for i, bar in enumerate(ax.patches):
                if plot_data.iloc[i]['retail_deficit'] > 0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
            
            plt.title('Retail Deficit/Surplus by ZIP Code (Top 20)')
            plt.xlabel('ZIP Code')
            plt.ylabel('Retail Deficit/Surplus (# of Businesses)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'retail_deficit_by_zip.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created retail deficit by ZIP chart: {chart_path}")
            
            # Expected vs actual retail
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            plt.scatter(self.data['expected_retail'], 
                       self.data['retail_businesses'], 
                       alpha=0.7)
            
            # Add diagonal line (expected = actual)
            max_val = max(self.data['expected_retail'].max(), 
                         self.data['retail_businesses'].max())
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            # Add labels for some points
            for i, row in self.data.iterrows():
                if abs(row['retail_deficit']) > self.data['retail_deficit'].abs().quantile(0.9):
                    plt.annotate(row['zip_code'], 
                                (row['expected_retail'], row['retail_businesses']),
                                xytext=(5, 5), textcoords='offset points')
            
            plt.title('Expected vs Actual Retail Businesses by ZIP Code')
            plt.xlabel('Expected Retail Businesses')
            plt.ylabel('Actual Retail Businesses')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'expected_vs_actual_retail.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created expected vs actual retail chart: {chart_path}")
            
            # Deficit category distribution
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of deficit categories
            deficit_counts = self.data['deficit_category'].value_counts().sort_index()
            deficit_counts.plot(kind='bar')
            
            plt.title('Distribution of ZIP Codes by Retail Deficit/Surplus Category')
            plt.xlabel('Category')
            plt.ylabel('Number of ZIP Codes')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'deficit_category_distribution.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created deficit category distribution chart: {chart_path}")
            
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
                    'population': row['population'],
                    'median_income': row['median_income'],
                    'median_income_fmt': format_currency(row['median_income']),
                    'retail_businesses': row['retail_businesses'],
                    'retail_per_capita': row['retail_per_capita'],
                    'expected_retail': row['expected_retail'],
                    'retail_deficit': row['retail_deficit'],
                    'retail_deficit_pct': row['retail_deficit_pct'],
                    'deficit_category': row['deficit_category']
                }
                
                zip_details.append(detail)
            
            # Sort by retail deficit (descending)
            zip_details.sort(key=lambda x: x['retail_deficit'], reverse=True)
            
            self.context['zip_details'] = zip_details
            logger.info(f"Prepared details for {len(zip_details)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error preparing ZIP code details: {str(e)}")
    
    def generate_report(self, output_filename=None):
        """
        Generate the Retail Deficit Analysis report.
        
        Args:
            output_filename (str, optional): Filename for the output report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use default filename if not provided
            if output_filename is None:
                output_filename = "retail_deficit_report.html"
            
            # Generate report using parent method
            return super().generate_report(output_filename)
            
        except Exception as e:
            logger.error(f"Error generating Retail Deficit Analysis report: {str(e)}")
            return False
