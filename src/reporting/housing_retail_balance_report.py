"""
Housing Retail Balance Report generator for the Chicago Population Analysis project.
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

class HousingRetailBalanceReport(BaseReport):
    """Generates reports analyzing the balance between housing and retail in Chicago ZIP codes."""
    
    def __init__(self, output_dir=None, template_dir=None):
        """
        Initialize the Housing Retail Balance Report generator.
        
        Args:
            output_dir (Path, optional): Directory to save report outputs
            template_dir (Path, optional): Directory containing report templates
        """
        super().__init__("Housing Retail Balance", output_dir, template_dir)
    
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
            required_cols = ['housing_units', 'retail_businesses']
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
        Flag rows with insufficient data for housing-retail balance analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with data_status column
        """
        try:
            # Create a copy to avoid modifying the original
            result = df.copy()
            
            # Define minimum thresholds
            min_housing = 100  # Minimum housing units
            min_retail = 5     # Minimum retail businesses
            
            # Flag rows with insufficient data
            result['data_status'] = 'ok'
            
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
            
            # Calculate balance metrics
            self._calculate_balance_metrics()
            
            # Create visualizations
            self._create_visualizations()
            
            # Prepare ZIP code details
            self._prepare_zip_details()
            
            logger.info("Report context prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing report context: {str(e)}")
    
    def _calculate_balance_metrics(self):
        """
        Calculate housing-retail balance metrics.
        """
        try:
            # Calculate retail per housing unit
            self.data['retail_per_housing'] = (
                self.data['retail_businesses'] / self.data['housing_units']
            ).round(4)
            
            # Calculate retail per capita if population exists
            if 'population' in self.data.columns:
                self.data['retail_per_capita'] = (
                    self.data['retail_businesses'] / self.data['population'] * 1000
                ).round(2)
            
            # Calculate housing per capita if population exists
            if 'population' in self.data.columns:
                self.data['housing_per_capita'] = (
                    self.data['housing_units'] / self.data['population']
                ).round(4)
            
            # Calculate summary statistics
            self.context['avg_retail_per_housing'] = self.data['retail_per_housing'].mean().round(4)
            self.context['median_retail_per_housing'] = self.data['retail_per_housing'].median().round(4)
            self.context['max_retail_per_housing'] = self.data['retail_per_housing'].max().round(4)
            self.context['min_retail_per_housing'] = self.data['retail_per_housing'].min().round(4)
            
            # Find ZIP codes with highest and lowest retail per housing
            max_idx = self.data['retail_per_housing'].idxmax()
            min_idx = self.data['retail_per_housing'].idxmin()
            
            self.context['max_retail_per_housing_zip'] = self.data.loc[max_idx, 'zip_code']
            self.context['min_retail_per_housing_zip'] = self.data.loc[min_idx, 'zip_code']
            
            # Calculate percentiles
            percentiles = [10, 25, 50, 75, 90]
            self.context['retail_per_housing_percentiles'] = {
                p: self.data['retail_per_housing'].quantile(p/100).round(4)
                for p in percentiles
            }
            
            # Classify ZIP codes by retail-housing balance
            self.data['balance_category'] = pd.cut(
                self.data['retail_per_housing'],
                bins=[0, 0.01, 0.02, 0.05, 0.1, float('inf')],
                labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            )
            
            # Count ZIP codes in each category
            balance_counts = self.data['balance_category'].value_counts().to_dict()
            self.context['balance_category_counts'] = balance_counts
            
            logger.info("Balance metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating balance metrics: {str(e)}")
    
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
            
            # Retail per housing unit by ZIP code
            plt.figure(figsize=(12, 8))
            
            # Sort by retail per housing
            plot_data = self.data.sort_values('retail_per_housing', ascending=False).head(20)
            
            # Create bar chart
            sns.barplot(x='zip_code', y='retail_per_housing', data=plot_data)
            plt.title('Retail Businesses per Housing Unit by ZIP Code (Top 20)')
            plt.xlabel('ZIP Code')
            plt.ylabel('Retail Businesses per Housing Unit')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'retail_per_housing_by_zip.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created retail per housing by ZIP chart: {chart_path}")
            
            # Housing vs retail scatter plot
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            plt.scatter(self.data['housing_units'], 
                       self.data['retail_businesses'], 
                       alpha=0.7)
            
            # Add labels for some points
            for i, row in self.data.iterrows():
                if (row['housing_units'] > self.data['housing_units'].quantile(0.9) or
                    row['retail_businesses'] > self.data['retail_businesses'].quantile(0.9)):
                    plt.annotate(row['zip_code'], 
                                (row['housing_units'], row['retail_businesses']),
                                xytext=(5, 5), textcoords='offset points')
            
            # Add trend line
            z = np.polyfit(self.data['housing_units'], 
                          self.data['retail_businesses'], 1)
            p = np.poly1d(z)
            plt.plot(self.data['housing_units'], 
                    p(self.data['housing_units']), 
                    "r--", alpha=0.5)
            
            plt.title('Housing Units vs Retail Businesses by ZIP Code')
            plt.xlabel('Housing Units')
            plt.ylabel('Retail Businesses')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'housing_vs_retail.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created housing vs retail chart: {chart_path}")
            
            # Balance category distribution
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of balance categories
            balance_counts = self.data['balance_category'].value_counts().sort_index()
            balance_counts.plot(kind='bar')
            
            plt.title('Distribution of ZIP Codes by Retail-Housing Balance Category')
            plt.xlabel('Balance Category')
            plt.ylabel('Number of ZIP Codes')
            plt.tight_layout()
            
            # Save chart
            chart_path = figures_dir / 'balance_category_distribution.png'
            plt.savefig(chart_path)
            plt.close()
            
            self.context['visualizations'].append(str(chart_path))
            logger.info(f"Created balance category distribution chart: {chart_path}")
            
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
                    'housing_units': row['housing_units'],
                    'retail_businesses': row['retail_businesses'],
                    'retail_per_housing': row['retail_per_housing'],
                    'balance_category': row['balance_category']
                }
                
                # Add population data if available
                if 'population' in self.data.columns:
                    detail['population'] = row['population']
                    detail['retail_per_capita'] = row['retail_per_capita']
                    detail['housing_per_capita'] = row['housing_per_capita']
                
                # Add income data if available
                if 'median_income' in self.data.columns:
                    detail['median_income'] = row['median_income']
                    detail['median_income_fmt'] = format_currency(row['median_income'])
                
                zip_details.append(detail)
            
            # Sort by retail per housing (descending)
            zip_details.sort(key=lambda x: x['retail_per_housing'], reverse=True)
            
            self.context['zip_details'] = zip_details
            logger.info(f"Prepared details for {len(zip_details)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error preparing ZIP code details: {str(e)}")
    
    def generate_report(self, output_filename=None):
        """
        Generate the Housing Retail Balance report.
        
        Args:
            output_filename (str, optional): Filename for the output report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use default filename if not provided
            if output_filename is None:
                output_filename = "housing_retail_balance_report.html"
            
            # Generate report using parent method
            return super().generate_report(output_filename)
            
        except Exception as e:
            logger.error(f"Error generating Housing Retail Balance report: {str(e)}")
            return False
