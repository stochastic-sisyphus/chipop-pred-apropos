"""
Chicago ZIP Summary Report generator for the Chicago Population Analysis project.
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

class ChicagoZipSummaryReport(BaseReport):
    """Generates summary reports for Chicago ZIP codes."""
    
    def __init__(self, output_dir=None, template_dir=None):
        """
        Initialize the Chicago ZIP Summary Report generator.
        
        Args:
            output_dir (Path, optional): Directory to save report outputs
            template_dir (Path, optional): Directory containing report templates
        """
        super().__init__("Chicago ZIP Summary", output_dir, template_dir)
    
    def _filter_valid_zips(self, df):
        """
        Filter dataframe to include only valid Chicago ZIP codes.
        
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
            
            logger.info(f"Filtered to {len(filtered_df)} Chicago ZIP codes")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering valid ZIP codes: {str(e)}")
            return None
    
    def _prepare_report_context(self):
        """
        Prepare context dictionary for report template.
        """
        try:
            # Call parent method to initialize context
            super()._prepare_report_context()
            
            # Calculate summary statistics
            self._calculate_summary_stats()
            
            # Create visualizations
            self._create_visualizations()
            
            # Prepare ZIP code details
            self._prepare_zip_details()
            
            logger.info("Report context prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing report context: {str(e)}")
    
    def _calculate_summary_stats(self):
        """
        Calculate summary statistics for the report.
        """
        try:
            # Population statistics
            if 'population' in self.data.columns:
                self.context['total_population'] = self.data['population'].sum()
                self.context['avg_population'] = self.data['population'].mean().round(0)
                self.context['max_population'] = self.data['population'].max()
                self.context['min_population'] = self.data['population'].min()
                
                # Find ZIP code with highest population
                max_pop_idx = self.data['population'].idxmax()
                self.context['max_population_zip'] = self.data.loc[max_pop_idx, 'zip_code']
            
            # Housing statistics
            if 'housing_units' in self.data.columns:
                self.context['total_housing'] = self.data['housing_units'].sum()
                self.context['avg_housing'] = self.data['housing_units'].mean().round(0)
                
                # Calculate housing density if population exists
                if 'population' in self.data.columns:
                    self.data['housing_density'] = (
                        self.data['housing_units'] / self.data['population'] * 1000
                    ).round(2)
                    self.context['avg_housing_density'] = self.data['housing_density'].mean().round(2)
            
            # Income statistics
            if 'median_income' in self.data.columns:
                self.context['avg_median_income'] = self.data['median_income'].mean().round(0)
                self.context['max_median_income'] = self.data['median_income'].max()
                self.context['min_median_income'] = self.data['median_income'].min()
                
                # Find ZIP code with highest median income
                max_inc_idx = self.data['median_income'].idxmax()
                self.context['max_income_zip'] = self.data.loc[max_inc_idx, 'zip_code']
                
                # Format currency values
                self.context['avg_median_income_fmt'] = format_currency(self.context['avg_median_income'])
                self.context['max_median_income_fmt'] = format_currency(self.context['max_median_income'])
                self.context['min_median_income_fmt'] = format_currency(self.context['min_median_income'])
            
            # Retail statistics
            if 'retail_businesses' in self.data.columns:
                self.context['total_retail'] = self.data['retail_businesses'].sum()
                self.context['avg_retail'] = self.data['retail_businesses'].mean().round(1)
                
                # Calculate retail per capita if population exists
                if 'population' in self.data.columns:
                    self.data['retail_per_capita'] = (
                        self.data['retail_businesses'] / self.data['population'] * 1000
                    ).round(2)
                    self.context['avg_retail_per_capita'] = self.data['retail_per_capita'].mean().round(2)
            
            # Count ZIP codes
            self.context['zip_count'] = len(self.data)
            
            logger.info("Summary statistics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
    
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
            
            # Population by ZIP code
            if 'population' in self.data.columns:
                # Sort by population
                plot_data = self.data.sort_values('population', ascending=False).head(15)
                
                # Create bar chart
                plt.figure(figsize=(10, 6))
                sns.barplot(x='zip_code', y='population', data=plot_data)
                plt.title('Population by ZIP Code (Top 15)')
                plt.xlabel('ZIP Code')
                plt.ylabel('Population')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save chart
                chart_path = figures_dir / 'population_by_zip.png'
                plt.savefig(chart_path)
                plt.close()
                
                self.context['visualizations'].append(str(chart_path))
                logger.info(f"Created population by ZIP chart: {chart_path}")
            
            # Median income by ZIP code
            if 'median_income' in self.data.columns:
                # Sort by median income
                plot_data = self.data.sort_values('median_income', ascending=False).head(15)
                
                # Create bar chart
                plt.figure(figsize=(10, 6))
                sns.barplot(x='zip_code', y='median_income', data=plot_data)
                plt.title('Median Income by ZIP Code (Top 15)')
                plt.xlabel('ZIP Code')
                plt.ylabel('Median Income ($)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save chart
                chart_path = figures_dir / 'median_income_by_zip.png'
                plt.savefig(chart_path)
                plt.close()
                
                self.context['visualizations'].append(str(chart_path))
                logger.info(f"Created median income by ZIP chart: {chart_path}")
            
            # Retail businesses by ZIP code
            if 'retail_businesses' in self.data.columns:
                # Sort by retail businesses
                plot_data = self.data.sort_values('retail_businesses', ascending=False).head(15)
                
                # Create bar chart
                plt.figure(figsize=(10, 6))
                sns.barplot(x='zip_code', y='retail_businesses', data=plot_data)
                plt.title('Retail Businesses by ZIP Code (Top 15)')
                plt.xlabel('ZIP Code')
                plt.ylabel('Number of Retail Businesses')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save chart
                chart_path = figures_dir / 'retail_by_zip.png'
                plt.savefig(chart_path)
                plt.close()
                
                self.context['visualizations'].append(str(chart_path))
                logger.info(f"Created retail by ZIP chart: {chart_path}")
            
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
                    'zip_code': row['zip_code']
                }
                
                # Add population data
                if 'population' in self.data.columns:
                    detail['population'] = row['population']
                
                # Add housing data
                if 'housing_units' in self.data.columns:
                    detail['housing_units'] = row['housing_units']
                    
                    if 'population' in self.data.columns:
                        detail['housing_density'] = (
                            row['housing_units'] / row['population'] * 1000
                        ).round(2)
                
                # Add income data
                if 'median_income' in self.data.columns:
                    detail['median_income'] = row['median_income']
                    detail['median_income_fmt'] = format_currency(row['median_income'])
                
                # Add retail data
                if 'retail_businesses' in self.data.columns:
                    detail['retail_businesses'] = row['retail_businesses']
                    
                    if 'population' in self.data.columns:
                        detail['retail_per_capita'] = (
                            row['retail_businesses'] / row['population'] * 1000
                        ).round(2)
                
                # Add employment data
                if 'employment_rate' in self.data.columns:
                    detail['employment_rate'] = row['employment_rate']
                    detail['employment_rate_fmt'] = format_percent(row['employment_rate'] / 100)
                
                zip_details.append(detail)
            
            # Sort by ZIP code
            zip_details.sort(key=lambda x: x['zip_code'])
            
            self.context['zip_details'] = zip_details
            logger.info(f"Prepared details for {len(zip_details)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error preparing ZIP code details: {str(e)}")
    
    def generate_report(self, output_filename=None):
        """
        Generate the Chicago ZIP Summary report.
        
        Args:
            output_filename (str, optional): Filename for the output report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use default filename if not provided
            if output_filename is None:
                output_filename = "chicago_zip_summary_report.html"
            
            # Generate report using parent method
            return super().generate_report(output_filename)
            
        except Exception as e:
            logger.error(f"Error generating Chicago ZIP Summary report: {str(e)}")
            return False
