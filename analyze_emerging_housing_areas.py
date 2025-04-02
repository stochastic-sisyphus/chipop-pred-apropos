#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmergingAreasAnalyzer:
    """
    Analyzes building permits and retail licenses to identify emerging housing areas 
    with potential retail deficits.
    """
    
    def __init__(self, data_dir='data', output_dir='output'):
        """Initialize with data paths"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.viz_dir = Path('visualizations')
        self.viz_dir.mkdir(exist_ok=True)
        
        # Input and output files
        self.permits_file = self.data_dir / 'building_permits.csv'
        self.licenses_file = self.data_dir / 'business_licenses.csv'
        self.pop_shifts_file = Path('population_shift_patterns.csv')
        
        # Output files
        self.emerging_areas_file = self.output_dir / 'emerging_housing_areas.csv'
        self.retail_deficit_file = self.output_dir / 'high_retail_deficit_areas.csv'
        self.ten_year_growth_file = self.output_dir / 'ten_year_growth_areas.csv'
        
        # Data frames
        self.permits_df = None
        self.licenses_df = None
        self.pop_shifts_df = None
        self.emerging_areas = None
        self.retail_deficit_areas = None
        self.ten_year_growth_areas = None
        
        # Time periods for comparison
        self.recent_period = (2020, 2025)  # Define what "recent" means
        self.historical_period = (2015, 2020)  # For comparison
        self.ten_year_start = 2015  # For 10-year growth analysis
        self.ten_year_end = 2025
        
        # Set up plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load permit and license data"""
        try:
            if self.permits_file.exists():
                logger.info(f"Loading building permits from {self.permits_file}")
                self.permits_df = pd.read_csv(self.permits_file)
                
                # Convert columns if needed
                if 'issue_date' in self.permits_df.columns:
                    self.permits_df['year'] = pd.to_datetime(self.permits_df['issue_date']).dt.year
                
                # Handle zip_code differently in building permits (it's contact_1_zipcode)
                if 'contact_1_zipcode' in self.permits_df.columns:
                    # Extract and clean zip code
                    self.permits_df['zip_code'] = self.permits_df['contact_1_zipcode'].astype(str).str[:5]
                    # Clean up invalid zip codes
                    self.permits_df['zip_code'] = self.permits_df['zip_code'].apply(
                        lambda x: x if x.isdigit() and len(x) == 5 else ''
                    )
                    # Remove rows with missing zip code
                    self.permits_df = self.permits_df[self.permits_df['zip_code'] != '']
                
                if 'zip_code' in self.permits_df.columns:
                    self.permits_df['zip_code'] = self.permits_df['zip_code'].astype(str)
            else:
                logger.error(f"Building permits file not found: {self.permits_file}")
            
            if self.licenses_file.exists():
                logger.info(f"Loading business licenses from {self.licenses_file}")
                self.licenses_df = pd.read_csv(self.licenses_file)
                
                # Convert columns if needed
                if 'license_start_date' in self.licenses_df.columns:
                    self.licenses_df['year'] = pd.to_datetime(self.licenses_df['license_start_date']).dt.year
                
                if 'zip_code' in self.licenses_df.columns:
                    # Clean up zip codes
                    self.licenses_df['zip_code'] = self.licenses_df['zip_code'].astype(str).str[:5]
                    # Clean up invalid zip codes
                    self.licenses_df['zip_code'] = self.licenses_df['zip_code'].apply(
                        lambda x: x if x.isdigit() and len(x) == 5 else ''
                    )
                    # Remove rows with missing zip code
                    self.licenses_df = self.licenses_df[self.licenses_df['zip_code'] != '']
                    self.licenses_df['zip_code'] = self.licenses_df['zip_code'].astype(str)
            else:
                logger.error(f"Business licenses file not found: {self.licenses_file}")
            
            if self.pop_shifts_file.exists():
                logger.info(f"Loading population shifts data from {self.pop_shifts_file}")
                self.pop_shifts_df = pd.read_csv(self.pop_shifts_file)
            else:
                logger.warning(f"Population shifts file not found: {self.pop_shifts_file}")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def analyze_emerging_housing_areas(self):
        """
        Identify ZIP codes with significant recent housing development
        but potentially inadequate retail development
        """
        if self.permits_df is None or self.licenses_df is None:
            logger.error("Permit or license data missing. Cannot analyze emerging areas.")
            return
        
        try:
            # Filter housing permits with focus on multi-family
            housing_permits = self.permits_df[
                self.permits_df['permit_type'].str.contains('PERMIT - NEW CONSTRUCTION', case=False, na=False) &
                self.permits_df['work_description'].str.contains('RESIDENTIAL|APARTMENT|CONDO|MULTI', case=False, na=False)
            ]
            
            # Further identify multi-family permits
            multi_family_permits = housing_permits[
                housing_permits['work_description'].str.contains('APARTMENT|CONDO|MULTI|UNIT', case=False, na=False) &
                ~housing_permits['work_description'].str.contains('SINGLE FAMILY|ONE FAMILY', case=False, na=False)
            ]
            
            # Filter retail licenses
            retail_licenses = self.licenses_df[
                self.licenses_df['license_description'].str.contains(
                    'RETAIL|FOOD|GROCERY|RESTAURANT|SHOP|STORE', 
                    case=False, na=False
                )
            ]
            
            # Group by ZIP code and time period
            recent_housing = housing_permits[
                (housing_permits['year'] >= self.recent_period[0]) & 
                (housing_permits['year'] <= self.recent_period[1])
            ].groupby('zip_code').size()
            
            recent_multi_family = multi_family_permits[
                (multi_family_permits['year'] >= self.recent_period[0]) & 
                (multi_family_permits['year'] <= self.recent_period[1])
            ].groupby('zip_code').size()
            
            historical_housing = housing_permits[
                (housing_permits['year'] >= self.historical_period[0]) & 
                (housing_permits['year'] <= self.historical_period[1])
            ].groupby('zip_code').size()
            
            historical_multi_family = multi_family_permits[
                (multi_family_permits['year'] >= self.historical_period[0]) & 
                (multi_family_permits['year'] <= self.historical_period[1])
            ].groupby('zip_code').size()
            
            recent_retail = retail_licenses[
                (retail_licenses['year'] >= self.recent_period[0]) & 
                (retail_licenses['year'] <= self.recent_period[1])
            ].groupby('zip_code').size()
            
            # Calculate growth in housing permits
            housing_growth = pd.DataFrame({
                'recent_housing': recent_housing,
                'historical_housing': historical_housing,
                'recent_multi_family': recent_multi_family,
                'historical_multi_family': historical_multi_family
            }).fillna(0)
            
            # Prevent division by zero
            housing_growth['historical_housing'] = housing_growth['historical_housing'].replace(0, 1)
            housing_growth['historical_multi_family'] = housing_growth['historical_multi_family'].replace(0, 0.1)
            
            housing_growth['growth_pct'] = (
                (housing_growth['recent_housing'] - housing_growth['historical_housing']) / 
                housing_growth['historical_housing'] * 100
            )
            
            housing_growth['multi_family_growth_pct'] = (
                (housing_growth['recent_multi_family'] - housing_growth['historical_multi_family']) / 
                housing_growth['historical_multi_family'] * 100
            )
            
            # Identify areas with new multi-family development that had little before
            housing_growth['is_new_multi_family'] = (
                (housing_growth['recent_multi_family'] >= 5) &  # At least 5 multi-family permits recently
                (housing_growth['historical_multi_family'] <= 2)  # Very few multi-family permits historically
            )
            
            # Calculate housing-to-retail ratio
            housing_retail_ratio = pd.DataFrame({
                'housing_permits': recent_housing,
                'multi_family_permits': recent_multi_family,
                'retail_licenses': recent_retail
            }).fillna(0)
            
            # Prevent division by zero
            housing_retail_ratio['retail_licenses'] = housing_retail_ratio['retail_licenses'].replace(0, 1)
            
            housing_retail_ratio['housing_retail_ratio'] = (
                housing_retail_ratio['housing_permits'] / housing_retail_ratio['retail_licenses']
            )
            
            housing_retail_ratio['multi_family_retail_ratio'] = (
                housing_retail_ratio['multi_family_permits'] / housing_retail_ratio['retail_licenses']
            )
            
            # Combine dataframes
            combined = housing_growth.join(housing_retail_ratio, how='outer').fillna(0)
            
            # Define criteria for emerging housing areas (more inclusive than before)
            self.emerging_areas = combined[
                ((combined['growth_pct'] > 20) |  # Lower threshold to 20% for general growth
                 (combined['multi_family_growth_pct'] > 100) |  # High multi-family growth
                 (combined['is_new_multi_family'] == True)) &  # Areas with new multi-family development
                (combined['recent_housing'] >= 5) &  # At least some recent activity
                (combined['housing_retail_ratio'] > 1.5)  # More housing than retail
            ].sort_values('multi_family_growth_pct', ascending=False)
            
            # Add ZIP code and format growth percentage
            self.emerging_areas = self.emerging_areas.reset_index()
            self.emerging_areas['growth_pct'] = self.emerging_areas['growth_pct'].round(1)
            self.emerging_areas['multi_family_growth_pct'] = self.emerging_areas['multi_family_growth_pct'].round(1)
            
            # Identify areas with high housing-to-retail imbalance
            self.retail_deficit_areas = combined[
                (combined['housing_permits'] >= 10) &  # Minimum housing activity
                (combined['housing_retail_ratio'] > 2) &  # Higher ratio than typical
                # Exclude areas that already had high retail provision historically
                ~combined.index.isin(retail_licenses.groupby('zip_code').size().nlargest(15).index)
            ].sort_values('housing_retail_ratio', ascending=False)
            
            self.retail_deficit_areas = self.retail_deficit_areas.reset_index()
            
            # Identify areas with new multi-family that didn't have it before
            self.new_multi_family_areas = combined[
                combined['is_new_multi_family'] == True
            ].sort_values('recent_multi_family', ascending=False).reset_index()
            
            # Save the results
            if not self.emerging_areas.empty:
                self.emerging_areas.to_csv(self.emerging_areas_file, index=False)
                logger.info(f"Emerging housing areas saved to {self.emerging_areas_file}")
            
            if not self.retail_deficit_areas.empty:
                self.retail_deficit_areas.to_csv(self.retail_deficit_file, index=False)
                logger.info(f"Areas with high retail deficit saved to {self.retail_deficit_file}")
            
            if not self.new_multi_family_areas.empty:
                self.new_multi_family_areas.to_csv(self.output_dir / 'new_multi_family_areas.csv', index=False)
                logger.info(f"New multi-family areas saved to {self.output_dir / 'new_multi_family_areas.csv'}")
            
            # Visualize the results
            self.visualize_emerging_areas()
            self.visualize_retail_deficit_areas()
            self.visualize_new_multi_family_areas()
            
            # Analyze 10-year permit growth
            self.analyze_ten_year_permit_growth(housing_permits)
            
            return self.emerging_areas, self.retail_deficit_areas
            
        except Exception as e:
            logger.error(f"Error analyzing emerging areas: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def analyze_ten_year_permit_growth(self, housing_permits):
        """
        Analyze areas with at least 20% increase in completed permit activity over 10-year period
        and calculate unit differences
        """
        try:
            logger.info("Analyzing 10-year permit growth (at least 20% increase)")
            
            # First 5 years period
            first_half = housing_permits[
                (housing_permits['year'] >= self.ten_year_start) & 
                (housing_permits['year'] < self.ten_year_start + 5)
            ]
            
            # Second 5 years period
            second_half = housing_permits[
                (housing_permits['year'] >= self.ten_year_end - 5) & 
                (housing_permits['year'] <= self.ten_year_end)
            ]
            
            # Count permits by ZIP code for each period
            first_half_counts = first_half.groupby('zip_code').size()
            second_half_counts = second_half.groupby('zip_code').size()
            
            # Create dataframe with both periods
            ten_year_growth = pd.DataFrame({
                'first_half_permits': first_half_counts,
                'second_half_permits': second_half_counts
            }).fillna(0)
            
            # Calculate absolute change in units
            ten_year_growth['unit_difference'] = ten_year_growth['second_half_permits'] - ten_year_growth['first_half_permits']
            
            # Calculate percentage change (handle division by zero)
            ten_year_growth['first_half_permits_adj'] = ten_year_growth['first_half_permits'].replace(0, 1)
            ten_year_growth['growth_pct'] = (
                (ten_year_growth['second_half_permits'] - ten_year_growth['first_half_permits']) / 
                ten_year_growth['first_half_permits_adj'] * 100
            ).round(1)
            
            # Filter for areas with at least 20% growth and minimum activity level
            self.ten_year_growth_areas = ten_year_growth[
                (ten_year_growth['growth_pct'] >= 20) &  # At least 20% growth
                (ten_year_growth['second_half_permits'] >= 10)  # Minimum recent activity
            ].sort_values('growth_pct', ascending=False)
            
            # Reset index to make ZIP code a column
            self.ten_year_growth_areas = self.ten_year_growth_areas.reset_index()
            
            # Save to CSV
            if not self.ten_year_growth_areas.empty:
                self.ten_year_growth_areas.to_csv(self.ten_year_growth_file, index=False)
                logger.info(f"10-year growth areas saved to {self.ten_year_growth_file}")
                
                # Visualize the results
                self.visualize_ten_year_growth()
            else:
                logger.warning("No areas found with significant 10-year growth")
                
            return self.ten_year_growth_areas
            
        except Exception as e:
            logger.error(f"Error analyzing 10-year permit growth: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_emerging_areas(self):
        """Create visualizations for emerging housing areas"""
        if self.emerging_areas is None or self.emerging_areas.empty:
            logger.warning("No emerging areas data to visualize")
            return
        
        try:
            # Top emerging housing areas
            plt.figure(figsize=(14, 8))
            top_areas = self.emerging_areas.head(10)
            
            # Create a bar chart
            ax = sns.barplot(
                x='zip_code', 
                y='growth_pct',
                data=top_areas,
                palette='viridis',
                hue='zip_code',
                legend=False
            )
            
            plt.title('Top ZIP Codes with Emerging Housing Development', fontsize=16)
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Housing Permit Growth (%)', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add data labels
            for i, row in enumerate(top_areas.itertuples()):
                ax.text(
                    i, row.growth_pct + 5, 
                    f"{row.growth_pct:.1f}%", 
                    ha='center', va='bottom',
                    fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'emerging_housing_areas.png', dpi=300)
            plt.close()
            
            # Housing growth vs. retail licenses
            plt.figure(figsize=(14, 8))
            
            # Create a scatter plot
            ax = sns.scatterplot(
                x='growth_pct',
                y='retail_licenses',
                size='housing_permits',
                hue='housing_retail_ratio',
                sizes=(50, 500),
                palette='viridis',
                data=self.emerging_areas
            )
            
            # Add ZIP code labels
            for i, row in enumerate(self.emerging_areas.itertuples()):
                ax.text(
                    row.growth_pct + 5, row.retail_licenses + 2, 
                    row.zip_code, 
                    fontsize=9
                )
            
            plt.title('Housing Growth vs. Retail Licenses in Emerging Areas', fontsize=16)
            plt.xlabel('Housing Permit Growth (%)', fontsize=14)
            plt.ylabel('Number of Retail Licenses', fontsize=14)
            plt.colorbar(ax.get_children()[0], label='Housing-to-Retail Ratio')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'housing_growth_vs_retail.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing emerging areas: {str(e)}")
    
    def visualize_retail_deficit_areas(self):
        """Create visualizations for areas with retail deficits"""
        if self.retail_deficit_areas is None or self.retail_deficit_areas.empty:
            logger.warning("No retail deficit areas data to visualize")
            return
        
        try:
            # Areas with highest housing-to-retail imbalance
            plt.figure(figsize=(14, 8))
            top_deficit_areas = self.retail_deficit_areas.head(10)
            
            # Create a bar chart
            ax = sns.barplot(
                x='zip_code', 
                y='housing_retail_ratio',
                data=top_deficit_areas,
                palette='plasma',
                hue='zip_code',
                legend=False
            )
            
            plt.title('ZIP Codes with Highest Housing-to-Retail Imbalance', fontsize=16)
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Housing-to-Retail Ratio', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add data labels
            for i, row in enumerate(top_deficit_areas.itertuples()):
                ax.text(
                    i, row.housing_retail_ratio + 0.2, 
                    f"{row.housing_retail_ratio:.1f}", 
                    ha='center', va='bottom',
                    fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'housing_retail_imbalance.png', dpi=300)
            plt.close()
            
            # Housing permits vs. retail licenses scatter plot
            plt.figure(figsize=(14, 8))
            
            # Create a scatter plot
            ax = sns.scatterplot(
                x='housing_permits',
                y='retail_licenses',
                size='housing_retail_ratio',
                hue='housing_retail_ratio',
                sizes=(50, 500),
                palette='plasma',
                data=self.retail_deficit_areas
            )
            
            # Add ZIP code labels
            for i, row in enumerate(self.retail_deficit_areas.itertuples()):
                ax.text(
                    row.housing_permits + 5, row.retail_licenses + 2, 
                    row.zip_code, 
                    fontsize=9
                )
            
            # Add a reference line for 1:1 ratio
            max_val = max(
                self.retail_deficit_areas['housing_permits'].max(),
                self.retail_deficit_areas['retail_licenses'].max()
            )
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='1:1 Ratio')
            
            plt.title('Housing Permits vs. Retail Licenses by ZIP Code', fontsize=16)
            plt.xlabel('Number of Housing Permits', fontsize=14)
            plt.ylabel('Number of Retail Licenses', fontsize=14)
            plt.colorbar(ax.get_children()[0], label='Housing-to-Retail Ratio')
            plt.legend(loc='upper left')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'housing_vs_retail_licenses.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing retail deficit areas: {str(e)}")
    
    def visualize_new_multi_family_areas(self):
        """Create visualizations for areas with new multi-family development"""
        if not hasattr(self, 'new_multi_family_areas') or self.new_multi_family_areas is None or self.new_multi_family_areas.empty:
            logger.warning("No new multi-family areas data to visualize")
            return
        
        try:
            # Top new multi-family areas
            plt.figure(figsize=(14, 8))
            top_areas = self.new_multi_family_areas.head(10)
            
            # Create a bar chart
            ax = sns.barplot(
                x='zip_code', 
                y='recent_multi_family',
                data=top_areas,
                palette='mako',
                hue='zip_code',
                legend=False
            )
            
            plt.title('Top ZIP Codes with New Multi-Family Development', fontsize=16)
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Recent Multi-Family Permits', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add data labels
            for i, row in enumerate(top_areas.itertuples()):
                ax.text(
                    i, row.recent_multi_family + 0.5, 
                    f"{int(row.recent_multi_family)}", 
                    ha='center', va='bottom',
                    fontweight='bold'
                )
                ax.text(
                    i, row.recent_multi_family/2,
                    f"Prior: {int(row.historical_multi_family)}",
                    ha='center', va='center',
                    color='white', fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'new_multi_family_areas.png', dpi=300)
            plt.close()
            
            # Multi-family to retail comparison
            plt.figure(figsize=(14, 8))
            
            # Create a scatter plot
            ax = sns.scatterplot(
                x='recent_multi_family',
                y='retail_licenses',
                size='multi_family_retail_ratio',
                hue='multi_family_retail_ratio',
                sizes=(50, 500),
                palette='mako',
                data=self.new_multi_family_areas
            )
            
            # Add ZIP code labels
            for i, row in enumerate(self.new_multi_family_areas.itertuples()):
                ax.text(
                    row.recent_multi_family + 0.5, row.retail_licenses + 0.5, 
                    row.zip_code, 
                    fontsize=9
                )
            
            plt.title('New Multi-Family Development vs. Retail Licenses', fontsize=16)
            plt.xlabel('Recent Multi-Family Permits', fontsize=14)
            plt.ylabel('Number of Retail Licenses', fontsize=14)
            plt.colorbar(ax.get_children()[0], label='Multi-Family to Retail Ratio')
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'multi_family_vs_retail.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing new multi-family areas: {str(e)}")
    
    def visualize_ten_year_growth(self):
        """Create visualization for areas with significant 10-year permit growth"""
        if self.ten_year_growth_areas is None or self.ten_year_growth_areas.empty:
            logger.warning("No 10-year growth areas data to visualize")
            return
        
        try:
            # Top areas by growth percentage
            plt.figure(figsize=(14, 8))
            top_areas = self.ten_year_growth_areas.head(15)
            
            # Create a bar chart
            ax = sns.barplot(
                x='zip_code', 
                y='growth_pct',
                data=top_areas,
                palette='magma',
                hue='zip_code',
                legend=False
            )
            
            plt.title(f'Top ZIP Codes with 10-Year Permit Growth (≥20%, {self.ten_year_start}-{self.ten_year_end})', fontsize=16)
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Permit Growth (%)', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add data labels
            for i, row in enumerate(top_areas.itertuples()):
                ax.text(
                    i, row.growth_pct + 5, 
                    f"{row.growth_pct:.1f}%", 
                    ha='center', va='bottom',
                    fontweight='bold'
                )
                ax.text(
                    i, row.growth_pct/2,
                    f"+{int(row.unit_difference)} units",
                    ha='center', va='center',
                    color='white', fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'ten_year_permit_growth.png', dpi=300)
            plt.close()
            
            # Unit difference chart
            plt.figure(figsize=(14, 8))
            top_unit_diff = self.ten_year_growth_areas.sort_values('unit_difference', ascending=False).head(15)
            
            # Create a bar chart
            ax = sns.barplot(
                x='zip_code', 
                y='unit_difference',
                data=top_unit_diff,
                palette='crest',
                hue='zip_code',
                legend=False
            )
            
            plt.title(f'Top ZIP Codes by Housing Unit Increase ({self.ten_year_start}-{self.ten_year_end})', fontsize=16)
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Unit Difference (Second 5 years - First 5 years)', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add data labels
            for i, row in enumerate(top_unit_diff.itertuples()):
                ax.text(
                    i, row.unit_difference + 2, 
                    f"+{int(row.unit_difference)}", 
                    ha='center', va='bottom',
                    fontweight='bold'
                )
                ax.text(
                    i, row.unit_difference/2,
                    f"{row.growth_pct:.1f}%",
                    ha='center', va='center',
                    color='white', fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'ten_year_unit_difference.png', dpi=300)
            plt.close()
            
            # Comparison chart showing both periods
            plt.figure(figsize=(16, 8))
            
            # Prepare data for grouped bar chart
            comparison_data = top_unit_diff.copy()
            
            # Create grouped bar chart
            index = np.arange(len(comparison_data))
            bar_width = 0.35
            
            first_bars = plt.bar(
                index, comparison_data['first_half_permits'], bar_width, 
                label=f'First 5 Years ({self.ten_year_start}-{self.ten_year_start+4})', 
                color='skyblue'
            )
            
            second_bars = plt.bar(
                index + bar_width, comparison_data['second_half_permits'], bar_width,
                label=f'Second 5 Years ({self.ten_year_end-4}-{self.ten_year_end})', 
                color='coral'
            )
            
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Number of Housing Permits', fontsize=14)
            plt.title(f'Housing Permit Comparison: First vs. Second 5-Year Period', fontsize=16)
            plt.xticks(index + bar_width / 2, comparison_data['zip_code'], rotation=45)
            plt.legend()
            
            # Add data labels
            for i, (first, second) in enumerate(zip(first_bars, second_bars)):
                plt.text(
                    first.get_x() + first.get_width()/2, 
                    first.get_height() + 2,
                    str(int(first.get_height())),
                    ha='center', va='bottom',
                    fontsize=9
                )
                plt.text(
                    second.get_x() + second.get_width()/2, 
                    second.get_height() + 2,
                    str(int(second.get_height())),
                    ha='center', va='bottom',
                    fontsize=9
                )
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'ten_year_permit_comparison.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing 10-year growth areas: {str(e)}")
    
    def analyze_downtown_vs_non_downtown(self):
        """Compare housing and retail development in downtown vs. non-downtown areas"""
        if self.permits_df is None or self.licenses_df is None:
            logger.error("Permit or license data missing. Cannot analyze downtown comparison.")
            return
        
        try:
            # Define downtown ZIP codes (example)
            downtown_zips = ['60601', '60602', '60603', '60604', '60605', '60606', '60607', '60616']
            
            # Filter housing permits
            housing_permits = self.permits_df[
                self.permits_df['permit_type'].str.contains('PERMIT - NEW CONSTRUCTION', case=False, na=False) &
                self.permits_df['work_description'].str.contains('RESIDENTIAL|APARTMENT|CONDO', case=False, na=False)
            ]
            
            # Filter retail licenses
            retail_licenses = self.licenses_df[
                self.licenses_df['license_description'].str.contains(
                    'RETAIL|FOOD|GROCERY|RESTAURANT|SHOP|STORE', 
                    case=False, na=False
                )
            ]
            
            # Group by downtown/non-downtown
            housing_permits['location'] = housing_permits['zip_code'].apply(
                lambda x: 'Downtown' if x in downtown_zips else 'Non-Downtown'
            )
            
            retail_licenses['location'] = retail_licenses['zip_code'].apply(
                lambda x: 'Downtown' if x in downtown_zips else 'Non-Downtown'
            )
            
            # Count permits and licenses
            housing_counts = housing_permits.groupby('location').size()
            retail_counts = retail_licenses.groupby('location').size()
            
            # Calculate ratios
            combined = pd.DataFrame({
                'housing_permits': housing_counts,
                'retail_licenses': retail_counts
            }).fillna(0)
            
            combined['retail_pct_of_housing'] = (
                combined['retail_licenses'] / combined['housing_permits'] * 100
            ).round(1)
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Create grouped bar chart
            index = np.arange(len(combined.index))
            bar_width = 0.35
            
            housing_bars = plt.bar(
                index, combined['housing_permits'], bar_width, 
                label='Housing Permits', color='skyblue'
            )
            
            retail_bars = plt.bar(
                index + bar_width, combined['retail_licenses'], bar_width,
                label='Retail Licenses', color='orange'
            )
            
            plt.xlabel('Location', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.title('Housing Permits and Retail Licenses: Downtown vs. Non-Downtown', fontsize=16)
            plt.xticks(index + bar_width / 2, combined.index)
            plt.legend()
            
            # Add data labels
            for i, bars in enumerate(zip(housing_bars, retail_bars)):
                housing_bar, retail_bar = bars
                plt.text(
                    housing_bar.get_x() + housing_bar.get_width()/2, 
                    housing_bar.get_height() + 5,
                    str(int(housing_bar.get_height())),
                    ha='center', va='bottom'
                )
                plt.text(
                    retail_bar.get_x() + retail_bar.get_width()/2, 
                    retail_bar.get_height() + 5,
                    str(int(retail_bar.get_height())),
                    ha='center', va='bottom'
                )
                
                # Add ratio
                plt.text(
                    index[i] + bar_width/2, 
                    max(housing_bar.get_height(), retail_bar.get_height()) + 50,
                    f"Retail is {combined['retail_pct_of_housing'].iloc[i]}% of Housing",
                    ha='center', va='bottom', fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'downtown_vs_non_downtown.png', dpi=300)
            plt.close()
            
            # Save the data
            combined.to_csv(self.output_dir / 'downtown_comparison.csv')
            logger.info(f"Downtown comparison saved to {self.output_dir / 'downtown_comparison.csv'}")
            
            return combined
            
        except Exception as e:
            logger.error(f"Error analyzing downtown comparison: {str(e)}")
            return None
    
    def identify_lowest_retail_provision(self):
        """
        Identify ZIP codes with the lowest retail-to-housing ratio
        compared to their neighborhood type
        """
        if self.retail_deficit_areas is None or self.retail_deficit_areas.empty:
            logger.warning("No retail deficit data to analyze")
            return
        
        try:
            # Calculate retail as percentage of housing permits
            self.retail_deficit_areas['retail_pct_of_housing'] = (
                self.retail_deficit_areas['retail_licenses'] / 
                self.retail_deficit_areas['housing_permits'] * 100
            ).round(1)
            
            # Get the lowest retail provision areas
            lowest_retail = self.retail_deficit_areas.sort_values('retail_pct_of_housing').head(10)
            
            # Create visualization
            plt.figure(figsize=(14, 8))
            
            # Create a bar chart
            ax = sns.barplot(
                x='zip_code', 
                y='retail_pct_of_housing',
                data=lowest_retail,
                palette='viridis',
                hue='zip_code',
                legend=False
            )
            
            plt.title('ZIP Codes with Lowest Retail-to-Housing Ratio', fontsize=16)
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Retail as % of Housing Permits', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add data labels
            for i, row in enumerate(lowest_retail.itertuples()):
                ax.text(
                    i, row.retail_pct_of_housing + 1, 
                    f"{row.retail_pct_of_housing:.1f}%", 
                    ha='center', va='bottom',
                    fontweight='bold'
                )
            
            # Add reference line for downtown benchmark
            if self.retail_deficit_areas['retail_pct_of_housing'].max() > 0:
                plt.axhline(
                    y=50, color='r', linestyle='--', 
                    label='Downtown Benchmark (50%)'
                )
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'lowest_retail_provision.png', dpi=300)
            plt.close()
            
            # Save the data
            lowest_retail.to_csv(self.output_dir / 'lowest_retail_provision.csv', index=False)
            logger.info(f"Lowest retail provision areas saved to {self.output_dir / 'lowest_retail_provision.csv'}")
            
            return lowest_retail
            
        except Exception as e:
            logger.error(f"Error identifying lowest retail provision: {str(e)}")
            return None

def main():
    analyzer = EmergingAreasAnalyzer()
    analyzer.analyze_emerging_housing_areas()
    analyzer.analyze_downtown_vs_non_downtown()
    analyzer.identify_lowest_retail_provision()
    
    logger.info("Analysis of emerging housing areas completed")

if __name__ == "__main__":
    main() 