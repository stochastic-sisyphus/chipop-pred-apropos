"""
Module for generating the ten-year growth analysis report for Chicago.
This module analyzes historical data and generates projections to create
a comprehensive report on Chicago's growth patterns and opportunities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import jinja2
from jinja2 import Template

from src.data_processing.processor import DataProcessor
from src.models.population_model import PopulationModel
from src.models.retail_model import RetailModel
from src.models.economic_model import EconomicModel
from src.visualization.visualizer import Visualizer
from src.utils.helpers import calculate_growth_rate, calculate_confidence_interval, resolve_column_name
from src.config import settings
from src.config.column_alias_map import column_aliases

logger = logging.getLogger(__name__)

class TenYearGrowthReport:
    """Generates the ten-year growth analysis report for Chicago."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize the report generator with necessary components."""
        self.data_dir = data_dir or settings.DATA_PROCESSED_DIR
        self.df = None
        self.report_data = {}
        
        self.data_processor = DataProcessor()
        self.population_model = PopulationModel()
        self.retail_model = RetailModel()
        self.economic_model = EconomicModel()
        self.visualizer = Visualizer()
        
        # Set up Jinja2 environment for templates
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(settings.REPORT_TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Define template and output paths
        self.templates = {
            'main': 'ten_year_growth_analysis.md',
            'executive_summary': 'EXECUTIVE_SUMMARY.md',
            'chicago_summary': 'chicago_zip_summary.md',
            'economic_impact': 'economic_impact_analysis.md',
            'housing_retail': 'housing_retail_balance_report.md',
            'retail_deficit': 'retail_deficit_analysis.md',
            'void_analysis': 'void_analysis.md'
        }
        
        # Create output paths
        self.output_paths = {
            name: settings.REPORTS_DIR / template_file
            for name, template_file in self.templates.items()
        }
        
    def _calculate_total_growth(self, df: pd.DataFrame, value_column: str, time_column: str = 'year', group_by: str = 'zip_code') -> pd.DataFrame:
        """
        Calculate total growth rate between first and last period for each group.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_column (str): Column containing values to calculate growth for
            time_column (str): Column containing time periods
            group_by (str): Column to group by (e.g. zip_code)
            
        Returns:
            pd.DataFrame: Dataframe with total growth rates
        """
        try:
            # Sort by time column within each group
            df = df.sort_values([group_by, time_column])

            # Get first and last values for each group
            first_values = df.groupby(group_by)[value_column].first()
            last_values = df.groupby(group_by)[value_column].last()

            # Calculate total growth rate
            total_growth = (last_values - first_values) / first_values

            return pd.DataFrame(
                {
                    group_by: total_growth.index,
                    f'{value_column}_total_growth': total_growth.values,
                }
            )
        except Exception as e:
            logger.error(f"Error calculating total growth: {str(e)}")
            return pd.DataFrame()
        
    def load_and_prepare_data(self):
        """Load and prepare data for the report."""
        try:
            # Load merged dataset
            df = pd.read_csv(settings.MERGED_DATA_PATH)
            if 'zip_code' not in df.columns:
                logger.error("zip_code column missing from merged dataset!")
                raise ValueError("zip_code column missing from merged dataset!")
            
            # Load processed datasets
            population_data = pd.read_csv(self.data_dir / "census_processed.csv")
            economic_data = pd.read_csv(self.data_dir / "economic_processed.csv")
            permits_data = pd.read_csv(self.data_dir / "permits_processed.csv")
            business_data = pd.read_csv(self.data_dir / "business_licenses_processed.csv")
            
            # Merge datasets
            self.df = population_data.merge(economic_data, on=['zip_code', 'year'], how='left')
            self.df = self.df.merge(permits_data, on=['zip_code', 'year'], how='left')
            self.df = self.df.merge(business_data, on=['zip_code', 'year'], how='left')
            
            # Fill missing values
            self.df = self.df.fillna(0)
            
            logger.info("Data loaded and prepared successfully")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
        
    def analyze_historical_trends(self):
        """Analyze historical trends for the report."""
        try:
            results = {}
            
            # Population growth analysis
            if self.df is not None:
                pop_growth = self._calculate_total_growth(
                    self.df, 
                    'total_population'
                )
                results['population'] = {
                    'total_growth': pop_growth['total_population_total_growth'].mean(),
                    'annual_growth': pop_growth['total_population_total_growth'].mean() / 100,
                    'by_zip': pop_growth.to_dict('records')
                }
            
                # GDP growth analysis
                gdp_growth = self._calculate_total_growth(
                    self.df,
                    'gdp'
                )
                results['economic'] = {
                    'gdp_growth': gdp_growth['gdp_total_growth'].mean(),
                    'annual_gdp_growth': gdp_growth['gdp_total_growth'].mean() / 100,
                    'by_zip': gdp_growth.to_dict('records')
                }
            
                # Permit activity analysis
                permit_growth = self._calculate_total_growth(
                    self.df,
                    'total_permits'
                )
                results['permits'] = {
                    'total_growth': permit_growth['total_permits_total_growth'].mean(),
                    'annual_growth': permit_growth['total_permits_total_growth'].mean() / 100,
                    'by_zip': permit_growth.to_dict('records')
                }
                
                # Business activity analysis
                if 'active_licenses' in self.df.columns:
                    business_growth = self._calculate_total_growth(
                        self.df,
                        'active_licenses'
                    )
                    results['business'] = {
                        'total_growth': business_growth['active_licenses_total_growth'].mean(),
                        'annual_growth': business_growth['active_licenses_total_growth'].mean() / 100,
                        'by_zip': business_growth.to_dict('records')
                    }
            
            logger.info("Historical trends analyzed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing historical trends: {str(e)}")
            return {}
        
    def generate_report(self, output_path: Path = None) -> bool:
        """
        Generate the ten year growth report.
        
        Args:
            output_path: Path to save report (default: settings.REPORTS_DIR)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting report generation")

            # Load data
            if not self.load_and_prepare_data():
                logger.error("Failed to load and prepare data")
                return False

            # Analyze trends
            trends = self.analyze_historical_trends()
            if not trends:
                logger.error("Failed to analyze historical trends")
                return False

            # Set output path
            if output_path is None:
                output_path = settings.REPORTS_DIR / "ten_year_growth_analysis.md"

            # Generate report content
            content = [
                "# Chicago Ten Year Growth Analysis",
                f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}\n",
            ]

            # Population section
            if 'population' in trends:
                content.append("## Population Growth")
                content.append(f"- Total Growth: {trends['population']['total_growth']:.1%}")
                content.append(f"- Annual Growth Rate: {trends['population']['annual_growth']:.1%}\n")

            # Economic section
            if 'economic' in trends:
                content.append("## Economic Growth")
                content.append(f"- GDP Growth: {trends['economic']['gdp_growth']:.1%}")
                content.append(f"- Annual GDP Growth: {trends['economic']['annual_gdp_growth']:.1%}\n")

            # Development section
            if 'permits' in trends:
                content.append("## Development Activity")
                content.append(f"- Permit Growth: {trends['permits']['total_growth']:.1%}")
                content.append(f"- Annual Permit Growth: {trends['permits']['annual_growth']:.1%}\n")

            # Business section
            if 'business' in trends:
                content.append("## Business Activity")
                content.append(f"- Business Growth: {trends['business']['total_growth']:.1%}")
                content.append(f"- Annual Business Growth: {trends['business']['annual_growth']:.1%}\n")

            # Save report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("\n".join(content))

            logger.info(f"Report saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return False
        
    def analyze_historical_trends(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze historical trends from 2010-2023."""
        logger.info("Analyzing historical trends")
        
        return {
            'population': self._analyze_population_trends(data),
            'development': self._analyze_development_trends(data),
            'economic': self._analyze_economic_trends(data),
            'market': self._analyze_market_trends(data)
        }
        
    def generate_current_analysis(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze current conditions (2023)."""
        logger.info("Analyzing current conditions")
        
        return {
            'population': self._analyze_current_population(data),
            'market': self._analyze_current_market(data),
            'economic': self._analyze_current_economic(data),
            'development': self._analyze_current_development(data)
        }
        
    def generate_projections(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Generate and analyze projections for 2024-2033."""
        logger.info("Generating future projections")
        
        return {
            'population': self._generate_population_projections(data),
            'development': self._generate_development_projections(data),
            'economic': self._generate_economic_projections(data),
            'market': self._generate_market_projections(data)
        }
        
    def analyze_impacts(self, current: Dict, projections: Dict) -> Dict[str, Dict]:
        """Analyze impacts of projected growth."""
        logger.info("Analyzing growth impacts")
        
        return {
            'housing': self._analyze_housing_impacts(current, projections),
            'retail': self._analyze_retail_impacts(current, projections),
            'economic': self._analyze_economic_impacts(current, projections),
            'community': self._analyze_community_impacts(current, projections)
        }
        
    def identify_growth_areas(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Identify and categorize growth areas."""
        logger.info("Identifying growth areas")
        
        return {
            'primary': self._identify_primary_growth_centers(data),
            'emerging': self._identify_emerging_markets(data),
            'stabilization': self._identify_stabilization_zones(data)
        }
        
    def generate_recommendations(self, 
                               impacts: Dict,
                               growth_areas: Dict) -> Dict[str, List[str]]:
        """Generate strategic recommendations."""
        logger.info("Generating recommendations")
        
        return {
            'strategic': self._generate_strategic_priorities(impacts, growth_areas),
            'implementation': self._generate_implementation_steps(impacts, growth_areas),
            'support': self._generate_support_requirements(impacts, growth_areas)
        }
        
    def _analyze_population_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze historical population trends and calculate growth rates."""
        # Calculate year-over-year growth rates
        data = data.sort_values('year')
        data['growth_rate'] = data.groupby('zip_code')['total_population'].pct_change()
        
        # Calculate average growth rate by ZIP code
        avg_growth_by_zip = data.groupby('zip_code')['growth_rate'].mean()
        
        # Identify high-growth and declining areas
        high_growth_zips = avg_growth_by_zip[avg_growth_by_zip > avg_growth_by_zip.mean() + avg_growth_by_zip.std()].index
        declining_zips = avg_growth_by_zip[avg_growth_by_zip < 0].index
        
        return {
            'avg_growth_rates': avg_growth_by_zip,
            'high_growth_areas': high_growth_zips.tolist(),
            'declining_areas': declining_zips.tolist()
        }
        
    def _analyze_development_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze historical development trends."""
        try:
            # Analyze permit activity
            permit_trends = {
                'total_permits': self._analyze_permit_volume(data),
                'permit_types': self._analyze_permit_types(data),
                'permit_values': self._analyze_permit_values(data)
            }
            
            # Analyze development patterns
            development_patterns = {
                'spatial_distribution': self._analyze_development_distribution(data)
            }
            
            # Analyze market indicators
            market_indicators = {
                'price_trends': {
                    'median_value': data['total_construction_cost'].median(),
                    'mean_value': data['total_construction_cost'].mean(),
                    'growth': self._calculate_total_growth(data, 'total_construction_cost')['gdp_total_growth'].mean()
                }
            }
            
            # Generate summary
            summary = {
                'total_permits': data['total_permits'].sum(),
                'total_value': data['total_construction_cost'].sum(),
                'residential_share': (data['residential_permits'].sum() / data['total_permits'].sum() * 100),
                'commercial_share': (data['commercial_permits'].sum() / data['total_permits'].sum() * 100),
                'growth_rate': self._calculate_total_growth(data, 'total_permits')['total_permits_total_growth'].mean()
            }
            
            return {
                'permits': permit_trends,
                'patterns': development_patterns,
                'market': market_indicators,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error analyzing development trends: {str(e)}")
            raise
            
    def _analyze_permit_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze building permit volume trends."""
        try:
            # Calculate yearly permit totals
            yearly_permits = data.groupby('year')['total_permits'].sum()
            
            # Calculate growth metrics
            volume_metrics = {
                'total_growth': self._calculate_total_growth(data, 'total_permits')['total_permits_total_growth'].mean(),
                'cagr': self._calculate_total_growth(data, 'total_permits')['total_permits_total_growth'].mean() / 100,
                'year_over_year': self._calculate_total_growth(data, 'total_permits')['total_permits_total_growth'].mean()
            }
            
            # Analyze seasonality
            seasonality = self._analyze_permit_seasonality(data)
            
            return {
                'yearly_totals': yearly_permits.to_dict(),
                'growth_metrics': volume_metrics,
                'seasonality': seasonality
            }
            
        except Exception as e:
            logger.error(f"Error analyzing permit volume: {str(e)}")
            raise
            
    def _analyze_permit_types(self, data: pd.DataFrame) -> Dict:
        """Analyze distribution of permit types."""
        try:
            # Calculate residential vs commercial distribution
            permit_types = pd.DataFrame({
                'residential': data['residential_permits'].sum(),
                'commercial': data['commercial_permits'].sum()
            }, index=[0])
            
            total_permits = permit_types.sum(axis=1)[0]
            type_distribution = (permit_types.iloc[0] / total_permits * 100).to_dict()
            
            # Analyze trends by type
            type_trends = {
                'residential': {
                    'growth': self._calculate_total_growth(data, 'residential_permits')['residential_permits_total_growth'].mean(),
                    'cagr': self._calculate_total_growth(data, 'residential_permits')['residential_permits_total_growth'].mean() / 100
                },
                'commercial': {
                    'growth': self._calculate_total_growth(data, 'commercial_permits')['commercial_permits_total_growth'].mean(),
                    'cagr': self._calculate_total_growth(data, 'commercial_permits')['commercial_permits_total_growth'].mean() / 100
                }
            }
            
            return {
                'distribution': type_distribution,
                'trends': type_trends
            }
            
        except Exception as e:
            logger.error(f"Error analyzing permit types: {str(e)}")
            raise
            
    def _analyze_permit_values(self, data: pd.DataFrame) -> Dict:
        """Analyze trends in permit values."""
        try:
            # Calculate value metrics
            value_metrics = {
                'total_value': data['total_construction_cost'].sum(),
                'mean_value': data['total_construction_cost'].mean(),
                'median_value': data['total_construction_cost'].median(),
                'value_growth': self._calculate_total_growth(data, 'total_construction_cost')['gdp_total_growth'].mean(),
                'value_cagr': self._calculate_total_growth(data, 'total_construction_cost')['gdp_total_growth'].mean() / 100
            }
            
            # Analyze value distribution
            value_distribution = {
                'percentiles': {
                    '25th': data['total_construction_cost'].quantile(0.25),
                    '50th': data['total_construction_cost'].quantile(0.50),
                    '75th': data['total_construction_cost'].quantile(0.75),
                    '90th': data['total_construction_cost'].quantile(0.90)
                }
            }
            
            # Analyze value trends by type
            value_by_type = {
                'residential': {
                    'total': data['residential_construction_cost'].sum(),
                    'mean': data['residential_construction_cost'].mean(),
                    'median': data['residential_construction_cost'].median()
                },
                'commercial': {
                    'total': data['commercial_construction_cost'].sum(),
                    'mean': data['commercial_construction_cost'].mean(),
                    'median': data['commercial_construction_cost'].median()
                }
            }
            
            return {
                'metrics': value_metrics,
                'distribution': value_distribution,
                'by_type': value_by_type
            }
            
        except Exception as e:
            logger.error(f"Error analyzing permit values: {str(e)}")
            raise
            
    def _analyze_development_distribution(self, data: pd.DataFrame) -> Dict:
        """Analyze spatial distribution of development."""
        try:
            # Analyze by ZIP code
            zip_metrics = data.groupby('zip_code').agg({
                'total_permits': 'sum',
                'total_construction_cost': 'sum'
            }).to_dict()
            
            # Calculate development density
            data['development_density'] = data['total_permits'] / data['total_population']
            
            # Identify development hotspots (top 10% by permit density)
            density_threshold = data['development_density'].quantile(0.9)
            hotspots = data[data['development_density'] > density_threshold]['zip_code'].unique().tolist()
            
            # Analyze clustering (adjacent ZIP codes with high activity)
            clusters = []  # This would require spatial data to properly implement
            
            return {
                'zip_metrics': zip_metrics,
                'hotspots': hotspots,
                'clusters': clusters
            }
            
        except Exception as e:
            logger.error(f"Error analyzing development distribution: {str(e)}")
            raise
            
    def _analyze_permit_seasonality(self, data: pd.DataFrame) -> Dict:
        """Analyze seasonality in permit activity."""
        try:
            # Extract month from year
            data['month'] = pd.to_datetime(data['year'].astype(str), format='%Y').dt.month
            
            # Calculate monthly averages
            monthly_avg = data.groupby('month')['total_permits'].mean()
            
            # Calculate seasonal indices
            yearly_avg = monthly_avg.mean()
            seasonal_indices = (monthly_avg / yearly_avg * 100).to_dict()
            
            # Identify peak and trough months
            peak_month = monthly_avg.idxmax()
            trough_month = monthly_avg.idxmin()
            
            return {
                'seasonal_indices': seasonal_indices,
                'peak_month': peak_month,
                'trough_month': trough_month,
                'seasonality_strength': (monthly_avg.max() - monthly_avg.min()) / yearly_avg
            }
            
        except Exception as e:
            logger.error(f"Error analyzing permit seasonality: {str(e)}")
            raise
        
    def _analyze_economic_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze economic trends."""
        try:
            # Analyze GDP trends
            gdp_trends = {
                'total_growth': self._calculate_total_growth(data, 'gdp')['gdp_total_growth'].mean(),
                'cagr': self._calculate_total_growth(data, 'gdp')['gdp_total_growth'].mean() / 100,
                'year_over_year': self._calculate_total_growth(data, 'gdp')['gdp_total_growth'].mean()
            }
            
            # Analyze income trends
            income_trends = {
                'per_capita_growth': self._calculate_total_growth(data, 'per_capita_income')['per_capita_income_total_growth'].mean(),
                'per_capita_cagr': self._calculate_total_growth(data, 'per_capita_income')['per_capita_income_total_growth'].mean() / 100,
                'household_growth': self._calculate_total_growth(data, 'median_household_income')['median_household_income_total_growth'].mean(),
                'household_cagr': self._calculate_total_growth(data, 'median_household_income')['median_household_income_total_growth'].mean() / 100
            }
            
            # Analyze labor force trends
            labor_trends = {
                'total_growth': self._calculate_total_growth(data, 'labor_force')['labor_force_total_growth'].mean(),
                'cagr': self._calculate_total_growth(data, 'labor_force')['labor_force_total_growth'].mean() / 100,
                'unemployment_rate': data['unemployment_rate'].mean(),
                'unemployment_change': self._calculate_total_growth(data, 'unemployment_rate')['unemployment_rate_total_growth'].mean()
            }
            
            return {
                'gdp': gdp_trends,
                'income': income_trends,
                'labor': labor_trends
            }
            
        except Exception as e:
            logger.error(f"Error analyzing economic trends: {str(e)}")
            raise
            
    def _analyze_employment_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze employment trends."""
        try:
            # Calculate labor force metrics
            labor_metrics = {
                'total_labor_force': data['labor_force'].sum(),
                'labor_force_growth': self._calculate_total_growth(data, 'labor_force')['labor_force_total_growth'].mean(),
                'labor_force_cagr': self._calculate_total_growth(data, 'labor_force')['labor_force_total_growth'].mean() / 100
            }
            
            # Calculate unemployment metrics
            unemployment_metrics = {
                'current_rate': data.sort_values('year')['unemployment_rate'].iloc[-1],
                'avg_rate': data['unemployment_rate'].mean(),
                'rate_change': self._calculate_total_growth(data, 'unemployment_rate')['unemployment_rate_total_growth'].mean()
            }
            
            # Calculate participation rate if population data available
            if 'total_population' in data.columns:
                participation_rate = (data['labor_force'] / data['total_population'] * 100).mean()
            else:
                participation_rate = None
            
            return {
                'labor_force': labor_metrics,
                'unemployment': unemployment_metrics,
                'participation_rate': participation_rate
            }
            
        except Exception as e:
            logger.error(f"Error analyzing employment trends: {str(e)}")
            raise
        
    def _analyze_income_distribution(self, data: pd.DataFrame) -> Dict:
        """Analyze income distribution using available data."""
        if 'median_household_income' not in data.columns:
            return {
                'distribution': None,
                'summary': {
                    'status': 'No income data available'
                }
            }
        try:
            current_year = data['year'].max()
            current_data = data[data['year'] == current_year]

            # Calculate income quintiles
            quintiles = current_data['median_household_income'].quantile([0.2, 0.4, 0.6, 0.8, 1.0])

            distribution = {
                'low_income': len(current_data[current_data['median_household_income'] <= quintiles[0.2]]),
                'lower_middle': len(current_data[(current_data['median_household_income'] > quintiles[0.2]) & 
                                               (current_data['median_household_income'] <= quintiles[0.4])]),
                'middle': len(current_data[(current_data['median_household_income'] > quintiles[0.4]) & 
                                         (current_data['median_household_income'] <= quintiles[0.6])]),
                'upper_middle': len(current_data[(current_data['median_household_income'] > quintiles[0.6]) & 
                                               (current_data['median_household_income'] <= quintiles[0.8])]),
                'high_income': len(current_data[current_data['median_household_income'] > quintiles[0.8]])
            }

            # Calculate summary statistics
            summary = {
                'median': current_data['median_household_income'].median(),
                'mean': current_data['median_household_income'].mean(),
                'std': current_data['median_household_income'].std(),
                'min': current_data['median_household_income'].min(),
                'max': current_data['median_household_income'].max()
            }

            return {
                'distribution': distribution,
                'summary': summary,
                'quintiles': quintiles.to_dict()
            }

        except Exception as e:
            logger.error(f"Error analyzing income distribution: {str(e)}")
            return {
                'distribution': None,
                'summary': {
                    'status': f'Error: {str(e)}'
                }
            }