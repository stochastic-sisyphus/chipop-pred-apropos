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

from src.data_processing.processor import DataProcessor
from src.models.population_model import PopulationModel
from src.models.retail_model import RetailModel
from src.models.economic_model import EconomicModel
from src.visualization.visualizer import Visualizer
from src.utils.helpers import calculate_growth_rate, calculate_confidence_interval
from src.config import settings

logger = logging.getLogger(__name__)

class TenYearGrowthReport:
    """Generates the ten-year growth analysis report for Chicago."""
    
    def __init__(self):
        """Initialize the report generator with necessary components."""
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
        
    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare all necessary data for the report."""
        logger.info("Loading and preparing data for report generation")
        
        try:
            # Load processed data
            historical_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')
            current_data = pd.read_csv(settings.PROCESSED_DATA_DIR / 'merged_dataset.csv')
            
            # Ensure year column exists and is properly formatted
            for df in [historical_data, current_data]:
                if 'year' not in df.columns:
                    # Try to extract year from date columns if available
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    if date_cols:
                        df['year'] = pd.to_datetime(df[date_cols[0]]).dt.year
                    else:
                        raise ValueError("No year or date column found in dataset")
                else:
                    # Ensure year is numeric
                    df['year'] = pd.to_numeric(df['year'], errors='coerce')
            
            # Load projections if available, otherwise create empty DataFrame
            try:
                projections = pd.read_csv(settings.PREDICTIONS_DIR / 'scenario_predictions.csv')
            except FileNotFoundError:
                logger.warning("No scenario predictions found, using empty DataFrame")
                projections = pd.DataFrame(columns=['year'])
            
            # Add required features for population modeling
            required_features = ['month', 'development_density']
            for df in [historical_data, current_data, projections]:
                if 'month' not in df.columns:
                    df['month'] = 1  # Default to January if no month specified
                if 'development_density' not in df.columns:
                    if 'housing_units' in df.columns and 'total_population' in df.columns:
                        df['development_density'] = df['housing_units'] / df['total_population'].replace(0, np.nan)
                    else:
                        df['development_density'] = 0.0  # Default value if required columns missing
                
                # Fill NaN values with appropriate defaults
                df['development_density'] = df['development_density'].fillna(0.0)
                df['month'] = df['month'].fillna(1)
            
            return {
                'historical': historical_data,
                'current': current_data,
                'projections': projections
            }
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
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
        
    def generate_report(self) -> None:
        """Generate the complete report."""
        try:
            logger.info("Starting report generation")
            
            # Load and prepare data
            logger.info("Loading and preparing data for report generation")
            data = self.load_and_prepare_data()
            
            # Analyze historical trends
            logger.info("Analyzing historical trends")
            historical_trends = self.analyze_historical_trends(data['historical'])
            
            # Analyze current conditions
            logger.info("Analyzing current conditions")
            current_analysis = self.generate_current_analysis(data['current'])
            
            # Generate projections
            logger.info("Generating future projections")
            projections = self.generate_projections(data['historical'])
            
            # Analyze impacts
            logger.info("Analyzing growth impacts")
            impacts = self.analyze_impacts(current_analysis, projections)
            
            # Identify growth areas
            logger.info("Identifying growth areas")
            growth_areas = self.identify_growth_areas(data)
            
            # Generate recommendations
            logger.info("Generating recommendations")
            recommendations = self.generate_recommendations(impacts, growth_areas)
            
            # Generate visualizations
            logger.info("Generating report visualizations")
            self._generate_report_visualizations(data, historical_trends, projections)
            
            # Prepare template context
            # Patch: Ensure total_growth, gdp_growth, cagr, employment_change are always present for templates
            def safe_get(d, keys, default=0.0):
                try:
                    for k in keys:
                        d = d[k]
                    return d if d is not None else default
                except Exception:
                    return default

            # Patch historical_trends for required keys
            if 'population' in historical_trends:
                if 'total_growth' not in historical_trends['population']:
                    historical_trends['population']['total_growth'] = 0.0
                if 'cagr' not in historical_trends['population']:
                    historical_trends['population']['cagr'] = 0.0
            else:
                historical_trends['population'] = {'total_growth': 0.0, 'cagr': 0.0}
            if 'economic' in historical_trends:
                if 'gdp_growth' not in historical_trends['economic']:
                    gdp_growth = safe_get(historical_trends['economic'], ['gdp', 'total_growth'], 0.0)
                    historical_trends['economic']['gdp_growth'] = gdp_growth
                if 'employment_change' not in historical_trends['economic']:
                    employment_change = safe_get(historical_trends['economic'], ['labor', 'total_growth'], 0.0)
                    historical_trends['economic']['employment_change'] = employment_change
            else:
                historical_trends['economic'] = {'gdp_growth': 0.0, 'employment_change': 0.0}

            context = {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'historical_trends': historical_trends,
                'current_analysis': current_analysis,
                'projections': {
                    'period_start': projections.get('period_start', ''),
                    'period_end': projections.get('period_end', ''),
                    'population': {
                        'scenarios': projections.get('scenarios', []),
                        'baseline': projections.get('baseline', {}),
                        'summary': projections.get('summary', {})
                    }
                },
                'impacts': impacts,
                'growth_areas': growth_areas,
                'recommendations': recommendations,
                'analysis_period': {
                    'start': '' if data['historical'].empty else data['historical']['year'].min(),
                    'end': '' if data['historical'].empty else data['historical']['year'].max()
                },
                'zip_summaries': [],
                'data_quality': {
                    'population_moe': 5.0,  # Default margin of error
                    'economic_update_frequency': 'annually',
                    'development_data_date': datetime.now().strftime('%Y-%m-%d')
                },
                'analysis_results': {},
            }
            
            # Generate each report
            for name, template_file in self.templates.items():
                try:
                    template = self.template_env.get_template(template_file)
                    report_content = template.render(**context)
                    
                    # Save generated report
                    output_path = self.output_paths[name]
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w') as f:
                        f.write(report_content)
                    
                    logger.info(f"Generated {name} report at {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error generating {name} report: {str(e)}")
                    continue
            
            logger.info("Report generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            raise
            
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
                    'growth': self._calculate_total_growth(data, 'total_construction_cost')
                }
            }
            
            # Generate summary
            summary = {
                'total_permits': data['total_permits'].sum(),
                'total_value': data['total_construction_cost'].sum(),
                'residential_share': (data['residential_permits'].sum() / data['total_permits'].sum() * 100),
                'commercial_share': (data['commercial_permits'].sum() / data['total_permits'].sum() * 100),
                'growth_rate': self._calculate_cagr(data, 'total_permits')
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
                'total_growth': self._calculate_total_growth(data, 'total_permits'),
                'cagr': self._calculate_cagr(data, 'total_permits'),
                'year_over_year': self._calculate_yoy_changes(data, 'total_permits')
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
                    'growth': self._calculate_total_growth(data, 'residential_permits'),
                    'cagr': self._calculate_cagr(data, 'residential_permits')
                },
                'commercial': {
                    'growth': self._calculate_total_growth(data, 'commercial_permits'),
                    'cagr': self._calculate_cagr(data, 'commercial_permits')
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
                'value_growth': self._calculate_total_growth(data, 'total_construction_cost'),
                'value_cagr': self._calculate_cagr(data, 'total_construction_cost')
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
                'total_growth': self._calculate_total_growth(data, 'gdp'),
                'cagr': self._calculate_cagr(data, 'gdp'),
                'year_over_year': self._calculate_yoy_changes(data, 'gdp')
            }
            
            # Analyze income trends
            income_trends = {
                'per_capita_growth': self._calculate_total_growth(data, 'per_capita_income'),
                'per_capita_cagr': self._calculate_cagr(data, 'per_capita_income'),
                'household_growth': self._calculate_total_growth(data, 'median_household_income'),
                'household_cagr': self._calculate_cagr(data, 'median_household_income')
            }
            
            # Analyze labor force trends
            labor_trends = {
                'total_growth': self._calculate_total_growth(data, 'labor_force'),
                'cagr': self._calculate_cagr(data, 'labor_force'),
                'unemployment_rate': data['unemployment_rate'].mean(),
                'unemployment_change': self._calculate_total_growth(data, 'unemployment_rate')
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
                'labor_force_growth': self._calculate_total_growth(data, 'labor_force'),
                'labor_force_cagr': self._calculate_cagr(data, 'labor_force')
            }
            
            # Calculate unemployment metrics
            unemployment_metrics = {
                'current_rate': data.sort_values('year')['unemployment_rate'].iloc[-1],
                'avg_rate': data['unemployment_rate'].mean(),
                'rate_change': self._calculate_total_growth(data, 'unemployment_rate')
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
        try:
            # Use median_household_income instead of detailed income brackets
            if 'median_household_income' not in data.columns:
                return {
                    'distribution': None,
                    'summary': {
                        'status': 'No income data available'
                    }
                }

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
        
    def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Analyze market conditions."""
        try:
            # Analyze price trends
            price_trends = {
                'housing_price_growth': self._calculate_total_growth(data, 'housing_price'),
                'retail_price_growth': self._calculate_total_growth(data, 'retail_price'),
                'price_correlation': data['housing_price'].corr(data['retail_price'])
            }
            
            # Analyze market activity
            market_activity = {
                'transaction_volume': self._analyze_transaction_volume(data),
                'market_velocity': self._analyze_market_velocity(data),
                'price_volatility': self._calculate_price_volatility(data)
            }
            
            # Analyze market balance
            market_balance = {
                'supply_demand_ratio': self._calculate_supply_demand_ratio(data),
                'market_absorption': self._calculate_market_absorption(data),
                'price_elasticity': self._calculate_price_elasticity(data)
            }
            
            return {
                'price_trends': price_trends,
                'activity': market_activity,
                'balance': market_balance
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            raise
            
    def _analyze_economic_development_impact(self, data: pd.DataFrame) -> Dict:
        """Analyze economic development impacts."""
        try:
            # Calculate direct impacts
            direct_impacts = {
                'job_creation': self._calculate_job_creation(data),
                'tax_revenue': self._calculate_tax_revenue(data),
                'economic_output': self._calculate_economic_output(data)
            }
            
            # Calculate multiplier effects
            multiplier_effects = {
                'indirect_jobs': self._calculate_indirect_jobs(data),
                'induced_spending': self._calculate_induced_spending(data),
                'total_economic_impact': self._calculate_total_economic_impact(data)
            }
            
            # Analyze spatial distribution
            spatial_impacts = {
                'impact_by_zip': self._calculate_zip_level_impacts(data),
                'impact_concentration': self._calculate_impact_concentration(data)
            }
            
            return {
                'direct': direct_impacts,
                'multiplier': multiplier_effects,
                'spatial': spatial_impacts
            }
            
        except Exception as e:
            logger.error(f"Error analyzing economic development impact: {str(e)}")
            raise
            
    def _generate_short_term_forecast(self, data: pd.DataFrame) -> Dict:
        """Generate short-term economic forecast."""
        try:
            # Generate baseline and scenario forecasts
            baseline = self._generate_baseline_forecast(data, period='short')
            scenarios = {
                'optimistic': self._generate_scenario_forecast(data, 'optimistic', period='short'),
                'pessimistic': self._generate_scenario_forecast(data, 'pessimistic', period='short')
            }
            
            return {
                'baseline': baseline,
                'scenarios': scenarios,
                'uncertainty': self._calculate_forecast_uncertainty(baseline, scenarios)
            }
            
        except Exception as e:
            logger.error(f"Error generating short-term forecast: {str(e)}")
            raise
            
    def _generate_medium_term_forecast(self, data: pd.DataFrame) -> Dict:
        """Generate medium-term economic forecast."""
        try:
            return self._extracted_from__generate_long_term_forecast_5(data, 'medium')
        except Exception as e:
            logger.error(f"Error generating medium-term forecast: {str(e)}")
            raise
            
    def _generate_long_term_forecast(self, data: pd.DataFrame) -> Dict:
        """Generate long-term economic forecast."""
        try:
            return self._extracted_from__generate_long_term_forecast_5(data, 'long')
        except Exception as e:
            logger.error(f"Error generating long-term forecast: {str(e)}")
            raise

    # TODO Rename this here and in `_generate_medium_term_forecast` and `_generate_long_term_forecast`
    def _extracted_from__generate_long_term_forecast_5(self, data, period):
        baseline = self._generate_baseline_forecast(data, period=period)
        scenarios = {
            'optimistic': self._generate_scenario_forecast(
                data, 'optimistic', period=period
            ),
            'pessimistic': self._generate_scenario_forecast(
                data, 'pessimistic', period=period
            ),
        }
        uncertainty = self._calculate_forecast_uncertainty(baseline, scenarios)
        return {
            'baseline': baseline,
            'scenarios': scenarios,
            'uncertainty': uncertainty,
        }
        
    def _analyze_current_population(self, data: pd.DataFrame) -> Dict:
        """Analyze current population conditions."""
        try:
            # Get most recent year's data
            latest_year = data['year'].max()
            current_data = data[data['year'] == latest_year]

            # Current population metrics
            current_metrics = {
                'total': current_data['total_population'].sum() if 'total_population' in data.columns else None,
                'population_density': self._calculate_population_density(data),
                'avg_household_income': current_data['median_household_income'].mean() if 'median_household_income' in data.columns else None,
                'total_labor_force': current_data['labor_force'].sum() if 'labor_force' in data.columns else None
            }
            
            # Current demographic composition
            demographics = {
                'population_growth': self._calculate_total_growth(data, 'total_population') if 'total_population' in data.columns else None,
                'income_growth': self._calculate_total_growth(data, 'median_household_income') if 'median_household_income' in data.columns else None,
                'labor_force_growth': self._calculate_total_growth(data, 'labor_force') if 'labor_force' in data.columns else None
            }
            
            # Spatial distribution
            spatial = {
                'density_metrics': self._calculate_population_density(data),
                'income_distribution': self._analyze_income_distribution(data) if 'median_household_income' in data.columns else None,
                'growth_patterns': self._analyze_growth_rate_trend(data, 'total_population') if 'total_population' in data.columns else None
            }
            
            # Remove None values from dictionaries
            current_metrics = {k: v for k, v in current_metrics.items() if v is not None}
            demographics = {k: v for k, v in demographics.items() if v is not None}
            spatial = {k: v for k, v in spatial.items() if v is not None}
            
            return {
                'metrics': current_metrics,
                'demographics': demographics,
                'spatial': spatial
            }
            
        except Exception as e:
            logger.error(f"Error analyzing current population: {str(e)}")
            raise
        
    def _analyze_current_market(self, data: pd.DataFrame) -> Dict:
        """Analyze current market conditions."""
        # Implementation details
        pass
        
    def _analyze_current_economic(self, data: pd.DataFrame) -> Dict:
        """Analyze current economic conditions."""
        # Implementation details
        pass
        
    def _analyze_current_development(self, data: pd.DataFrame) -> Dict:
        """Analyze current development conditions."""
        # Implementation details
        pass
        
    def _generate_population_projections(self, data: pd.DataFrame) -> Dict:
        """Generate population projections."""
        try:
            # Initialize population model if not already done
            if not hasattr(self, 'population_model'):
                from src.models.population_model import PopulationModel
                self.population_model = PopulationModel()

            # Create a copy to avoid modifying original data
            train_data = data.copy()

            # Add required features before training
            if 'month' not in train_data.columns:
                train_data['month'] = train_data['year'].astype(str).str[-2:].astype(int)  # Extract month from year
            if 'development_density' not in train_data.columns:
                train_data['development_density'] = train_data['housing_units'] / train_data['total_population'] if 'housing_units' in train_data.columns else 0.0

            # Ensure model is trained with all required features
            feature_list = [col for col in train_data.columns if col != 'total_population']
            if not self.population_model.train(train_data, feature_list=feature_list):
                logger.error("Failed to train population model")
                return {}

            # Generate base predictions using the same features
            predictions_df = self.population_model.generate_predictions()
            if predictions_df is None or predictions_df.empty:
                logger.error("Failed to generate base predictions")
                return {}

            # Store predictions for scenario generation
            self.population_model.predictions = predictions_df

            # Generate baseline projections
            baseline = self.population_model.generate_baseline_projection(df=predictions_df)

            # Generate scenario-based projections
            scenarios = {}
            if not baseline.empty:
                scenarios = {
                    'high_growth': self.population_model.generate_high_growth_scenario(train_data),
                    'low_growth': self.population_model.generate_low_growth_scenario(train_data),
                    'moderate_growth': self.population_model.generate_moderate_growth_scenario(train_data)
                }

            # Analyze projection components
            components = {
                'natural_growth': self._analyze_natural_growth_component(train_data),
                'migration': self._analyze_migration_component(train_data),
                'development_impact': self._analyze_development_impact(train_data)
            }

            # Generate ZIP-level projections
            zip_projections = self._generate_zip_level_projections(train_data)

            # Generate summary
            summary = self._generate_projection_summary(
                baseline,
                scenarios,
                components,
                zip_projections
            )

            return {
                'baseline': baseline,
                'scenarios': scenarios,
                'components': components,
                'zip_level': zip_projections,
                'summary': summary,
                'period_start': None if train_data.empty else train_data['year'].min(),
                'period_end': None if train_data.empty else train_data['year'].max()
            }
        except Exception as e:
            logger.error(f"Error generating population projections: {str(e)}")
            return {}
            
    def _calculate_total_growth(self, data: pd.DataFrame, column: str) -> float:
        """Calculate total growth over the entire period."""
        try:
            yearly_totals = data.groupby('year')[column].sum()
            if len(yearly_totals) < 2:
                return 0.0
                
            start_value = yearly_totals.iloc[0]
            end_value = yearly_totals.iloc[-1]
            
            if start_value == 0 or pd.isna(start_value) or pd.isna(end_value):
                return 0.0
                
            return ((end_value - start_value) / start_value) * 100
            
        except Exception as e:
            logger.error(f"Error calculating total growth: {str(e)}")
            return 0.0
            
    def _calculate_cagr(self, data: pd.DataFrame, column: str) -> float:
        """Calculate Compound Annual Growth Rate."""
        try:
            yearly_totals = data.groupby('year')[column].sum()
            if len(yearly_totals) < 2:
                return 0.0
                
            years = len(yearly_totals) - 1
            start_value = yearly_totals.iloc[0]
            end_value = yearly_totals.iloc[-1]
            
            if start_value <= 0 or pd.isna(start_value) or pd.isna(end_value):
                return 0.0
                
            return (((end_value / start_value) ** (1/years)) - 1) * 100
            
        except Exception as e:
            logger.error(f"Error calculating CAGR: {str(e)}")
            return 0.0
            
    def _calculate_yoy_changes(self, data: pd.DataFrame, column: str) -> Dict[int, float]:
        """Calculate year-over-year changes."""
        try:
            yearly_totals = data.groupby('year')[column].sum()
            yoy_changes = {}
            
            for i in range(1, len(yearly_totals)):
                year = yearly_totals.index[i]
                prev_year = yearly_totals.index[i-1]
                current_value = yearly_totals[year]
                prev_value = yearly_totals[prev_year]
                
                if prev_value == 0 or pd.isna(prev_value) or pd.isna(current_value):
                    yoy_changes[year] = 0.0
                else:
                    yoy_changes[year] = ((current_value - prev_value) / prev_value) * 100
                    
            return yoy_changes
            
        except Exception as e:
            logger.error(f"Error calculating YoY changes: {str(e)}")
            return {}
            
    def _analyze_growth_rate_trend(self, data: pd.DataFrame, column: str) -> Dict:
        """Analyze trend in growth rates."""
        try:
            # Calculate year-over-year changes
            yoy_changes = self._calculate_yoy_changes(data, column)
            
            # Calculate trend statistics
            values = list(yoy_changes.values())
            
            return {
                'yoy_changes': yoy_changes,
                'mean_growth': np.mean(values),
                'median_growth': np.median(values),
                'std_growth': np.std(values),
                'trend_direction': 'increasing' if np.polyfit(range(len(values)), values, 1)[0] > 0 
                                 else 'decreasing',
                'volatility': np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else np.inf
            }
            
        except Exception as e:
            logger.error(f"Error analyzing growth rate trend: {str(e)}")
            raise
            
    def _analyze_age_distribution(self, data: pd.DataFrame) -> Dict:
        """Analyze age distribution of the population."""
        try:
            # Define age columns we expect to have
            age_columns = ['age_0_17', 'age_18_34', 'age_35_64', 'age_65_plus']

            # Check which age columns are actually available
            available_age_cols = [col for col in age_columns if col in data.columns]

            if not available_age_cols:
                logger.warning("No age distribution columns found in data")
                # Return a simplified analysis based on total population
                latest_year = data['year'].max()
                current_data = data[data['year'] == latest_year]

                return {
                    'total_population': current_data['total_population'].sum(),
                    'population_growth': self._calculate_cagr(data, 'total_population'),
                    'distribution': None,
                    'trends': self._analyze_growth_rate_trend(data, 'total_population')
                }

            # If we have age columns, perform detailed analysis
            latest_year = data['year'].max()
            current_data = data[data['year'] == latest_year]

            # Calculate distribution for available age groups
            total_pop = current_data[available_age_cols].sum().sum()
            distribution = {}

            for col in available_age_cols:
                age_group = col.replace('age_', '').replace('_', '-')
                distribution[age_group] = {
                    'count': current_data[col].sum(),
                    'percentage': (current_data[col].sum() / total_pop) * 100
                }

            growth_rates = {
                col.replace('age_', '').replace('_', '-'): self._calculate_cagr(
                    data, col
                )
                for col in available_age_cols
            }
            return {
                'total_population': total_pop,
                'distribution': distribution,
                'growth_rates': growth_rates,
                'trends': self._analyze_growth_rate_trend(data, 'total_population')
            }

        except Exception as e:
            logger.error(f"Error analyzing age distribution: {str(e)}")
            return {
                'total_population': None,
                'distribution': None,
                'growth_rates': None,
                'trends': None
            }
            
    def _analyze_household_composition(self, data: pd.DataFrame) -> Dict:
        """Analyze changes in household composition."""
        try:
            household_cols = ['single_person_households', 'family_households', 
                            'non_family_households']

            return self._extracted_from__analyze_household_composition_7(
                data, household_cols
            )
        except Exception as e:
            logger.error(f"Error analyzing household composition: {str(e)}")
            raise

    # TODO Rename this here and in `_analyze_age_distribution` and `_analyze_household_composition`
    def _extracted_from__analyze_household_composition_7(self, data, arg1):
        start_year = data['year'].min()
        end_year = data['year'].max()
        start_dist = data[data['year'] == start_year][arg1].sum()
        end_dist = data[data['year'] == end_year][arg1].sum()
        changes = ((end_dist - start_dist) / start_dist * 100).to_dict()
        return {
            'start_distribution': start_dist.to_dict(),
            'end_distribution': end_dist.to_dict(),
            'percent_changes': changes,
        }
            
    def _generate_development_projections(self, data: pd.DataFrame) -> Dict:
        """Generate development projections."""
        # Implementation details
        pass
        
    def _generate_economic_projections(self, data: pd.DataFrame) -> Dict:
        """Generate economic projections."""
        # Implementation details
        pass
        
    def _generate_market_projections(self, data: pd.DataFrame) -> Dict:
        """Generate market projections."""
        # Implementation details
        pass
        
    def _analyze_housing_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze housing market impacts."""
        # Implementation details
        pass
        
    def _analyze_retail_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze retail sector impacts."""
        # Implementation details
        pass
        
    def _analyze_economic_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze economic impacts."""
        # Implementation details
        pass
        
    def _analyze_community_impacts(self, current: Dict, projections: Dict) -> Dict:
        """Analyze community impacts."""
        # Implementation details
        pass
        
    def _identify_primary_growth_centers(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """Identify primary growth centers."""
        # Implementation details
        pass
        
    def _identify_emerging_markets(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """Identify emerging markets."""
        # Implementation details
        pass
        
    def _identify_stabilization_zones(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """Identify stabilization zones."""
        # Implementation details
        pass
        
    def _generate_strategic_priorities(self, impacts: Dict, growth_areas: Dict) -> List[str]:
        """Generate strategic priorities."""
        # Implementation details
        pass
        
    def _generate_implementation_steps(self, impacts: Dict, growth_areas: Dict) -> List[str]:
        """Generate implementation steps."""
        # Implementation details
        pass
        
    def _generate_support_requirements(self, impacts: Dict, growth_areas: Dict) -> List[str]:
        """Generate support requirements."""
        # Implementation details
        pass
        
    def _generate_report_visualizations(self,
                                      data: Dict[str, pd.DataFrame],
                                      historical_trends: Dict,
                                      projections: Dict) -> None:
        """Generate visualizations for the report."""
        logger.info("Generating report visualizations")
        
        viz_dir = settings.VISUALIZATIONS_DIR
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Population trends over time
            total_pop_by_year = data['historical'].groupby('year')['total_population'].sum()
            self.visualizer.create_trend_plot(
                total_pop_by_year,
                'Year',
                'Total Population',
                'Chicago Population Trends',
                viz_dir / 'population_trends.png'
            )
            
            # Growth rates by ZIP code
            growth_by_zip = data['historical'].groupby('zip_code')['total_population'].pct_change().mean()
            self.visualizer.create_map_plot(
                growth_by_zip,
                column='growth_rate',
                title='Population Growth Rate by ZIP Code',
                output_path=str(viz_dir / 'growth_rates_map.png'),
                cmap='RdYlBu',
                legend_title='Average Annual Growth Rate (%)'
            )
            
            # Projected population by ZIP code
            if isinstance(projections, pd.DataFrame) and 'year' in projections.columns:
                proj_2030 = projections[projections['year'] == 2030].set_index('zip_code')['projected_population']
                self.visualizer.create_map_plot(
                    proj_2030,
                    column='projected_population',
                    title='Projected 2030 Population by ZIP Code',
                    output_path=str(viz_dir / 'population_2030_map.png'),
                    cmap='viridis',
                    legend_title='Projected Population'
                )
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def _populate_report_template(self,
                                template: str,
                                historical_trends: Dict,
                                current_analysis: Dict,
                                projections: Dict,
                                impacts: Dict,
                                growth_areas: Dict,
                                recommendations: Dict) -> str:
        """Populate the report template with analyzed data."""
        logger.info("Populating report template")
        
        try:
            # Format date
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Replace template sections with actual content
            report = template.replace('{date}', current_date)
            
            # Add content for each section
            # Historical Analysis
            report = self._add_historical_analysis(report, historical_trends)
            
            # Current Conditions
            report = self._add_current_conditions(report, current_analysis)
            
            # Growth Projections
            report = self._add_growth_projections(report, projections)
            
            # Impact Analysis
            report = self._add_impact_analysis(report, impacts)
            
            # Growth Areas
            report = self._add_growth_areas(report, growth_areas)
            
            # Recommendations
            report = self._add_recommendations(report, recommendations)
            
            return report
            
        except Exception as e:
            logger.error(f"Error populating report template: {str(e)}")
            raise
            
    def _add_historical_analysis(self, report: str, trends: Dict) -> str:
        """Add historical analysis content to the report."""
        # Implementation details
        pass
        
    def _add_current_conditions(self, report: str, current: Dict) -> str:
        """Add current conditions content to the report."""
        # Implementation details
        pass
        
    def _add_growth_projections(self, report: str, projections: Dict) -> str:
        """Add growth projections content to the report."""
        # Implementation details
        pass
        
    def _add_impact_analysis(self, report: str, impacts: Dict) -> str:
        """Add impact analysis content to the report."""
        # Implementation details
        pass
        
    def _add_growth_areas(self, report: str, areas: Dict) -> str:
        """Add growth areas content to the report."""
        # Implementation details
        pass
        
    def _add_recommendations(self, report: str, recommendations: Dict) -> str:
        """Add recommendations content to the report."""
        # Implementation details
        pass
        
    def _analyze_market_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze market trends."""
        try:
            # Calculate construction cost trends
            cost_trends = {
                'total_value': data['total_construction_cost'].sum(),
                'mean_value': data['total_construction_cost'].mean(),
                'median_value': data['total_construction_cost'].median(),
                'value_growth': self._calculate_total_growth(data, 'total_construction_cost'),
                'value_cagr': self._calculate_cagr(data, 'total_construction_cost')
            }
            
            # Calculate permit trends
            permit_trends = {
                'total_permits': data['total_permits'].sum(),
                'residential_share': (data['residential_permits'].sum() / data['total_permits'].sum() * 100),
                'commercial_share': (data['commercial_permits'].sum() / data['total_permits'].sum() * 100),
                'permit_growth': self._calculate_total_growth(data, 'total_permits'),
                'permit_cagr': self._calculate_cagr(data, 'total_permits')
            }
            
            # Calculate market indicators
            market_indicators = {
                'population_growth': self._calculate_total_growth(data, 'total_population'),
                'income_growth': self._calculate_total_growth(data, 'median_household_income'),
                'gdp_growth': self._calculate_total_growth(data, 'gdp'),
                'unemployment_rate': data['unemployment_rate'].mean()
            }
            
            return {
                'construction_costs': cost_trends,
                'permits': permit_trends,
                'indicators': market_indicators
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market trends: {str(e)}")
            raise
        
    def _calculate_population_density(self, data: pd.DataFrame) -> Dict:
        """Calculate population density metrics."""
        try:
            # Calculate basic density metrics
            density_metrics = {
                'total_population': data['total_population'].sum(),
                'mean_density': data['total_population'].mean(),
                'median_density': data['total_population'].median()
            }
            
            # Calculate density distribution
            distribution = {
                'percentiles': {
                    '25th': data['total_population'].quantile(0.25),
                    '50th': data['total_population'].quantile(0.50),
                    '75th': data['total_population'].quantile(0.75),
                    '90th': data['total_population'].quantile(0.90)
                }
            }
            
            # Calculate growth metrics
            growth_metrics = {
                'total_growth': self._calculate_total_growth(data, 'total_population'),
                'cagr': self._calculate_cagr(data, 'total_population'),
                'year_over_year': self._calculate_yoy_changes(data, 'total_population')
            }
            
            return {
                'metrics': density_metrics,
                'distribution': distribution,
                'growth': growth_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating population density: {str(e)}")
            raise

    def _generate_baseline_forecast(self, data: pd.DataFrame, period: str = 'short') -> Dict:
        """Generate baseline forecast for the specified period."""
        try:
            logger.info(f"Generating {period}-term baseline forecast...")

            # Define forecast periods
            period_lengths = {
                'short': 3,  # 3 years
                'medium': 5,  # 5 years
                'long': 10   # 10 years
            }

            if period not in period_lengths:
                logger.error(f"Invalid period '{period}'. Must be one of {list(period_lengths.keys())}")
                return {}

            # Get the most recent year's data
            latest_year = data['year'].max()
            forecast_years = period_lengths[period]

            # Prepare baseline data
            baseline_data = data[data['year'] == latest_year].copy()
            baseline_data['forecast_year'] = baseline_data['year'] + forecast_years

            # Calculate growth rates from historical data
            historical_growth = {
                'population': self._calculate_cagr(data, 'total_population'),
                'income': self._calculate_cagr(data, 'median_household_income'),
                'employment': self._calculate_cagr(data, 'labor_force'),
                'gdp': self._calculate_cagr(data, 'gdp')
            }

            # Apply growth rates to generate forecast
            for metric, growth_rate in historical_growth.items():
                if growth_rate is not None:
                    col_name = metric
                    if col_name in baseline_data.columns:
                        baseline_data[f'{col_name}_forecast'] = baseline_data[col_name] * (1 + growth_rate) ** forecast_years

            # Calculate forecast metrics
            forecast_metrics = {
                'population_forecast': baseline_data['total_population_forecast'].sum() if 'total_population_forecast' in baseline_data.columns else None,
                'income_forecast': baseline_data['median_household_income_forecast'].mean() if 'median_household_income_forecast' in baseline_data.columns else None,
                'employment_forecast': baseline_data['labor_force_forecast'].sum() if 'labor_force_forecast' in baseline_data.columns else None,
                'gdp_forecast': baseline_data['gdp_forecast'].sum() if 'gdp_forecast' in baseline_data.columns else None
            }

            # Calculate growth metrics
            growth_metrics = {
                'population_growth': historical_growth['population'],
                'income_growth': historical_growth['income'],
                'employment_growth': historical_growth['employment'],
                'gdp_growth': historical_growth['gdp']
            }

            return {
                'metrics': forecast_metrics,
                'growth_rates': growth_metrics,
                'forecast_year': latest_year + forecast_years,
                'base_year': latest_year,
                'period': period
            }

        except Exception as e:
            logger.error(f"Error generating baseline forecast: {str(e)}")
            return {}

    def _analyze_natural_growth_component(self, data: pd.DataFrame) -> Dict:
        """Analyze natural population growth (births - deaths)."""
        try:
            # Get most recent year's data
            latest_year = data['year'].max()
            current_data = data[data['year'] == latest_year]
            
            # Calculate natural growth metrics
            metrics = {
                'total_population': current_data['total_population'].sum(),
                'growth_rate': self._calculate_cagr(data, 'total_population'),
                'year_over_year_change': self._calculate_yoy_changes(data, 'total_population')
            }
            
            return {
                'metrics': metrics,
                'trends': self._analyze_growth_rate_trend(data, 'total_population'),
                'distribution': self._analyze_age_distribution(data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing natural growth component: {str(e)}")
            return {}

    def _analyze_migration_component(self, data: pd.DataFrame) -> Dict:
        """Analyze population changes due to migration."""
        try:
            # Calculate migration metrics
            migration_data = data.copy()
            migration_data['population_change'] = migration_data.groupby('zip_code')['total_population'].diff()
            
            # Remove natural growth component (if available)
            if 'natural_growth' in migration_data.columns:
                migration_data['net_migration'] = migration_data['population_change'] - migration_data['natural_growth']
            else:
                migration_data['net_migration'] = migration_data['population_change']
            
            metrics = {
                'total_net_migration': migration_data['net_migration'].sum(),
                'average_annual_migration': migration_data.groupby('year')['net_migration'].mean(),
                'migration_rate': migration_data['net_migration'].mean() / migration_data['total_population'].mean()
            }
            
            return {
                'metrics': metrics,
                'trends': self._analyze_migration_trends(migration_data),
                'patterns': self._analyze_migration_patterns(migration_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing migration component: {str(e)}")
            return {}

    def _analyze_development_impact(self, data: pd.DataFrame) -> Dict:
        """Analyze how development affects population growth."""
        try:
            # Calculate development impact metrics
            metrics = {
                'total_new_units': data['housing_units'].sum(),
                'avg_household_size': data['total_population'].mean() / data['housing_units'].mean(),
                'development_correlation': data.groupby('year').agg({
                    'total_population': 'sum',
                    'housing_units': 'sum'
                }).corr().iloc[0,1]
            }
            
            return {
                'metrics': metrics,
                'trends': self._analyze_development_trends(data),
                'impact': self._calculate_development_impact(data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing development impact: {str(e)}")
            return {}

    def _calculate_development_impact(self, data: pd.DataFrame) -> Dict:
        """Calculate the impact of development on population growth."""
        try:
            development_metrics = [
                'housing_units',
                'residential_permits',
                'total_construction_cost'
            ]

            correlations = {
                metric: data['total_population'].corr(data[metric])
                for metric in development_metrics
                if metric in data.columns
            }
            # Calculate elasticity (% change in population / % change in development)
            elasticities = {}
            for metric in development_metrics:
                if metric in data.columns:
                    pct_change_pop = data['total_population'].pct_change()
                    pct_change_dev = data[metric].pct_change()
                    elasticities[metric] = (pct_change_pop / pct_change_dev).mean()

            return {
                'correlations': correlations,
                'elasticities': elasticities,
                'significance': self._calculate_impact_significance(correlations)
            }

        except Exception as e:
            logger.error(f"Error calculating development impact: {str(e)}")
            return {}

    def _calculate_impact_significance(self, correlations: Dict[str, float]) -> Dict[str, str]:
        """Calculate the significance of development impacts."""
        try:
            significance = {}
            for metric, correlation in correlations.items():
                if abs(correlation) > 0.7:
                    significance[metric] = 'Strong'
                elif abs(correlation) > 0.4:
                    significance[metric] = 'Moderate'
                else:
                    significance[metric] = 'Weak'
            return significance
            
        except Exception as e:
            logger.error(f"Error calculating impact significance: {str(e)}")
            return {}

    def _generate_zip_level_projections(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Generate ZIP code level population projections."""
        try:
            logger.info("Generating ZIP code level projections...")
            
            # Get unique ZIP codes
            zip_codes = data['zip_code'].unique()
            
            # Initialize projections dictionary
            projections = {}
            
            for zip_code in zip_codes:
                # Get ZIP code specific data
                zip_data = data[data['zip_code'] == zip_code].copy()
                
                if zip_data.empty:
                    continue
                
                # Calculate growth metrics
                growth_metrics = {
                    'population_growth': self._calculate_cagr(zip_data, 'total_population'),
                    'income_growth': self._calculate_cagr(zip_data, 'median_household_income'),
                    'employment_growth': self._calculate_cagr(zip_data, 'labor_force')
                }
                
                # Get current metrics
                latest_year = zip_data['year'].max()
                current_metrics = zip_data[zip_data['year'] == latest_year].iloc[0].to_dict()
                
                # Generate projections
                projection_years = range(latest_year + 1, latest_year + 11)
                projected_values = {}
                
                for year in projection_years:
                    years_forward = year - latest_year
                    
                    # Project each metric
                    projected_values[year] = {
                        'population': current_metrics['total_population'] * (1 + growth_metrics['population_growth']) ** years_forward,
                        'income': current_metrics['median_household_income'] * (1 + growth_metrics['income_growth']) ** years_forward,
                        'employment': current_metrics['labor_force'] * (1 + growth_metrics['employment_growth']) ** years_forward
                    }
                
                # Store projections for this ZIP code
                projections[zip_code] = {
                    'current_metrics': current_metrics,
                    'growth_metrics': growth_metrics,
                    'projected_values': projected_values
                }
            
            return projections
            
        except Exception as e:
            logger.error(f"Error generating ZIP level projections: {str(e)}")
            return {}

    def _generate_projection_summary(self, baseline: pd.DataFrame, scenarios: Dict, components: Dict, zip_projections: Dict) -> Dict:
        """Generate a summary of all projections."""
        try:
            logger.info("Generating projection summary...")

            # Summarize baseline projections
            baseline_summary = {}
            if not baseline.empty:
                baseline_summary = {
                    'total_population': baseline['baseline_population'].sum(),
                    'avg_growth_rate': baseline['baseline_population'].pct_change().mean(),
                    'final_year_population': baseline[baseline['year'] == baseline['year'].max()]['baseline_population'].sum()
                }

            scenario_summary = {
                name: {
                    'total_population': scenario_df['predicted_population'].sum(),
                    'avg_growth_rate': scenario_df['predicted_population']
                    .pct_change()
                    .mean(),
                    'final_year_population': scenario_df[
                        scenario_df['year'] == scenario_df['year'].max()
                    ]['predicted_population'].sum(),
                }
                for name, scenario_df in scenarios.items()
                if not scenario_df.empty
            }
            # Summarize components
            component_summary = {
                'natural_growth': components.get('natural_growth', {}).get('metrics', {}),
                'migration': components.get('migration', {}).get('metrics', {}),
                'development': components.get('development_impact', {}).get('metrics', {})
            }

            # Summarize ZIP-level projections
            zip_summary = {}
            for zip_code, projection in zip_projections.items():
                if 'projected_values' in projection:
                    final_year = max(projection['projected_values'].keys())
                    zip_summary[zip_code] = {
                        'current_population': projection['current_metrics'].get('total_population'),
                        'projected_population': projection['projected_values'][final_year].get('population'),
                        'growth_rate': projection['growth_metrics'].get('population_growth')
                    }

            # Generate overall summary
            return {
                'baseline': baseline_summary,
                'scenarios': scenario_summary,
                'components': component_summary,
                'zip_level': zip_summary,
                'period_start': None if baseline.empty else baseline['year'].min(),
                'period_end': None if baseline.empty else baseline['year'].max()
            }

        except Exception as e:
            logger.error(f"Error generating projection summary: {str(e)}")
            return {}

    def _analyze_migration_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze migration trends over time."""
        try:
            trends = data.groupby('year').agg({
                'net_migration': ['sum', 'mean', 'std'],
                'total_population': 'mean'
            })
            
            return {
                'annual_net_migration': trends['net_migration']['sum'].to_dict(),
                'migration_rate': (trends['net_migration']['sum'] / trends['total_population']).to_dict(),
                'migration_volatility': trends['net_migration']['std'].to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing migration trends: {str(e)}")
            return {}

    def _analyze_migration_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze spatial patterns of migration."""
        try:
            patterns = data.groupby('zip_code').agg({
                'net_migration': ['sum', 'mean'],
                'total_population': 'mean'
            })
            
            # Identify high migration areas
            high_migration = patterns[
                patterns['net_migration']['sum'] > patterns['net_migration']['sum'].quantile(0.75)
            ].index.tolist()
            
            # Identify low migration areas
            low_migration = patterns[
                patterns['net_migration']['sum'] < patterns['net_migration']['sum'].quantile(0.25)
            ].index.tolist()
            
            return {
                'high_migration_areas': high_migration,
                'low_migration_areas': low_migration,
                'migration_intensity': (patterns['net_migration']['sum'] / patterns['total_population']).to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing migration patterns: {str(e)}")
            return {} 