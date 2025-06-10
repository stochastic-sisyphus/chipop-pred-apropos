"""
Zoning Impact Model for Chicago ZIP Codes

This model analyzes how zoning regulations affect housing development:
- Identifies areas where zoning suppresses housing potential
- Calculates maximum buildable units by zone
- Assesses up-zoning and down-zoning impacts
- Provides binary and scale measurements of zoning constraints
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import json

logger = logging.getLogger(__name__)

class ZoningImpactModel:
    """
    Analyzes zoning impact on housing development potential.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("output/models/zoning_impact")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Key neighborhoods with known zoning issues
        self.zoning_constrained_areas = {
            '60637': 'Woodlawn',  # Affordable housing ordinance impact
            '60607': 'West Loop',  # Up-zoned for high-rise
            '60654': 'River North',  # Up-zoned for high-rise
            '60614': 'Lincoln Park',  # Restrictive zoning
            '60657': 'Lakeview',  # Mixed zoning
        }
        
        # Zoning categories and their typical constraints
        self.zoning_types = {
            'RS': {'name': 'Residential Single-Unit', 'max_units_per_acre': 8, 'max_height': 30},
            'RT': {'name': 'Residential Two-Flat', 'max_units_per_acre': 20, 'max_height': 38},
            'RM': {'name': 'Residential Multi-Unit', 'max_units_per_acre': 65, 'max_height': 47},
            'DX': {'name': 'Downtown Mixed-Use', 'max_units_per_acre': 400, 'max_height': 'unlimited'},
            'PMD': {'name': 'Planned Manufacturing', 'max_units_per_acre': 0, 'max_height': 'n/a'}
        }
        
        self.results = {}
    
    def analyze_zoning_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how zoning affects housing development potential.
        """
        logger.info("Analyzing zoning impact on housing development")
        
        analysis_results = []
        
        for zip_code in data['zip_code'].unique():
            zip_data = data[data['zip_code'] == zip_code]
            
            # Calculate actual vs potential development
            actual_metrics = self._calculate_actual_development(zip_data)
            potential_metrics = self._calculate_potential_development(zip_data, zip_code)
            
            # Zoning constraint analysis
            constraint_score = self._calculate_constraint_score(
                actual_metrics, potential_metrics
            )
            
            # Binary measurement (is zoning limiting development?)
            is_constrained = constraint_score > 0.3  # 30% below potential
            
            # Scale measurement
            constraint_level = self._categorize_constraint_level(constraint_score)
            
            # Special area considerations
            special_factors = self._identify_special_factors(zip_code, zip_data)
            
            analysis_results.append({
                'zip_code': zip_code,
                'neighborhood': self.zoning_constrained_areas.get(zip_code, 'Other'),
                'actual_units_per_acre': actual_metrics['units_per_acre'],
                'potential_units_per_acre': potential_metrics['units_per_acre'],
                'actual_recent_permits': actual_metrics['recent_permits'],
                'potential_annual_permits': potential_metrics['annual_capacity'],
                'constraint_score': constraint_score,
                'is_constrained': is_constrained,
                'constraint_level': constraint_level,
                'zoning_efficiency': 1 - constraint_score,
                'special_factors': special_factors,
                'development_gap': potential_metrics['units_per_acre'] - actual_metrics['units_per_acre'],
                'permit_utilization': actual_metrics['recent_permits'] / potential_metrics['annual_capacity'] if potential_metrics['annual_capacity'] > 0 else 0
            })
        
        results_df = pd.DataFrame(analysis_results)
        self.results['zoning_analysis'] = results_df
        
        return results_df
    
    def _calculate_actual_development(self, zip_data: pd.DataFrame) -> dict:
        """Calculate actual development metrics."""
        metrics = {
            'units_per_acre': 0,
            'recent_permits': 0,
            'permit_growth': 0,
            'avg_unit_size': 0
        }
        
        # Housing density
        if 'housing_units' in zip_data.columns:
            current_units = zip_data['housing_units'].iloc[-1] if len(zip_data) > 0 else 0
            # Assume average ZIP code is ~10 square miles
            metrics['units_per_acre'] = current_units / (10 * 640)  # 640 acres per sq mile
        
        # Recent permit activity
        if 'permit_year' in zip_data.columns:
            recent_years = max(zip_data['permit_year']) - 2 if len(zip_data) > 0 else 0
            recent_permits = zip_data[zip_data['permit_year'] >= recent_years]
            
            if len(recent_permits) > 0:
                metrics['recent_permits'] = len(recent_permits)
                
                # Calculate growth trend
                permit_counts = zip_data.groupby('permit_year').size()
                if len(permit_counts) > 2:
                    years = list(range(len(permit_counts)))
                    counts = permit_counts.values
                    slope = np.polyfit(years, counts, 1)[0]
                    metrics['permit_growth'] = slope / np.mean(counts) if np.mean(counts) > 0 else 0
                
                # Average unit size
                if 'unit_count' in recent_permits.columns:
                    metrics['avg_unit_size'] = recent_permits['unit_count'].mean()
        
        return metrics
    
    def _calculate_potential_development(self, zip_data: pd.DataFrame, zip_code: str) -> dict:
        """Calculate potential development based on zoning and market conditions."""
        metrics = {
            'units_per_acre': 0,
            'annual_capacity': 0,
            'max_height': 0,
            'market_demand': 0
        }
        
        # Determine zoning category (simplified)
        if zip_code in ['60601', '60602', '60603', '60604', '60605', '60607', '60654']:
            # Downtown/high-rise zones
            zoning = 'DX'
            metrics['units_per_acre'] = 200  # Conservative for high-rise
            metrics['annual_capacity'] = 50   # High permit capacity
            metrics['max_height'] = 600      # Feet
        elif zip_code in ['60614', '60657']:
            # Restrictive residential
            zoning = 'RT'
            metrics['units_per_acre'] = 20
            metrics['annual_capacity'] = 10
            metrics['max_height'] = 38
        elif zip_code == '60637':  # Woodlawn
            # Special case - affordable housing ordinance
            zoning = 'RM'
            metrics['units_per_acre'] = 40  # Reduced from normal RM due to ordinance
            metrics['annual_capacity'] = 15
            metrics['max_height'] = 47
        else:
            # Standard multi-unit residential
            zoning = 'RM'
            metrics['units_per_acre'] = 65
            metrics['annual_capacity'] = 20
            metrics['max_height'] = 47
        
        # Adjust for market demand
        if 'population' in zip_data.columns and len(zip_data) > 1:
            pop_growth = zip_data['population'].pct_change().mean()
            demand_multiplier = 1 + min(pop_growth * 10, 0.5)  # Cap at 50% increase
            metrics['annual_capacity'] *= demand_multiplier
            metrics['market_demand'] = demand_multiplier
        
        return metrics
    
    def _calculate_constraint_score(self, actual: dict, potential: dict) -> float:
        """
        Calculate how constrained development is (0 = no constraint, 1 = fully constrained).
        """
        scores = []
        
        # Density constraint
        if potential['units_per_acre'] > 0:
            density_gap = 1 - (actual['units_per_acre'] / potential['units_per_acre'])
            scores.append(max(0, min(1, density_gap)))
        
        # Permit utilization
        if potential['annual_capacity'] > 0:
            permit_gap = 1 - (actual['recent_permits'] / potential['annual_capacity'])
            scores.append(max(0, min(1, permit_gap)))
        
        # Growth constraint (negative growth despite potential)
        if actual['permit_growth'] < 0 and potential['market_demand'] > 1:
            scores.append(0.8)  # High constraint score
        
        return np.mean(scores) if scores else 0
    
    def _categorize_constraint_level(self, score: float) -> str:
        """Categorize constraint level based on score."""
        if score >= 0.7:
            return 'Severely Constrained'
        elif score >= 0.5:
            return 'Significantly Constrained'
        elif score >= 0.3:
            return 'Moderately Constrained'
        elif score >= 0.1:
            return 'Slightly Constrained'
        else:
            return 'Unconstrained'
    
    def _identify_special_factors(self, zip_code: str, zip_data: pd.DataFrame) -> list:
        """Identify special factors affecting development."""
        factors = []
        
        # Known special cases
        if zip_code == '60637':
            factors.append('Woodlawn Affordable Housing Ordinance')
            factors.append('Obama Center Impact Zone')
        elif zip_code in ['60607', '60654']:
            factors.append('Up-zoned for High-Rise Development')
        elif zip_code in ['60614']:
            factors.append('Historic District Restrictions')
        
        # Transit-oriented development
        if zip_code in ['60601', '60602', '60603', '60607', '60654']:
            factors.append('Transit-Oriented Development Zone')
        
        # Recent rezoning activity (simplified detection)
        if 'permit_year' in zip_data.columns:
            recent_permits = zip_data[zip_data['permit_year'] >= 2020]
            if len(recent_permits) > 0 and 'unit_count' in recent_permits.columns:
                avg_recent_units = recent_permits['unit_count'].mean()
                historical_units = zip_data[zip_data['permit_year'] < 2020]['unit_count'].mean() if 'unit_count' in zip_data.columns else 0
                
                if avg_recent_units > historical_units * 2:
                    factors.append('Recent Up-zoning Activity')
        
        return factors
    
    def identify_opportunity_zones(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """Identify zones with highest potential for zoning reform."""
        logger.info("Identifying zoning reform opportunity zones")
        
        # Calculate opportunity score
        analysis_df['opportunity_score'] = (
            analysis_df['development_gap'] * 0.4 +
            analysis_df['constraint_score'] * 0.3 +
            (1 - analysis_df['permit_utilization']) * 0.3
        )
        
        # Normalize to 0-1 scale
        if analysis_df['opportunity_score'].max() > 0:
            analysis_df['opportunity_score'] = (
                analysis_df['opportunity_score'] / 
                analysis_df['opportunity_score'].max()
            )
        
        # Identify top opportunities
        opportunities = analysis_df[
            (analysis_df['constraint_score'] > 0.3) &
            (analysis_df['development_gap'] > 20)
        ].sort_values('opportunity_score', ascending=False)
        
        self.results['opportunity_zones'] = opportunities
        
        return opportunities
    
    def create_visualizations(self, analysis_df: pd.DataFrame):
        """Create visualizations for zoning impact analysis."""
        logger.info("Creating zoning impact visualizations")
        
        # 1. Constraint levels distribution
        self._plot_constraint_distribution(analysis_df)
        
        # 2. Actual vs potential development
        self._plot_development_gap(analysis_df)
        
        # 3. Permit utilization
        self._plot_permit_utilization(analysis_df)
        
        # 4. Opportunity zones
        self._plot_opportunity_zones(analysis_df)
        
        # 5. Special factors impact
        self._plot_special_factors(analysis_df)
        
        # 6. Zoning efficiency map
        self._plot_zoning_efficiency(analysis_df)
    
    def _plot_constraint_distribution(self, analysis_df: pd.DataFrame):
        """Plot distribution of constraint levels."""
        plt.figure(figsize=(10, 6))
        
        constraint_counts = analysis_df['constraint_level'].value_counts()
        colors = {
            'Unconstrained': '#2ca02c',
            'Slightly Constrained': '#90EE90',
            'Moderately Constrained': '#ff7f0e',
            'Significantly Constrained': '#ff6347',
            'Severely Constrained': '#d62728'
        }
        
        # Order categories
        order = ['Unconstrained', 'Slightly Constrained', 'Moderately Constrained',
                'Significantly Constrained', 'Severely Constrained']
        ordered_counts = [constraint_counts.get(level, 0) for level in order]
        ordered_colors = [colors[level] for level in order]
        
        plt.bar(range(len(order)), ordered_counts, color=ordered_colors)
        plt.xticks(range(len(order)), order, rotation=45, ha='right')
        plt.ylabel('Number of ZIP Codes')
        plt.title('Distribution of Zoning Constraint Levels')
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        total = sum(ordered_counts)
        for i, count in enumerate(ordered_counts):
            if count > 0:
                plt.text(i, count + 0.5, f'{count/total*100:.1f}%', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'constraint_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_development_gap(self, analysis_df: pd.DataFrame):
        """Plot actual vs potential development."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top constrained areas
        top_constrained = analysis_df.nlargest(15, 'development_gap')
        
        # Plot 1: Development gap
        y_pos = np.arange(len(top_constrained))
        ax1.barh(y_pos, top_constrained['actual_units_per_acre'], 
                alpha=0.7, label='Actual', color='#1f77b4')
        ax1.barh(y_pos, top_constrained['potential_units_per_acre'], 
                alpha=0.5, label='Potential', color='#ff7f0e')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['neighborhood']} ({row['zip_code']})" 
                            for _, row in top_constrained.iterrows()])
        ax1.set_xlabel('Units per Acre')
        ax1.set_title('Actual vs Potential Housing Density - Top 15 Constrained Areas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot of all areas
        scatter = ax2.scatter(analysis_df['actual_units_per_acre'],
                            analysis_df['potential_units_per_acre'],
                            c=analysis_df['constraint_score'],
                            cmap='RdYlGn_r',
                            s=100,
                            alpha=0.6)
        
        # Add diagonal line (actual = potential)
        max_val = max(analysis_df['potential_units_per_acre'].max(), 
                     analysis_df['actual_units_per_acre'].max())
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Actual = Potential')
        
        ax2.set_xlabel('Actual Units per Acre')
        ax2.set_ylabel('Potential Units per Acre')
        ax2.set_title('Housing Density: Actual vs Potential')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Constraint Score')
        
        # Annotate special areas
        for _, row in analysis_df[analysis_df['zip_code'].isin(self.zoning_constrained_areas.keys())].iterrows():
            ax2.annotate(row['neighborhood'], 
                        (row['actual_units_per_acre'], row['potential_units_per_acre']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'development_gap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_permit_utilization(self, analysis_df: pd.DataFrame):
        """Plot permit utilization rates."""
        plt.figure(figsize=(12, 8))
        
        # Filter out extreme outliers for better visualization
        filtered_df = analysis_df[analysis_df['permit_utilization'] <= 2]
        
        # Sort by utilization
        sorted_df = filtered_df.sort_values('permit_utilization')
        
        # Color based on constraint level
        colors = []
        for _, row in sorted_df.iterrows():
            if row['permit_utilization'] < 0.3:
                colors.append('#d62728')  # Very low utilization
            elif row['permit_utilization'] < 0.7:
                colors.append('#ff7f0e')  # Low utilization
            else:
                colors.append('#2ca02c')  # Good utilization
        
        plt.bar(range(len(sorted_df)), sorted_df['permit_utilization'], color=colors)
        
        # Add reference lines
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Full Utilization')
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Target (70%)')
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.3, label='Low Utilization')
        
        plt.xlabel('ZIP Code (sorted by utilization)')
        plt.ylabel('Permit Utilization Rate')
        plt.title('Building Permit Utilization vs Zoning Capacity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Annotate extremes
        if len(sorted_df) > 0:
            worst = sorted_df.iloc[0]
            plt.annotate(f"{worst['neighborhood']}\n({worst['zip_code']})", 
                        xy=(0, worst['permit_utilization']),
                        xytext=(5, 0.1), textcoords='offset points',
                        fontsize=8, ha='left')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'permit_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_opportunity_zones(self, analysis_df: pd.DataFrame):
        """Plot zoning reform opportunity zones."""
        opportunities = self.identify_opportunity_zones(analysis_df)
        
        if len(opportunities) == 0:
            return
        
        top_opportunities = opportunities.head(20)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Opportunity scores
        y_pos = np.arange(len(top_opportunities))
        bars = ax1.barh(y_pos, top_opportunities['opportunity_score'])
        
        # Color bars based on constraint level
        for i, (_, row) in enumerate(top_opportunities.iterrows()):
            if row['constraint_level'] == 'Severely Constrained':
                bars[i].set_color('#d62728')
            elif row['constraint_level'] == 'Significantly Constrained':
                bars[i].set_color('#ff6347')
            else:
                bars[i].set_color('#ff7f0e')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{row['neighborhood']} ({row['zip_code']})" 
                            for _, row in top_opportunities.iterrows()])
        ax1.set_xlabel('Opportunity Score')
        ax1.set_title('Top 20 Zoning Reform Opportunity Zones')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Components breakdown
        components = ['development_gap', 'constraint_score', 'permit_utilization']
        component_data = top_opportunities[components].values.T
        
        # Normalize for stacked bar
        component_data[2] = 1 - component_data[2]  # Convert utilization to underutilization
        
        ax2.bar(range(len(top_opportunities)), component_data[0], 
               label='Development Gap', alpha=0.8)
        ax2.bar(range(len(top_opportunities)), component_data[1], 
               bottom=component_data[0], label='Constraint Score', alpha=0.8)
        ax2.bar(range(len(top_opportunities)), component_data[2], 
               bottom=component_data[0] + component_data[1], 
               label='Permit Underutilization', alpha=0.8)
        
        ax2.set_xticks(range(len(top_opportunities)))
        ax2.set_xticklabels([row['zip_code'] for _, row in top_opportunities.iterrows()], 
                           rotation=45)
        ax2.set_ylabel('Score Components')
        ax2.set_title('Opportunity Score Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'opportunity_zones.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_special_factors(self, analysis_df: pd.DataFrame):
        """Visualize impact of special factors."""
        # Count special factors
        factor_counts = {}
        for factors_list in analysis_df['special_factors']:
            for factor in factors_list:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        if not factor_counts:
            return
        
        plt.figure(figsize=(12, 6))
        
        factors = list(factor_counts.keys())
        counts = list(factor_counts.values())
        
        plt.bar(range(len(factors)), counts, color='steelblue')
        plt.xticks(range(len(factors)), factors, rotation=45, ha='right')
        plt.ylabel('Number of ZIP Codes')
        plt.title('Special Factors Affecting Housing Development')
        plt.grid(True, alpha=0.3)
        
        # Add count labels
        for i, count in enumerate(counts):
            plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'special_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_zoning_efficiency(self, analysis_df: pd.DataFrame):
        """Plot zoning efficiency scores."""
        plt.figure(figsize=(14, 8))
        
        # Sort by efficiency
        sorted_df = analysis_df.sort_values('zoning_efficiency', ascending=False)
        
        # Create color gradient
        colors = plt.cm.RdYlGn(sorted_df['zoning_efficiency'])
        
        plt.bar(range(len(sorted_df)), sorted_df['zoning_efficiency'], color=colors)
        
        # Add reference lines
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, 
                   label='Good Efficiency (70%)')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, 
                   label='Moderate Efficiency (50%)')
        plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, 
                   label='Poor Efficiency (30%)')
        
        plt.xlabel('ZIP Code (sorted by efficiency)')
        plt.ylabel('Zoning Efficiency Score')
        plt.title('Zoning Efficiency by ZIP Code')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Annotate best and worst
        if len(sorted_df) > 0:
            best = sorted_df.iloc[0]
            worst = sorted_df.iloc[-1]
            
            plt.annotate(f"Best: {best['neighborhood']}\n({best['zip_code']})", 
                        xy=(0, best['zoning_efficiency']),
                        xytext=(5, -0.05), textcoords='offset points',
                        fontsize=8, ha='left')
            
            plt.annotate(f"Worst: {worst['neighborhood']}\n({worst['zip_code']})", 
                        xy=(len(sorted_df)-1, worst['zoning_efficiency']),
                        xytext=(-5, 0.05), textcoords='offset points',
                        fontsize=8, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'zoning_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, analysis_df: pd.DataFrame) -> dict:
        """Generate comprehensive zoning impact report."""
        logger.info("Generating zoning impact report")
        
        # Calculate summary statistics
        constrained_zips = analysis_df[analysis_df['is_constrained']]
        severely_constrained = analysis_df[
            analysis_df['constraint_level'] == 'Severely Constrained'
        ]
        
        report = {
            'summary': {
                'total_zip_codes': len(analysis_df),
                'constrained_zip_codes': len(constrained_zips),
                'severely_constrained': len(severely_constrained),
                'average_constraint_score': analysis_df['constraint_score'].mean(),
                'average_efficiency': analysis_df['zoning_efficiency'].mean(),
                'total_development_gap': analysis_df['development_gap'].sum()
            },
            
            'constraint_distribution': analysis_df['constraint_level'].value_counts().to_dict(),
            
            'top_constrained_areas': analysis_df.nlargest(10, 'constraint_score')[
                ['zip_code', 'neighborhood', 'constraint_score', 'constraint_level', 
                 'development_gap', 'special_factors']
            ].to_dict('records'),
            
            'reform_opportunities': self.results.get('opportunity_zones', pd.DataFrame()).head(10)[
                ['zip_code', 'neighborhood', 'opportunity_score', 'development_gap',
                 'permit_utilization']
            ].to_dict('records') if 'opportunity_zones' in self.results else [],
            
            'special_considerations': {
                'woodlawn_impact': 'Affordable housing ordinance limiting development',
                'downtown_upzoning': 'West Loop and River North successfully up-zoned',
                'transit_oriented': f"{len(analysis_df[analysis_df['special_factors'].apply(lambda x: 'Transit-Oriented Development Zone' in x)])} ZIP codes near transit"
            },
            
            'recommendations': self._generate_recommendations(analysis_df)
        }
        
        # Save detailed results
        analysis_df.to_csv(self.output_dir / 'zoning_impact_analysis.csv', index=False)
        
        with open(self.output_dir / 'zoning_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.results['report'] = report
        return report
    
    def _generate_recommendations(self, analysis_df: pd.DataFrame) -> list:
        """Generate policy recommendations based on analysis."""
        recommendations = []
        
        # High constraint areas
        high_constraint = analysis_df[analysis_df['constraint_score'] > 0.5]
        if len(high_constraint) > 0:
            recommendations.append({
                'category': 'High Priority Reform',
                'description': f"Consider zoning reform in {len(high_constraint)} severely constrained ZIP codes",
                'impact': f"Could add {high_constraint['development_gap'].sum():.0f} units per acre",
                'areas': high_constraint.head(5)['zip_code'].tolist()
            })
        
        # Low permit utilization
        low_utilization = analysis_df[analysis_df['permit_utilization'] < 0.3]
        if len(low_utilization) > 0:
            recommendations.append({
                'category': 'Permit Process Reform',
                'description': f"Streamline permit process in {len(low_utilization)} ZIP codes with low utilization",
                'impact': 'Could increase annual permit issuance by 70%',
                'areas': low_utilization.head(5)['zip_code'].tolist()
            })
        
        # Special case: Woodlawn
        if '60637' in analysis_df['zip_code'].values:
            woodlawn = analysis_df[analysis_df['zip_code'] == '60637'].iloc[0]
            if woodlawn['constraint_score'] > 0.5:
                recommendations.append({
                    'category': 'Ordinance Review',
                    'description': 'Review Woodlawn Affordable Housing Ordinance impact',
                    'impact': 'Current ordinance may be suppressing development',
                    'areas': ['60637']
                })
        
        return recommendations
    
    def run_analysis(self, data: pd.DataFrame) -> bool:
        """Run complete zoning impact analysis."""
        try:
            logger.info("Starting zoning impact analysis")
            
            # Analyze zoning impact
            analysis_df = self.analyze_zoning_impact(data)
            
            # Identify opportunities
            self.identify_opportunity_zones(analysis_df)
            
            # Create visualizations
            self.create_visualizations(analysis_df)
            
            # Generate report
            self.generate_report(analysis_df)
            
            logger.info("Zoning impact analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in zoning impact analysis: {str(e)}")
            return False 