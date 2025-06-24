"""
Income Distribution Model for Chicago ZIP Codes

This model analyzes and predicts income distribution changes with focus on:
- Gentrification indicators
- Income inequality trends
- Neighborhood transition patterns
- Displacement risk assessment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IncomeDistributionModel:
    """
    Analyzes income distribution changes and gentrification patterns.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("output/models/income_distribution")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Income brackets for distribution analysis
        self.income_brackets = [
            (0, 25000, 'Very Low'),
            (25000, 50000, 'Low'),
            (50000, 75000, 'Moderate'),
            (75000, 100000, 'Middle'),
            (100000, 150000, 'Upper Middle'),
            (150000, float('inf'), 'High')
        ]
        
        # Gentrification indicators
        self.gentrification_thresholds = {
            'income_growth_rate': 0.04,  # 4% annual growth
            'rent_growth_rate': 0.05,    # 5% annual growth
            'education_change': 0.02,     # 2% increase in college-educated
            'displacement_threshold': 0.1  # 10% population change
        }
        
        self.results = {}
    
    def analyze_income_distribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze income distribution patterns and changes over time.
        """
        logger.info("Analyzing income distribution patterns")
        
        analysis_results = []
        
        for zip_code in data['zip_code'].unique():
            zip_data = data[data['zip_code'] == zip_code]
            
            # Calculate income metrics
            if 'median_income' in zip_data.columns and len(zip_data) > 0:
                # Ensure we have data before accessing indices
                income_series = zip_data['median_income'].dropna()
                if len(income_series) > 0:
                    current_income = income_series.iloc[-1]
                    initial_income = income_series.iloc[0] if len(income_series) > 0 else current_income
                    income_growth_rate = self._calculate_growth_rate(income_series)
                    income_volatility = income_series.std() / income_series.mean() if income_series.mean() != 0 else 0
                else:
                    # No valid income data
                    current_income = 0
                    initial_income = 0
                    income_growth_rate = 0
                    income_volatility = 0
            else:
                current_income = 0
                initial_income = 0
                income_growth_rate = 0
                income_volatility = 0
            
            # Calculate gentrification indicators
            gentrification_score = self._calculate_gentrification_score(
                zip_data, income_growth_rate
            )
            
            # Displacement risk
            displacement_risk = self._calculate_displacement_risk(
                zip_data, income_growth_rate
            )
            
            # Income inequality metrics
            if 'income_inequality' in zip_data.columns and len(zip_data) > 0:
                inequality_series = zip_data['income_inequality'].dropna()
                inequality_index = inequality_series.iloc[-1] if len(inequality_series) > 0 else self._estimate_inequality(current_income)
            else:
                inequality_index = self._estimate_inequality(current_income)
            
            analysis_results.append({
                'zip_code': zip_code,
                'current_median_income': current_income,
                'income_growth_rate': income_growth_rate,
                'income_volatility': income_volatility,
                'initial_bracket': self._get_income_bracket(initial_income),
                'current_bracket': self._get_income_bracket(current_income),
                'bracket_change': self._calculate_bracket_change(self._get_income_bracket(initial_income), self._get_income_bracket(current_income)),
                'gentrification_score': gentrification_score,
                'displacement_risk': displacement_risk,
                'inequality_index': inequality_index
            })
        
        results_df = pd.DataFrame(analysis_results)
        self.results['income_analysis'] = results_df
        
        return results_df
    
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate compound annual growth rate."""
        if len(series) < 2:
            return 0
        
        years = len(series) - 1
        if years == 0 or series.iloc[0] == 0:
            return 0
        
        return (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    
    def _get_income_bracket(self, income: float) -> str:
        """Determine income bracket."""
        for lower, upper, label in self.income_brackets:
            if lower <= income < upper:
                return label
        return 'Unknown'
    
    def _calculate_bracket_change(self, initial: str, current: str) -> int:
        """Calculate change in income brackets."""
        bracket_order = ['Very Low', 'Low', 'Moderate', 'Middle', 'Upper Middle', 'High']
        
        if initial in bracket_order and current in bracket_order:
            return bracket_order.index(current) - bracket_order.index(initial)
        return 0
    
    def _calculate_gentrification_score(self, zip_data: pd.DataFrame, income_growth: float) -> float:
        """
        Calculate gentrification score based on multiple indicators.
        Score ranges from 0 (no gentrification) to 1 (high gentrification).
        """
        score_components = []
        
        # Income growth component
        income_score = min(income_growth / self.gentrification_thresholds['income_growth_rate'], 1)
        score_components.append(income_score)
        
        # Housing cost changes
        if 'housing_units' in zip_data.columns and len(zip_data) > 1:
            housing_growth = zip_data['housing_units'].pct_change().mean()
            housing_score = min(housing_growth / 0.03, 1)  # 3% threshold
            score_components.append(housing_score)
        
        # Retail/business changes
        if 'retail_businesses' in zip_data.columns and len(zip_data) > 1:
            retail_growth = zip_data['retail_businesses'].pct_change().mean()
            retail_score = min(retail_growth / 0.04, 1)  # 4% threshold
            score_components.append(retail_score)
        
        return np.mean(score_components) if score_components else 0
    
    def _calculate_displacement_risk(self, zip_data: pd.DataFrame, income_growth: float) -> str:
        """
        Calculate displacement risk based on income growth and population changes.
        """
        risk_score = 0
        
        # High income growth increases displacement risk
        if income_growth > self.gentrification_thresholds['income_growth_rate']:
            risk_score += 0.4
        
        # Population changes
        if 'population' in zip_data.columns and len(zip_data) > 1:
            pop_change = zip_data['population'].pct_change().mean()
            if abs(pop_change) > self.gentrification_thresholds['displacement_threshold']:
                risk_score += 0.3
        
        # Current income level (lower income areas at higher risk)
        if 'median_income' in zip_data.columns and len(zip_data) > 0:
            income_series = zip_data['median_income'].dropna()
            if len(income_series) > 0:
                current_income = income_series.iloc[-1]
                if current_income < 50000:
                    risk_score += 0.3
        
        # Categorize risk
        if risk_score >= 0.7:
            return 'High'
        elif risk_score >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _estimate_inequality(self, median_income: float) -> float:
        """
        Estimate income inequality index based on median income.
        This is a simplified approximation when detailed data is not available.
        """
        # Simple estimation: lower median income areas tend to have higher inequality
        if median_income == 0:
            return 0.5
        
        # Normalize and invert (lower income = higher inequality)
        normalized = min(median_income / 100000, 1)
        return 1 - (normalized * 0.6)  # Scale to 0.4-1.0 range
    
    def identify_gentrification_zones(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and classify gentrification zones using clustering.
        """
        logger.info("Identifying gentrification zones")
        
        # Features for clustering
        feature_cols = [
            'income_growth_rate', 'gentrification_score', 
            'current_median_income', 'inequality_index'
        ]
        
        # Prepare data
        X = analysis_df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        analysis_df['gentrification_cluster'] = clusters
        
        # Label clusters based on characteristics
        cluster_labels = self._label_gentrification_clusters(analysis_df)
        analysis_df['gentrification_type'] = analysis_df['gentrification_cluster'].map(cluster_labels)
        
        # Identify high-priority zones
        high_priority = analysis_df[
            (analysis_df['gentrification_score'] > 0.6) & 
            (analysis_df['displacement_risk'] == 'High')
        ]
        
        self.results['gentrification_zones'] = high_priority
        self.results['cluster_analysis'] = analysis_df
        
        return high_priority
    
    def _label_gentrification_clusters(self, df: pd.DataFrame) -> dict:
        """Label clusters based on their characteristics."""
        cluster_stats = df.groupby('gentrification_cluster').agg({
            'income_growth_rate': 'mean',
            'gentrification_score': 'mean',
            'current_median_income': 'mean',
            'displacement_risk': lambda x: (x == 'High').sum() / len(x)
        })
        
        labels = {}
        for cluster in cluster_stats.index:
            stats = cluster_stats.loc[cluster]
            
            if stats['gentrification_score'] > 0.7:
                labels[cluster] = 'Active Gentrification'
            elif stats['income_growth_rate'] > 0.03 and stats['current_median_income'] < 60000:
                labels[cluster] = 'Early Stage Gentrification'
            elif stats['displacement_risk'] > 0.5:
                labels[cluster] = 'High Displacement Risk'
            elif stats['current_median_income'] > 80000:
                labels[cluster] = 'Established High Income'
            else:
                labels[cluster] = 'Stable/Low Change'
        
        return labels
    
    def create_visualizations(self, analysis_df: pd.DataFrame, data: pd.DataFrame):
        """Create comprehensive visualizations for income distribution analysis."""
        logger.info("Creating income distribution visualizations")
        
        # 1. Income distribution heatmap over time
        self._plot_income_heatmap(data)
        
        # 2. Gentrification score map
        self._plot_gentrification_scores(analysis_df)
        
        # 3. Income bracket transitions
        self._plot_bracket_transitions(analysis_df)
        
        # 4. Displacement risk analysis
        self._plot_displacement_risk(analysis_df)
        
        # 5. Income inequality trends
        self._plot_inequality_trends(analysis_df)
        
        # 6. Cluster analysis visualization
        self._plot_gentrification_clusters(analysis_df)
    
    def _plot_income_heatmap(self, data: pd.DataFrame):
        """Create heatmap of income changes over time by ZIP code."""
        # Prepare data for heatmap
        income_matrix = []
        zip_codes = []
        
        for zip_code in sorted(data['zip_code'].unique()):
            zip_data = data[data['zip_code'] == zip_code]
            if 'year' in zip_data.columns and 'median_income' in zip_data.columns:
                yearly_income = zip_data.groupby('year')['median_income'].first()
                if len(yearly_income) > 0:
                    income_matrix.append(yearly_income.values)
                    zip_codes.append(zip_code)
        
        if income_matrix:
            # Create heatmap
            plt.figure(figsize=(12, 10))
            income_array = np.array(income_matrix)
            
            # Normalize for better visualization
            income_normalized = (income_array - income_array.mean()) / income_array.std()
            
            sns.heatmap(income_normalized, 
                       yticklabels=zip_codes,
                       cmap='RdYlBu_r',
                       center=0,
                       cbar_kws={'label': 'Normalized Median Income'})
            
            plt.title('Income Distribution Changes Over Time by ZIP Code')
            plt.xlabel('Year Index')
            plt.ylabel('ZIP Code')
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'income_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_gentrification_scores(self, analysis_df: pd.DataFrame):
        """Visualize gentrification scores by ZIP code."""
        top_gentrifying = analysis_df.nlargest(20, 'gentrification_score')
        
        plt.figure(figsize=(12, 8))
        
        colors = ['#d62728' if risk == 'High' else '#ff7f0e' if risk == 'Medium' else '#2ca02c' 
                 for risk in top_gentrifying['displacement_risk']]
        
        bars = plt.barh(range(len(top_gentrifying)), 
                        top_gentrifying['gentrification_score'],
                        color=colors)
        
        plt.yticks(range(len(top_gentrifying)), top_gentrifying['zip_code'])
        plt.xlabel('Gentrification Score')
        plt.title('Top 20 ZIP Codes by Gentrification Score')
        plt.xlim(0, 1)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', label='High Displacement Risk'),
            Patch(facecolor='#ff7f0e', label='Medium Displacement Risk'),
            Patch(facecolor='#2ca02c', label='Low Displacement Risk')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'gentrification_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bracket_transitions(self, analysis_df: pd.DataFrame):
        """Visualize income bracket transitions."""
        # Count transitions
        transition_matrix = pd.crosstab(
            analysis_df['initial_bracket'], 
            analysis_df['current_bracket']
        )
        
        # Reorder brackets
        bracket_order = ['Very Low', 'Low', 'Moderate', 'Middle', 'Upper Middle', 'High']
        existing_initial = [b for b in bracket_order if b in transition_matrix.index]
        existing_current = [b for b in bracket_order if b in transition_matrix.columns]
        transition_matrix = transition_matrix.loc[existing_initial, existing_current]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix, 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   cbar_kws={'label': 'Number of ZIP Codes'})
        
        plt.title('Income Bracket Transitions')
        plt.xlabel('Current Income Bracket')
        plt.ylabel('Initial Income Bracket')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'bracket_transitions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_displacement_risk(self, analysis_df: pd.DataFrame):
        """Visualize displacement risk analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Risk distribution
        risk_counts = analysis_df['displacement_risk'].value_counts()
        colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
        
        ax1.pie(risk_counts.values, 
               labels=risk_counts.index,
               colors=[colors[risk] for risk in risk_counts.index],
               autopct='%1.1f%%',
               startangle=90)
        ax1.set_title('Distribution of Displacement Risk')
        
        # Risk vs Income
        for risk in ['Low', 'Medium', 'High']:
            risk_data = analysis_df[analysis_df['displacement_risk'] == risk]
            ax2.scatter(risk_data['current_median_income'], 
                       risk_data['gentrification_score'],
                       label=risk, 
                       alpha=0.6,
                       color=colors[risk],
                       s=80)
        
        ax2.set_xlabel('Current Median Income ($)')
        ax2.set_ylabel('Gentrification Score')
        ax2.set_title('Displacement Risk by Income and Gentrification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'displacement_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inequality_trends(self, analysis_df: pd.DataFrame):
        """Visualize income inequality trends."""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot: Inequality vs Income
        scatter = plt.scatter(analysis_df['current_median_income'], 
                            analysis_df['inequality_index'],
                            c=analysis_df['gentrification_score'],
                            cmap='viridis',
                            s=100,
                            alpha=0.6)
        
        plt.xlabel('Current Median Income ($)')
        plt.ylabel('Inequality Index')
        plt.title('Income Inequality vs Median Income')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Gentrification Score')
        
        # Add trend line
        x = analysis_df['current_median_income'].values
        y = analysis_df['inequality_index'].values
        z = np.polyfit(x[~np.isnan(x) & ~np.isnan(y)], 
                      y[~np.isnan(x) & ~np.isnan(y)], 1)
        p = np.poly1d(z)
        plt.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'inequality_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gentrification_clusters(self, analysis_df: pd.DataFrame):
        """Visualize gentrification clustering results."""
        if 'gentrification_type' not in analysis_df.columns:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Cluster distribution
        cluster_counts = analysis_df['gentrification_type'].value_counts()
        ax1.bar(range(len(cluster_counts)), cluster_counts.values)
        ax1.set_xticks(range(len(cluster_counts)))
        ax1.set_xticklabels(cluster_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of ZIP Codes')
        ax1.set_title('Distribution of Gentrification Types')
        ax1.grid(True, alpha=0.3)
        
        # Cluster characteristics
        colors = plt.cm.Set3(np.linspace(0, 1, len(analysis_df['gentrification_type'].unique())))
        
        for i, gtype in enumerate(analysis_df['gentrification_type'].unique()):
            cluster_data = analysis_df[analysis_df['gentrification_type'] == gtype]
            ax2.scatter(cluster_data['income_growth_rate'] * 100,
                       cluster_data['gentrification_score'],
                       label=gtype,
                       color=colors[i],
                       s=100,
                       alpha=0.6)
        
        ax2.set_xlabel('Income Growth Rate (%)')
        ax2.set_ylabel('Gentrification Score')
        ax2.set_title('Gentrification Clusters by Key Metrics')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'gentrification_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, analysis_df: pd.DataFrame) -> dict:
        """Generate comprehensive income distribution report."""
        logger.info("Generating income distribution report")
        
        report = {
            'summary': {
                'total_zip_codes': len(analysis_df),
                'average_income': f"${analysis_df['current_median_income'].mean():,.0f}",
                'income_range': f"${analysis_df['current_median_income'].min():,.0f} - ${analysis_df['current_median_income'].max():,.0f}",
                'average_growth_rate': f"{analysis_df['income_growth_rate'].mean() * 100:.2f}%"
            },
            
            'gentrification': {
                'high_gentrification_zones': len(analysis_df[analysis_df['gentrification_score'] > 0.6]),
                'high_displacement_risk': len(analysis_df[analysis_df['displacement_risk'] == 'High']),
                'top_gentrifying': analysis_df.nlargest(5, 'gentrification_score')[
                    ['zip_code', 'gentrification_score', 'displacement_risk']
                ].to_dict('records')
            },
            
            'income_mobility': {
                'upward_mobility': len(analysis_df[analysis_df['bracket_change'] > 0]),
                'downward_mobility': len(analysis_df[analysis_df['bracket_change'] < 0]),
                'stable': len(analysis_df[analysis_df['bracket_change'] == 0]),
                'largest_improvements': analysis_df.nlargest(5, 'bracket_change')[
                    ['zip_code', 'initial_bracket', 'current_bracket']
                ].to_dict('records')
            },
            
            'inequality': {
                'average_inequality_index': analysis_df['inequality_index'].mean(),
                'high_inequality_zones': len(analysis_df[analysis_df['inequality_index'] > 0.7]),
                'inequality_correlation': self._calculate_inequality_correlation(analysis_df)
            }
        }
        
        # Save detailed results
        analysis_df.to_csv(self.output_dir / 'income_distribution_analysis.csv', index=False)
        
        with open(self.output_dir / 'income_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.results['report'] = report
        return report
    
    def _calculate_inequality_correlation(self, df: pd.DataFrame) -> str:
        """Calculate correlation between inequality and other factors."""
        correlations = {
            'income_level': df['inequality_index'].corr(df['current_median_income']),
            'gentrification': df['inequality_index'].corr(df['gentrification_score']),
            'growth_rate': df['inequality_index'].corr(df['income_growth_rate'])
        }
        
        # Find strongest correlation
        strongest = max(correlations.items(), key=lambda x: abs(x[1]))
        return f"{strongest[0]}: {strongest[1]:.3f}"
    
    def run_analysis(self, data: pd.DataFrame) -> bool:
        """Run complete income distribution analysis."""
        try:
            logger.info("Starting income distribution analysis")
            
            # Analyze income distribution
            analysis_df = self.analyze_income_distribution(data)
            
            # Identify gentrification zones
            self.identify_gentrification_zones(analysis_df)
            
            # Create visualizations
            self.create_visualizations(analysis_df, data)
            
            # Generate report
            self.generate_report(analysis_df)
            
            logger.info("Income distribution analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in income distribution analysis: {str(e)}")
            return False 