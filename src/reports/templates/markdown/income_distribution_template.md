# Income Distribution & Gentrification Analysis Report
*Generated: {{ timestamp }}*

## Executive Summary

This report analyzes income distribution changes and gentrification patterns across Chicago ZIP codes, focusing on:
- Income bracket transitions
- Gentrification indicators and scores
- Displacement risk assessment
- Income inequality trends

### Key Findings
- **Total ZIP Codes Analyzed**: {{ summary.total_zip_codes }}
- **Average Income**: {{ summary.average_income }}
- **Income Range**: {{ summary.income_range }}
- **Average Growth Rate**: {{ summary.average_growth_rate }}
- **High Gentrification Zones**: {{ gentrification.high_gentrification_zones }}
- **High Displacement Risk Areas**: {{ gentrification.high_displacement_risk }}

## Gentrification Analysis

### Top Gentrifying ZIP Codes

| ZIP Code | Gentrification Score | Displacement Risk | Current Income | Growth Rate |
|----------|---------------------|-------------------|----------------|-------------|
{% for area in gentrification.top_gentrifying %}
| {{ area.zip_code }} | {{ area.gentrification_score|round(2) }} | {{ area.displacement_risk }} | ${{ area.current_median_income|format_number }} | {{ (area.income_growth_rate * 100)|round(1) }}% |
{% endfor %}

### Gentrification Types Distribution

| Type | Count | Percentage |
|------|-------|------------|
{% for type, count in cluster_distribution.items() %}
| {{ type }} | {{ count }} | {{ (count / summary.total_zip_codes * 100)|round(1) }}% |
{% endfor %}

## Income Mobility Analysis

### Bracket Transitions
- **Upward Mobility**: {{ income_mobility.upward_mobility }} ZIP codes
- **Downward Mobility**: {{ income_mobility.downward_mobility }} ZIP codes
- **Stable**: {{ income_mobility.stable }} ZIP codes

### Largest Improvements

| ZIP Code | Initial Bracket | Current Bracket | Change |
|----------|----------------|-----------------|--------|
{% for improvement in income_mobility.largest_improvements %}
| {{ improvement.zip_code }} | {{ improvement.initial_bracket }} | {{ improvement.current_bracket }} | ↑{{ improvement.bracket_change }} |
{% endfor %}

## Displacement Risk Analysis

### Risk Distribution
- **High Risk**: {{ displacement_stats.high_risk }} ZIP codes ({{ displacement_stats.high_risk_pct }}%)
- **Medium Risk**: {{ displacement_stats.medium_risk }} ZIP codes ({{ displacement_stats.medium_risk_pct }}%)
- **Low Risk**: {{ displacement_stats.low_risk }} ZIP codes ({{ displacement_stats.low_risk_pct }}%)

### High-Priority Areas (High Gentrification + High Displacement Risk)

| ZIP Code | Neighborhood | Population | Median Income | Risk Factors |
|----------|--------------|------------|---------------|--------------|
{% for area in high_priority_areas %}
| {{ area.zip_code }} | {{ area.neighborhood|default('N/A', true) }} | {{ area.population|format_number }} | ${{ area.median_income|format_number }} | {{ area.risk_factors|join(', ') }} |
{% endfor %}

## Income Inequality Analysis

- **Average Inequality Index**: {{ inequality.average_inequality_index|round(3) }}
- **High Inequality Zones**: {{ inequality.high_inequality_zones }}
- **Strongest Correlation**: {{ inequality.inequality_correlation }}

### Inequality Trends
{% if inequality.trend == 'increasing' %}
⚠️ **Warning**: Income inequality is increasing across {{ inequality.affected_zips }} ZIP codes
{% else %}
✓ Income inequality remains stable or decreasing
{% endif %}

## Neighborhood-Specific Analysis

### Bronzeville (60615)
{% if neighborhoods.bronzeville %}
- **Current Median Income**: ${{ neighborhoods.bronzeville.current_income|format_number }}
- **Income Growth Rate**: {{ neighborhoods.bronzeville.growth_rate }}%
- **Gentrification Score**: {{ neighborhoods.bronzeville.gentrification_score|round(2) }}
- **Displacement Risk**: {{ neighborhoods.bronzeville.displacement_risk }}
{% endif %}

### Woodlawn (60637)
{% if neighborhoods.woodlawn %}
- **Current Median Income**: ${{ neighborhoods.woodlawn.current_income|format_number }}
- **Income Growth Rate**: {{ neighborhoods.woodlawn.growth_rate }}%
- **Gentrification Score**: {{ neighborhoods.woodlawn.gentrification_score|round(2) }}
- **Displacement Risk**: {{ neighborhoods.woodlawn.displacement_risk }}
- **Special Factors**: Obama Center impact zone
{% endif %}

### West Loop (60607)
{% if neighborhoods.west_loop %}
- **Current Median Income**: ${{ neighborhoods.west_loop.current_income|format_number }}
- **Income Growth Rate**: {{ neighborhoods.west_loop.growth_rate }}%
- **Gentrification Score**: {{ neighborhoods.west_loop.gentrification_score|round(2) }}
- **Status**: {{ neighborhoods.west_loop.status|default('Established high-income area', true) }}
{% endif %}

## Visualizations

![Income Distribution Heatmap]({{ figures_path }}/income_heatmap.png)
*Figure 1: Income distribution changes over time by ZIP code*

![Gentrification Scores]({{ figures_path }}/gentrification_scores.png)
*Figure 2: Top 20 ZIP codes by gentrification score with displacement risk indicators*

![Income Bracket Transitions]({{ figures_path }}/bracket_transitions.png)
*Figure 3: Income bracket transition matrix showing mobility patterns*

![Displacement Risk Analysis]({{ figures_path }}/displacement_risk_analysis.png)
*Figure 4: Displacement risk distribution and correlation with income*

![Income Inequality Trends]({{ figures_path }}/inequality_trends.png)
*Figure 5: Income inequality index vs median income with gentrification overlay*

![Gentrification Clusters]({{ figures_path }}/gentrification_clusters.png)
*Figure 6: Machine learning clustering of gentrification patterns*

## Policy Implications

{% if gentrification.high_displacement_risk > 10 %}
### ⚠️ Urgent Action Required
With {{ gentrification.high_displacement_risk }} ZIP codes at high displacement risk, consider:
1. **Affordable Housing Preservation**: Protect existing affordable units
2. **Inclusionary Zoning**: Require affordable units in new developments
3. **Community Land Trusts**: Preserve long-term affordability
4. **Tenant Protections**: Strengthen rent control and just-cause eviction policies
{% endif %}

### Recommended Interventions by ZIP Code

{% for intervention in recommended_interventions %}
- **{{ intervention.zip_code }}**: {{ intervention.recommendation }}
{% endfor %}

## Methodology

This analysis employs:
- **Compound Annual Growth Rate (CAGR)** for income trends
- **K-means Clustering** for gentrification pattern identification
- **Multi-factor Scoring** for gentrification and displacement risk
- **20-year Historical Analysis** for trend identification

## Data Sources

- **US Census Bureau**: American Community Survey (ACS) 5-year estimates
- **Chicago Department of Buildings**: Permit data for development indicators
- **Chicago Business Licenses**: Retail development patterns
- **FRED Economic Data**: Economic indicators and employment rates

---
*This analysis is part of the Chicago Housing Pipeline project. For methodology details or custom analysis requests, please contact the project team.* 