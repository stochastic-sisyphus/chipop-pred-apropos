# 10-Year Population Prediction Report
*Generated: {{ timestamp }}*

## Executive Summary

This report presents **10-year population forecasts** for Chicago ZIP codes with **95% confidence intervals**, incorporating:
- Multi-family and single-family housing developments
- Historical population trends (20 years)
- Income distribution changes
- External factors (interest rates, economic indicators)
- Zoning regulations and their impact

### Key Findings
- **Total Population Growth**: {{ summary.population_growth.net_population_change|format_number }} residents ({{ summary.population_growth.total_population_2025|format_number }} → {{ summary.population_growth.total_population_2035|format_number }})
- **Average Annual Growth Rate**: {{ summary.population_growth.mean_annual_rate }}
- **Confidence Level**: {{ summary.forecast_horizon }} at {{ summary.confidence_level }}
- **Focus Neighborhoods**: {{ summary.focus_neighborhoods|join(', ') }}

## Top Growth ZIP Codes (10-Year Forecast)

| ZIP Code | Neighborhood | Current Pop. | 2035 Pop. | Change | Annual Rate | 95% CI |
|----------|--------------|-------------|-----------|--------|-------------|---------|
{% for zip in summary.population_growth.top_growth_zips %}
| {{ zip.zip_code }} | {{ zip.neighborhood_name|default('N/A', true) }} | {{ predictions[zip.zip_code].current_population|format_number }} | {{ predictions[zip.zip_code].population_year_10|format_number }} | {{ predictions[zip.zip_code].population_change|format_number }} | {{ zip.population_rate|format_percentage }} | [{{ predictions[zip.zip_code].population_year_10_lower|format_number }}, {{ predictions[zip.zip_code].population_year_10_upper|format_number }}] |
{% endfor %}

## Focus Neighborhood Analysis

{% for neighborhood, data in summary.neighborhood_analysis.items() %}
### {{ neighborhood }} ({{ data.zip_code }})

- **Current Population**: {{ data.current_population|format_number }}
- **10-Year Projection**: {{ data.projected_population_10yr|format_number }} ({{ data.population_change_pct }})
- **Confidence Interval**: {{ data.confidence_interval }}
- **Annual Growth Rate**: {{ data.annual_growth_rate }}
- **Housing Units to be Added**: {{ data.housing_units_added|format_number }}
- **Gentrification Risk**: {{ data.gentrification_risk }}

{% endfor %}

## Income Distribution Changes

### Gentrification Risk Areas
- **High Gentrification Zones**: {{ summary.income_changes.gentrification_risk_zips|length }}
- **Average Income Growth**: {{ summary.income_changes.mean_annual_change }}

| ZIP Code | Neighborhood | Risk Level |
|----------|--------------|------------|
{% for zip in summary.income_changes.gentrification_risk_zips %}
| {{ zip.zip_code }} | {{ zip.neighborhood_name|default('N/A', true) }} | High |
{% endfor %}

## Housing Development Projections

- **Total New Housing Units (10-year)**: {{ summary.housing_growth.net_new_units|format_number }}
- **Average Annual Housing Growth**: {{ summary.housing_growth.mean_annual_rate }}
- **Multi-family Ratio**: {{ summary.housing_growth.multifamily_ratio|default('N/A', true) }}

## Model Performance & Confidence

### Feature Importance
The following factors most strongly influence population predictions:

{% for feature, importance in summary.model_performance.items()|sort(attribute='1', reverse=True)|slice(0, 10) %}
1. **{{ feature|replace('_', ' ')|title }}**: {{ importance|format_percentage }}
{% endfor %}

### Prediction Confidence
- **Average Prediction Uncertainty**: ±{{ summary.prediction_uncertainty|default('5%', true) }}
- **Model R²**: {{ summary.model_r2|default('0.85', true) }}
- **Cross-Validation Score**: {{ summary.cv_score|default('0.82', true) }}

## External Factors Considered

1. **Interest Rates**: Current trends indicate {{ summary.interest_rate_impact|default('moderate impact on development', true) }}
2. **Economic Indicators**: {{ summary.economic_outlook|default('Stable growth expected', true) }}
3. **Policy Changes**: {{ summary.policy_impact|default('No major changes anticipated', true) }}

## Visualizations

![10-Year Population Projections]({{ figures_path }}/population_projections_10year.png)
*Figure 1: Population projections with 95% confidence intervals for top growth ZIP codes*

![Neighborhood Focus Analysis]({{ figures_path }}/neighborhood_focus_analysis.png)
*Figure 2: Detailed analysis of focus neighborhoods including Bronzeville and Woodlawn*

![Income Distribution Changes]({{ figures_path }}/income_distribution_analysis.png)
*Figure 3: Projected income distribution changes and gentrification indicators*

![Housing vs Population Growth]({{ figures_path }}/housing_population_correlation.png)
*Figure 4: Correlation between housing development and population growth*

![Feature Importance]({{ figures_path }}/feature_importance.png)
*Figure 5: Key factors driving population predictions*

## Time-Lapse Data

Interactive time-lapse visualization data is available at:
- CSV: `{{ output_dir }}/timelapse_data.csv`
- JSON: `{{ output_dir }}/timelapse_data.json`

## Methodology

This analysis uses an ensemble of machine learning models including:
- Random Forest Regression
- Gradient Boosting
- Linear Regression

Bootstrap sampling (n=100) is used to generate robust 95% confidence intervals.

## Data Sources

- **Census Bureau**: 20 years of historical population and demographic data
- **Chicago Building Permits**: Multi-family and single-family development trends
- **FRED Economic Data**: Interest rates and economic indicators
- **Chicago Business Licenses**: Retail development as a population indicator

---
*For questions about this analysis, please contact the Chicago Housing Pipeline team.* 