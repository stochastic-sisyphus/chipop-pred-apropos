# {{ title }}

*Generated on: {{ generation_date }}*

## Executive Summary

{{ executive_summary }}

## Population Forecast Analysis

{{ population_forecast.summary }}

{% if population_forecast.population_stats %}
### Population Statistics

| Metric | Value |
|--------|-------|
{% for key, value in population_forecast.population_stats.items() %}
| {{ key }} | {{ "%.2f"|format(value) if value is number and key == 'growth_rate' else ("{:,}".format(value|int) if value is number else value) }} |
{% endfor %}
{% endif %}

{% if population_forecast.top_emerging_zips %}
### Top Emerging ZIP Codes

| ZIP Code | Growth Rate (%) |
|----------|----------------|
{% for zip_info in population_forecast.top_emerging_zips %}
| {{ zip_info.zip_code }} | {{ "%.2f"|format(zip_info.population_growth_rate) }}% |
{% endfor %}
{% endif %}

{% if visualizations.population_forecast %}
![Population Forecast Visualization]({{ visualizations.population_forecast }})

*Figure 1: Population forecast trends for Chicago.*
{% endif %}

## Retail Gap Analysis

{{ retail_gap.summary }}

{% if retail_gap.gap_stats %}
### Retail Gap Statistics

| Metric | Value |
|--------|-------|
{% for key, value in retail_gap.gap_stats.items() %}
| {{ key }} | {{ "%.3f"|format(value) if value is number and 'gap' in key else value }} |
{% endfor %}
{% endif %}

{% if retail_gap.opportunity_zones %}
### Retail Opportunity Zones

| ZIP Code | Gap Score |
|----------|-----------|
{% for zone in retail_gap.opportunity_zones %}
| {{ zone.zip_code }} | {{ "%.3f"|format(zone.retail_gap_score) }} |
{% endfor %}
{% endif %}

{% if visualizations.retail_gap %}
![Retail Gap Visualization]({{ visualizations.retail_gap }})

*Figure 2: Retail gap analysis for Chicago ZIP codes.*
{% endif %}

## Retail Void Analysis

{{ retail_void.summary }}

{% if retail_void.void_stats %}
### Retail Void Statistics

| Metric | Value |
|--------|-------|
{% for key, value in retail_void.void_stats.items() %}
| {{ key }} | {{ "%.3f"|format(value) if value is number and 'leakage' in key else ("%.1f"|format(value) + '%' if value is number and 'percentage' in key else value) }} |
{% endfor %}
{% endif %}

{% if retail_void.void_zones %}
### Retail Void Zones

| ZIP Code | Leakage Ratio |
|----------|---------------|
{% for zone in retail_void.void_zones %}
| {{ zone.zip_code }} | {{ "%.3f"|format(zone.leakage_ratio) if zone.leakage_ratio is defined else "N/A" }} |
{% endfor %}
{% endif %}

{% if visualizations.retail_void %}
![Retail Void Visualization]({{ visualizations.retail_void }})

*Figure 3: Retail void analysis for Chicago ZIP codes.*
{% endif %}

## Recommendations

{% if recommendations %}
{% for recommendation in recommendations %}
- {{ recommendation }}
{% endfor %}
{% else %}
No recommendations available.
{% endif %}

{% if visualizations.combined_dashboard %}
## Interactive Dashboard

A comprehensive interactive dashboard is available at: [Chicago Housing Pipeline Dashboard]({{ visualizations.combined_dashboard }})
{% endif %}

## Methodology

{{ methodology }}

---

*Chicago Housing Pipeline & Population Shift Project*

*© {{ generation_date.split('-')[0] }} All Rights Reserved*