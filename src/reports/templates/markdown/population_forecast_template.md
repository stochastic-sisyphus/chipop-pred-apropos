# {{ title }}

*Generated on: {{ generation_date }}*

## Executive Summary

{{ summary }}

## Population Forecast Overview

{% if population_stats %}
| Metric | Value |
|--------|-------|
| Latest Historical Year | {{ population_stats.latest_historical_year }} |
| Latest Forecast Year | {{ population_stats.latest_forecast_year }} |
| Historical Population | {{ "{:,}".format(population_stats.historical_population|int) }} |
| Forecast Population | {{ "{:,}".format(population_stats.forecast_population|int) }} |
| Projected Growth Rate | {{ "%.2f"|format(population_stats.growth_rate) }}% |
{% endif %}

{% if visualizations.overall_trend %}
![Population Trend and Forecast]({{ visualizations.overall_trend }})

*Figure 1: Historical population trend and future forecast for Chicago.*
{% endif %}

## Top Emerging ZIP Codes

{% if top_emerging_zips %}
| ZIP Code | Growth Rate (%) | Current Population | Forecast Population |
|----------|----------------|-------------------|---------------------|
{% for zip_info in top_emerging_zips %}
| {{ zip_info.zip_code }} | {{ "%.2f"|format(zip_info.population_growth_rate) }}% | {{ "{:,}".format(zip_info.current_population|int) }} | {{ "{:,}".format(zip_info.forecast_population|int) }} |
{% endfor %}
{% else %}
No top emerging ZIP codes identified.
{% endif %}

{% if visualizations.top_emerging_zips %}
![Top Emerging ZIP Codes by Growth Rate]({{ visualizations.top_emerging_zips }})

*Figure 2: ZIP codes with the highest projected population growth rates.*
{% endif %}

{% if visualizations.zip_comparison %}
![Population Trends by ZIP Code]({{ visualizations.zip_comparison }})

*Figure 3: Population trends for selected ZIP codes.*
{% endif %}

{% if visualizations.growth_heatmap %}
![Growth Rate Heatmap]({{ visualizations.growth_heatmap }})

*Figure 4: Heatmap showing population growth rates by ZIP code and year.*
{% endif %}

## Forecast Metrics and Accuracy

{% if forecast_metrics %}
| Metric | Value |
|--------|-------|
{% for key, value in forecast_metrics.items() %}
| {{ key }} | {{ value }} |
{% endfor %}
{% else %}
No forecast metrics available.
{% endif %}

## Methodology

{{ methodology }}

{% if visualizations.interactive_dashboard %}
## Interactive Dashboard

An interactive dashboard is available at: [Population Forecast Dashboard]({{ visualizations.interactive_dashboard }})
{% endif %}

---

*Chicago Housing Pipeline & Population Shift Project*

*© {{ generation_date.split('-')[0] }} All Rights Reserved*