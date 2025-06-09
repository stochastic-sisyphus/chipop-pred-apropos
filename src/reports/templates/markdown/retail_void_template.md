# {{ title }}

*Generated on: {{ date }}*

## Executive Summary

{{ summary }}

## Retail Void Analysis Overview

{% if void_stats %}
| Metric | Value |
|--------|-------|
{% if void_stats.mean_leakage is defined %}
| Mean Leakage Ratio | {{ "%.3f"|format(void_stats.mean_leakage) }} |
| Median Leakage Ratio | {{ "%.3f"|format(void_stats.median_leakage) }} |
| Minimum Leakage Ratio | {{ "%.3f"|format(void_stats.min_leakage) }} |
| Maximum Leakage Ratio | {{ "%.3f"|format(void_stats.max_leakage) }} |
| Standard Deviation | {{ "%.3f"|format(void_stats.std_leakage) }} |
{% endif %}
{% if void_stats.void_count is defined %}
| Void Zones Count | {{ void_stats.void_count }} |
| Total ZIP Codes | {{ void_stats.total_zips }} |
| Void Percentage | {{ "%.1f"|format(void_stats.void_percentage) }}% |
{% endif %}
{% endif %}

{% if visualizations.leakage_distribution %}
![Spending Leakage Distribution]({{ visualizations.leakage_distribution }})

*Figure 1: Distribution of spending leakage across Chicago ZIP codes.*
{% endif %}

## Retail Void Zones

{% if void_zones %}
| ZIP Code | Leakage Ratio | Retail per Capita | Population |
|----------|---------------|-------------------|------------|
{% for zone in void_zones %}
| {{ zone.zip_code }} | {{ "%.3f"|format(zone.leakage_ratio) if zone.leakage_ratio is defined else "N/A" }} | {{ "%.4f"|format(zone.retail_per_capita) if zone.retail_per_capita is defined else "N/A" }} | {{ "{:,}".format(zone.population|int) if zone.population is defined else "N/A" }} |
{% endfor %}
{% else %}
No retail void zones identified.
{% endif %}

{% if visualizations.void_zones %}
![Retail Void Zones]({{ visualizations.void_zones }})

*Figure 2: ZIP codes identified as retail void zones.*
{% endif %}

## Category Void Analysis

{% if category_voids %}
{% if category_voids is mapping %}
| Category | ZIP Codes with Voids | Average Void Size |
|----------|----------------------|-------------------|
{% for category, voids in category_voids.items() %}
| {{ category }} | {{ voids|length }} | {{ "%.3f"|format(voids|map(attribute='void_size')|sum / voids|length if voids else 0) }} |
{% endfor %}
{% else %}
| Category | ZIP Code | Void Size |
|----------|----------|-----------|
{% for void in category_voids %}
| {{ void.category if void.category is defined else "N/A" }} | {{ void.zip_code if void.zip_code is defined else "N/A" }} | {{ "%.3f"|format(void.void_size) if void.void_size is defined else "N/A" }} |
{% endfor %}
{% endif %}
{% else %}
No category voids available.
{% endif %}

{% if visualizations.category_voids %}
![Retail Category Voids]({{ visualizations.category_voids }})

*Figure 3: Analysis of retail category voids by type.*
{% endif %}

## Spending Leakage Patterns

{% if leakage_patterns %}
{% if leakage_patterns is mapping %}
| Metric | Value |
|--------|-------|
{% for key, value in leakage_patterns.items() if key not in ['high_leakage_zips', 'low_leakage_zips'] %}
| {{ key }} | {{ "%.3f"|format(value) if value is number else value }} |
{% endfor %}

{% if leakage_patterns.high_leakage_zips %}
### High Leakage ZIP Codes

{{ leakage_patterns.high_leakage_zips|join(', ') }}
{% endif %}

{% if leakage_patterns.low_leakage_zips %}
### Low Leakage ZIP Codes (Attraction)

{{ leakage_patterns.low_leakage_zips|join(', ') }}
{% endif %}
{% else %}
| ZIP Code | Overall Leakage | Top Category |
|----------|----------------|-------------|
{% for pattern in leakage_patterns %}
| {{ pattern.zip_code if pattern.zip_code is defined else "N/A" }} | {{ "%.3f"|format(pattern.overall_leakage) if pattern.overall_leakage is defined else "N/A" }} | {{ pattern.top_category if pattern.top_category is defined else "N/A" }} |
{% endfor %}
{% endif %}
{% else %}
No leakage patterns available.
{% endif %}

{% if visualizations.leakage_flow %}
![Spending Leakage Flow]({{ visualizations.leakage_flow }})

*Figure 4: Analysis of spending leakage flow patterns.*
{% endif %}

## Methodology

{{ methodology|default('Standard retail void analysis methodology was applied.') }}

{% if visualizations.interactive_dashboard %}
## Interactive Dashboard

An interactive dashboard is available at: [Retail Void Dashboard]({{ visualizations.interactive_dashboard }})
{% endif %}

---

*Chicago Housing Pipeline & Population Shift Project*

*© {{ date.split('-')[0] }} All Rights Reserved*
