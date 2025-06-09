# {{ title }}

*Generated on: {{ date }}*

## Executive Summary

{{ summary }}

{% if visualizations %}
## Key Visualizations

{% for name, path in visualizations.items() %}
### {{ name }}

![{{ name }}]({{ path }})

{% endfor %}
{% endif %}

## Top Emerging ZIP Codes

{% if top_emerging_zips %}
{% for zip_info in top_emerging_zips %}
### {{ zip_info.zip_code }}

- **Growth Score:** {{ zip_info.growth_score|default('N/A') }}
- **New Units:** {{ zip_info.new_units|default('N/A') }}
{% if zip_info.description %}
- **Description:** {{ zip_info.description }}
{% endif %}

{% endfor %}
{% else %}
No top emerging ZIP codes identified.
{% endif %}

## Growth Metrics

{% if growth_metrics %}
| Metric | Value |
|--------|-------|
{% for metric, value in growth_metrics.items() %}
| {{ metric }} | {{ value }} |
{% endfor %}
{% else %}
No growth metrics available.
{% endif %}

## Building Permit Analysis

{% if permit_analysis %}
| Metric | Value |
|--------|-------|
{% for metric, value in permit_analysis.items() %}
| {{ metric }} | {{ value }} |
{% endfor %}
{% else %}
No building permit analysis available.
{% endif %}

## Methodology

{{ methodology|default('Standard multifamily growth analysis methodology was applied.') }}

---

*Chicago Housing Pipeline & Population Shift Project*

*© {{ date.split('-')[0] }} All Rights Reserved*
