# {{ title }}

*Generated on: {{ date }}*

## Executive Summary

{{ pipeline_summary|default('This report summarizes the findings from the Chicago Housing Pipeline & Population Shift Project analysis.') }}

## Key Findings

{% if key_findings %}
{% for finding in key_findings %}
- {{ finding }}
{% endfor %}
{% else %}
No key findings available.
{% endif %}

## Strategic Development Zones

### Top Emerging ZIP Codes for Multifamily Growth

{% if top_emerging_zips %}
{% for zip_info in top_emerging_zips[:5] %}
#### {{ zip_info.zip_code }}

**Growth Score:** {{ zip_info.growth_score|default('N/A') }}

{% if zip_info.description %}
{{ zip_info.description }}
{% endif %}

{% endfor %}
{% else %}
No top emerging ZIP codes identified.
{% endif %}

### Top Retail Opportunity Zones

{% if opportunity_zones %}
{% for zone in opportunity_zones[:5] %}
#### {{ zone.zip_code }}

**Opportunity Score:** {{ zone.opportunity_score|default('N/A') }}

{% if zone.description %}
{{ zone.description }}
{% endif %}

{% endfor %}
{% else %}
No retail opportunity zones identified.
{% endif %}

### Top Retail Void Zones

{% if void_zones %}
{% for zone in void_zones[:5] %}
#### {{ zone.zip_code }}

**Leakage Score:** {{ zone.leakage_score|default('N/A') }}

{% if zone.description %}
{{ zone.description }}
{% endif %}

{% endfor %}
{% else %}
No retail void zones identified.
{% endif %}

## Model Summaries

### Multifamily Growth Analysis

{{ model_summaries.multifamily_growth|default('Multifamily growth analysis identified emerging areas for residential development.') }}

### Retail Gap Analysis

{{ model_summaries.retail_gap|default('Retail gap analysis identified areas with retail development potential.') }}

### Retail Void Analysis

{{ model_summaries.retail_void|default('Retail void analysis identified specific retail categories that are underrepresented in different areas.') }}

## Methodology

{{ methodology|default('Standard methodology was applied across all analyses.') }}

---

*Chicago Housing Pipeline & Population Shift Project*

*© {{ date.split('-')[0] }} All Rights Reserved*
