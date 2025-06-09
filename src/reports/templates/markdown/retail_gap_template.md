# {{ title }}

*Generated on: {{ date }}*

## Executive Summary

{{ summary }}

## Retail Gap Analysis Overview

{% if gap_stats %}
| Metric | Value |
|--------|-------|
| Mean Gap Score | {{ "%.3f"|format(gap_stats.mean_gap) }} |
| Median Gap Score | {{ "%.3f"|format(gap_stats.median_gap) }} |
| Minimum Gap Score | {{ "%.3f"|format(gap_stats.min_gap) }} |
| Maximum Gap Score | {{ "%.3f"|format(gap_stats.max_gap) }} |
| Standard Deviation | {{ "%.3f"|format(gap_stats.std_gap) }} |
| Opportunity Zones Count | {{ gap_stats.opportunity_count }} |
| Saturated Markets Count | {{ gap_stats.saturated_count }} |
{% endif %}

{% if visualizations.gap_distribution %}
![Retail Gap Score Distribution]({{ visualizations.gap_distribution }})

*Figure 1: Distribution of retail gap scores across Chicago ZIP codes.*
{% endif %}

## Retail Opportunity Zones

{% if opportunity_zones %}
| ZIP Code |
|----------|
{% for zone in opportunity_zones %}
| {{ zone }} |
{% endfor %}
{% else %}
No retail opportunity zones identified.
{% endif %}

{% if visualizations.opportunity_zones %}
![Top Retail Opportunity Zones]({{ visualizations.opportunity_zones }})

*Figure 2: ZIP codes with the highest retail development potential.*
{% endif %}

## Retail Cluster Analysis

{% if retail_clusters %}
| Cluster | ZIP Codes | Avg. Retail per Capita | Avg. Population |
|---------|-----------|------------------------|-----------------|
{% for cluster in retail_clusters %}
| {{ cluster.retail_cluster }} | {{ cluster.zip_count }} | {{ "%.4f"|format(cluster.retail_per_capita) }} | {{ "{:,}".format(cluster.population|int) }} |
{% endfor %}
{% else %}
No retail clusters available.
{% endif %}

{% if visualizations.retail_clusters %}
![Retail Cluster Analysis]({{ visualizations.retail_clusters }})

*Figure 3: Analysis of retail clusters across Chicago.*
{% endif %}

## Category Gap Analysis

{% if category_gaps %}
| Category | ZIP Codes with Gaps | Average Gap Size |
|----------|---------------------|------------------|
{% for category, gaps in category_gaps.items() %}
| {{ category }} | {{ gaps|length }} | {{ "%.3f"|format(gaps|map(attribute='gap_size')|sum / gaps|length if gaps else 0) }} |
{% endfor %}
{% else %}
No category gaps available.
{% endif %}

{% if visualizations.category_gaps %}
![Retail Category Gaps]({{ visualizations.category_gaps }})

*Figure 4: Analysis of retail category gaps by type.*
{% endif %}

## Methodology

{{ methodology|default('Standard retail gap analysis methodology was applied.') }}

{% if visualizations.interactive_dashboard %}
## Interactive Dashboard

An interactive dashboard is available at: [Retail Gap Dashboard]({{ visualizations.interactive_dashboard }})
{% endif %}

---

*Chicago Housing Pipeline & Population Shift Project*

*© {{ date.split('-')[0] }} All Rights Reserved*
