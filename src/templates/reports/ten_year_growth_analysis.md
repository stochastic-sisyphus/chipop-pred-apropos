# Chicago Ten-Year Growth Analysis Report
Generated on {{ generation_date }} at {{ generation_time }}

{% if notes %}
> **WARNING:** The following metrics were missing or defaulted: {{ notes|join('; ') }}
---
{% endif %}

## Executive Summary

{% if historical_trends.get('population') %}
Over the past decade ({{ historical_trends.population.get('period_start', 'N/A') }} to {{ historical_trends.population.get('period_end', 'N/A') }}), Chicago has experienced a {{ "%.1f"|format(historical_trends.population.get('total_growth', 0) * 100) }}% change in population, with a compound annual growth rate (CAGR) of {{ "%.2f"|format(historical_trends.population.get('cagr', 0) * 100) }}%.
{% endif %}

## Historical Trends Analysis

### Population Trends
{% if historical_trends.get('population') %}
- Total Growth: {{ "%.1f"|format(historical_trends.population.get('total_growth', 0) * 100) }}%
- CAGR: {{ "%.2f"|format(historical_trends.population.get('cagr', 0) * 100) }}%
- Key Demographic Shifts:
  {% for shift in historical_trends.population.get('demographic_shifts', []) %}
  - {{ shift }}
  {% endfor %}
{% endif %}

### Development Trends
{% if historical_trends.get('development') %}
- Permit Volume: {{ historical_trends.development.get('total_permits', 'N/A') }} permits issued
- Construction Value: ${{ "%.2f"|format(historical_trends.development.get('total_value', 0) / 1e9) }} billion
- Key Development Types:
  {% for type in historical_trends.development.get('types', []) %}
  - {{ type.name }}: {{ "%.1f"|format(type.percentage * 100) }}%
  {% endfor %}
{% endif %}

### Economic Trends
{% if historical_trends.get('economic') %}
- GDP Growth: {{ "%.1f"|format(historical_trends.economic.get('gdp_growth', 0) * 100) }}%
- Employment Change: {{ "%.1f"|format(historical_trends.economic.get('employment_change', 0) * 100) }}%
- Income Growth: {{ "%.1f"|format(historical_trends.economic.get('income_growth', 0) * 100) }}%
{% endif %}

## Current Analysis ({{ current_analysis.get('year', 'N/A') }})

### Population Metrics
{% if current_analysis.get('population') %}
- Total Population: {{ "{:,}".format(current_analysis.population.get('total', 0)) }}
- Population Density: {{ "{:,.1f}".format(current_analysis.population.get('density', 0)) }} per sq mile
- Household Size: {% if current_analysis.population.get('avg_household_size') is not none %}{{ "%.2f"|format(current_analysis.population.get('avg_household_size', 0)) }}{% else %}N/A{% endif %}
{% endif %}

### Development Activity
{% if current_analysis.get('development') %}
- Active Permits: {% if current_analysis.development.get('active_permits') is not none %}{{ "{:,}".format(current_analysis.development.get('active_permits', 0)) }}{% else %}N/A{% endif %}
- Under Construction: {% if current_analysis.development.get('units_under_construction') is not none %}{{ "{:,}".format(current_analysis.development.get('units_under_construction', 0)) }}{% else %}N/A{% endif %} units
- Pipeline Value: {% if current_analysis.development.get('pipeline_value') is not none %}${{ "%.2f".format(current_analysis.development.get('pipeline_value', 0) / 1e9) }} billion{% else %}N/A{% endif %}
{% endif %}

## Growth Projections ({{ projections.get('period_start', 'N/A') }} - {{ projections.get('period_end', 'N/A') }})

### Population Projections
{% if projections.get('population') %}
{% for scenario in projections.population.get('scenarios', []) %}
#### {{ scenario.name|title }} Scenario
- Population Change: {{ "%.1f"|format(scenario.get('population_change', 0) * 100) }}%
- Total Population by {{ projections.get('period_end', 'N/A') }}: {{ "{:,}".format(scenario.get('final_population', 0)) }}
{% endfor %}
{% endif %}

### Development Projections
{% if projections.get('development') %}
- Expected New Units: {{ "{:,}".format(projections.development.get('total_new_units', 0)) }}
- Projected Investment: ${{ "%.2f".format(projections.development.get('total_investment', 0) / 1e9) }} billion
{% endif %}

## Impact Analysis

### Housing Impacts
{% if impacts.get('housing') %}
{% for impact in impacts.housing %}
- {{ impact }}
{% endfor %}
{% endif %}

### Economic Impacts
{% if impacts.get('economic') %}
{% for impact in impacts.economic %}
- {{ impact }}
{% endfor %}
{% endif %}

## Growth Areas

### Primary Growth Centers
{% if growth_areas.get('primary') %}
{% for area in growth_areas.primary %}
- {{ area }}
{% endfor %}
{% endif %}

### Emerging Markets
{% if growth_areas.get('emerging') %}
{% for market in growth_areas.emerging %}
- {{ market }}
{% endfor %}
{% endif %}

## Recommendations

### Strategic Priorities
{% if recommendations.get('strategic') %}
{% for priority in recommendations.strategic %}
1. {{ priority }}
{% endfor %}
{% endif %}

### Implementation Steps
{% if recommendations.get('implementation') %}
{% for step in recommendations.implementation %}
1. {{ step }}
{% endfor %}
{% endif %}

### Support Requirements
{% if recommendations.get('support') %}
{% for req in recommendations.support %}
- {{ req }}
{% endfor %}
{% endif %}

---
*This report was automatically generated by the Chicago Population Analysis Pipeline*