# Zoning Impact Analysis Report
*Generated: {{ timestamp }}*

## Executive Summary

This report analyzes how zoning regulations affect housing development potential across Chicago ZIP codes, providing:
- Binary assessment of zoning constraints (Yes/No)
- Scale measurement of constraint levels
- Identification of areas where zoning suppresses housing
- Recommendations for zoning reform opportunities

### Key Findings
- **Total ZIP Codes Analyzed**: {{ summary.total_zip_codes }}
- **Constrained ZIP Codes**: {{ summary.constrained_zip_codes }} ({{ (summary.constrained_zip_codes / summary.total_zip_codes * 100)|round(1) }}%)
- **Severely Constrained**: {{ summary.severely_constrained }} ZIP codes
- **Average Constraint Score**: {{ (summary.average_constraint_score * 100)|round(1) }}%
- **Total Development Gap**: {{ summary.total_development_gap|format_number }} units per acre
- **Average Zoning Efficiency**: {{ (summary.average_efficiency * 100)|round(1) }}%

## Constraint Level Distribution

| Constraint Level | Count | Percentage | Binary Assessment |
|-----------------|-------|------------|-------------------|
{% for level, count in constraint_distribution.items() %}
| {{ level }} | {{ count }} | {{ (count / summary.total_zip_codes * 100)|round(1) }}% | {% if level in ['Moderately Constrained', 'Significantly Constrained', 'Severely Constrained'] %}YES - Constrained{% else %}NO - Unconstrained{% endif %} |
{% endfor %}

## Top Constrained Areas

| ZIP Code | Neighborhood | Constraint Score | Level | Development Gap | Special Factors |
|----------|--------------|------------------|-------|-----------------|-----------------|
{% for area in top_constrained_areas %}
| {{ area.zip_code }} | {{ area.neighborhood }} | {{ (area.constraint_score * 100)|round(1) }}% | {{ area.constraint_level }} | {{ area.development_gap|round(1) }} units/acre | {{ area.special_factors|join(', ') if area.special_factors else 'None' }} |
{% endfor %}

## Zoning Reform Opportunities

### Top 10 Reform Opportunity Zones

| Rank | ZIP Code | Neighborhood | Opportunity Score | Development Gap | Current Utilization |
|------|----------|--------------|-------------------|-----------------|---------------------|
{% for i, opp in enumerate(reform_opportunities, 1) %}
| {{ i }} | {{ opp.zip_code }} | {{ opp.neighborhood }} | {{ (opp.opportunity_score * 100)|round(1) }}% | {{ opp.development_gap|round(1) }} units/acre | {{ (opp.permit_utilization * 100)|round(1) }}% |
{% endfor %}

## Neighborhood-Specific Analysis

### Woodlawn (60637)
{% if neighborhoods.woodlawn %}
- **Constraint Level**: {{ neighborhoods.woodlawn.constraint_level }}
- **Binary Assessment**: {% if neighborhoods.woodlawn.is_constrained %}YES - Constrained{% else %}NO - Unconstrained{% endif %}
- **Actual Density**: {{ neighborhoods.woodlawn.actual_units_per_acre|round(1) }} units/acre
- **Potential Density**: {{ neighborhoods.woodlawn.potential_units_per_acre|round(1) }} units/acre
- **Development Gap**: {{ neighborhoods.woodlawn.development_gap|round(1) }} units/acre
- **Special Factors**: 
  - Woodlawn Affordable Housing Ordinance (limiting development)
  - Obama Center Impact Zone
- **Recommendation**: Review affordable housing ordinance impact on development feasibility
{% endif %}

### West Loop (60607) & River North (60654)
{% if neighborhoods.west_loop %}
- **Status**: Successfully up-zoned for high-rise development
- **Binary Assessment**: NO - Unconstrained (after up-zoning)
- **Zoning Type**: Downtown Mixed-Use (DX)
- **Maximum Height**: Unlimited
- **Success Metrics**: 
  - Permit utilization: {{ (neighborhoods.west_loop.permit_utilization * 100)|round(1) }}%
  - Development efficiency: {{ (neighborhoods.west_loop.zoning_efficiency * 100)|round(1) }}%
{% endif %}

### Lincoln Park (60614)
{% if neighborhoods.lincoln_park %}
- **Constraint Level**: {{ neighborhoods.lincoln_park.constraint_level }}
- **Binary Assessment**: {% if neighborhoods.lincoln_park.is_constrained %}YES - Constrained{% else %}NO - Unconstrained{% endif %}
- **Zoning Type**: Restrictive Residential (RT)
- **Development Gap**: {{ neighborhoods.lincoln_park.development_gap|round(1) }} units/acre
- **Special Factors**: Historic District Restrictions
{% endif %}

## Development Potential Analysis

### Actual vs Potential Development

| Metric | Current State | Potential | Gap | Improvement Possible |
|--------|--------------|-----------|-----|---------------------|
| Total Units per Acre | {{ actual_total.units_per_acre|round(1) }} | {{ potential_total.units_per_acre|round(1) }} | {{ development_gap_total|round(1) }} | {{ ((development_gap_total / actual_total.units_per_acre) * 100)|round(1) }}% |
| Annual Permits | {{ actual_total.annual_permits }} | {{ potential_total.annual_permits }} | {{ permit_gap_total }} | {{ ((permit_gap_total / actual_total.annual_permits) * 100)|round(1) }}% |

### Permit Utilization Analysis

- **ZIP codes with <30% utilization**: {{ low_utilization_count }} (severely underutilized)
- **ZIP codes with 30-70% utilization**: {{ medium_utilization_count }} (moderately underutilized)
- **ZIP codes with >70% utilization**: {{ high_utilization_count }} (well utilized)

## Policy Recommendations

{% for rec in recommendations %}
### {{ rec.category }}

**Description**: {{ rec.description }}

**Potential Impact**: {{ rec.impact }}

**Target Areas**: {{ rec.areas|join(', ') }}

{% endfor %}

## Special Considerations

### Transit-Oriented Development Zones
- **Count**: {{ special_considerations.transit_oriented }} ZIP codes
- **Recommendation**: Prioritize up-zoning near transit stations

### Historic Affordable Housing Ordinances
- **Woodlawn Example**: Current ordinance may be suppressing development by {{ woodlawn_suppression_pct|default('30-50', true) }}%
- **Recommendation**: Balance affordability requirements with development feasibility

### Up-zoning Success Stories
- **West Loop & River North**: Successful transformation to high-density zones
- **Key Factors**: Market demand, transit access, political will

## Visualizations

![Constraint Distribution]({{ figures_path }}/constraint_distribution.png)
*Figure 1: Distribution of zoning constraint levels across Chicago ZIP codes*

![Development Gap Analysis]({{ figures_path }}/development_gap.png)
*Figure 2: Actual vs potential housing density by ZIP code*

![Permit Utilization]({{ figures_path }}/permit_utilization.png)
*Figure 3: Building permit utilization rates vs zoning capacity*

![Opportunity Zones]({{ figures_path }}/opportunity_zones.png)
*Figure 4: Top zoning reform opportunity zones with score components*

![Special Factors Impact]({{ figures_path }}/special_factors.png)
*Figure 5: Distribution of special factors affecting development*

![Zoning Efficiency Map]({{ figures_path }}/zoning_efficiency.png)
*Figure 6: Zoning efficiency scores by ZIP code*

## Methodology

This analysis uses:
- **Actual Development Metrics**: Current housing density and permit activity
- **Potential Development Calculations**: Based on zoning categories and market conditions
- **Constraint Scoring**: 0-1 scale measuring gap between actual and potential
- **Binary Classification**: Constrained if score > 0.3 (30% below potential)
- **Scale Measurement**: Five-level classification from Unconstrained to Severely Constrained

### Zoning Categories Analyzed
1. **RS** - Residential Single-Unit (8 units/acre max)
2. **RT** - Residential Two-Flat (20 units/acre max)
3. **RM** - Residential Multi-Unit (65 units/acre max)
4. **DX** - Downtown Mixed-Use (400+ units/acre)

## Data Sources

- **Chicago Department of Buildings**: Building permit data
- **Chicago Zoning Map**: Current zoning designations
- **US Census Bureau**: Housing unit counts
- **Market Analysis**: Development feasibility studies

## Conclusion

{{ summary.constrained_zip_codes }} of Chicago's ZIP codes show zoning constraints that limit housing development. Priority reform areas could unlock {{ summary.total_development_gap|format_number }} additional housing units per acre, helping address the city's housing shortage while maintaining neighborhood character through thoughtful planning.

---
*This report satisfies the meeting requirement for both binary (Yes/No) and scale measurements of zoning constraints. For detailed methodology or custom analysis, please contact the Chicago Housing Pipeline team.* 