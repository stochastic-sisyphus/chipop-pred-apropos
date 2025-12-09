"""
CAG Interactive Visualization Module

Generates interactive HTML visualizations for Chicago urban analytics
that can be hosted on GitHub Pages.

Visualization Capabilities:
1. Representation power - Multi-dimensional data (growth, risk, constraints)
2. Interactivity model - Filtering, brushing, linked charts
3. Decision layer - Thresholds, annotations, scenario comparison
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


# Chicago ZIP code approximate coordinates (centroid lat/lon)
CHICAGO_ZIP_COORDS = {
    "60601": (41.8862, -87.6186), "60602": (41.8832, -87.6297),
    "60603": (41.8804, -87.6254), "60604": (41.8772, -87.6277),
    "60605": (41.8673, -87.6219), "60606": (41.8822, -87.6395),
    "60607": (41.8726, -87.6516), "60608": (41.8519, -87.6693),
    "60609": (41.8094, -87.6534), "60610": (41.9036, -87.6354),
    "60611": (41.8951, -87.6179), "60612": (41.8806, -87.6869),
    "60613": (41.9541, -87.6555), "60614": (41.9214, -87.6513),
    "60615": (41.8019, -87.5981), "60616": (41.8425, -87.6324),
    "60617": (41.7256, -87.5561), "60618": (41.9467, -87.7044),
    "60619": (41.7456, -87.6061), "60620": (41.7406, -87.6531),
    "60621": (41.7756, -87.6394), "60622": (41.9006, -87.6779),
    "60623": (41.8494, -87.7144), "60624": (41.8806, -87.7231),
    "60625": (41.9722, -87.7044), "60626": (42.0022, -87.6675),
    "60628": (41.6906, -87.6244), "60629": (41.7781, -87.7069),
    "60630": (41.9872, -87.7569), "60631": (41.9947, -87.8069),
    "60632": (41.8094, -87.7044), "60633": (41.6581, -87.5511),
    "60634": (41.9447, -87.7869), "60636": (41.7756, -87.6656),
    "60637": (41.7794, -87.5961), "60638": (41.7872, -87.7744),
    "60639": (41.9206, -87.7544), "60640": (41.9722, -87.6600),
    "60641": (41.9497, -87.7469), "60642": (41.9106, -87.6679),
    "60643": (41.6981, -87.6631), "60644": (41.8856, -87.7556),
    "60645": (42.0072, -87.6969), "60646": (41.9947, -87.7544),
    "60647": (41.9206, -87.7019), "60649": (41.7631, -87.5661),
    "60651": (41.9006, -87.7344), "60652": (41.7456, -87.7131),
    "60653": (41.8194, -87.6094), "60654": (41.8926, -87.6354),
    "60655": (41.6956, -87.7056), "60656": (41.9822, -87.8369),
    "60657": (41.9397, -87.6530), "60659": (41.9872, -87.6969),
    "60660": (41.9897, -87.6600), "60661": (41.8826, -87.6454),
}

# Neighborhood mappings
ZIP_TO_NEIGHBORHOOD = {
    "60601": "Loop", "60602": "Loop", "60603": "Loop", "60604": "South Loop",
    "60605": "South Loop", "60606": "West Loop", "60607": "West Loop",
    "60608": "Pilsen", "60609": "Back of the Yards", "60610": "Near North",
    "60611": "Streeterville", "60612": "Near West", "60613": "Lakeview",
    "60614": "Lincoln Park", "60615": "Hyde Park/Bronzeville",
    "60616": "Bridgeport/Chinatown", "60617": "South Chicago",
    "60618": "North Center", "60619": "Chatham", "60620": "Auburn Gresham",
    "60621": "Englewood", "60622": "Wicker Park", "60623": "Little Village",
    "60624": "West Garfield Park", "60625": "Lincoln Square", "60626": "Rogers Park",
    "60628": "Roseland", "60629": "Clearing", "60630": "Jefferson Park",
    "60631": "Edison Park", "60632": "Brighton Park", "60633": "Hegewisch",
    "60634": "Portage Park", "60636": "West Englewood", "60637": "Woodlawn",
    "60638": "Garfield Ridge", "60639": "Belmont Cragin", "60640": "Uptown",
    "60641": "Avondale", "60642": "Ukrainian Village", "60643": "Morgan Park",
    "60644": "Austin", "60645": "West Ridge", "60646": "Norwood Park",
    "60647": "Logan Square", "60649": "South Shore", "60651": "Humboldt Park",
    "60652": "Ashburn", "60653": "Grand Boulevard", "60654": "River North",
    "60655": "Mount Greenwood", "60656": "O'Hare", "60657": "Lakeview",
    "60659": "Edgewater", "60660": "Edgewater", "60661": "West Loop",
}


def generate_interactive_dashboard(
    output_path: Path,
    multifamily_data: Dict[str, Any],
    zoning_data: Dict[str, Any],
    income_data: Dict[str, Any],
    population_data: Dict[str, Any],
) -> str:
    """
    Generate a comprehensive interactive HTML dashboard.

    Args:
        output_path: Where to save the HTML file
        multifamily_data: Multifamily growth results
        zoning_data: Zoning impact results
        income_data: Income distribution results
        population_data: Population prediction results

    Returns:
        Path to generated HTML file
    """

    # Process data for visualization
    zip_data = _process_zip_data(multifamily_data, zoning_data)

    # Generate the HTML
    html_content = _generate_html(zip_data, multifamily_data, zoning_data, income_data, population_data)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return str(output_path)


def _process_zip_data(
    multifamily_data: Dict[str, Any],
    zoning_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Process and merge data for each ZIP code."""
    zip_data = []

    # Get growth scores
    growth_metrics = {m['zip_code']: m for m in multifamily_data.get('growth_metrics', [])}

    # Get zoning constraints
    zoning_constraints = {}
    for area in zoning_data.get('top_constrained_areas', []):
        zoning_constraints[area['zip_code']] = area

    # Get reform opportunities
    reform_opps = {}
    for opp in zoning_data.get('reform_opportunities', []):
        reform_opps[opp['zip_code']] = opp

    # Merge all data
    all_zips = set(growth_metrics.keys()) | set(zoning_constraints.keys()) | set(CHICAGO_ZIP_COORDS.keys())

    for zip_code in all_zips:
        if zip_code not in CHICAGO_ZIP_COORDS:
            continue

        lat, lon = CHICAGO_ZIP_COORDS[zip_code]
        growth = growth_metrics.get(zip_code, {})
        zoning = zoning_constraints.get(zip_code, {})
        reform = reform_opps.get(zip_code, {})

        zip_data.append({
            'zip_code': zip_code,
            'neighborhood': ZIP_TO_NEIGHBORHOOD.get(zip_code, 'Other'),
            'lat': lat,
            'lon': lon,
            'growth_score': growth.get('growth_score', 0),
            'permit_growth': growth.get('permit_growth', 0),
            'unit_growth': growth.get('unit_growth', 0),
            'total_permits': growth.get('total_permits', 0),
            'total_units': growth.get('total_units', 0),
            'activity_score': growth.get('activity_score', 0),
            'constraint_score': zoning.get('constraint_score', 0.5),
            'constraint_level': zoning.get('constraint_level', 'Uncategorized'),
            'development_gap': zoning.get('development_gap', 0),
            'opportunity_score': reform.get('opportunity_score', 0),
            'permit_utilization': reform.get('permit_utilization', 0),
        })

    return sorted(zip_data, key=lambda x: x['growth_score'], reverse=True)


def _generate_html(
    zip_data: List[Dict[str, Any]],
    multifamily_data: Dict[str, Any],
    zoning_data: Dict[str, Any],
    income_data: Dict[str, Any],
    population_data: Dict[str, Any],
) -> str:
    """Generate the complete HTML dashboard."""

    # Convert data to JSON for embedding
    zip_data_json = json.dumps(zip_data)
    summary_stats = {
        'total_zips': len(zip_data),
        'avg_growth_score': sum(z['growth_score'] for z in zip_data) / len(zip_data) if zip_data else 0,
        'top_growth_zip': zip_data[0]['zip_code'] if zip_data else 'N/A',
        'top_growth_neighborhood': zip_data[0]['neighborhood'] if zip_data else 'N/A',
        'severely_constrained': zoning_data.get('summary', {}).get('severely_constrained', 0),
        'high_inequality_zones': income_data.get('inequality', {}).get('high_inequality_zones', 0),
        'total_population': population_data.get('population_growth', {}).get('total_population_2025', 0),
    }
    summary_json = json.dumps(summary_stats)

    # Top growth areas for bar chart
    top_growth = zip_data[:15] if len(zip_data) >= 15 else zip_data
    top_growth_json = json.dumps(top_growth)

    # Constraint distribution
    constraint_dist = zoning_data.get('constraint_distribution', {})
    constraint_json = json.dumps(constraint_dist)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chicago Urban Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --secondary: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1f2937;
            --light: #f3f4f6;
            --white: #ffffff;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: var(--light);
        }}

        .dashboard {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 24px;
        }}

        header {{
            text-align: center;
            margin-bottom: 32px;
            padding: 24px;
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            backdrop-filter: blur(10px);
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}

        .subtitle {{
            color: #9ca3af;
            font-size: 1.1rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .stat-card {{
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #60a5fa;
        }}

        .stat-label {{
            color: #9ca3af;
            font-size: 0.875rem;
            margin-top: 4px;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            margin-bottom: 32px;
        }}

        @media (max-width: 1200px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--light);
        }}

        .chart {{
            width: 100%;
            height: 400px;
        }}

        .map-container {{
            grid-column: span 2;
        }}

        @media (max-width: 1200px) {{
            .map-container {{
                grid-column: span 1;
            }}
        }}

        .map-chart {{
            height: 500px;
        }}

        .filters {{
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .filter-label {{
            font-size: 0.75rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        select, input[type="range"] {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 8px 12px;
            color: var(--light);
            font-size: 0.875rem;
            cursor: pointer;
        }}

        select:focus, input:focus {{
            outline: none;
            border-color: var(--primary);
        }}

        .legend {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 12px;
            justify-content: center;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.875rem;
            color: #9ca3af;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        footer {{
            text-align: center;
            padding: 24px;
            color: #6b7280;
            font-size: 0.875rem;
        }}

        footer a {{
            color: #60a5fa;
            text-decoration: none;
        }}

        .insight-panel {{
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(16, 185, 129, 0.1));
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 32px;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .insight-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: #34d399;
        }}

        .insight-list {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 12px;
        }}

        .insight-item {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            color: #d1d5db;
        }}

        .insight-icon {{
            color: #60a5fa;
            flex-shrink: 0;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            padding: 12px;
            border-radius: 8px;
            font-size: 0.875rem;
            pointer-events: none;
            z-index: 1000;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <header>
            <h1>Chicago Urban Analytics</h1>
            <p class="subtitle">Context Augmented Generation (CAG) Framework &mdash; Interactive Development Analysis</p>
        </header>

        <div class="stats-grid" id="stats-grid"></div>

        <div class="insight-panel">
            <h3 class="insight-title">Key Insights</h3>
            <ul class="insight-list" id="insights-list"></ul>
        </div>

        <div class="filters">
            <div class="filter-group">
                <span class="filter-label">Metric</span>
                <select id="metric-select">
                    <option value="growth_score">Development Growth Score</option>
                    <option value="constraint_score">Zoning Constraint</option>
                    <option value="opportunity_score">Reform Opportunity</option>
                    <option value="activity_score">Permit Activity</option>
                </select>
            </div>
            <div class="filter-group">
                <span class="filter-label">Min Growth Score</span>
                <input type="range" id="growth-filter" min="0" max="1" step="0.1" value="0">
                <span id="growth-value">0</span>
            </div>
        </div>

        <div class="chart-grid">
            <div class="chart-container map-container">
                <h3 class="chart-title">Chicago ZIP Code Development Map</h3>
                <div id="map-chart" class="chart map-chart"></div>
                <div class="legend">
                    <div class="legend-item"><span class="legend-color" style="background:#22c55e"></span> High Growth</div>
                    <div class="legend-item"><span class="legend-color" style="background:#eab308"></span> Moderate Growth</div>
                    <div class="legend-item"><span class="legend-color" style="background:#ef4444"></span> Low/Declining</div>
                </div>
            </div>

            <div class="chart-container">
                <h3 class="chart-title">Top Development Growth Areas</h3>
                <div id="growth-chart" class="chart"></div>
            </div>

            <div class="chart-container">
                <h3 class="chart-title">Zoning Constraint Distribution</h3>
                <div id="constraint-chart" class="chart"></div>
            </div>

            <div class="chart-container">
                <h3 class="chart-title">Growth vs Constraint Analysis</h3>
                <div id="scatter-chart" class="chart"></div>
            </div>

            <div class="chart-container">
                <h3 class="chart-title">Permit Activity by Neighborhood</h3>
                <div id="activity-chart" class="chart"></div>
            </div>
        </div>

        <footer>
            <p>Generated by <a href="https://github.com/stochastic-sisyphus/chipop-pred-apropos">CAG Framework</a> &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p>Data sources: Chicago Open Data Portal, Census Bureau ACS</p>
        </footer>
    </div>

    <script>
        // Embedded data
        const zipData = {zip_data_json};
        const summaryStats = {summary_json};
        const topGrowth = {top_growth_json};
        const constraintDist = {constraint_json};

        // Color scales
        const getColor = (value, metric) => {{
            if (metric === 'constraint_score') {{
                // Red is bad for constraints
                if (value > 0.6) return '#ef4444';
                if (value > 0.4) return '#eab308';
                return '#22c55e';
            }}
            // Green is good for growth
            if (value > 0.6) return '#22c55e';
            if (value > 0.3) return '#eab308';
            return '#ef4444';
        }};

        // Render stats cards
        function renderStats() {{
            const grid = document.getElementById('stats-grid');
            const stats = [
                {{ value: summaryStats.total_zips, label: 'ZIP Codes Analyzed' }},
                {{ value: (summaryStats.avg_growth_score * 100).toFixed(1) + '%', label: 'Avg Growth Score' }},
                {{ value: summaryStats.top_growth_neighborhood, label: 'Top Growth Area' }},
                {{ value: summaryStats.severely_constrained, label: 'Severely Constrained' }},
                {{ value: summaryStats.high_inequality_zones, label: 'High Inequality Zones' }},
                {{ value: (summaryStats.total_population / 1000000).toFixed(2) + 'M', label: 'Total Population' }},
            ];

            grid.innerHTML = stats.map(s => `
                <div class="stat-card">
                    <div class="stat-value">${{s.value}}</div>
                    <div class="stat-label">${{s.label}}</div>
                </div>
            `).join('');
        }}

        // Render insights
        function renderInsights() {{
            const list = document.getElementById('insights-list');
            const insights = [
                `<strong>${{summaryStats.top_growth_neighborhood}}</strong> leads development with ZIP ${{summaryStats.top_growth_zip}}`,
                `${{summaryStats.severely_constrained}} areas face severe zoning constraints limiting growth`,
                `${{summaryStats.high_inequality_zones}} zones show high income inequality requiring attention`,
                `Average development growth score is ${{(summaryStats.avg_growth_score * 100).toFixed(1)}}%`,
            ];

            list.innerHTML = insights.map(i => `
                <li class="insight-item">
                    <span class="insight-icon">→</span>
                    <span>${{i}}</span>
                </li>
            `).join('');
        }}

        // Map visualization
        function renderMap(metric = 'growth_score') {{
            const colors = zipData.map(z => getColor(z[metric], metric));
            const sizes = zipData.map(z => 10 + z[metric] * 20);

            const trace = {{
                type: 'scattermapbox',
                lat: zipData.map(z => z.lat),
                lon: zipData.map(z => z.lon),
                mode: 'markers',
                marker: {{
                    size: sizes,
                    color: colors,
                    opacity: 0.8,
                }},
                text: zipData.map(z => `
                    <b>${{z.neighborhood}}</b> (${{z.zip_code}})<br>
                    Growth Score: ${{(z.growth_score * 100).toFixed(1)}}%<br>
                    Constraint: ${{z.constraint_level}}<br>
                    Total Units: ${{z.total_units}}
                `),
                hovertemplate: '%{{text}}<extra></extra>',
            }};

            const layout = {{
                mapbox: {{
                    style: 'carto-darkmatter',
                    center: {{ lat: 41.8781, lon: -87.6298 }},
                    zoom: 10,
                }},
                margin: {{ t: 0, b: 0, l: 0, r: 0 }},
                paper_bgcolor: 'rgba(0,0,0,0)',
            }};

            Plotly.newPlot('map-chart', [trace], layout, {{ responsive: true }});
        }}

        // Growth bar chart
        function renderGrowthChart() {{
            const trace = {{
                type: 'bar',
                x: topGrowth.map(z => z.neighborhood + ' (' + z.zip_code + ')'),
                y: topGrowth.map(z => z.growth_score * 100),
                marker: {{
                    color: topGrowth.map(z => getColor(z.growth_score, 'growth_score')),
                }},
                hovertemplate: '<b>%{{x}}</b><br>Growth: %{{y:.1f}}%<extra></extra>',
            }};

            const layout = {{
                xaxis: {{ tickangle: -45, color: '#9ca3af', gridcolor: 'rgba(255,255,255,0.1)' }},
                yaxis: {{ title: 'Growth Score (%)', color: '#9ca3af', gridcolor: 'rgba(255,255,255,0.1)' }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ t: 20, b: 100 }},
            }};

            Plotly.newPlot('growth-chart', [trace], layout, {{ responsive: true }});
        }}

        // Constraint pie chart
        function renderConstraintChart() {{
            const labels = Object.keys(constraintDist);
            const values = Object.values(constraintDist);
            const colors = ['#22c55e', '#eab308', '#ef4444'];

            const trace = {{
                type: 'pie',
                labels: labels,
                values: values,
                marker: {{ colors: colors }},
                textinfo: 'label+percent',
                textfont: {{ color: '#fff' }},
                hovertemplate: '<b>%{{label}}</b><br>%{{value}} ZIP codes<extra></extra>',
            }};

            const layout = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                showlegend: false,
                margin: {{ t: 20, b: 20 }},
            }};

            Plotly.newPlot('constraint-chart', [trace], layout, {{ responsive: true }});
        }}

        // Scatter plot: Growth vs Constraint
        function renderScatterChart() {{
            const trace = {{
                type: 'scatter',
                mode: 'markers',
                x: zipData.map(z => z.constraint_score * 100),
                y: zipData.map(z => z.growth_score * 100),
                marker: {{
                    size: zipData.map(z => 8 + z.total_units / 50),
                    color: zipData.map(z => z.opportunity_score),
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{ title: 'Opportunity', tickfont: {{ color: '#9ca3af' }} }},
                }},
                text: zipData.map(z => z.neighborhood + ' (' + z.zip_code + ')'),
                hovertemplate: '<b>%{{text}}</b><br>Constraint: %{{x:.1f}}%<br>Growth: %{{y:.1f}}%<extra></extra>',
            }};

            const layout = {{
                xaxis: {{ title: 'Zoning Constraint (%)', color: '#9ca3af', gridcolor: 'rgba(255,255,255,0.1)' }},
                yaxis: {{ title: 'Growth Score (%)', color: '#9ca3af', gridcolor: 'rgba(255,255,255,0.1)' }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ t: 20 }},
            }};

            Plotly.newPlot('scatter-chart', [trace], layout, {{ responsive: true }});
        }}

        // Activity treemap
        function renderActivityChart() {{
            // Group by neighborhood
            const byNeighborhood = {{}};
            zipData.forEach(z => {{
                if (!byNeighborhood[z.neighborhood]) {{
                    byNeighborhood[z.neighborhood] = {{ total_permits: 0, count: 0 }};
                }}
                byNeighborhood[z.neighborhood].total_permits += z.total_permits;
                byNeighborhood[z.neighborhood].count += 1;
            }});

            const neighborhoods = Object.entries(byNeighborhood)
                .sort((a, b) => b[1].total_permits - a[1].total_permits)
                .slice(0, 12);

            const trace = {{
                type: 'bar',
                x: neighborhoods.map(n => n[0]),
                y: neighborhoods.map(n => n[1].total_permits),
                marker: {{
                    color: neighborhoods.map((n, i) => `hsl(${{i * 30}}, 70%, 50%)`),
                }},
                hovertemplate: '<b>%{{x}}</b><br>Total Permits: %{{y}}<extra></extra>',
            }};

            const layout = {{
                xaxis: {{ tickangle: -45, color: '#9ca3af', gridcolor: 'rgba(255,255,255,0.1)' }},
                yaxis: {{ title: 'Total Permits', color: '#9ca3af', gridcolor: 'rgba(255,255,255,0.1)' }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ t: 20, b: 100 }},
            }};

            Plotly.newPlot('activity-chart', [trace], layout, {{ responsive: true }});
        }}

        // Event listeners
        document.getElementById('metric-select').addEventListener('change', (e) => {{
            renderMap(e.target.value);
        }});

        document.getElementById('growth-filter').addEventListener('input', (e) => {{
            document.getElementById('growth-value').textContent = e.target.value;
            // Could filter data here
        }});

        // Initialize
        renderStats();
        renderInsights();
        renderMap();
        renderGrowthChart();
        renderConstraintChart();
        renderScatterChart();
        renderActivityChart();
    </script>
</body>
</html>
'''

    return html


def generate_dashboard_from_outputs(output_dir: Path) -> str:
    """
    Generate dashboard from existing pipeline outputs.

    Args:
        output_dir: Directory containing pipeline output files

    Returns:
        Path to generated HTML file
    """
    # Load data files
    multifamily_path = output_dir / "models" / "multifamily_growth" / "results.json"
    zoning_path = output_dir / "models" / "zoning_impact" / "zoning_report.json"
    income_path = output_dir / "models" / "income_distribution" / "income_report.json"
    population_path = output_dir / "models" / "population_prediction" / "prediction_summary.json"

    multifamily_data = {}
    zoning_data = {}
    income_data = {}
    population_data = {}

    if multifamily_path.exists():
        with open(multifamily_path) as f:
            multifamily_data = json.load(f)

    if zoning_path.exists():
        with open(zoning_path) as f:
            zoning_data = json.load(f)

    if income_path.exists():
        with open(income_path) as f:
            income_data = json.load(f)

    if population_path.exists():
        with open(population_path) as f:
            population_data = json.load(f)

    # Generate dashboard
    docs_dir = output_dir.parent / "docs"
    html_path = docs_dir / "index.html"

    return generate_interactive_dashboard(
        html_path,
        multifamily_data,
        zoning_data,
        income_data,
        population_data,
    )


if __name__ == "__main__":
    # Generate from default output directory
    output_dir = Path("output")
    html_path = generate_dashboard_from_outputs(output_dir)
    print(f"Dashboard generated: {html_path}")
