# Chicago Housing Pipeline

A data pipeline that analyzes housing trends, retail gaps, and population shifts in Chicago using real data from Census, FRED, and Chicago Data Portal APIs.

## Quick Start

```bash
# Clone repository
git clone https://github.com/stochastic-sisyphus/chipop-pred-apropos.git
cd chipop-pred-apropos

# Install dependencies
pip install -r requirements.txt

# Set up API keys (get from respective websites)
export CENSUS_API_KEY='your_census_key'
export FRED_API_KEY='your_fred_key'
export CHICAGO_DATA_TOKEN='your_chicago_token'

# Run pipeline
python main.py
```

## Latest Results (June 8, 2025)

### Data Collection
- **56** Chicago ZIP codes analyzed
- **1,324** FRED economic indicators
- **10,000** building permits
- **10,000** business licenses  
- **17,660** retail sales records
- **2,400** consumer spending records

### Key Findings

#### 🏗️ Multifamily Growth (10 Emerging ZIP Codes)
| ZIP Code | Growth Score | Permit Growth | Unit Growth |
|----------|-------------|---------------|-------------|
| 60602 | 0.82 | +200% | +39.4 units |
| 60654 | 0.70 | +14% | +27.0 units |
| 60661 | 0.62 | +20% | +10.8 units |
| 60605 | 0.59 | +9% | +5.9 units |
| 60607 | 0.57 | +55% | +2.5 units |

**Model Metrics:**
- Average permit growth: 11.9%
- Average unit growth: 1.42 units
- Growth score range: 0.18 - 0.82

#### 🛍️ Retail Gap Analysis (11 Opportunity Zones)
Top ZIP codes with unmet retail demand:
- **60640** - Highest gap score (1.00)
- **60615** - Significant retail opportunity
- **60649** - Underserved market
- Average retail per capita: 45.54 sq ft
- Average retail per housing unit: 71.53 sq ft

#### 📍 Retail Void Analysis
3 ZIP codes identified with retail category voids:
- **60601** - Missing grocery, clothing stores
- **60603** - Retail desert zones
- **60602** - Limited retail categories

**Void Statistics:**
- Average void score: 0.27
- Max void count: 2 categories
- Positive leakage: 17.9% of zones

### Generated Outputs

#### Data Files (7)
- `output/data/top_multifamily_zips.csv` - Top 10 emerging ZIP codes with growth scores
- `output/data/retail_lag_zips.csv` - 11 retail opportunity zones  
- `output/data/migration_flows.json` - Population flow patterns between zones
- `output/data/loop_adjusted_permit_balance.csv` - Downtown permit analysis
- `output/forecasts/population_forecast.csv` - 5-year population projections
- `output/forecasts/scenario_forecasts.csv` - Growth scenario modeling
- `output/maps/development_map.geojson` - Interactive map of development zones

#### Reports (4)
- `output/reports/multifamily_growth_report_20250608_204847.md`
- `output/reports/retail_gap_report_20250608_204847.md`  
- `output/reports/retail_void_report_20250608_204847.md`
- `output/reports/summary_report_20250608_204847.md`

#### Visualizations - Multifamily Growth (3)
- `top_emerging_zips.png` - Bar chart of top 10 emerging ZIP codes
- `growth_comparison.png` - Permit growth comparison  
- `units_comparison.png` - Unit growth analysis

#### Visualizations - Retail Gap (10)
- `retail_gap_score.png` - Gap scores by ZIP code
- `retail_housing_comparison.png` - Retail vs housing density
- `cluster_analysis.png` - K-means clustering results
- `actual_vs_predicted.png` - Model predictions
- `opportunity_zones.png` - Top opportunity zones map
- `retail_gap_distribution.png` - Distribution histogram
- Plus 4 additional analysis charts

#### Visualizations - Retail Void (8)  
- `void_count.png` - Void counts by ZIP code
- `leakage_distribution.png` - Spending leakage patterns
- `category_voids.png` - Missing retail categories
- `retail_per_capita.png` - Per capita retail analysis
- `top_leakage_zips.png` - Highest leakage zones
- Plus 3 additional analysis charts

#### Model Results (3 directories)
- `output/models/multifamily_growth/` - Growth metrics, predictions
- `output/models/retail_gap/` - Gap analysis, opportunity zones
- `output/models/retail_void/` - Void zones, leakage analysis

## Project Structure

```
chi/
├── main.py              # Run the pipeline
├── requirements.txt     # Dependencies
├── src/                 # Source code
│   ├── models/         # Analysis models
│   ├── data_collection/# API collectors
│   └── pipeline/       # Pipeline logic
├── data/               # Data storage
└── output/             # Results
    ├── data/          # CSV outputs
    ├── reports/       # Analysis reports
    ├── visualizations/# Charts
    └── maps/          # GeoJSON files
```

## API Keys Required

Get free API keys from:
- [Census API](https://api.census.gov/data/key_signup.html)
- [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
- [Chicago Data Portal](https://data.cityofchicago.org/profile/app_tokens)

## Sample Output

### Top Multifamily ZIP Codes (top_multifamily_zips.csv)
```csv
zip_code,growth_score,permit_growth,unit_growth,recent_permits,recent_units
60602,0.818,+200%,+39.4,40,84
60622,0.792,+27%,+200%,42,114
60647,0.787,-2%,+77%,44,83
60607,0.736,+29%,+1058%,36,463
60654,0.731,+82%,+82%,31,31
```

### Pipeline Summary
- **Exit Code**: 0 (Success)
- **Execution Time**: ~20 seconds
- **Data Validated**: 10 datasets, 0 rejected
- **Models Run**: 3 (all successful)
- **Files Generated**: 7 data files + 4 reports + 21 visualizations

## License

MIT 