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

## Latest Results (June 8, 2025 - 22:27:56)

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
| 60602 | 0.84 | +2.4% | +77.6% |
| 60622 | 0.80 | +30.3% | +202.6% |
| 60647 | 0.78 | -4.3% | +72.9% |
| 60607 | 0.74 | +27.6% | +1031.7% |
| 60654 | 0.70 | +72.2% | +72.2% |

**Model Metrics:**
- Average permit growth: 9.3%
- Average unit growth: 1.37 units
- Growth score range: 0.18 - 0.84

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
- `output/reports/multifamily_growth_report_20250608_222756.md`
- `output/reports/retail_gap_report_20250608_222756.md`  
- `output/reports/retail_void_report_20250608_222756.md`
- `output/reports/summary_report_20250608_222756.md`

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
60602,0.843,0.024,0.776,43,87
60622,0.797,0.303,2.026,43,115
60647,0.779,-0.043,0.729,44,83
60607,0.741,0.276,10.317,37,464
60654,0.703,0.722,0.722,31,31
```

### Pipeline Summary
- **Exit Code**: 0 (Success)
- **Execution Time**: ~20 seconds
- **Data Validated**: 10 datasets, 0 rejected
- **Models Run**: 3 (all successful)
- **Files Generated**: 7 data files + 4 reports + 21 visualizations

## License

MIT 