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

#### 🛍️ Retail Gap Analysis (11 Opportunity Zones)
Top ZIP codes with unmet retail demand:
- **60640** - Highest gap score (1.00)
- **60615** - Significant retail opportunity
- **60649** - Underserved market
- Average retail per capita: 45.5 sq ft
- Average retail per housing unit: 71.5 sq ft

#### 📍 Retail Void Analysis
3 ZIP codes identified with retail category voids:
- **60601** - Missing grocery, clothing stores
- **60603** - Retail desert zones
- **60602** - Limited retail categories

### Generated Outputs

#### Data Files (7)
- `top_multifamily_zips.csv` - Emerging development zones
- `retail_lag_zips.csv` - Retail opportunity areas
- `population_forecast.csv` - 5-year population projections
- `scenario_forecasts.csv` - Growth scenarios
- `loop_adjusted_permit_balance.csv` - Downtown analysis
- `migration_flows.json` - Population movement patterns
- `development_map.geojson` - Interactive development map

#### Reports (4)
- Multifamily Growth Analysis
- Retail Gap Analysis  
- Retail Void Analysis
- Executive Summary

#### Visualizations (8)
- Growth trend charts
- Retail gap heatmaps
- Clustering analysis
- Development maps

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

## License

MIT 