# Example Pipeline Outputs

This directory contains example outputs from running the Chicago Housing Pipeline with production data.

## Contents

### Reports (`output/reports/`)
- **`multifamily_growth_report_20250607_202754.md`** - Comprehensive analysis of multifamily housing growth patterns with identified emerging ZIP codes
- **`retail_gap_report_20250607_204655.md`** - **FULLY POPULATED!** Complete retail gap analysis with 11 opportunity zones, statistical metrics, and cluster analysis
- **`retail_void_report_20250607_204655.md`** - **FULLY POPULATED!** Complete retail void analysis with 3 void zones, leakage patterns (-0.430 mean), and statistical overview  
- **`summary_report_20250607_204655.md`** - **FULLY POPULATED!** Complete executive summary with meaningful data across all models

### Visualizations (`output/visualizations/`)
- **`multifamily_growth/`** - Charts showing:
  - `growth_comparison.png` - Growth trends comparison across ZIP codes
  - `top_emerging_zips.png` - Top emerging areas for multifamily development
  - `units_comparison.png` - Housing unit growth analysis
- **`retail_void/`** - **NEW!** Retail void analysis charts:
  - `leakage_distribution.png` - Distribution of spending leakage across ZIP codes
  - `void_count.png` - Retail void zone identification

### Data Outputs (`output/data/`)
- **`top_multifamily_zips.csv`** - Top ZIP codes for multifamily development with growth metrics
- **`retail_lag_zips.csv`** - ZIP codes with retail gaps/opportunities
- **`migration_flows.json`** - Population migration patterns data
- **`loop_adjusted_permit_balance.csv`** - Permit activity analysis

### Maps (`output/maps/`)
- **`development_map.geojson`** - Geographic visualization of development opportunities

### Models (`output/models/`)
- **`results.json`** - Model execution results and key metrics

## Pipeline Run Details

- **Generated**: June 7, 2025 at 20:27:54
- **Data Sources**: Census API, FRED API, Chicago Data Portal
- **Models Executed**: Multifamily Growth, Retail Gap, Retail Void
- **Total ZIP Codes Analyzed**: 56
- **Emerging Areas Identified**: 10

## Key Insights

- **Top Emerging ZIP Code**: 60633 with growth score of -0.75
- **Opportunity Areas**: 11 ZIP codes identified with retail gaps
- **Data Coverage**: Successfully integrated building permits, business licenses, and economic indicators

## Usage

These files demonstrate the pipeline's capabilities and can be used to:
- Understand the analysis methodology
- Review report formats and visualizations
- Examine data structure and outputs
- Validate pipeline functionality

To generate fresh outputs, run:
```bash
python main.py
``` 