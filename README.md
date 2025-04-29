# Chicago Population Analysis Pipeline

A comprehensive tool for analyzing and predicting population shifts in Chicago based on housing pipeline data, demographic trends, and economic indicators. This project provides structured forecasts and visualizations to inform urban planning, housing development strategy, and real estate investment prioritization.

## Features

- **Data Collection & Integration**
  - Chicago Data Portal (building permits, zoning)
  - U.S. Census Bureau (population, demographics)
  - Federal Reserve Economic Data (FRED) (economic indicators)

- **Analysis Components**
  - ZIP-level population forecasting
  - Scenario-based modeling
  - Early detection of emerging neighborhoods
  - Housing-retail imbalance analysis
  - Evidence-based decision-making support

- **Outputs**
  - Structured CSV data files
  - Interactive visualizations
  - Comprehensive reports
  - Optional interactive dashboard

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt
- API keys:
  - Census Bureau API key
  - FRED API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chipop-final.git
cd chipop-final
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file to add your API keys
```

## Usage

Run the complete analysis pipeline:
```bash
python main.py
```

## Project Structure

```
chipop-final/
├── data/                  # Data storage
│   ├── raw/              # Original data
│   ├── interim/          # Intermediate data
│   └── processed/        # Final datasets
├── src/                  # Source code
│   ├── data_collection/  # Data collection modules
│   ├── data_processing/  # Data processing modules
│   ├── models/          # Model definitions
│   ├── visualization/   # Visualization code
│   ├── utils/          # Utility functions
│   └── config/         # Configuration settings
├── output/             # Generated analysis
│   ├── models/        # Trained models
│   ├── visualizations/ # Generated graphics
│   └── reports/      # Generated reports
├── logs/             # Log files
├── tests/           # Test suite
├── .env            # Environment variables
├── .env.example   # Example environment variables
├── README.md     # Project documentation
└── requirements.txt # Project dependencies
```

## Output Files

### CSV Outputs (output/)
- feature_importance.csv - Ranked feature contributions
- population_shift_patterns.csv - ZIP-level changes
- retail_deficit_predictions.csv - Retail gaps by ZIP
- scenario_predictions.csv - Economic scenario forecasts
- zip_summary.csv - Master ZIP-level summary
- And more...

### Visualizations (output/visualizations/)
- Population trends
- Permit activity
- Economic indicators
- Housing-retail balance
- Interactive maps

### Reports (output/reports/)
- EXECUTIVE_SUMMARY.md - Key findings
- chicago_population_analysis_report.md - Technical report
- economic_impact_analysis.md - Economic study
- housing_retail_balance_report.md - Development analysis
- chicago_zip_summary.md - ZIP-level breakdown

## Data Sources

1. **Building Permits**
   - Source: Chicago Data Portal
   - Dataset: Building Permits (ydr8-5enu)
   - Updates: Daily

2. **Zoning Information**
   - Source: Chicago Data Portal
   - Dataset: Zoning Ordinance (dj47-wfun)
   - Updates: As changes occur

3. **Population & Demographics**
   - Source: U.S. Census Bureau
   - Dataset: American Community Survey (ACS)
   - Updates: Annual

4. **Economic Indicators**
   - Source: Federal Reserve Economic Data (FRED)
   - Datasets: Interest rates, Consumer confidence
   - Updates: Monthly/Quarterly

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- City of Chicago Data Portal for providing open access to city data
- U.S. Census Bureau for demographic and population data
- Federal Reserve Economic Data (FRED) for economic indicators 