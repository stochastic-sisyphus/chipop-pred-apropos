# Chicago Population Shift Analysis

A comprehensive tool for analyzing and predicting population shifts in Chicago based on housing pipeline data, demographic trends, and economic indicators.

## Features

- Data collection from multiple sources:
  - Chicago Data Portal (building permits, zoning)
  - U.S. Census Bureau (population, demographics)
  - Federal Reserve Economic Data (FRED)
- Integrated data processing pipeline
- Population shift modeling with scenario analysis
- Detailed impact analysis by zip code
- Visualization capabilities

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt
- API keys:
  - Census Bureau API key
  - FRED API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/stochastic-sisyphus/chipop-pred-apropos.git
cd chipop-pred-apropos
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

1. Run the complete analysis pipeline:
```bash
python src/run_population_analysis.py
```

2. View results in the output directory:
- model_metrics.csv: Model performance metrics
- feature_importance.csv: Feature importance analysis
- top_impacted_areas.csv: Areas with highest predicted population changes
- scenario_impact.csv: Impact analysis under different scenarios

## Data Sources

1. Building Permits:
- Source: Chicago Data Portal
- Dataset: Building Permits (ydr8-5enu)
- Updates: Daily

2. Zoning Information:
- Source: Chicago Data Portal
- Dataset: Zoning Ordinance (dj47-wfun)
- Updates: As changes occur

3. Population & Demographics:
- Source: U.S. Census Bureau
- Dataset: American Community Survey (ACS)
- Updates: Annual

4. Economic Indicators:
- Source: Federal Reserve Economic Data (FRED)
- Datasets: Interest rates, Consumer confidence
- Updates: Monthly/Quarterly

## Analysis Components

1. Data Collection:
- Automated collection from multiple sources
- Data validation and quality checks
- Historical data retrieval (20-year lookback)

2. Data Processing:
- Cleaning and standardization
- Feature engineering
- Dataset integration

3. Modeling:
- Random Forest regression
- Feature importance analysis
- Cross-validation

4. Scenario Analysis:
- Optimistic scenario (low interest rates, high confidence)
- Neutral scenario (current conditions)
- Pessimistic scenario (high interest rates, low confidence)

## Project Structure

```
chipop/
├── data/               # Data storage
│   ├── raw/           # Original, immutable data
│   ├── interim/       # Intermediate data
│   └── processed/     # Final, canonical data sets
├── src/               # Source code
│   ├── data_processing/   # Data collection and processing
│   ├── models/           # Model definitions and training
│   ├── utils/           # Utility functions
│   └── visualization/   # Visualization code
├── output/            # Generated analysis
│   ├── models/       # Trained model files
│   └── visualizations/ # Generated graphics
├── tests/            # Test suite
├── .env              # Environment variables
├── .gitignore        # Git ignore rules
├── DASHBOARD.html    # Main dashboard
├── README.md         # Project documentation
└── main.py          # Main entry point

Key Files:
- main.py: Primary script to run the complete analysis pipeline
- DASHBOARD.html: Interactive visualization of results
- .env: Configuration for API keys and settings
```

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
