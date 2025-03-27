# Chicago Population Analysis (ChiPop)

A comprehensive analysis of population shifts in Chicago, focusing on building permits, economic factors, and demographic changes to predict future population trends by ZIP code.

## Project Overview

This project analyzes historical data on building permits, economic indicators, and demographic changes to model and predict population shifts across Chicago ZIP codes. It employs machine learning to forecast future population change and generates multiple scenarios (optimistic, neutral, pessimistic) to support urban planning and development decisions.

## Key Features

- **Future Population Prediction**: Uses current-year features to predict next year's population change (t+1)
- **Multi-scenario Forecasting**: Generates optimistic, neutral, and pessimistic scenarios for 1-year and 2-year prediction windows
- **Economic Impact Analysis**: Assesses how economic factors influence population growth patterns
- **ZIP Code-level Insights**: Provides detailed analysis for each ZIP code's growth potential
- **Visualization Suite**: Comprehensive charts showing relationships between permits and future growth

## Data Sources

- Building permit data from Chicago's Open Data Portal
- Census population data by ZIP code
- Economic indicators from Federal Reserve Economic Data (FRED)
- Housing indices and mortgage rates

## Setup and Running

### Prerequisites

- Python 3.9+
- Required libraries listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd chipop

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file to add your API keys
```

### Running the Analysis

To run the full pipeline:

```bash
python run_chicago_analysis.py --steps all
```

For specific steps:

```bash
python run_chicago_analysis.py --steps data
python run_chicago_analysis.py --steps process
python run_chicago_analysis.py --steps model
python run_chicago_analysis.py --steps visualize
```

## Project Structure

```
chipop/
├── src/                  # Source code
│   ├── data_collection.py   # Data collection utilities
│   ├── data_processing.py   # Data cleaning and transformation
│   ├── analysis.py          # Analysis functions
│   ├── modeling.py          # Prediction models
│   └── visualization.py     # Visualization utilities
├── data/                 # Raw input data
├── output/               # Generated output
│   ├── models/              # Trained models and metrics
│   ├── visualizations/      # Generated charts and visualizations
│   └── reports/             # Analysis reports
├── reports/              # Markdown reports
│   ├── chicago_population_analysis_report.md
│   ├── chicago_zip_summary.md
│   └── economic_impact_analysis.md
├── dashboard/            # Interactive dashboard files
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Key Reports

- **Chicago Population Analysis Report**: Comprehensive overview of findings and methodology
- **ZIP Code Summary**: Detailed breakdown of growth patterns by neighborhood
- **Economic Impact Analysis**: Assessment of how economic factors influence population shifts

## Modeling Methodology

The project uses an ensemble machine learning approach to predict future population changes:

1. **Target Variable**: `population_change_next_year` - the percentage change in population for the next year
2. **Features**: Current year economic indicators, permit activity, and demographic metrics
3. **Train-Test Split**: Time-based split with training data up to 2020 and testing on more recent years
4. **Scenarios**: Adjustments to key economic factors to generate multiple prediction scenarios
5. **Prediction Windows**: Both 1-year and 2-year forecasts are generated

## License

Copyright (c) 2025 [stochastic-sisyphus]. Created as part of development work for Baker Enterprises.

## Acknowledgments

- Chicago Data Portal for providing open data
- Census Bureau for demographic data
- Federal Reserve Economic Data (FRED) for economic indicators
