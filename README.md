# Chicago Population Analysis

This project analyzes population shifts in Chicago, focusing on building permits, economic factors, and demographic changes to predict future population trends by ZIP code.

## Project Overview

The Chicago Population Analysis uses historical data on building permits, economic indicators, and demographic changes to model and predict population shifts across Chicago ZIP codes. It employs machine learning to forecast future population change and generates multiple scenarios to support urban planning and development decisions.

## Key Features

- **Future Population Prediction**: Uses current-year features to predict next year's population change (t+1)
- **Multi-scenario Forecasting**: Generates optimistic, neutral, and pessimistic scenarios for 1-year and 2-year prediction windows
- **Economic Impact Analysis**: Assesses how economic factors influence population growth patterns
- **ZIP Code-level Insights**: Provides detailed analysis for each ZIP code's growth potential
- **Visualization Suite**: Comprehensive charts showing relationships between permits and future growth
- **Housing-Retail Balance Analysis**: Identifies areas with housing development outpacing retail growth
- **10-Year Permit Growth Analysis**: Tracks areas with significant (≥20%) permit growth over the decade
- **Unit Difference Calculation**: Quantifies absolute housing unit increases between time periods

## Data Sources

- Building permit data from Chicago's Open Data Portal
- Census population data by ZIP code
- Economic indicators from Federal Reserve Economic Data (FRED)
- Housing indices and mortgage rates
- Business license data for retail establishment tracking

## Setup and Running

### Prerequisites

- Python 3.9+
- Required libraries listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
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

To run just the emerging housing areas analysis:

```bash
python chipop/analyze_emerging_housing_areas.py
```

### Code Quality and Linting

This project uses linting tools to maintain code quality:

- **Python**: We follow PEP 8 style guidelines. You can lint Python files with:
  ```bash
  # For Python files
  pip install pylint
  pylint src/*.py
  ```

- **JavaScript**: We use ESLint for JavaScript files. Run:
  ```bash
  # For JavaScript files (if any)
  npm install eslint --save-dev
  npx eslint .
  ```

The `.eslintignore` file excludes non-JavaScript files from linting. You can fix most issues automatically with:
```bash
npx eslint . --fix
```

### Recent Changes and Fixes

Recent improvements to the codebase include:

1. **New Analysis Features**:
   - Added 10-year permit growth analysis with 20% growth threshold
   - Implemented unit difference calculation between time periods
   - Created new visualizations for long-term housing development patterns

2. **Modeling Fixes**:
   - Fixed return value handling in `train_model()` function
   - Improved scenario generation for future population predictions
   - Added proper validation for data preparation steps

3. **Code Quality**:
   - Added comprehensive docstrings to all functions
   - Implemented consistent error handling
   - Enhanced logging for better debugging

4. **Pipeline Integration**:
   - Consolidated multiple pipeline scripts
   - Fixed dependency handling
   - Added validation steps between pipeline stages

5. **Documentation**:
   - Improved README with detailed instructions
   - Added ESLint configuration
   - Updated inline code comments

6. **Error Handling**:
   - Improved graceful degradation when data sources are missing
   - Added validation for merged dataset integrity
   - Fixed unpacking errors in model training functions

## Project Structure

```
chicago/
├── src/                  # Source code
│   ├── data_collection.py   # Data collection utilities
│   ├── data_processing.py   # Data cleaning and transformation
│   ├── analysis.py          # Analysis functions
│   ├── modeling.py          # Prediction models
│   └── visualization.py     # Visualization utilities
├── chipop/               # Chicago Population Analysis
│   ├── analyze_emerging_housing_areas.py  # Housing-retail balance analysis
│   ├── output/               # Generated CSV outputs
│   ├── visualizations/       # Generated charts
│   └── reports/              # Analysis reports
├── output/               # Generated output
│   ├── data/                # Raw and processed data files
│   ├── models/              # Trained models and metrics
│   ├── visualizations/      # Generated charts and visualizations
│   └── reports/             # Analysis reports
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks for exploration
└── tests/                # Unit tests
```

## Modeling Methodology

The project uses an ensemble machine learning approach to predict future population changes:

1. **Target Variable**: `population_change_next_year` - the percentage change in population for the next year
2. **Features**: Current year economic indicators, permit activity, and demographic metrics
3. **Train-Test Split**: Time-based split with training data up to 2020 and testing on more recent years
4. **Scenarios**: Adjustments to key economic factors to generate multiple prediction scenarios
5. **Prediction Windows**: Both 1-year and 2-year forecasts are generated

## Key Visualizations

- **Permit vs Future Population Change**: Shows how current year permit activity correlates with next year's population change
- **ZIP Code Heatmap**: Correlations between permit activity and future growth by ZIP category
- **Scenario Predictions**: Distribution of predicted changes across different economic scenarios
- **Top Growing ZIP Codes**: Identifies areas with highest predicted growth potential
- **10-Year Permit Growth**: Charts showing areas with ≥20% permit growth over the decade
- **Housing-Retail Balance**: Visualizes imbalances between housing development and retail establishments

## Reports

The analysis generates these key reports:
- **Chicago Population Analysis Report**: Comprehensive overview of findings and methodology
- **ZIP Code Summary**: Detailed breakdown of growth patterns by neighborhood
- **Economic Impact Analysis**: Assessment of how economic factors influence population shifts
- **Housing-Retail Balance Report**: Analysis of retail deficits in emerging housing areas
- **10-Year Growth Analysis**: Long-term housing development patterns by ZIP code
=======
Copyright (c) 2025 [stochastic-sisyphus]. Created as part of development work for Baker Enterprises.


