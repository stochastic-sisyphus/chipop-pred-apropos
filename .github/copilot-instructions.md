# Copilot Instructions for Chicago Housing Pipeline (ChiPop Pred Apropos)

## Project Overview

This is a data pipeline that analyzes housing trends, retail gaps, and population shifts in Chicago using real data from Census, FRED, and Chicago Data Portal APIs. The project identifies:
- **Emerging multifamily development zones** (growth scores for ZIPs)
- **Retail gap opportunities** (areas with unmet retail demand)
- **Retail void analysis** (missing retail categories by ZIP)

## Project Structure

```
chipop-pred-apropos/
├── main.py                    # Entry point - Run the pipeline
├── requirements.txt           # Python dependencies
├── setup_api_keys.py         # API key configuration helper
├── src/                      # Source code
│   ├── config/              # Configuration settings
│   ├── data_collection/     # API collectors (Census, FRED, Chicago Data Portal)
│   ├── data_processing/     # Data transformation and cleaning
│   ├── data_validation/     # Data quality checks
│   ├── features/            # Feature engineering
│   ├── models/              # Analysis models
│   │   ├── multifamily_growth_model.py
│   │   ├── retail_gap_model.py
│   │   ├── retail_void_model.py
│   │   └── population_forecast_model.py
│   ├── pipeline/            # Pipeline orchestration
│   ├── reporting/           # Report generation (Markdown format)
│   ├── utils/               # Helper utilities
│   └── visualization/       # Chart and map generation
├── data/                    # Data storage
│   ├── raw/                # Raw API data
│   ├── interim/            # Partially processed data
│   ├── processed/          # Final clean datasets
│   └── sample/             # Sample/test data
├── output/                  # Generated outputs
│   ├── data/               # CSV results
│   ├── reports/            # Markdown reports
│   ├── visualizations/     # PNG charts
│   └── models/             # Model metrics and predictions
├── tests/                   # Test suite
└── examples/               # Example scripts
```

## Key Technologies

- **Language**: Python 3.8+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, lightgbm
- **Geospatial**: geopandas, uszipcode
- **Visualization**: matplotlib, seaborn, plotly
- **APIs**: census, fredapi, sodapy (Socrata)
- **Testing**: pytest

## Setup and Running

### Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Set up API keys (get free keys from respective websites):
   - Census API: https://api.census.gov/data/key_signup.html
   - FRED API: https://fred.stlouisfed.org/docs/api/api_key.html
   - Chicago Data Portal: https://data.cityofchicago.org/profile/app_tokens

3. Configure environment variables:
```bash
export CENSUS_API_KEY='your_census_key'
export FRED_API_KEY='your_fred_key'
export CHICAGO_DATA_TOKEN='your_chicago_token'
```

### Running the Pipeline
```bash
python main.py
```

### Running Tests
```bash
# Run all tests with pytest (recommended - can run unittest tests)
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py

# Run with verbose output
pytest -v tests/

# Or run with unittest directly
python -m unittest tests.test_pipeline
```

## Code Style and Conventions

### General Guidelines
- **Python Style**: Follow PEP 8
- **Line Length**: Max 120 characters (existing code uses this)
- **Imports**: Group by standard library, third-party, local modules
- **Type Hints**: Use where it improves clarity, but not required throughout
- **Docstrings**: Use triple-quoted strings for modules, classes, and functions

### Naming Conventions
- **Files**: Use snake_case (e.g., `multifamily_growth_model.py`)
- **Classes**: Use PascalCase (e.g., `MultifamilyGrowthModel`)
- **Functions/Variables**: Use snake_case (e.g., `run_analysis`, `top_emerging_zips`)
- **Constants**: Use UPPER_SNAKE_CASE (e.g., `CENSUS_API_KEY`, `DATA_DIR`)

### Logging
- Use the `loguru` or standard `logging` library
- Log at appropriate levels: DEBUG, INFO, WARNING, ERROR
- Example:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Starting analysis for ZIP codes")
```

### Error Handling
- Use try-except blocks for API calls and file operations
- Provide meaningful error messages
- Log errors with traceback information
- Return success/failure booleans from major operations

## Working with Models

### Model Base Class Pattern
All analysis models inherit from `BaseModel` and should:
1. Accept an `output_dir` parameter
2. Implement a `run_analysis(data)` method that returns boolean success
3. Store results in `self.results` dictionary
4. Generate visualizations in the output directory
5. Return top findings (e.g., `self.top_emerging_zips`)

### Example Model Structure
```python
class MyModel(BaseModel):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.results = {}
    
    def run_analysis(self, data):
        try:
            # Analysis logic here
            self.results = {...}
            return True
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return False
```

## Working with Data Collection

### API Collectors
- Located in `src/data_collection/`
- Each collector handles a specific API (Census, FRED, Chicago Data Portal)
- Implement retry logic with exponential backoff
- Cache responses to avoid redundant API calls
- Return pandas DataFrames

### Data Validation
- Use `src/data_validation/` to validate data quality
- Check for required columns, data types, and value ranges
- Log validation warnings but don't fail pipeline unless critical

## Working with Reports

### Report Format
- **Generate Markdown reports**, not HTML
- Use clear headings (# ## ###) and tables
- Include key metrics, findings, and visualizations
- Save to `output/reports/` with timestamp suffix
- Example filename: `multifamily_growth_report_20250608_222756.md`

### Report Content Structure
1. Title and metadata (date, time, data sources)
2. Executive summary
3. Key findings with metrics
4. Tables of top ZIP codes or opportunities
5. Data quality notes
6. Methodology notes (if applicable)

## Testing Guidelines

### Test Structure
- Tests are in `tests/` directory
- Use `unittest` framework (existing tests) or `pytest` (can run both)
- Create synthetic test data in `setUpClass` method
- Clean up test outputs in `tearDownClass` method

### What to Test
- Directory structure exists
- Models run successfully with test data
- Reports generate in correct format (Markdown, not HTML)
- Visualizations are created
- Pipeline completes end-to-end
- Data validation works correctly

### Running Tests
- Tests should work without API keys (use sample/cached data)
- Tests create temporary output in `test_output/` directory
- Tests should be fast (<5 minutes total)

## Common Tasks

### Adding a New Model
1. Create new file in `src/models/` (e.g., `new_model.py`)
2. Inherit from `BaseModel`
3. Implement `run_analysis(data)` method
4. Add model to pipeline in `src/pipeline/pipeline.py`
5. Create corresponding report in `src/reporting/`
6. Add tests in `tests/test_pipeline.py`

### Adding a New Data Source
1. Create collector in `src/data_collection/` (e.g., `new_collector.py`)
2. Implement API client with error handling and retries
3. Add caching logic
4. Add data validation checks
5. Update pipeline to use new collector
6. Add tests for the collector

### Modifying Visualizations
- Charts are in `src/visualization/`
- Use matplotlib/seaborn for static charts (PNG output)
- Use plotly for interactive charts (HTML output)
- Save all charts to `output/visualizations/` or model-specific subdirectories
- Use consistent color schemes and styling

## API Rate Limits and Caching

### Rate Limits
- **Census API**: 500 requests per day
- **FRED API**: No official limit, but be respectful
- **Chicago Data Portal**: 1000 requests per rolling hour period

### Caching Strategy
- Cache all API responses to `data/raw/` or `data/interim/`
- Use timestamp-based cache invalidation
- Implement cache refresh via `AutoRefresher` class
- Check cache first before making API calls

## Data Pipeline Flow

1. **Data Collection** (`src/data_collection/`)
   - Census demographics
   - FRED economic indicators
   - Building permits
   - Business licenses
   - Retail sales

2. **Data Processing** (`src/data_processing/`)
   - Clean and standardize
   - Handle missing values
   - Normalize formats

3. **Data Validation** (`src/data_validation/`)
   - Check data quality
   - Validate schemas
   - Flag anomalies

4. **Feature Engineering** (`src/features/`)
   - Calculate growth rates
   - Create composite scores
   - Generate spatial features

5. **Model Analysis** (`src/models/`)
   - Multifamily growth scoring
   - Retail gap identification
   - Retail void detection
   - Population forecasting

6. **Reporting** (`src/reporting/`)
   - Generate Markdown reports
   - Create visualizations
   - Export data tables

7. **Output** (`output/`)
   - CSV data files
   - Markdown reports
   - PNG visualizations
   - Model metrics

## Dependencies and Environment

### Core Dependencies
- pandas >= 2.2.3
- numpy >= 1.24.4
- scikit-learn >= 1.6.1
- matplotlib >= 3.10.1
- geopandas >= 1.0.1

### Optional Dependencies
- cmdstanpy (for Bayesian modeling)
- lightgbm (for gradient boosting)

### Virtual Environment
Recommended to use virtual environment:
```bash
python -m venv venv
source venv/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Security and Privacy

- **Never commit API keys** to the repository
- Use environment variables or `.env` file (add to `.gitignore`)
- Sample data should not contain PII
- Cache files in `data/` are gitignored

## Known Issues and Limitations

- Some FRED series IDs may change over time (check FRED documentation)
- Census API can be slow during peak times
- Chicago Data Portal occasionally has downtime
- Large datasets (>10K records) may require pagination

## Output Files

### Data Files (`output/data/`)
- `top_multifamily_zips.csv` - Top 10 emerging ZIP codes
- `retail_lag_zips.csv` - Retail opportunity zones
- `migration_flows.json` - Population flow patterns
- `population_forecast.csv` - 5-year population projections

### Reports (`output/reports/`)
- `multifamily_growth_report_YYYYMMDD_HHMMSS.md`
- `retail_gap_report_YYYYMMDD_HHMMSS.md`
- `retail_void_report_YYYYMMDD_HHMMSS.md`
- `summary_report_YYYYMMDD_HHMMSS.md`

### Visualizations (`output/visualizations/`)
- Various PNG charts organized by model type
- GeoJSON maps for spatial analysis

## Performance Considerations

- Data collection can take 15-30 minutes with fresh API calls
- Use cached data for development/testing to speed up iterations
- Large ZIP code analysis (all Chicago ZIPs ~57) can be memory intensive
- Consider chunking for very large datasets

## Contact and Resources

- **Repository**: https://github.com/stochastic-sisyphus/chipop-pred-apropos
- **Census API Docs**: https://www.census.gov/data/developers/data-sets.html
- **FRED API Docs**: https://fred.stlouisfed.org/docs/api/
- **Chicago Data Portal**: https://data.cityofchicago.org/

## Additional Notes for Copilot

- When generating new code, match the existing code style and patterns
- Preserve existing error handling and logging patterns
- Test changes with sample data before using live API calls
- Update documentation if adding new features or changing behavior
- Follow the existing model/report/visualization structure for consistency
- Always generate Markdown reports (not HTML) for publication-ready output
- Use existing utility functions in `src/utils/` when possible
