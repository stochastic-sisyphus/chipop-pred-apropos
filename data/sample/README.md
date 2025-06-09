# Sample Data Setup Instructions

This directory contains sample data files for the Chicago Housing Pipeline & Population Shift Project. These files are provided to demonstrate the expected format and structure of data required by the pipeline.

## Required Data Files

The pipeline requires the following data files to be placed in this directory:

1. `census_data.csv` - Census demographic data for Chicago ZIP codes
2. `economic_data.csv` - Economic indicators from FRED API
3. `building_permits.csv` - Building permit data from Chicago Data Portal
4. `business_licenses.csv` - Business license data from Chicago Data Portal

## Data Format Requirements

### census_data.csv
- Must include columns: `zip_code`, `population`, `median_income`, `households`
- ZIP codes must be Chicago ZIP codes

### economic_data.csv
- Must include columns: `date`, `series_id`, `value`
- Should contain economic indicators like housing prices, employment rates, etc.

### building_permits.csv
- Must include columns: `id`, `permit_type`, `address`, `zip_code`, `issue_date`, `estimated_cost`, `unit_count`
- Should contain recent building permit data for Chicago

### business_licenses.csv
- Must include columns: `id`, `license_id`, `account_number`, `legal_name`, `business_name`, `license_code`, `license_description`, `business_activity`, `address`, `zip_code`, `ward`, `precinct`, `license_status`, `application_type`, `application_created_date`, `application_requirements_complete`, `payment_date`, `license_start_date`, `expiration_date`, `license_approved_for_issuance`, `date_issued`, `license_status_change_date`
- Should contain business license data for Chicago

## Obtaining Real Data

To obtain real data for the pipeline:

1. **Census Data**: Download from the U.S. Census Bureau's American Community Survey (ACS)
   - Visit: https://data.census.gov/
   - Select Chicago area ZIP codes
   - Download demographic data including population, income, and households

2. **Economic Data**: Download from FRED (Federal Reserve Economic Data)
   - Visit: https://fred.stlouisfed.org/
   - Search for Chicago economic indicators
   - Download data in CSV format

3. **Building Permits**: Download from Chicago Data Portal
   - Visit: https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu
   - Download recent building permit data in CSV format

4. **Business Licenses**: Download from Chicago Data Portal
   - Visit: https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr
   - Download recent business license data in CSV format

## API Keys

Some data sources require API keys. Set these as environment variables:

```bash
export CENSUS_API_KEY="your_census_api_key"
export FRED_API_KEY="your_fred_api_key"
export CHICAGO_DATA_PORTAL_TOKEN="your_chicago_data_portal_token"
export BEA_KEY="your_bea_key"
```

## Sample Data Files

Sample data files are provided in this directory to demonstrate the expected format. These files contain a limited subset of real data sufficient for testing the pipeline. For production use, replace these with complete datasets from the sources mentioned above.

## Data Validation

The pipeline includes validation checks to ensure your data meets the requirements. If validation fails, the pipeline will report specific issues that need to be addressed rather than generating synthetic data.
