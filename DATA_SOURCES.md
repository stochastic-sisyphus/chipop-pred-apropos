# Chicago Housing Pipeline Data Sources Documentation

This document provides comprehensive information about all data sources used in the Chicago Housing Pipeline & Population Shift Project. It includes API endpoints, series IDs, alternatives, and last verification dates to ensure maintainability and transparency.

## FRED API Data Sources

### Housing Price Indices

#### Primary Source
- **Source**: FRED (Federal Reserve Economic Data)
- **Series ID**: CHXRSA
- **Description**: S&P CoreLogic Case-Shiller IL-Chicago Home Price Index
- **URL**: [https://fred.stlouisfed.org/series/CHXRSA](https://fred.stlouisfed.org/series/CHXRSA)
- **Frequency**: Monthly
- **Last Verified**: June 4, 2025

#### Alternative Sources
1. **ATNHPIUS16980Q**
   - **Description**: FHFA House Price Index for Chicago-Naperville-Elgin, IL-IN-WI (MSA)
   - **URL**: [https://fred.stlouisfed.org/series/ATNHPIUS16980Q](https://fred.stlouisfed.org/series/ATNHPIUS16980Q)
   - **Frequency**: Quarterly

2. **CHXRHTSA**
   - **Description**: Home Price Index (High Tier) for Chicago, Illinois
   - **URL**: [https://fred.stlouisfed.org/series/CHXRHTSA](https://fred.stlouisfed.org/series/CHXRHTSA)
   - **Frequency**: Monthly

3. **CHXRCSA**
   - **Description**: Condo Price Index for Chicago, Illinois
   - **URL**: [https://fred.stlouisfed.org/series/CHXRCSA](https://fred.stlouisfed.org/series/CHXRCSA)
   - **Frequency**: Monthly

### Retail Sales Data

#### Primary Source
- **Source**: FRED (Federal Reserve Economic Data)
- **Series ID**: RSAFS
- **Description**: Advance Retail Sales: Retail and Food Services, Total
- **URL**: [https://fred.stlouisfed.org/series/RSAFS](https://fred.stlouisfed.org/series/RSAFS)
- **Frequency**: Monthly
- **Last Verified**: June 4, 2025
- **Notes**: National data used as proxy for Chicago retail sales (Chicago-specific series CARTSI was discontinued)

#### Alternative Sources
1. **RETSCHUS**
   - **Description**: Retail Sales: Total (Chain-Type) in United States
   - **URL**: [https://fred.stlouisfed.org/series/RETSCHUS](https://fred.stlouisfed.org/series/RETSCHUS)
   - **Frequency**: Monthly

### Other Economic Indicators

1. **CHIC917URN**
   - **Description**: Unemployment Rate in Chicago-Naperville-Elgin, IL-IN-WI (MSA)
   - **URL**: [https://fred.stlouisfed.org/series/CHIC917URN](https://fred.stlouisfed.org/series/CHIC917URN)
   - **Frequency**: Monthly

2. **NGMP16980**
   - **Description**: Total GDP for Chicago-Naperville-Elgin, IL-IN-WI (MSA)
   - **URL**: [https://fred.stlouisfed.org/series/NGMP16980](https://fred.stlouisfed.org/series/NGMP16980)
   - **Frequency**: Annual

3. **CUURA207SA0**
   - **Description**: CPI for Chicago-Naperville-Elgin, IL-IN-WI
   - **URL**: [https://fred.stlouisfed.org/series/CUURA207SA0](https://fred.stlouisfed.org/series/CUURA207SA0)
   - **Frequency**: Bi-monthly

## Census Bureau Data Sources

### Vacancy Data

#### Primary Source
- **Source**: Census Bureau American Community Survey (ACS)
- **Table**: B25002 (Occupancy Status)
- **Description**: Housing occupancy status including vacancy data
- **URL**: [https://data.census.gov/table/ACSDT1Y2022.B25002](https://data.census.gov/table/ACSDT1Y2022.B25002)
- **API Endpoint**: `https://api.census.gov/data/{year}/acs/acs5`
- **Variables**: 
  - B25002_001E: Total housing units
  - B25002_002E: Occupied units
  - B25002_003E: Vacant units
- **Frequency**: Annual
- **Last Verified**: June 4, 2025

#### Alternative Sources
1. **RRVRUSQ156N (FRED)**
   - **Description**: Rental Vacancy Rate in the United States
   - **URL**: [https://fred.stlouisfed.org/series/RRVRUSQ156N](https://fred.stlouisfed.org/series/RRVRUSQ156N)
   - **Frequency**: Quarterly
   - **Notes**: National data, not Chicago-specific

2. **Chicago Data Portal - Vacant Buildings**
   - **Dataset ID**: 7nii-7srd
   - **Description**: Vacant and Abandoned Buildings Violations
   - **URL**: [https://data.cityofchicago.org/Buildings/Vacant-and-Abandoned-Buildings-Violations/7nii-7srd](https://data.cityofchicago.org/Buildings/Vacant-and-Abandoned-Buildings-Violations/7nii-7srd)
   - **Frequency**: Daily
   - **Notes**: Provides data on vacant buildings with violations, not overall vacancy rates

### Population and Demographic Data

- **Source**: Census Bureau American Community Survey (ACS)
- **Description**: Population, income, housing, and demographic data
- **URL**: [https://www.census.gov/programs-surveys/acs/data.html](https://www.census.gov/programs-surveys/acs/data.html)
- **API Endpoint**: `https://api.census.gov/data/{year}/acs/acs5`
- **Frequency**: Annual
- **Last Verified**: June 4, 2025

## Chicago Data Portal Sources

### Building Permits

- **Dataset ID**: ydr8-5enu
- **Description**: Building permits issued by the Department of Buildings
- **URL**: [https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu](https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu)
- **Frequency**: Daily
- **Last Verified**: June 4, 2025

### Business Licenses

- **Dataset ID**: r5kz-chrr
- **Description**: Business licenses issued by the Department of Business Affairs and Consumer Protection
- **URL**: [https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr](https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr)
- **Frequency**: Daily
- **Last Verified**: June 4, 2025

### Affordable Housing

- **Dataset ID**: uahe-iimk
- **Description**: Affordable rental housing developments in Chicago
- **URL**: [https://data.cityofchicago.org/Community-Economic-Development/Affordable-Rental-Housing-Developments/s6ha-ppgi](https://data.cityofchicago.org/Community-Economic-Development/Affordable-Rental-Housing-Developments/s6ha-ppgi)
- **Frequency**: Quarterly
- **Last Verified**: June 4, 2025

## API Configuration and Retry Settings

### API Keys
The following environment variables are used for API authentication:
- `CENSUS_API_KEY`: Census Bureau API key
- `FRED_API_KEY`: Federal Reserve Economic Data API key
- `CHICAGO_DATA_TOKEN`: Chicago Data Portal API token
- `HUD_API_KEY`: Department of Housing and Urban Development API key
- `BEA_API_KEY`: Bureau of Economic Analysis API key

### Retry Settings
All API calls use the following retry configuration:
- **Maximum Retries**: 5
- **Initial Delay**: 2 seconds
- **Backoff Factor**: 2 (exponential backoff)
- **Maximum Delay**: 60 seconds
- **Jitter**: 10% (to prevent thundering herd problem)

## Data Update Frequency

| Data Source | Update Frequency | Last Updated |
|-------------|------------------|--------------|
| FRED Economic Data | Monthly | Automatic |
| Census ACS Data | Annual | When new ACS data is released |
| Building Permits | Daily | Automatic |
| Business Licenses | Daily | Automatic |
| Vacancy Data | Annual | When new ACS data is released |

## Fallback Mechanisms

The pipeline implements the following fallback mechanisms when primary data sources are unavailable:

1. **Housing Price Index**: Falls back to alternative Chicago-specific indices, then to national indices if needed
2. **Retail Sales**: Falls back to national retail sales data when Chicago-specific data is unavailable
3. **Vacancy Data**: Falls back to national vacancy rates when local data is unavailable
4. **All Sources**: Falls back to cached data when API calls fail
5. **Last Resort**: Falls back to sample data when all other options fail

## Data Validation

All data sources undergo the following validation steps:
1. Schema validation to ensure required columns are present
2. Data type validation to ensure consistent types across the pipeline
3. Range validation to ensure values are within expected ranges
4. Completeness validation to ensure sufficient data for analysis

## Maintenance Guidelines

1. **Regular Verification**: Verify all API endpoints and series IDs quarterly
2. **Update Documentation**: Update this document when any data source changes
3. **Monitor Logs**: Regularly check logs for API failures or data quality issues
4. **Update Sample Data**: Refresh sample data annually to reflect current patterns
5. **API Key Rotation**: Rotate API keys according to provider recommendations
