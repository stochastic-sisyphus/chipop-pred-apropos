# Chicago Housing Pipeline & Population Shift Project - Todo List

## Data Collection
- [ ] Fix Census API integration for all Chicago ZIP codes
- [ ] Implement proper FRED API data collection with retry logic
- [ ] Add migration data collection and integration
- [ ] Implement BEA retail GDP data collection
- [ ] Enhance retail vacancy data collection
- [ ] Add robust error handling for all API calls

## Data Processing
- [ ] Implement ZIP code derivation for building permits data
- [ ] Replace hardcoded column dependencies with dynamic column detection
- [ ] Fix data merging to handle missing or incomplete data files
- [ ] Implement proper validation for processed data
- [ ] Add geocoding capabilities for address-to-ZIP conversion

## Model Implementation
- [ ] Fix MultifamilyGrowthModel to handle missing columns dynamically
- [ ] Enhance RetailVoidModel to properly integrate with demographic data
- [ ] Improve visualization quality and completeness
- [ ] Add spatial analysis and mapping capabilities
- [ ] Implement progressive fallback logic for model parameters

## Report Generation
- [ ] Fix template-data mismatches in all report modules
- [ ] Implement robust context preparation with defensive handling
- [ ] Enhance markdown template generation
- [ ] Improve visualization integration in reports
- [ ] Add validation for report outputs

## Pipeline Architecture
- [ ] Eliminate all synthetic data generation
- [ ] Implement proper error handling and reporting
- [ ] Add comprehensive validation against project requirements
- [ ] Enhance pipeline resilience to continue despite non-critical errors
- [ ] Implement proper logging throughout the pipeline

## Project Requirements
- [ ] Ensure identification of top 5 ZIPs with new multifamily growth
- [ ] Complete retail development gaps analysis
- [ ] Implement flagging of high housing growth + low retail areas
- [ ] Add 10-year population forecasts
- [ ] Implement attribution of new development vs. baseline growth
- [ ] Add migration maps and flow analysis
- [ ] Implement scenario-based projections
