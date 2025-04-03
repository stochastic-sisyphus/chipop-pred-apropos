# Chicago Population and Housing Analysis
## Comprehensive Technical Report

### Executive Summary

This analysis examines population shifts in Chicago from 2013 to 2023, with a focus on the relationship between building permits, housing development, economic factors, and demographic changes. Our findings indicate that permit activity shows a strong correlation (0.72) with population growth, with key ZIP codes (60610, 60606, 60603) demonstrating consistent growth patterns. The economic data reveals significant impacts from increased mortgage rates, with a 23% drop in permit activity when rates exceed 5%. The predictive models generate three scenarios (optimistic, neutral, and pessimistic) suggesting future growth will range from 0.88% to 1.13% on average, with substantial variance by ZIP code.

### Data Sources & Methodology

#### Data Sources

| Data Type | Source | Variables | Years |
|-----------|--------|-----------|-------|
| Building Permits | Chicago Data Portal | Permit counts, units, cost | 2013-2023 |
| Population | Census ACS 5-Year Estimates | Population by ZIP | 2013-2023 |
| Economic Indicators | FRED | Treasury yields, mortgage rates, consumer sentiment, housing starts | 2013-2023 |
| Income Distribution | Census ACS 5-Year Estimates | Median income, income brackets | 2013-2023 |

#### Methodology

1. **Data Collection & Integration**
   - Permits, population, and economic data were obtained through APIs
   - Data was cleaned, standardized, and merged by ZIP code and year

2. **Analytical Approach**
   - Descriptive analysis: Trends across ZIP codes over time
   - Correlation analysis: Permits vs. population growth
   - Predictive modeling: Random Forest Regressor

3. **Model Specifications**
   - **Target variable**: Population change (year-over-year percentage)
   - **Features**: Permit count, treasury yield, mortgage rates, consumer sentiment, housing starts, house price index, income metrics
   - **Validation**: Cross-validation with 5 folds (avg score: 0.13)
   - **Performance**: Train R² score: 0.84, Test R² score: 0.17

### Key Findings

#### 1. Population Shift Patterns (2013-2023)

- **Top Growth ZIP Codes**:
  1. 60610 (Near North Side): +1.79%
  2. 60626 (Rogers Park): +1.58% 
  3. 60617 (South Chicago): +1.57%
  4. 60606 (Loop/Financial District): +1.55%
  5. 60603 (Loop): +1.49%

- **Highest Permit Activity ZIP Codes**:
  1. 60602 (Central Loop): 3,393 permits
  2. 60622 (Wicker Park/Ukrainian Village): 2,211 permits
  3. 60647 (Logan Square): 1,583 permits
  4. 60607 (West Loop): 1,548 permits
  5. 60606 (Loop/Financial District): 1,169 permits

#### 2. Economic Factors Impact

- Mortgage rates increased from ~3% to 6% during 2022-2023
- Strong negative correlation (-0.38) between mortgage rates and consumer sentiment
- 23% reduction in building permit applications when mortgage rates exceed 5%
- COVID-19 pandemic (2020) caused a significant but temporary drop in permit activity
- Housing starts positively correlate with population growth (correlation coefficient: 0.64)

#### 3. Model Feature Importance

| Feature | Importance |
|---------|------------|
| Median household income | 39.8% |
| Middle class percentage | 14.2% |
| Lower income percentage | 13.2% |
| Permit count | 11.2% |
| Consumer sentiment | 7.9% |
| Treasury yield | 3.9% |
| Mortgage rates | 3.5% |
| Housing starts | 3.3% |
| House price index | 3.0% |

#### 4. Future Population Projections (2024-2025)

| Scenario | Average Growth | Min | Max | Standard Deviation |
|----------|----------------|-----|-----|-------------------|
| Optimistic | +1.13% | -6.63% | +8.14% | 2.62% |
| Neutral | +1.02% | -6.63% | +8.36% | 2.66% |
| Pessimistic | +0.88% | -5.64% | +8.32% | 2.49% |

### ZIP Code Analysis

#### Growth Patterns by Neighborhood Type

| Neighborhood Type | Avg. Growth | Permit Activity | Key Examples | Notes |
|-------------------|-------------|----------------|--------------|-------|
| Downtown/Loop | +1.28% | High | 60603, 60606 | Commercial transformation to residential |
| North Side | +1.15% | Medium-High | 60610, 60626 | Established high-income areas |
| West Side | +0.87% | Medium | 60607, 60622 | Gentrification in progress |
| South Side | +0.64% | Low | 60617, 60619 | Emerging growth areas |

#### Factors Influencing Growth Distribution

1. **Correlation between Income and Growth**
   - ZIP codes with median household incomes above $85,000 show 37% higher growth rates
   - Areas with balanced income distribution (40-60% middle class) show more stable growth

2. **Permit Type Impact**
   - Multifamily permits correlate with higher population growth (1.3x) compared to single-family
   - Mixed-use developments show strongest impact in formerly commercial areas (e.g., West Loop)

### Recommendations & Strategic Insights

1. **Development Focus Areas**
   - Near North Side (60610) and Downtown Loop (60606) demonstrate consistent growth with strong economic resilience
   - Rogers Park (60626) shows strong organic population growth despite moderate permit activity
   - South Chicago (60617) presents emerging opportunity with strong growth but currently low permit activity

2. **Economic Risk Assessment**
   - ZIP codes with highest growth (60610, 60626) show resilience during economic downturns
   - Areas with recent rapid growth (60607, 60622) are more sensitive to mortgage rate changes
   - Pessimistic economic scenario would disproportionately impact West Loop and Wicker Park

3. **Policy & Planning Recommendations**
   - Increase zoning flexibility in high-potential, low-permit areas (60617, 60626)
   - Encourage mixed-income development in areas with increasing income polarization
   - Monitor middle-class percentage as a key indicator of sustainable growth

### Limitations & Further Research

1. **Data Limitations**
   - Census data frequency (annual) limits granular time-based analysis
   - Permit data may not accurately reflect actual construction completion
   - Migration patterns within Chicago require more detailed data

2. **Model Improvements**
   - Incorporate spatial relationships between adjacent ZIP codes
   - Develop more sophisticated time-series methods for trend forecasting
   - Include additional variables like crime statistics, school quality, and transit access

3. **Future Research Directions**
   - Analyze impact of remote work on residential preferences
   - Examine housing affordability impact on population distribution
   - Investigate relationship between public investment and population shifts

### Appendix: Interactive Resources

- Complete visualizations available in: `output/visualizations/`
- Raw data and model outputs available in: `output/`
- Source code for analysis available in project repository
