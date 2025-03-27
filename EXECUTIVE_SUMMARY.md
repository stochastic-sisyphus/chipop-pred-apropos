# Chicago Population Shifts Analysis
## Executive Summary

This document summarizes the key findings from our comprehensive analysis of Chicago's population shifts from 2013-2023, with projections for 2024-2025.

## 📊 Key Findings

### Population Shift Patterns

1. **Top Growth ZIP Codes**:
   - **60610** (Near North Side): +1.79% annual growth
   - **60626** (Rogers Park): +1.58% annual growth
   - **60617** (South Chicago): +1.57% annual growth
   - **60606** (Loop/Financial District): +1.55% annual growth
   - **60603** (Loop): +1.49% annual growth

2. **Highest Permit Activity**:
   - **60602** (Central Loop): 3,393 permits
   - **60622** (Wicker Park): 2,211 permits
   - **60647** (Logan Square): 1,583 permits
   - **60607** (West Loop): 1,548 permits
   - **60606** (Loop/Financial District): 1,169 permits

### Economic Impact Analysis

1. **Mortgage Rate Influence**:
   - Each 1% increase in mortgage rates leads to 0.15-0.20% decrease in population growth
   - 23% reduction in permit activity when rates exceed 5%
   - Strong negative correlation (-0.38) between mortgage rates and consumer sentiment

2. **Key Economic Indicators**:
   - Consumer sentiment correlates strongly with population growth (+0.72)
   - Housing starts serve as leading indicators with 6-9 month lag time for local impact

### Future Population Projections (2024-2025)

| Scenario | Average Growth | Min | Max | Standard Deviation |
|----------|----------------|-----|-----|-------------------|
| Optimistic | +1.13% | -6.63% | +8.14% | 2.62% |
| Neutral | +1.02% | -6.63% | +8.36% | 2.66% |
| Pessimistic | +0.88% | -5.64% | +8.32% | 2.49% |

### Model Feature Importance

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

## 📈 Neighborhood Growth Patterns

| Neighborhood Type | Avg. Growth | Permit Activity | Key Examples | Notes |
|-------------------|-------------|----------------|--------------|-------|
| Downtown/Loop | +1.28% | High | 60603, 60606 | Commercial transformation to residential |
| North Side | +1.15% | Medium-High | 60610, 60626 | Established high-income areas |
| West Side | +0.87% | Medium | 60607, 60622 | Gentrification in progress |
| South Side | +0.64% | Low | 60617, 60619 | Emerging growth areas |

## 💡 Strategic Recommendations

1. **Development Focus Areas**:
   - **Near North Side (60610)** and **Downtown Loop (60606)** show consistent growth with economic resilience
   - **Rogers Park (60626)** demonstrates strong organic growth despite moderate permit activity
   - **South Chicago (60617)** presents emerging opportunity with strong growth but currently low permit activity

2. **Economic Risk Assessment**:
   - Areas with highest growth (60610, 60626) show resilience during economic downturns
   - Recently rapid-growth areas (60607, 60622) are more sensitive to mortgage rate changes
   - Pessimistic economic scenario would disproportionately impact West Loop and Wicker Park

3. **Policy & Planning Recommendations**:
   - Increase zoning flexibility in high-potential, low-permit areas (60617, 60626)
   - Encourage mixed-income development in areas with increasing income polarization
   - Monitor middle-class percentage as a key indicator of sustainable growth

## 📁 Detailed Reports & Resources

- **Comprehensive Technical Report**: `output/reports/chicago_population_analysis_report.md`
- **ZIP Code Analysis**: `output/reports/chicago_zip_summary.md`
- **Economic Impact Analysis**: `output/reports/economic_impact_analysis.md`
- **Visualizations**: `output/visualizations/`
- **Model Outputs**: `output/models/`
- **Raw Data**: `output/merged_dataset.csv`

## 📝 Recent Improvements

- Fixed column name inconsistencies between datasets
- Implemented enhanced scenario generation methodology
- Added comprehensive ZIP code-level analysis of economic resilience
- Ensured pipeline consistency from data collection through visualization

---

*Analysis completed March 2025*