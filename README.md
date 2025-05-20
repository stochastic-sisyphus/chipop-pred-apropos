# Chicago Population Forecasting and Retail Demand Analysis

## This Repository Is Temporarily Private as of May 19, 2025. 

### The public version will be re-published by the end of May (pending negotiations).

This project was previously public and is currently hidden due to its recent adaptation for enterprise urban planning engagement. As of **May 19, 2025**, the pipeline includes deal-specific logic and structural conventions under active negotiation.

A cleaned public clone will be re-published by the **end of May 2025** (fingers crossed), preserving all core functionality, modeling, outputs, and insights—excluding only the modules and business logic specific to confidential contract work.

*This work supports investment planning for Chicago’s future, particularly in historically underserved communities. I’d rather temporarily pull it than risk compromising that.*

---

## Project Summary

This is a full-scale forecasting and simulation pipeline designed to model population growth and retail demand across Chicago at the ZIP-code level. It integrates a wide range of city, zoning, and economic data, transforming raw records into forecasts, dashboards, and markdown reports that support planning, investment, and policy decisions.

The system was built to balance exploratory research with stakeholder-facing presentation. It is modular, version-controlled, and structured for reproducibility.

### System Capabilities

- Ingests multi-source urban data: permits, licenses, census, zoning, economic indicators  
- Performs ZIP-level feature engineering with derived balance scores and custom indices  
- Trains modular regressors (population, housing, retail leakage)  
- Generates interactive dashboards and executive markdown reports  
- Supports scenario-based simulation, evaluation, and presentation  
- Archives model artifacts, predictions, metrics, and visualizations per run  

---

## Directory Structure

This snapshot reflects the real layout and contents of the repository *minus some, according to redaction plans*. Files and logic currently under review have been excluded from the public snapshot.

.
├── README.md
├── requirements.txt
├── main.py
├── eda.ipynb
├── permits_missing_zip.csv
├── local_zip_cache.json
├── scripts/
│   ├── clean_duplicates.py
│   └── run_clean_archive.sh
├── logs/
│   └── chipop.log
├── tests/
│   └── test_basic.py
├── data/
│   ├── raw/
│   │   ├── building_permits.csv
│   │   ├── business_licenses.csv
│   │   ├── census_data.csv
│   │   ├── economic_indicators.csv
│   │   └── zoning_data.csv
│   ├── interim/
│   │   ├── debug_housing_retail_merge.csv
│   │   └── debug_ten_year_growth_merge.csv
│   └── processed/
│       ├── bea_retail_gdp.csv
│       ├── business_licenses_processed.csv
│       ├── census_processed.csv
│       ├── economic_processed.csv
│       ├── housing_metrics.csv
│       ├── hud_usps_vacancy.csv
│       ├── merged_dataset.csv
│       ├── permits_processed.csv
│       ├── retail_business_count.csv
│       ├── retail_deficit.csv
│       ├── retail_demand_per_zip.csv
│       ├── retail_permits_summary.csv
│       ├── retail_space.csv
│       ├── retail_summary_metrics.json
│       ├── unresolved_addresses.csv
│       └── zoning_processed.csv
├── output/
│   ├── analysis_results/
│   │   ├── high_growth_areas.csv
│   │   ├── retail_deficit_areas.csv
│   │   ├── top_impacted_areas.csv
│   │   └── scenario_summary.csv
│   ├── dashboard.html
│   ├── directory_structure.json
│   ├── pipeline_summary.md
│   ├── pipeline_summary.json
│   ├── model_metrics/
│   │   ├── economic_feature_importance.csv
│   │   ├── economic_model_metrics.csv
│   │   ├── feature_importance.csv
│   │   ├── housing_feature_importance.csv
│   │   ├── population_feature_importance.csv
│   │   ├── retail_deficit_feature_importance.csv
│   │   └── retail_metrics.csv
│   ├── predictions/
│   │   ├── feature_importance.csv
│   │   ├── high_leakage_areas.csv
│   │   ├── housing_demand_predictions.csv
│   │   ├── population_predictions.csv
│   │   ├── population_scenarios.csv
│   │   ├── population_shift_patterns.csv
│   │   ├── retail_demand_predictions.csv
│   │   ├── retail_deficit_predictions.csv
│   │   ├── retail_spending_leakage.csv
│   │   ├── retail_housing_opportunity.csv
│   │   ├── scenario_predictions.csv
│   │   ├── scenario_summary.csv
│   │   ├── top_impacted_areas.csv
│   │   └── zip_summary.csv
│   ├── reports/
│   │   ├── EXECUTIVE_SUMMARY.md
│   │   ├── chicago_population_analysis_report.md
│   │   ├── chicago_zip_summary.md
│   │   ├── economic_impact_analysis.md
│   │   ├── housing_retail_balance_report.md
│   │   ├── retail_deficit_analysis.md
│   │   ├── ten_year_growth_analysis.md
│   │   └── void_analysis.md
│   ├── trained_models/
│   │   ├── economic_model.joblib
│   │   ├── housing_model.joblib
│   │   ├── population_model.joblib
│   │   ├── retail_model.joblib
│   │   └── *_scaler.joblib
│   ├── visualizations/
│   │   ├── balance_analysis/
│   │   │   ├── balance_category_pie.png
│   │   │   ├── balance_score_by_zip.png
│   │   │   └── housing_vs_retail_scatter.png
│   │   ├── permits_by_year.png
│   │   ├── permits_by_type.png
│   │   ├── business_activity.html
│   │   ├── population_trends.html
│   │   ├── population_trends.png
│   │   ├── retail_deficit_map.html
│   │   └── scenario_impact.html
│   └── run_archives/
│       └── run_archive_*.zip
├── src/
│   ├── config/
│   │   ├── column_alias_map.py
│   │   └── settings.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── collector.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── processor.py
│   │   └── zoning.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── economic_model.py
│   │   ├── housing_model.py
│   │   ├── population_model.py
│   │   └── retail_model.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── dynamic_runner.py
│   │   └── pipeline.py
│   ├── reporting/
│   │   ├── chicago_zip_summary.py
│   │   ├── housing_retail_balance_report.py
│   │   ├── retail_deficit_report.py
│   │   └── ten_year_growth_report.py
│   ├── templates/
│   │   └── [report markdown templates]
│   ├── utils/
│   │   ├── helpers.py
│   │   └── validate_data.py
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py

---

## Key Outputs

- **Forecasts** for population, housing demand, and retail growth across ZIP codes  
- **Economic imbalance analysis** with leakage, surplus, and trend overlays  
- **Auto-generated reports** in Markdown format for each scenario run  
- **Interactive HTML dashboards** with embedded plots and choropleths  
- **Model metrics** and feature importances by target and domain  
- **Versioned archives** of trained models, scenario predictions, and results  

---

## Status and Release Plan

The current repository includes modules and design structures specific to a real contract negotiation and delivery. These are being segmented and redacted to prepare for a public-facing version. The cleaned release will preserve the forecasting framework, data pipeline, modeling system, and all non-sensitive outputs—ensuring reproducibility and end-to-end insight delivery without jeopardizing any urban planning contract negotiations or buisness data specific to parties involved with making some exciting plans for Chicago's future, especially in historically underserved communities. I would rather pull this project from visibility while I sterilize it than compromise long overdue investment in our communities. :--)

**I repeat! The public version will be re-published hopefully end of May 25. It has been hidden as of May 19 2025**
