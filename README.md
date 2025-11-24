# Credit Risk Analytics Dashboard

## Overview
This is a comprehensive credit risk analytics dashboard built with Streamlit that identifies key default risk factors, segments customers by risk level, and provides early warning indicators for proactive intervention.

## Features
- **Executive Summary**: Key portfolio metrics and insights
- **Risk Segmentation**: Customer classification into risk categories
- **Key Risk Factors**: Feature importance analysis from multiple models
- **Early Warning System**: Real-time monitoring indicators
- **Financial Impact Analysis**: ROI calculations for intervention strategies
- **Model Performance**: Comparison of ML models with focus on F2 score

## Data Preprocessing
The dashboard includes detailed documentation of all preprocessing steps:
- Data quality issue resolution (8 issues fixed)
- Feature engineering (7 new risk indicators created)
- Class imbalance handling (SMOTE, class weights, F2 optimization)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure all model files are present:
- lr_model.pkl
- rf_model.pkl
- xgb_model.pkl
- scaler.pkl
- All CSV data files

3. Run the dashboard:
```bash
streamlit run credit_risk_dashboard.py
```

## Model Performance
- **Logistic Regression**: F2 Score = 0.556
- **Random Forest**: F2 Score = 0.378
- **XGBoost**: F2 Score = 0.572 (Best)
- **Ensemble**: Weighted average for production use

## Key Insights
- 22.1% overall default rate in the portfolio
- 48.8% of customers classified as high/very high risk
- Payment delays are the strongest predictor of default
- Estimated $20.3M net benefit from targeted intervention program

## Risk Segments
- **Low Risk**: 4.6% default rate
- **Medium Risk**: 10.9% default rate
- **High Risk**: 20.2% default rate
- **Very High Risk**: 53.1% default rate

## Early Warning Thresholds
- PAY_0 > 1 (Recent payment delay)
- DELAYED_MONTHS_COUNT > 3 (Pattern of delays)
- CREDIT_UTILIZATION > 0.8 (High credit usage)
- MAX_PAYMENT_DELAY > 2 (Historical delays)
- PAYMENT_RATIO < 0.1 (Low payment capacity)

## Financial Impact
- Total credit exposure: $5.0B
- Potential loss (60% LGD): $300M
- Intervention cost: $5M
- Prevented losses: $25.3M
- Net benefit: $20.3M
- ROI: 406%

## Usage
The dashboard provides interactive filtering options:
- Risk segment selection
- Age range filtering
- Credit limit range filtering
- Real-time metric updates

## Disclaimer
This dashboard is for analytical purposes. All credit decisions should be reviewed by qualified personnel.
