# Model Artifacts

This directory contains the final SARIMA models for CH4 (methane) concentration forecasting.

## Files

### 1. `CH4_best_sarima.pkl`  
**Model**: SARIMA(2,1,0)(1,0,2,52)  
**Purpose**: Best-performing model on training data  
**Selection Criteria**:  
- Initial candidate from `auto_arima`  
- Manually refined through:  
  - Residual diagnostics (ACF/PACF, QQ plots)  
  - Statistical tests (Ljung-Box p=0.11, Heteroskedasticity p=0.27)  
  - Metric comparison (AIC=-18713, BIC=-18679)  
**Training Period**: 1983-05-08 to 2023-12-19  
**Key Characteristics**:  
- Weekly seasonality (52 periods)  
- Non-seasonal AR(2) + I(1) differencing  
- Seasonal MA terms at lags 52 and 104  

### 2. `CH4_full_dataset_best_sarima.pkl`  
**Model**: SARIMA(1,1,0)(1,0,2,52)  
**Purpose**: Production model fit on full dataset  
**Optimization**:  
- Simplified from training model based on:  
  - Improved AIC (-18880 vs -18713)  
  - Lower RMSE (4.139 vs 4.160)  
  - Cleaner residual correlogram  
**Training Period**: 1983-05-08 to 2025-04-06  
**Notable Changes**:  
- Reduced to AR(1) after full-data analysis  
- Maintained seasonal MA structure  
- Better handling of heavy-tailed residuals  

## Usage

```python
import joblib

# Load model
model = joblib.load('CH4_full_dataset_best_sarima.pkl')['results']

# Forecast
forecast = model.get_forecast(steps=52)