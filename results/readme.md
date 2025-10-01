# Model Artifacts

This directory contains the final SARIMA model(s) for CH4 (methane) concentration forecasting.

## Files

### `CH4_train_best_model_config.json`  
**Model**: SARIMA(1,1,0)(1,0,0,52)  
**Purpose**: Best-performing model on training data  
**Selection Criteria**:  
- Initial candidate from `auto_arima`
- Final candidate from `TimeSeriesSplit CV`  
  - Error tests on log transformed data (RMSE = 0.007166, MAE = 0.005686)  
  - Statistical tests on log transformed data (Ljung-Box mean p=1.0)  
  - Metric comparison on log transformed data (AIC=-11048, BIC=-11031)  
**Training Period**: 1983-05-06 to 2024-12-31  
**Key Characteristics**:  
- Weekly seasonality (52 periods)  
- Non-seasonal AR(1) + I(1) differencing  
- Seasonal AR term at lag 52  

## Usage

```python
import json

# Load model configuration
with open('../results/CH4_train_best_model_config.json', 'r') as f:
    config = json.load(f)

# Forecast
forecast = model.get_forecast(steps=52)