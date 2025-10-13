# Model Artifacts

This directory contains the final SARIMA model(s) for CH4 (methane) concentration forecasting.  The best model is also significantly more complex with 9 orders vs 3 for the simple model.  The best, simple model is included as an option due to its near equal performance compared to the best model and significant lesser complexity.  

## Files

### `CH4_train_best_model_config.json`  
**Model**: SARIMA(7,1,0)(1,1,1,52)  
**Purpose**: Best-performing model on training data  
**Selection Criteria**:  
- Initial candidate from `auto_arima`
- Final candidate from `TimeSeriesSplit CV` and Pareto Frontier
  - Metrics from TimeSeriesSplit CV on log transformed data
    - RMSE_mean = 0.010205
    - MAE_mean = 0.007906
    - AIC_mean = -5001.42
    - BIC_mean = -4958.21
    - Ljung-Box pval mean at lag 1 = 0.993217
    - Ljung-Box pval mean at lag 5 = 1.0
    - Ljung-Box pval mean at lag 10 = 1.0
    - Ljung-Box pval mean at lag 52 = 0.000155
**Training Period**: 1983-05-06 to 2024-12-31  
**Key Characteristics**:  
- Annual seasonality (52 datapoints per year)  
- Non-seasonal AR(7) + I(1) differencing  
- Seasonal AR and MA terms for lag 52 + I(1) differencing 

### `CH4_train_best_simple_model_config.json`  
**Model**: SARIMA(1,1,1)(0,1,1,52)  
**Purpose**: Best-performing model on training data  
**Selection Criteria**:  
- Initial candidate from `auto_arima`
- Final candidate from `TimeSeriesSplit CV` and Pareto Frontier
  - Metrics from TimeSeriesSplit CV on log transformed data
    - RMSE_mean = 0.010530
    - MAE_mean = 0.008095
    - AIC_mean = -5023.31
    - BIC_mean = -5006.03
    - Ljung-Box pval mean at lag 1 = 0.993269
    - Ljung-Box pval mean at lag 5 = 1.0
    - Ljung-Box pval mean at lag 10 = 1.0
    - Ljung-Box pval mean at lag 52 = 0.000153
**Training Period**: 1983-05-06 to 2024-12-31  
**Key Characteristics**:  
- Annual seasonality (52 datapoints per year)  
- Non-seasonal AR(1) + I(1) differencing + MA(1)   
- Seasonal AR(0) + I(1) differencing + MA(1) for lag 52 

## Usage

```python
import json

# Load model configuration
with open('../results/CH4_train_best_model_config.json', 'w') as f:
    json.dump(model_config_best, f, indent=2)
    
with open('../results/CH4_train_best_simple_model_config.json', 'w') as f:
    json.dump(model_config_best_simple, f, indent=2)

# Forecast
forecast = model.get_forecast(steps=52)