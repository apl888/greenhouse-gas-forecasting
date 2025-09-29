# src/forecast_metrics.py

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def forecast_metrics(y_true, y_pred):
    '''
    Calculate forecast metrics: MSE, MAE, RMSE, MAPE, and R-squared.

    Parameters:
    y_true (pd.Series): Actual values.
    y_pred (pd.Series): Predicted values.
    y_pred_conf_int (pd.DataFrame): Forecast confidence intervals.

    Returns:
    dict: Dictionary of forecast metrics.
    '''
    # convert pandas objcts to numpy arrays
    if hasattr(y_true, 'values'):
        y_true = y_true.values.flatten() # handles pandas series and DataFrames
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values.flatten()
        
    # calculations 
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': round(mae, 3),
        'MSE': round(mse, 3), 
        'RMSE': round(rmse, 3),
        'MAPE': round(mape, 3)
    }