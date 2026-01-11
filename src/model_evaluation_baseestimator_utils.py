import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_skill_score(actual, predicted, train_data, seasonal_period):
    """Calculate skill score vs seasonal naive"""
    naive = seasonal_naive_forecast(train_data, len(actual), seasonal_period)
    model_rmse = np.sqrt(mean_squared_error(actual, predicted))
    naive_rmse = np.sqrt(mean_squared_error(actual, naive))
    return (1 - model_rmse / naive_rmse) * 100

def nash_sutcliffe_efficiency(observed, predicted):
    numerator = ((observed - predicted) ** 2).sum()
    denominator = ((observed - observed.mean()) ** 2).sum()
    return 1 - numerator / denominator

def calculate_mase(actual, predicted, train_data, seasonal_period):
    """Mean Absolute Scaled Error"""
    mae = mean_absolute_error(actual, predicted)
    naive_errors = np.abs(np.diff(train_data.values, n=seasonal_period))
    scale = np.mean(naive_errors)
    return mae / scale if scale != 0 else np.nan

def calculate_smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error"""
    return 200 * np.mean(np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))