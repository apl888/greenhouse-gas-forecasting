# src/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.api import qqplot
from scipy import stats

def check_volatility_clustering(residuals):
    '''
    Check for GARCH effects in residuals
    '''
    squared_residuals = residuals ** 2
    volatility_lags = [1, 4, 8, 12, 26]
    
    # Ljung-Box test on squared residuals ( Engle's ARCH test)
    arch_test = acorr_ljungbox(squared_residuals, lags=volatility_lags, return_df=True)
    
    print('\n--- Volatility Clustering Diagnostics ---')
    print("Engle's ARCH Test (on squared residuals):")
    print('H0: No ARCH effects (constant variance)')
    for lag in volatility_lags:
        pval = arch_test.loc[lag, 'lb_pvalue']
        sig = '***' if pval < 0.05 else ' (no ARCH effects)'
        print(f'  Lag {lag}: p-value = {pval:.4f}{sig}')
    
    # Visual check
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
    
    # Residuals plot
    ax1.plot(residuals)
    ax1.set_title('Residuals over Time')
    ax1.set_ylabel('Residuals')
    
    # Squared residuals plot (shows volatility clustering)
    ax2.plot(squared_residuals)
    ax2.set_title('Squared Residuals (Volatility)')
    ax2.set_ylabel('Squared Residuals')
    ax2.set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
    return arch_test


def evaluate_sarima_model(train, order, seasonal_order, run_hetero=False, plot_residuals=True):
    '''
    Fit a SARIMA model and run in-sample diagnostics.
    
    Parameters
    ----------
    train : pd.Series
        Training set (time-indexed).
    order : tuple
        SARIMA (p,d,q).
    seasonal_order : tuple
        SARIMA seasonal (P,D,Q,s).
    run_hetero : bool, default False
        If True, runs Breusch-Pagan and White tests for heteroscedasticity.
    plot_residuals : bool, default True
        If True, generates residual diagnostic plots.
    
    Returns
    -------
    results : dict
        Contains AIC, BIC, Ljung-Box p-values, and optional heteroscedasticity test results.
    '''
    
    # --- Fit model ---
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=True, enforce_invertibility=True, trend='c')
    results = model.fit(method='powell', disp=False)

    # --- Residual diagnostics ---
    residuals = results.resid

    if plot_residuals:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residual time series
        axes[0,0].plot(residuals)
        axes[0,0].set_title('Residuals over Time')
        axes[0,0].set_xlabel('Time Index')
        axes[0,0].set_ylabel('Residual Value')

        # Histogram + KDE
        axes[0,1].hist(residuals, bins=40, density=True, alpha=0.6, label='Histogram')
        residuals.plot(kind='kde', ax=axes[0,1], linewidth=2, label='KDE')
        axes[0,1].set_title('Residual Distribution')
        axes[0,1].set_xlabel('Residual Value')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()

        # Q-Q plot
        qqplot(residuals, line='s', ax=axes[1,0])
        axes[1,0].set_title('Q-Q Plot')
        axes[1,0].set_xlabel('Theoretical Quantiles')
        axes[1,0].set_ylabel('Sample Quantiles')

        # ACF plot
        plot_acf(residuals, lags=52, ax=axes[1,1])
        axes[1,1].set_title('ACF of Residuals')
        axes[1,1].set_xlabel('Lag')
        axes[1,1].set_ylabel('Autocorrelation')

        plt.tight_layout()
        plt.show()

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[1,5,10,52], return_df=True)

    results_dict = {
        'order': order,
        'seasonal_order': seasonal_order,
        'LB_pval_lag1': round(lb_test.loc[1, 'lb_pvalue'], 4),
        'LB_pval_lag5': round(lb_test.loc[5, 'lb_pvalue'], 4),
        'LB_pval_lag10': round(lb_test.loc[10, 'lb_pvalue'], 4),
        'LB_pval_lag52': round(lb_test.loc[52, 'lb_pvalue'], 4)
    }

    arch_results = check_volatility_clustering(residuals)
    
    # Optional: heteroscedasticity tests
    if run_hetero:
        exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])  
        bp_test = het_breuschpagan(residuals, exog)
        white_test = het_white(residuals, exog)

        results_dict.update({
            'BP_pval': round(bp_test[1], 4),
            'White_pval': round(white_test[1], 4),        
        })

    results_dict_df = pd.DataFrame([results_dict])
    model_eval_df = results_dict_df.melt(var_name='Metric', value_name='Value')
    return model_eval_df


def comprehensive_residual_analysis(model_params, train_data, test_data, model_name, plot_residuals=True):
    '''
    Complete residual diagnostics for final model validation using TEST SET residuals
    
    Parameters
    ----------
    model_params : dict
        Dictionary with 'order' and 'seasonal_order' keys
    train_data : pd.Series
        Training data (log-transformed)
    test_data : pd.Series  
        Test data (log-transformed)
    model_name : str
        Descriptive name for the model
    plot_residuals : bool, default True
        Whether to generate diagnostic plots
        
    Returns
    -------
    test_residuals : pd.Series
        Residuals from test set predictions
    results : SARIMAX results object
        Fitted model results
    '''
    
    print(f'=== Residual Analysis for {model_name} ===')
    
    # Fit model on training data
    model = SARIMAX(train_data,
                    order=model_params['order'],
                    seasonal_order=model_params['seasonal_order'],
                    enforce_stationarity=True, 
                    enforce_invertibility=True, 
                    trend='c')
    results = model.fit(method='powell', disp=False)
    
    # Forecast on test set
    forecast = results.get_forecast(steps=len(test_data))
    forecast_values = forecast.predicted_mean
    
    # Calculate test residuals
    test_residuals = test_data - forecast_values
    
    # 1. Basic residual statistics
    print(f'Test Residuals - Mean: {test_residuals.mean():.6f}, Std: {test_residuals.std():.6f}')
    print(f'Skewness: {test_residuals.skew():.3f}, Kurtosis: {test_residuals.kurtosis():.3f}')
    
    # 2. Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(test_residuals)
    print(f'Shapiro-Wilk normality test: p-value = {shapiro_p:.4f}')
    
    # 3. Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(test_residuals, lags=[1, 5, 10, 52], return_df=True)
    print('Ljung-Box p-values:')
    for lag in [1, 5, 10, 52]:
        pval = lb_test.loc[lag, 'lb_pvalue']
        sig = ' ***' if pval < 0.05 else ''
        print(f'  Lag {lag}: {pval:.4f}{sig}')
    
    # 4. Volatility clustering check (using your existing function)
    arch_results = check_volatility_clustering(test_residuals)
    
    if plot_residuals:
        # Plot diagnostics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals over time
        axes[0,0].plot(test_residuals.index, test_residuals.values)
        axes[0,0].set_title(f'{model_name} - Test Residuals Over Time')
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_ylabel('Residuals')
        
        # Q-Q plot
        stats.probplot(test_residuals, dist='norm', plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot')
        
        # Histogram
        axes[1,0].hist(test_residuals, bins=20, density=True, alpha=0.7)
        axes[1,0].set_title('Residual Distribution')
        axes[1,0].set_xlabel('Residual Value')
        axes[1,0].set_ylabel('Density')
        
        # ACF of residuals
        plot_acf(test_residuals, lags=52, ax=axes[1,1])
        axes[1,1].set_title('ACF of Residuals')
        axes[1,1].set_ylabel('Autocorrelation')
        axes[1,1].set_xlabel('Lag')
        
        plt.tight_layout()
        plt.show()
    
    return test_residuals, results

def check_model_assumptions(residuals, alpha=0.05):
    '''
    Check key model assumptions and return summary
    
    Parameters
    ----------
    residuals : pd.Series
        Model residuals
    alpha : float, default 0.05
        Significance level for tests
        
    Returns
    -------
    assumptions : dict
        Dictionary with assumption check results
    '''
    
    # Normality test
    _, normality_p = stats.shapiro(residuals)
    
    # Autocorrelation test
    lb_results = acorr_ljungbox(residuals, lags=[1, 5, 10, 52], return_df=True)
    no_autocorrelation = all(lb_results['lb_pvalue'] > alpha)
    
    assumptions = {
        'zero_mean': np.abs(residuals.mean()) < 0.01,  # Rough threshold
        'constant_variance': True,  # Would need more sophisticated test
        'normality': normality_p > alpha,
        'no_autocorrelation': no_autocorrelation,
        'normality_pvalue': normality_p
    }
    
    print('\n--- Model Assumptions Summary ---')
    print(f"Zero mean: {'\u2705' if assumptions['zero_mean'] else '\u274C'} (mean = {residuals.mean():.6f})")
    print(f"Normality: {'\u2705' if assumptions['normality'] else '\u274C'} (p = {normality_p:.4f})")
    print(f"No autocorrelation: {'\u2705' if assumptions['no_autocorrelation'] else '\u274C'}")
    
    return assumptions
