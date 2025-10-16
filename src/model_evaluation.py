# src/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import qqplot
from scipy import stats

# === 1. Volatility / ARCH Check ===
def test_volatility_clustering(residuals, plot=True):
    '''
    Check for GARCH effects in residuals
    '''
    squared_residuals = residuals ** 2
    lags = [1, 4, 8, 12, 26]
    
    # Ljung-Box test on squared residuals ( Engle's ARCH test)
    arch_test = acorr_ljungbox(squared_residuals, lags=lags, return_df=True)
    
    print('\n--- Volatility Clustering Diagnostics ---')
    print("Engle's ARCH Test (on squared residuals):")
    print('H0: No ARCH effects (constant variance)')
    for lag in lags:
        pval = arch_test.loc[lag, 'lb_pvalue']
        sig = '***' if pval < 0.05 else ''
        print(f'  Lag {lag}: p-value = {pval:.4f}{sig}')
    
    # Visual check
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(residuals, color='steelblue')
        axes[0].set_title("Residuals Over Time")
        axes[0].set_ylabel("Residuals")

        axes[1].plot(squared_residuals, color='tomato')
        axes[1].set_title("Squared Residuals (Volatility Clustering Check)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Squared Residuals")

        plt.tight_layout()
        plt.show()
    
    return arch_test

# === 2. In-sample residual diagnostics ===
def in_sample_resid_analysis(train, order, seasonal_order, run_hetero=False):
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
    '''
    
    # --- Fit model ---
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=True, enforce_invertibility=True, trend='c')
    results = model.fit(method='powell', disp=False)
    residuals = results.resid

    # --- Plot residual diagnostics ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residual time series
    axes[0,0].plot(residuals)
    axes[0,0].set_title('Residuals over Time')
    axes[0,0].set_xlabel('Time Index')
    axes[0,0].set_ylabel('Residual Value')

    # Histogram + KDE
    axes[0,1].hist(residuals, bins=40, density=True, alpha=0.6, label='Hist')
    residuals.plot(kind='kde', ax=axes[0,1], linewidth=2, label='KDE')
    axes[0,1].set_title('Residual Distribution', fontsize=14)
    axes[0,1].set_xlabel('Residual Value', fontsize=12)
    axes[0,1].set_ylabel('Density', fontsize=12)
    axes[0,1].legend()

    # Q-Q plot
    qqplot(residuals, line='s', ax=axes[1,0])
    axes[1,0].set_title('Q-Q Plot', fontsize=14)
    axes[1,0].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[1,0].set_ylabel('Sample Quantiles', fontsize=12)

    # ACF plot
    plot_acf(residuals, lags=55, ax=axes[1,1])
    axes[1,1].set_title('ACF of Residuals', fontsize=14)
    axes[1,1].set_xlabel('Lag', fontsize=12)
    axes[1,1].set_ylabel('Autocorrelation', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[1,5,10,52], return_df=True)
    print('\n--- Ljung-Box Test ---')
    for lag in [1,5,10,52]:
        print(f'lag {lag}: p = {lb.loc[lag, 'lb_pvalue']:.4f}')
    
    test_check_volatility_clustering(residuals)
    
    # Optional: heteroscedasticity tests
    if run_hetero:
        exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])  
        bp_p = het_breuschpagan(residuals, exog)[1]
        white_p = het_white(residuals, exog)[1]
        print(f"\nBreusch-Pagan p = {bp_p:.4f}, White p = {white_p:.4f}")

    return results

# === 3. Out-of-sample residual diagnostics ===
def comprehensive_residual_analysis(model_params, train_data, test_data, model_name):
    '''
    Complete residual diagnostics for test set
    '''
    
    print(f'=== Out-of-Sample Residual Analysis: {model_name} ===')
    
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
    predictions = forecast.predicted_mean
    
    # Calculate test residuals
    residuals = test_data - predictions
    
    # 1. Basic residual statistics
    print(f'Residual mean: {residuals.mean():.5f}, std: {residuals.std():.5f}')
    print(f'Skew: {residuals.skew():.3f}, Kurtosis: {residuals.kurtosis():.3f}')
    
    # 2. Normality tests
    shapiro_p = stats.shapiro(residuals)[1]
    print(f'Shapiro-Wilk normality test: p-value = {shapiro_p:.4f}')
    
    # 3. Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(residuals, lags=[1, 5, 10, 52], return_df=True)
    print('Ljung-Box test p-values:')
    for lag in [1, 5, 10, 52]:
        print(f'  Lag {lag}: p = {lb.loc[lag, 'lb_pvalue']:.4f}')
    
    # 4. Volatility clustering check (using existing function)
    test_volatility_clustering(residuals, plot=False)
    
    # Plot residual diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
    # Residuals over time
    axes[0,0].plot(residuals.index, residuals.values)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_title(f'{model_name} - Test Residuals Over Time', fontsize=14)
    axes[0,0].set_ylabel('Residual', fontsize=12)
    axes[0,0].set_xlabel('Date', fontsize=12)
        
    # Q-Q plot
    stats.probplot(residuals, dist='norm', plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot', fontsize=14)
    axes[0,1].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[0,1].set_ylabel('Sample Quantiles', fontsize=12)
        
    # Histogram
    axes[1,0].hist(residuals, bins=25, density=True, alpha=0.7)
    axes[1,0].set_title('Residual Distribution', fontsize=14)
    axes[1,0].set_xlabel('Residual Value', fontsize=12)
    axes[1,0].set_ylabel('Density', fontsize=12)
        
    # ACF of residuals
    plot_acf(residuals, lags=55, ax=axes[1,1])
    axes[1,1].set_title('ACF of Residuals', fontsize=14)
    axes[1,1].set_ylabel('Autocorrelation', fontsize=12)
    axes[1,1].set_xlabel('Lag', fontsize=12)
        
    plt.tight_layout()
    plt.show()
    
    return residuals, results

def summarize_model_assumptions(residuals, alpha=0.05):
    '''
    Evaluate normality, zero-mean, and autocorrelation assumptions.
    '''
    
    # Normality test
    _, normality_p = stats.shapiro(residuals)
    
    # Autocorrelation test
    lb = acorr_ljungbox(residuals, lags=[1, 5, 10, 52], return_df=True)
    no_autocorr = all(lb['lb_pvalue'] > alpha)
    
    assumptions = {
        'zero_mean': abs(residuals.mean()) < 0.01,  # Rough threshold
        'normality': normality_p > alpha,
        'no_autocorrelation': no_autocorr,
        'normality_pvalue': normality_p
    }
    
    print('\n--- Model Assumptions Summary ---')
    print(f"Zero mean: {'\u2705' if assumptions['zero_mean'] else '\u274C'} (mean = {residuals.mean():.6f})")
    print(f"Normality: {'\u2705' if assumptions['normality'] else '\u274C'} (p = {normality_p:.4f})")
    print(f"No autocorrelation: {'\u2705' if assumptions['no_autocorrelation'] else '\u274C'}")
    
    return assumptions
