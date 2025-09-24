# src/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats


def get_seasonal_ljung_box_lags(residuals, seasonal_period=52, max_lags=None):
    """
    Generate intelligent lags for Ljung-Box test on seasonal data
    """
    if max_lags is None:
        max_lags = len(residuals) // 2  # Statistical rule of thumb
    
    # Key multiples of seasonal period
    seasonal_multiples = [1, 2, 3, 4]  # Up to 4 years if data allows
    lags_to_test = []
    
    for multiple in seasonal_multiples:
        lag = multiple * seasonal_period
        if lag < max_lags:
            lags_to_test.append(lag)
    
    # Add some intermediate lags for short-term checking
    intermediate_lags = [1, 4, 13, 26]  # Week, month, quarter, half-year
    for lag in intermediate_lags:
        if lag not in lags_to_test and lag < max_lags:
            lags_to_test.append(lag)
    
    # Sort and return
    return sorted(lags_to_test)


def evaluate_sarima_model(series, order, seasonal_order):
    '''
    Comprehensive SARIMA model evaluation
    '''
    # fit the model
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)  
    
    print(f"\n{'='*50}")
    print(f'Evaluating SARIMA{order}{seasonal_order}')
    print(f"{'='*50}\n")
    
    # basic metrics
    print(f'AIC: {results.aic:.3f}')
    print(f'BIC: {results.bic:.3f}')
    print(f'Log-Likelihood: {results.llf:.3f}')
    
    # coefficient significance 
    print('\nCoefficient Significance:')
    for param, pval in zip(results.param_names, results.pvalues):
        sig ='\u2713' if pval < 0.05 else '\u2717'  # \u2713 is unicode for check mark, \u2717 for "x"
        print(f'   {param}: p-value = {pval:.4f} {sig}')
        
    # residual diagnostics
    residuals = results.resid.dropna()
        
    # Ljung-Box test for autocorrelation
    seasonal_period = seasonal_order[3]    
    lags_to_test = get_seasonal_ljung_box_lags(residuals, seasonal_period)
    
    lb_test = acorr_ljungbox(residuals, lags=lags_to_test, return_df=True)
    
    critical_lags = [1, seasonal_period]
    for lag in critical_lags:
        if lag in lb_test.index:
            pval = lb_test.loc[lag, 'lb_pvalue']
            lb_sig = '\u2713' if pval > 0.05 else '\u2717'
            print(f'\nLjung-Box (lag {lag}): p-value = {pval:.4f} {lb_sig}')
            
    min_pval = lb_test['lb_pvalue'].min()
    overall_sig = '\u2713' if min_pval > 0.05 else '\u2717'
    print(f'Overall Ljung-Box (min p-value): {min_pval:.4f} {overall_sig}')
    
    # heteroscedasticity
    X = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])
    bp_test = het_breuschpagan(residuals, X)
    bp_sig = '\u2713' if bp_test[1] > 0.05 else '\u2717'
    print(f'Breusch-Pagan Test: p-value = {bp_test[1]:.4f} {bp_sig}')
    
    # normality (Jarque-Bera test)
    jb_test = jarque_bera(residuals)
    jb_sig = '\u2713' if jb_test[1] > 0.05 else '\u2717'
    print(f'Jarque-Bera Test: p-value = {jb_test[1]:.4f} {jb_sig}')
    
    # comprehensive plots
    fig, axes = plt.subplots(3,2, figsize=(15,12))
    
    # residual plot
    axes[0,0].plot(residuals)
    axes[0,0].set_title('Residuals over Time')
    
    # ACF and PACF of residuals
    plot_acf(residuals, ax=axes[0,1], lags=min(52, len(residuals)//2))
    axes[0,1].set_title('ACF of Residuals')
    
    plot_pacf(residuals, ax=axes[1,0], lags=min(52, len(residuals)//2))
    axes[1,0].set_title('PACF of Residuals')
    
    # Q-Q plot for probability distribution
    stats.probplot(residuals, dist='norm', plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot')
    
    # histogram
    axes[2,0].hist(residuals, bins=20, density=True, alpha=0.7)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[2,0].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()))
    axes[2,0].set_title('Residual Distribution')
    
    # leave extra subplot empty, or remove it
    axes[2,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results