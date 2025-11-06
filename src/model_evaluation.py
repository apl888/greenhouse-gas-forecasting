# src/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white, normal_ad
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import qqplot
from scipy import stats
import time

# === 1. Model Validation via TimeSerisSplitCV ===
def evaluate_models_tscv(models_list, data, exog=None, n_splits=5, test_size=52, gap=13): # 3 month gap between train and validation sets to prevent leakage
    '''
    Evaluate specified candidate SARIMA/SARIMAX models on time series CV folds.
    Supports optional exogenous regressors (e.g., Fourier terms, weather, etc.).
    '''
    start_time = time.perf_counter()
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    
    print(f'Evaluating {len(models_list)} models with {n_splits} folds each')
    print(f'Total iterations: {len(models_list) * n_splits}\n')

    # Check split sizes
    for i, (train_idx, val_idx) in enumerate(tscv.split(data)):
        print(f'Fold {i+1}: Train={len(train_idx)} ({len(train_idx)/52:.1f} years), '
              f'Val={len(val_idx)} ({len(val_idx)/52:.1f} years), '
              f'Total={len(train_idx) + len(val_idx)}')
    print()
    
    results_summary = []
    
    for params in tqdm(models_list, desc='Models'):
        rmse_scores, mae_scores = [], []
        aic_values, bic_values = [], []
        lb_pval_lag1, lb_pval_lag5, lb_pval_lag10, lb_pval_lag52  = [], [], [], []
        seasonal_strengths = []
        
        successful_folds = 0
        
        for train_idx, val_idx in tscv.split(data):
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]
            
            # --- slice exogenous terms ---
            if exog is not None:
                exog_train = exog.iloc[train_idx]
                exog_val = exog.iloc[val_idx]
            else:
                exog_train = exog_val = None
            
            try:
                # suppress parameter initialization warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", 
                        message="Non-stationary starting autoregressive parameters")
                    warnings.filterwarnings("ignore", 
                        message="Non-invertible starting MA parameters")
                    warnings.filterwarnings("ignore", 
                        message="Non-invertible starting seasonal moving average")
                    warnings.filterwarnings("ignore", 
                        message="Too few observations to estimate starting parameters")
                    # keep convergence warnings visible
                    warnings.filterwarnings("always", category=ConvergenceWarning)
             
                    model = SARIMAX(
                        train_fold, 
                        exog=exog_train,
                        order=params['order'], 
                        seasonal_order=params['seasonal_order'],
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    )
    
                    results = model.fit(
                        disp=False, 
                        maxiter=2000,
                        method='lbfgs'
                    )
                
                # Check convergence
                converged = results.mle_retvals.get('converged', False)
                if not converged:
                    print(f"Model {params['order']}, {params['seasonal_order']} did not converge on fold {successful_folds+1}")
                    continue # skip non-coverged folds
                
                successful_folds += 1
                
                # Forecast and calculate metrics
                forecast = results.get_forecast(steps=len(val_fold), exog=exog_val)
                forecast_values = forecast.predicted_mean
                
                rmse = np.sqrt(mean_squared_error(val_fold, forecast_values))
                mae = mean_absolute_error(val_fold, forecast_values)

                # Ljung-Box test with safety check
                if len(results.resid) >= 52:
                    try:
                        lb_test_comprehensive = acorr_ljungbox(results.resid, lags=[1,5,10,52], return_df=True)
                        lb_pval_lag1.append(lb_test_comprehensive.loc[1, 'lb_pvalue'])
                        lb_pval_lag5.append(lb_test_comprehensive.loc[5, 'lb_pvalue'])
                        lb_pval_lag10.append(lb_test_comprehensive.loc[10, 'lb_pvalue'])
                        lb_pval_lag52.append(lb_test_comprehensive.loc[52, 'lb_pvalue'])
                    except Exception as lb_error:
                        print(f'Ljung-Box test failed on fold {fold_index+1}: {lb_error}')
                
                rmse_scores.append(rmse)
                mae_scores.append(mae)
                aic_values.append(results.aic)
                bic_values.append(results.bic)
                
                # Seasonal strength
                if len(forecast_values) >= 52:
                    seasonal_strengths.append(np.std(forecast_values))
                    
            except Exception as e:
                print(f"Model {params['order']}, {params['seasonal_order']} failed on fold {successful_folds+1}: {e}")
                continue
        
        # Only include models with sufficient successful folds
        if successful_folds >= 3:  # At least 3/5 folds converged
            convergence_rate = successful_folds / n_splits
            
            results_summary.append({
                'order': params['order'],
                'seasonal_order': params['seasonal_order'],
                'successful_folds': successful_folds,
                'convergence_rate': convergence_rate,
                'RMSE_mean': np.mean(rmse_scores) if rmse_scores else np.nan,
                'RMSE_std': np.std(rmse_scores) if rmse_scores else np.nan,
                'MAE_mean': np.mean(mae_scores) if mae_scores else np.nan,
                'AIC_mean': np.mean(aic_values) if aic_values else np.nan,
                'BIC_mean': np.mean(bic_values) if bic_values else np.nan,
                'Seasonal_Strength': np.mean(seasonal_strengths) if seasonal_strengths else 0,
                'LB_lag1_pval_mean': np.mean(lb_pval_lag1) if lb_pval_lag1 else np.nan,
                'LB_lag5_pval_mean': np.mean(lb_pval_lag5) if lb_pval_lag5 else np.nan,
                'LB_lag10_pval_mean': np.mean(lb_pval_lag10) if lb_pval_lag10 else np.nan,
                'LB_lag52_pval_mean': np.mean(lb_pval_lag52) if lb_pval_lag52 else np.nan
            })


    # get the total duration and duration per model
    end_time = time.perf_counter()     
    total_duration = end_time - start_time

    print(f'\nTotal duration: {total_duration / 60:.2f} minutes')
    print(f'Average duration per model: {(total_duration / 60) / len(models_list):.2f} minutes')
    print(f"\nNumber of models successfully evaluated: {len(results_summary)}")
        
    return pd.DataFrame(results_summary)

# === 1. Volatility / ARCH Check ===
def test_volatility_clustering(residuals, plot=False):
    '''
    Check for GARCH effects in residuals
    '''
    squared_residuals = residuals ** 2
    lags = [1, 4, 8, 12, 26]
    
    # Ljung-Box test on squared residuals ( Engle's ARCH test)
    arch_test = acorr_ljungbox(squared_residuals, lags=lags, return_df=True)
    
    print('\n--- Volatility Clustering Diagnostics ---')
    print("Engle's ARCH Test (on squared residuals):")
    print('(H0: No ARCH effects (constant variance))')
    for lag in lags:
        pval = arch_test.loc[lag, 'lb_pvalue']
        sig = '***' if pval < 0.05 else ''
        print(f'\tLag {lag}: p-value = {pval:.4f}{sig}')
    
    # Visual check
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(residuals)
        axes[0].set_title('Residuals Over Time', fontsize=14)
        axes[0].set_ylabel('Residuals', fontsize=12)

        axes[1].plot(squared_residuals)
        axes[1].set_title('Squared Residuals (Volatility Clustering Check)', fontsize=14)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Squared Residuals', fontsize=12)

        plt.tight_layout()
        plt.show()
    
    return arch_test

# === 2. In-sample residual diagnostics ===
def in_sample_resid_analysis(train, order, seasonal_order, exog=None, run_hetero=False, trim_first=False):
    '''
    Fit a SARIMA model and run in-sample diagnostics.
    
    Parameters
    ----------
    train          : pd.Series
        Training set (time-indexed).
    order          : tuple
        SARIMA (p,d,q).
    seasonal_order : tuple
        SARIMA seasonal (P,D,Q,s).
    exog           : pd.DataFrame or np.ndarray, optional
        Exogenous variables for SARIMAX (must align with 'train' index)
    run_hetero     : bool, default False
        If True, runs Breusch-Pagan and White tests for heteroscedasticity.
    trim_first     : bool, default False
        If True, drops the first residual before diagnostic plots and tests.
    '''
    
    # --- Fit model ---
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    exog=exog, enforce_stationarity=True, enforce_invertibility=True, 
                    trend='c')
    results = model.fit(method='lbfgs', disp=False)
    fitted_values = results.fittedvalues
    residuals = results.resid
    
    # optionally trim the first residual (if it is a model warm-up artifact)
    if trim_first:
        fitted_values = fitted_values.iloc[1:]
        residuals = residuals.iloc[1:]
        print('Note: First residual removed before plotting and diagnostics.\n')
        
    # --- Compute in-sample accuracy metrics ---
    rmse = np.sqrt(mean_squared_error(train.iloc[-len(fitted_values):], fitted_values))
    mae = mean_absolute_error(train.iloc[-len(fitted_values):], fitted_values)
    mape = np.mean(np.abs((train.iloc[-len(fitted_values):] - fitted_values) / 
                          train.iloc[-len(fitted_values):])) * 100
    r2 = r2_score(train.iloc[-len(fitted_values):], fitted_values)

    print(f'\n--- In-Sample Accuracy ---')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE:  {mae:.3f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'R2:   {r2:.3f}')

    # --- Plot fitted vs actual ---
    plt.figure(figsize=(12,5))
    plt.plot(train, label='Actual', color='black', linewidth=2)
    plt.plot(fitted_values, label='Fitted', color='red', linestyle='--')
    plt.title('In-Sample Fitted vs Actual', fontsize=14)
    plt.ylabel('Concentration', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot residual diagnostics ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residual time series
    axes[0,0].plot(residuals)
    axes[0,0].set_title('Residuals over Time')
    axes[0,0].set_xlabel('Time Index')
    axes[0,0].set_ylabel('Residual Value')

    # Histogram + KDE
    axes[0,1].hist(residuals, bins=30, density=True, alpha=0.8, label='Hist')
    residuals.plot(kind='kde', ax=axes[0,1], linewidth=2, alpha=0.6, label='KDE')
    # Theoretical normal curve
    x_vals = np.linspace(residuals.min(), residuals.max(), 200)
    normal_pdf = stats.norm.pdf(x_vals, loc=residuals.mean(), scale=residuals.std())
    axes[0,1].plot(x_vals, normal_pdf, color='r', linestyle='--', label='Normal PDF')
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
    
    # --- Statistical diagnostic tests ---
    print('\n--- Residual Diagnostic Tests ---')
    
    # Durbin-Watson 
    dw_stat = durbin_watson(residuals)
    print(f'Durbin-Watson statistic: {dw_stat:.3f}')
    print('\t< 2: positive autocorrelation \n\t~= 2: no autocorrelation \n\t> 2: negative autocorrelation\n')
    
    # Jarque-Bera
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    print('Jarque-Bera test:')
    print(f'\tJB = {jb_stat:.2f}')
    print(f'\tp = {jb_pvalue:.4f}')
    print(f'\tSkewness = {skew:.3f}')
    print(f'\tkurtosis = {kurtosis:.3f}')

    # Ljung-Box test
    lb = acorr_ljungbox(residuals, lags=[1,5,10,52], return_df=True)
    print('\n--- Autocorrelation Diagnostics ---')
    print('Ljung-Box Test:')
    print('(H0: No autocorrelation up to specified lags)')
    for lag in [1,5,10,52]:
        print(f"\tlag {lag}: p = {lb.loc[lag, 'lb_pvalue']:.4f}")
        
    # Optional volatility clustering test
    try:
        test_volatility_clustering(residuals)
    except NameError:
        pass
    
    # Optional: heteroscedasticity tests
    if run_hetero:
        exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])  
        bp_p = het_breuschpagan(residuals, exog)[1]
        white_p = het_white(residuals, exog)[1]
        print('\n--- Heteroscedasticity Tests ---')
        print(f'Breusch-Pagan: p = {bp_p:.4f}')
        print(f'White: p = {white_p:.4f}')

    return results

# === 3. Out-of-sample residual diagnostics ===
def out_of_sample_resid_analysis(train_data, test_data, order, seasonal_order, model_name):
    '''
    Complete residual diagnostics for test set
    '''
    
    print(f'=== Out-of-Sample Residual Analysis: {model_name} ===')
    
    # Fit model on training data
    model = SARIMAX(train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True, 
                    enforce_invertibility=True, 
                    trend='c')
    results = model.fit(method='lbfgs', disp=False)
    
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
    lb = acorr_ljungbox(residuals, lags=[1,5,10,52], return_df=True)
    print('\n--- Autocorrelation Diagnostics ---')
    print('Ljung-Box Test')
    print('H0: No autocorrelation up to specified lags')
    for lag in [1,5,10,52]:
        print(f'lag {lag}: p = {lb.loc[lag, 'lb_pvalue']:.4f}')
    
    # 4. Volatility clustering check (using existing function)
    test_volatility_clustering(residuals, plot=False)
    
    # Plot residual diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
    # Residuals over time
    axes[0,0].plot(residuals.index, residuals.values)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_title(f'{model_name} - Prediction Test Residuals Over Time', fontsize=14)
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

# === 4. Summary assumption check ===
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
