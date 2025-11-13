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
        forecast_volatility = []
        
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
                    forecast_volatility.append(np.std(forecast_values))
                    
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
                'forecast_volatility': np.mean(forecast_volatility) if forecast_volatility else 0,
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
    model = SARIMAX(train, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    exog=exog, 
                    enforce_stationarity=True, 
                    enforce_invertibility=True, 
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
    plt.figure(figsize=(12,4))
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
    print('\n=== Residual Diagnostic Tests ===')
    
    # Durbin-Watson 
    # This is largely redundant with Ljung-Box test at lag 1 (see below).  
    # I will leave the code here in commented-out form in case the test becomes desirable.  
    # dw_stat = durbin_watson(residuals)
    # print(f'Durbin-Watson statistic: {dw_stat:.3f}')
    # print('\t< 2: positive autocorrelation \n\t~= 2: no autocorrelation \n\t> 2: negative autocorrelation\n')
    
    # Jarque-Bera
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    print('\n--- Distribution Diagnostics ---')
    print('Jarque-Bera test:')
    print(f'\tJB = {jb_stat:.2f}')
    print(f'\tp = {jb_pvalue:.4f}')
    print(f'\tSkewness = {skew:.3f}')
    print(f'\tkurtosis = {kurtosis:.3f}')

    # Ljung-Box test
    lb = acorr_ljungbox(residuals, lags=[1,4,13,26,52], return_df=True)
    print('\n--- Autocorrelation Diagnostics ---')
    print('Ljung-Box Test:')
    print('(H0: No autocorrelation up to specified lags)')
    for lag in [1,4,13,26,52]:
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
def out_of_sample_resid_analysis(train_data, 
                                 test_data, 
                                 order, 
                                 seasonal_order, 
                                 model_name,
                                 exog_train=None,
                                 exog_test=None,
                                 run_hetero=False,
                                 plot_forecast=True):
    '''
    Complete residual diagnostics for test set
    
    Parameters
    ----------
    train_data     : pd.Series
        Training dataset
    test_data      : pd.Series  
        Test dataset (typically 52 weeks)
    order          : tuple
        SARIMA (p,d,q) order
    seasonal_order : tuple
        SARIMA seasonal (P,D,Q,s) order  
    model_name     : str
        Name for labeling plots/results
    run_hetero     : bool, default False
        Whether to run heteroscedasticity tests
    plot_forecast  : bool, default True
        Whether to plot forecast vs actual comparison
    '''
    
    print(f'=== Out-of-Sample Residual Analysis: {model_name} ===')
    print(f'Test set size: {len(test_data)} observations\n')
    
    # Fit model on training data
    model = SARIMAX(train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog_train if exog_train is not None else None,
                    enforce_stationarity=True, 
                    enforce_invertibility=True, 
                    trend='c')
    results = model.fit(method='lbfgs', disp=False)
    
    # get forecast with confidence intervals
    forecast_obj = results.get_forecast(
        steps=len(test_data),
        exog=exog_test if exog_test is not None else None
        )
    predictions = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    
    # Calculate test residuals
    residuals = test_data - predictions
    
    #--- accuracy metrics ---
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    mae = mean_absolute_error(test_data, predictions)
    mape = np.mean(np.abs(residuals / test_data)) * 100
    r2 = r2_score(test_data, predictions)
    
    print('--- Out-of-Sample Forecast Accuracy ---')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE:  {mae:.3f}') 
    print(f'MAPE: {mape:.2f}%')
    print(f'R²:   {r2:.3f}')   
    
    #--- forecast coverage (for uncertainty quantification) ---
    coverage = np.mean((test_data >= conf_int.iloc[:,0]) & 
                       (test_data <= conf_int.iloc[:,1]))
    print(f'95% CI coverage: {coverage:.1%} (ideal: 95%)')
    
    # plot forecast vs actual
    if plot_forecast:
        plt.figure(figsize=(12,6))
        plt.plot(test_data.index, test_data.values, label='Actual', marker='o', markersize=4)
        plt.plot(predictions.index, predictions.values, label='Forecast', linestyle='--', linewidth=2)
        plt.fill_between(conf_int.index, 
                         conf_int.iloc[:,0], 
                         conf_int.iloc[:,1],
                         color='r',
                         alpha=0.2,
                         label='95% Confidence Interval')
        plt.title(f'{model_name} - Out-of-Sample Forecast vs Actual', fontsize=14)
        plt.ylabel('Concentration', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
                         
    #--- residual statistical analysis ---
    print('\n--- Residual Statistical Properties ---')
    print(f'Mean: {residuals.mean():.5f}')
    print(f'Std:  {residuals.std():.5f}')
    print(f'Skew: {residuals.skew():.3f}')
    print(f'Kurtosis: {residuals.kurtosis():.3f}')
    
    # --- normality tests ---
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    jb_stat, jb_p, _, _ = jarque_bera(residuals)
    
    print('\n--- Normality Tests ---')
    print(f'Shapiro-Wilk: p = {shapiro_p:.4f}')
    print(f'Jarque-Bera:  p = {jb_p:.4f}')
    
    # --- autocorrelation analysis ---
    max_lag = min(52, len(residuals) - 2)
    lags = [lag for lag in [1, 4, 13, 26, 52] if lag <= max_lag]
    lb = acorr_ljungbox(residuals, lags=lags, return_df=True)
    
    print('\n--- Autocorrelation Diagnostics ---')
    print('Ljung-Box Test (H0: No autocorrelation)')
    significant_lags = []
    for lag in [1, 4, 13, 26, 52]:
        p_val = lb.loc[lag, 'lb_pvalue']
        significance = '***' if p_val < 0.05 else ''
        if p_val < 0.05:
            significant_lags.append(lag)
        print(f'  Lag {lag:2d}: p = {p_val:.4f} {significance}')
    
    if significant_lags:
        print(f'\n"\u26A0"  Significant autocorrelation detected at lags: {significant_lags}')
    else:
        print('"\u2705" No significant autocorrelation detected')
        
    # --- stationarity check on residuals ---
    adf_stat, adf_p, _, _, _, _ = adfuller(residuals)
    print(f'\n--- Stationarity Check ---')
    print(f'ADF test on residuals: p = {adf_p:.4f}')
    print('"\u2705" Residuals are stationary' if adf_p < 0.05 else '"\u26A0"  Residuals may not be stationary')
    
     # --- optional: heteroscedasticity tests ---
    if run_hetero:
        # Use time trend as exogenous variable for the test
        exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])
        bp_p = het_breuschpagan(residuals, exog)[1]
        white_p = het_white(residuals, exog)[1]
        
        print('\n--- Heteroscedasticity Tests ---')
        print(f'Breusch-Pagan: p = {bp_p:.4f}')
        print(f'White test:    p = {white_p:.4f}')
        print('"\u2705" Constant variance' if bp_p > 0.05 else '"\u26A0"  Possible heteroscedasticity')   
    
    # --- volatility clustering check ---
    try:
        test_volatility_clustering(residuals, plot=False)
    except NameError:
        print('\n"\u26A0"  Volatility clustering test function not available')
    
    # --- residual diagnostic plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{model_name} - Out-of-Sample Residual Diagnostics', fontsize=16, y=1.02)
        
    # 1. residuals over time
    axes[0,0].plot(residuals.index, residuals.values)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].fill_between(residuals.index, -2*residuals.std(), 2*residuals.std(), 
                          alpha=0.2, color='gray', label='±2 sigma')
    axes[0,0].set_title('Residuals Over Time', fontsize=14)
    axes[0,0].set_ylabel('Residual Value', fontsize=12)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
        
    # 2. Q-Q plot
    stats.probplot(residuals, dist='norm', plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot', fontsize=14)
    axes[0,1].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[0,1].set_ylabel('Sample Quantiles', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
        
    # 3. histogram
    axes[1,0].hist(residuals, bins=min(20, len(residuals)//3), density=True, alpha=0.7)
    
    # Overlay normal distribution
    xmin, xmax = axes[1,0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, residuals.mean(), residuals.std())
    axes[1,0].plot(x, p, 'k', linewidth=2, label='Normal dist')
    axes[1,0].set_title('Residual Distribution', fontsize=14)
    axes[1,0].set_xlabel('Residual Value', fontsize=12)
    axes[1,0].set_ylabel('Density', fontsize=12)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
        
    # ACF of residuals
    plot_acf(residuals, lags=min(40, len(residuals)-1), ax=axes[1,1])
    axes[1,1].set_title('ACF of Residuals', fontsize=14)
    axes[1,1].set_ylabel('Autocorrelation', fontsize=12)
    axes[1,1].set_xlabel('Lag', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
    
    # --- return comprehensive results ---
    analysis_results = {
        'residuals': residuals,
        'predictions': predictions,
        'confidence_intervals': conf_int,
        'accuracy_metrics': {
            'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'coverage': coverage
        },
        'normality_tests': {'shapiro_p': shapiro_p, 'jarque_bera_p': jb_p},
        'autocorrelation_tests': lb,
        'stationarity_test': adf_p
    }
    
    return analysis_results, results

# === 4. Summary assumption check ===
def summarize_model_assumptions(residuals, alpha=0.05):
    '''
    Evaluate normality, zero-mean, and autocorrelation assumptions.
    '''
    
    # Normality test
    _, normality_p = stats.shapiro(residuals)
    
    # Autocorrelation test
    lb = acorr_ljungbox(residuals, lags=[1,4,13,26,52], return_df=True)
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
