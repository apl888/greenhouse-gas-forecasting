# This edit adds one-step-ahead prediction in the evaluate_models_tscv function
# lines 123-133

# change enforce_stationarity and enforc_invertibility to 'False' to possibly better handle
# model warm-up issues.  

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
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import qqplot
from sktime.forecasting.tbats import TBATS
from sktime.performance_metrics.forecasting.probabilistic import EmpiricalCoverage
from sktime.forecasting.naive import NaiveForecaster
from scipy import stats
import time

# === 1. Model Validation via rolling-origin with expanding window ===
def rolling_origin_evaluation(
    y,
    order,
    seasonal_order,
    start_train_size,
    horizons=(1, 13, 52),
    step=13,
    enforce_stationarity=False,
    enforce_invertibility=False
):
    """
    Rolling-origin (walk-forward) evaluation with expanding window.

    Parameters
    ----------
    y : pd.Series
        Time series (weekly).
    order : tuple
        SARIMA nonseasonal order.
    seasonal_order : tuple
        SARIMA seasonal order (P, D, Q, s).
    start_train_size : int
        Initial number of observations to train on.
    horizons : iterable
        Forecast horizons (in weeks).
    step : int
        Step size between forecast origins.
    """

    results = []

    for t in range(start_train_size, len(y) - max(horizons), step):
        train = y.iloc[:t]

        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            trend='n',
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )

        res = model.fit(disp=False)

        for h in horizons:
            forecast = res.get_forecast(steps=h)
            y_pred = forecast.predicted_mean.iloc[-1]
            y_true = y.iloc[t + h - 1]

            results.append({
                'origin' : y.index[t],
                'horizon': h,
                'y_true' : y_true,
                'y_pred' : y_pred,
                'error'  : y_true - y_pred
            })

    return pd.DataFrame(results)

# === 2. Model Validation via TimeSerisSplitCV ===
def evaluate_models_tscv(
    models_list, 
    data, 
    exog=None, 
    n_splits=5, 
    test_size=52, 
    gap=13,                    # 3 month gap between train and validation sets to prevent leakage
    burn_in_period=52,
    estimation_method='lbfgs',
    enforce_stationarity=True,
    enforce_invertibility=True,
    trend='n',
    maxiter=300): 
    '''
    Evaluate specified candidate SARIMA/SARIMAX models on time series CV folds.
    Supports optional exogenous regressors (e.g., Fourier terms, weather, etc.).
    
        
    Parameters
    ----------
    burn_in_period : int, default=52
        Fixed number of initial residuals to exclude from diagnostics, primarily due to the initial model warm-up period
        (prevents data-dependent decisions)
    estimation_method : str, default='lbfgs'
        Better handling of initial values and warm-up period
    trend : str, default='n' 
        Use 't' for data with a more linear trend, as is the case with atmospheric CH4 time series data
    '''
    start_time = time.perf_counter()

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

    print(f"Evaluating {len(models_list)} models with {n_splits} folds each")
    print(f"Total iterations: {len(models_list) * n_splits}\n")

    # preview fold sizes
    for i, (train_idx, val_idx) in enumerate(tscv.split(data)):
        print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
    print()
    results_summary = []

    for params in tqdm(models_list, desc='Models'):
        rmse_scores, mae_scores = [], []
        aic_values, bic_values = [], []
        lb_pval_lag1, lb_pval_lag5, lb_pval_lag10, lb_pval_lag52 = [], [], [], []
        forecast_volatility = []

        successful_folds = 0

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(data)):

            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]

            # --- Handle exogenous regressors ---
            if exog is not None:
                exog_train = exog.iloc[train_idx]
                exog_val   = exog.iloc[val_idx]
            else:
                exog_train = exog_val = None

            # Must have at least 52 future points
            if len(val_fold) < test_size:
                continue

            # Restrict validation to first 52-step horizon
            val_target = val_fold.iloc[:test_size]
            if exog_val is not None:
                exog_val_h = exog_val.iloc[:test_size]
            else:exog_val_h = None

            try:
                # Build model
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                        message="Non-invertible starting")
                    warnings.filterwarnings("ignore",
                        message="Non-stationary starting")
                    warnings.filterwarnings("ignore",
                        message="Too few observations")
                    warnings.filterwarnings("always", category=ConvergenceWarning)

                    model = SARIMAX(
                        train_fold,
                        exog=exog_train,
                        order=params["order"],
                        seasonal_order=params["seasonal_order"],
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility,
                        trend=trend
                    )

                    results = model.fit(
                        disp=False,
                        maxiter=maxiter,
                        method=estimation_method
                    )

                converged = results.mle_retvals.get("converged", False)
                if not converged:
                    print(f"Model {params['order'], params['seasonal_order']} "
                          f"did not converge on fold {fold_idx+1}")
                    continue

                successful_folds += 1

                # --- True 52-step ahead forecast ---
                forecast = results.get_forecast(steps=test_size, exog=exog_val_h)
                forecast_values = forecast.predicted_mean

                # --- Accuracy ---
                rmse = np.sqrt(mean_squared_error(val_target, forecast_values))
                mae  = mean_absolute_error(val_target, forecast_values)

                rmse_scores.append(rmse)
                mae_scores.append(mae)
                aic_values.append(results.aic)
                bic_values.append(results.bic)

                # --- Residual diagnostics (filtered one-step-ahead) ---
                try:
                    pred = results.get_prediction(
                        start=train_fold.index[0],
                        end=train_fold.index[-1],
                        exog=exog_train,
                        dynamic=False
                    )
                    filt_mean = pred.predicted_mean
                    resid = train_fold.loc[filt_mean.index] - filt_mean

                    if len(resid) > burn_in_period:
                        stable_resid = resid.iloc[burn_in_period:]
                        
                        if len(stable_resid) >= 52:
                            lb = acorr_ljungbox(stable_resid,
                                                lags=[1, 4, 13, 26, 52],
                                                return_df=True)

                            lb_pval_lag1.append(lb.loc[1,  'lb_pvalue'])
                            lb_pval_lag5.append(lb.loc[4,  'lb_pvalue'])
                            lb_pval_lag10.append(lb.loc[13, 'lb_pvalue'])
                            lb_pval_lag10.append(lb.loc[26, 'lb_pvalue'])
                            lb_pval_lag52.append(lb.loc[52, 'lb_pvalue'])

                except Exception as e:
                    print(f"Ljung-Box failed on fold {fold_idx+1}: {e}")

                # approx seasonal "volatility"
                forecast_volatility.append(np.std(forecast_values))

            except Exception as e:
                print(f"Model {params['order'], params['seasonal_order']} "
                      f"failed on fold {fold_idx+1}: {e}")
                continue

        # require at least 3 converged folds
        if successful_folds >= 3:

            results_summary.append({
                'order': params['order'],
                'seasonal_order': params['seasonal_order'],
                'successful_folds': successful_folds,
                'convergence_rate': successful_folds / n_splits,
                'RMSE_mean': np.mean(rmse_scores),
                'RMSE_std': np.std(rmse_scores),
                'MAE_mean': np.mean(mae_scores),
                'AIC_mean': np.mean(aic_values),
                'BIC_mean': np.mean(bic_values),
                'forecast_volatility': np.mean(forecast_volatility),
                'LB_lag1_pval_mean': np.mean(lb_pval_lag1) if lb_pval_lag1 else np.nan,
                'LB_lag5_pval_mean': np.mean(lb_pval_lag5) if lb_pval_lag5 else np.nan,
                'LB_lag10_pval_mean': np.mean(lb_pval_lag10) if lb_pval_lag10 else np.nan,
                'LB_lag52_pval_mean': np.mean(lb_pval_lag52) if lb_pval_lag52 else np.nan
            })

    total_time = time.perf_counter() - start_time
    print(f"\nTotal duration: {total_time/60:.2f} minutes")
    print(f"Average per model: {(total_time/60)/len(models_list):.2f} minutes")

    print(f"\nNumber of models successfully evaluated: {len(results_summary)}")

    return pd.DataFrame(results_summary)

# === 3. Volatility / ARCH Check ===
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

# === 4. Make Fourier terms ===
def make_fourier_terms(index, 
                       period=52, 
                       K=1, 
                       start=0):     # start=0 is a global time index to anchor Fourier terms
    '''
    index  : pd.DatetimeIndex (e.g. weekly, DataFrame observations index)
    period : int (e.g., 52 weeks)
    K      : number of harmonic pairs (1..K)
    start  : integer offset for the first observation (global time anchor)
    returns: DataFrame of shape (len(index), 2*K) with sin_k and cos_k columns
    '''
    t = np.arange(start, start + len(index))
    fourier = {}
    for k in range(1, K + 1):
        angle = 2 * np.pi * k * t / period
        fourier[f'sin_{k}'] = np.sin(angle)
        fourier[f'cos_{k}'] = np.cos(angle)
    return pd.DataFrame(fourier, index=index)

# === 5. In-sample residual diagnostics ===
def in_sample_resid_analysis(train, 
                             order, 
                             seasonal_order, 
                             exog=None, 
                             run_hetero=False, 
                             burn_in_period=52,       # changed from trim_first to fixed burn-in
                             estimation_method='lbfgs',
                             maxiter=300,
                             enforce_stationarity=True, 
                             enforce_invertibility=True,
                             trend='n'
                             ):
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
    burn_in_period : int, default=52
        Fixed number of initial residuals to exclude (prevents data leakage)
    estimation_method : str, default='lbfgs'
        Better handling of initial values
    '''
    
    # --- Fit model ---
    model = SARIMAX(train, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    exog=exog, 
                    enforce_stationarity=enforce_stationarity, 
                    enforce_invertibility=enforce_invertibility, 
                    trend=trend)
    
    results = model.fit(method=estimation_method, maxiter=maxiter, disp=False)

    # use all residuals for model fitting assessment
    all_residuals = results.resid
    all_fitted = results.fittedvalues 
    
    # use stable residuals for diagnostics
    if len(all_residuals) > burn_in_period:
        stable_residuals = all_residuals.iloc[burn_in_period:]
        stable_fitted = all_fitted.iloc[burn_in_period:]
        stable_train = train.iloc[burn_in_period:]
    else:
        stable_residuals = all_residuals
        stable_fitted = all_fitted
        stable_train = train
        
    print(f'Using {len(stable_residuals)} residuals for diagnostics '
          f'(first {burn_in_period} excluded as burn_in)')
        
    # --- Compute in-sample accuracy on stable period ---
    rmse = np.sqrt(mean_squared_error(stable_train, stable_fitted))
    mae = mean_absolute_error(stable_train, stable_fitted)
    mape = np.mean(np.abs((stable_train - stable_fitted) / stable_train)) * 100
    r2 = r2_score(stable_train, stable_fitted)

    print(f'\n--- In-Sample Accuracy (after burn-in) ---')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE:  {mae:.3f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'R2:   {r2:.3f}')

    # --- Plot both full and stable periods ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full series
    ax1.plot(train, label='Actual', color='black', linewidth=2)
    ax1.plot(all_fitted, label='Fitted', color='red', linestyle='--', alpha=0.7)
    ax1.axvline(stable_train.index[0], color='blue', linestyle=':', 
                label=f'Burn-in end ({burn_in_period} points)')
    ax1.set_title('Full In-Sample Fitted vs Actual', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stable period only
    ax2.plot(stable_train, label='Actual', color='black', linewidth=2)
    ax2.plot(stable_fitted, label='Fitted', color='red', linestyle='--')
    ax2.set_title('Stable Period (After Burn-in)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # --- Plot residual diagnostics ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residual time series
    axes[0,0].plot(stable_residuals)
    axes[0,0].set_title('Residuals over Time')
    axes[0,0].set_xlabel('Time Index')
    axes[0,0].set_ylabel('Residual Value')

    # Histogram + KDE
    axes[0,1].hist(stable_residuals, bins=30, density=True, alpha=0.8, label='Hist')
    stable_residuals.plot(kind='kde', ax=axes[0,1], linewidth=2, alpha=0.6, label='KDE')
    
    # Theoretical normal curve
    x_vals = np.linspace(stable_residuals.min(), stable_residuals.max(), 200)
    normal_pdf = stats.norm.pdf(x_vals, loc=stable_residuals.mean(), scale=stable_residuals.std())
    axes[0,1].plot(x_vals, normal_pdf, color='r', linestyle='--', label='Normal PDF')
    axes[0,1].set_title('Residual Distribution', fontsize=14)
    axes[0,1].set_xlabel('Residual Value', fontsize=12)
    axes[0,1].set_ylabel('Density', fontsize=12)
    axes[0,1].legend()

    # Q-Q plot
    qqplot(stable_residuals, line='s', ax=axes[1,0])
    axes[1,0].set_title('Q-Q Plot', fontsize=14)
    axes[1,0].set_xlabel('Theoretical Quantiles', fontsize=12)
    axes[1,0].set_ylabel('Sample Quantiles', fontsize=12)

    # ACF plot
    plot_acf(stable_residuals, lags=55, ax=axes[1,1])
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
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(stable_residuals)
    print('\n--- Distribution Diagnostics ---')
    print('Jarque-Bera test:')
    print(f'\tJB = {jb_stat:.2f}')
    print(f'\tp = {jb_pvalue:.4f}')
    print(f'\tSkewness = {skew:.3f}')
    print(f'\tkurtosis = {kurtosis:.3f}')

    # Ljung-Box test
    lb = acorr_ljungbox(stable_residuals, lags=[1,4,13,26,52], return_df=True)
    print('\n--- Autocorrelation Diagnostics ---')
    print('Ljung-Box Test:')
    print('(H0: No autocorrelation up to specified lags)')
    for lag in [1,4,13,26,52]:
        print(f"\tlag {lag}: p = {lb.loc[lag, 'lb_pvalue']:.4f}")
        
    # Optional volatility clustering test
    try:
        test_volatility_clustering(stable_residuals)
    except NameError:
        pass
    
    # Optional: heteroscedasticity tests
    if run_hetero:
        exog = np.column_stack([np.ones(len(stable_residuals)), np.arange(len(stable_residuals))])  
        bp_p = het_breuschpagan(stable_residuals, exog)[1]
        white_p = het_white(stable_residuals, exog)[1]
        print('\n--- Heteroscedasticity Tests ---')
        print(f'Breusch-Pagan: p = {bp_p:.4f}')
        print(f'White: p = {white_p:.4f}')

    return results, stable_residuals

# === 6. Out-of-sample residual diagnostics ===
def out_of_sample_resid_analysis(train_data, 
                                 test_data, 
                                 order, 
                                 seasonal_order, 
                                 model_name,
                                 exog_train=None,
                                 exog_test=None,
                                 run_hetero=False,
                                 plot_forecast=True,
                                 maxiter=300,
                                 estimation_method='lbfgs',  
                                 enforce_stationarity=True,           
                                 enforce_invertibility=True,
                                 trend='n',
                                 seasonal_length=52):        
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
    print(f'Using estimation method: {estimation_method}')
    print(f'Test set size: {len(test_data)} observations\n')
    
    # Fit model on training data
    model = SARIMAX(train_data,
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog_train if exog_train is not None else None,
                    enforce_stationarity=enforce_stationarity, 
                    enforce_invertibility=enforce_invertibility, 
                    trend=trend)
    results = model.fit(method=estimation_method, maxiter=maxiter, disp=False)
    
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
    mape = np.mean(np.abs(residuals / test_data.replace(0, np.nan))) * 100
    r2 = r2_score(test_data, predictions)
    
    print('--- Out-of-Sample Forecast Accuracy ---')
    print(f'RMSE: {rmse:.3f}')
    print(f'MAE:  {mae:.3f}') 
    print(f'R²:   {r2:.3f}')   
    
    print('\n--- Horizon-specific RMSE ---')
    for h in [1,13,26,52]:
        if h <= len(test_data):
            rmse_h = np.sqrt(
                mean_squared_error(
                    test_data.iloc[:h],
                    predictions.iloc[:h]
                    )
                )
            print(f'h = {h:2d}: RMSE = {rmse_h:.3f}')
            
    print('\n--- Seasonal naive benchmark comparison ---')
    
    if len(train_data) < seasonal_length:
        raise ValueError('Training data shorter than one seasonal cycle')
    
    seasonal_naive_forecast = train_data.iloc[-seasonal_length:].values
    seasonal_naive_forecast = np.tile(
        seasonal_naive_forecast,
        int(np.ceil(len(test_data) / seasonal_length))
    )[:len(test_data)]                                 # trim to exact forecast length in case test data is longer
    
    seasonal_naive_forecast = pd.Series(
        seasonal_naive_forecast,
        index=test_data.index
    )
        
    naive_rmse = np.sqrt(mean_squared_error(test_data, seasonal_naive_forecast))
    
    print(f'Seasonal naive RMSE: {naive_rmse:.3f}')
    print(f'Skill vs naive: {(1 - rmse / naive_rmse) * 100:.2f}%')
    print('\n     - skill > 0% --> the model beats naive')
    print('     - skill = 0% --> the model adds little value')
    print('     - skill < 0% --> the model is worse than naive')   
    
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
    for lag in lags:
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

# === 7. Summary assumption check ===
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
