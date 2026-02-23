# =========================================================
# model_evaluation.py
# =========================================================
# PURPOSE:
# Unified, modular evaluation framework for
# SARIMA / ETS / TBATS mean models + optional GARCH volatility
# =========================================================
# Benchmark set:
# Core deterministic accuracy
# - RMSE
# - Horizon RMSE
# - NSE (Nash-Sutcliffe Efficiency)
# - Skill vs seasonal naive
# 
# Probabilistic accuracy
# - Interval coverage
# - CRPS (Continuous Ranked Probability Score)
#
# Comparative testing
# - DM (Diebold-Mariano test)
# ==========================================================

import numpy as np
import pandas as pd
import logging
import time
import os
import pickle
from tqdm.auto import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

from sktime.forecasting.tbats import TBATS
from sktime.forecasting.ets import AutoETS

from arch import arch_model
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =========================================================
# Metric helpers (standardized across models)
# =========================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def nash_sutcliffe_efficiency(y_true, y_pred, y_train):
    """
    Classical Nash-Sutcliffe Efficiency.
    Out-of-Sample.
    Benchmark = training-period mean.
    NOT appropriate for seasonal rolling-origin forecast comparison.
    """
    y_bar = np.mean(y_train)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_bar) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - numerator / denominator

# Note:  NSE is computed relative to the in-sample climatological mean, 
# consistent with hydrological and environmental forecasting standards.

def nse_vs_naive(y_true, y_pred, y_naive):
    """
    Nash-Sutcliffe efficiency relative to seasonal naive benchmark.
    Appropriate for rolling-origin forecast evaluation.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_naive) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - numerator / denominator

# Note: For reporting on methane data, standard NSE can be misleading. Because the CH4 trend 
# is so aggressive, almost any model that captures a basic upward slope will have a very high 
# standard NSE, simply because the "mean of the data" is a very poor predictor for a trending series.

def skill_vs_naive(y_true, y_pred, y_naive):
    """
    RMSE skill relative to naive benchmark
    """
    return 1 - rmse(y_true, y_pred) / rmse(y_true, y_naive)

# example notebook usage of metric helpers
#
# y_pred = model_forecast
# rmse_val = rmse(y_test, y_pred)
# mae_val  = mae(y_test, y_pred)
# nse_val  = nash_sutcliffe_efficiency(y_test, y_pred)
#
# y_naive = seasonal_naive_forecast(y_train, len(y_test), sp=52)
# skill   = skill_vs_naive(y_test, y_pred, y_naive)


# =========================================================
# Seasonal naive benchmark
# =========================================================

def seasonal_naive_forecast(y_train, horizon, sp):
    """
    Seasonal naive forecast:
    repeats last observed seasonal cycle.
    """
    last_season = y_train[-sp:]
    reps = int(np.ceil(horizon / sp))
    forecast = np.tile(last_season, reps)[:horizon]
    return forecast

# Example notebook usage:
#
# y_naiv e= seasonal_naive_forecast(
#     y_train=y_train,
#     horizon=len(y_test),
#     sp=52
# )
#
# naive_rmse = rmse(y_test, y_naive)

# =========================================================
# CRPS (Gaussian)
# =========================================================

def crps_gaussian(y, mu, sigma):
    """
    Continuous Ranked Probability Score (CRPS)
    Closed-form CRPS for Gaussian predictive distribution.

    Parameters
    ----------
    y     : array-like (observed)
    mu    : array-like (forecast mean)
    sigma : array-like (forecast std)

    Returns
    -------
    float : mean CRPS
    """
    y = np.asarray(y)
    mu = np.asarray(mu)
    sigma = np.maximum(np.asarray(sigma), 1e-8)

    z = (y - mu) / sigma
    crps = sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )

    return np.mean(crps)

# Example notebook usage:
#
# mean, sigma, _ = forecast_mean_model(
#     fitted_model,
#     model_type="sarima",
#     horizon=len(y_test)
# )
#
# crps = crps_gaussian(
#     y=y_test,
#     mu=mean,
#     sigma=sigma
# )

# =========================================================
# PIT (Gaussian)
# =========================================================

def pit_gaussian(y, mu, sigma):
    '''
    Calculate the Probability Integral Transform (PIT) for a Gaussian distribution.

    The PIT is the Value of the cumulative distribution function (CDF) 
    evaluated at the observed value 'y'. For a well-calibrated model, 
     the PIT values across a dataset should follow a Uniform(0, 1) distribution.

    Parameters
    ----------
    y : float or array_like
        The actual observed value(s).
    mu : float or array_like
        The mean (forecasted location) of the Gaussian distribution.
    sigma : float or array_like
        The standard deviation (forecasted scale) of the Gaussian 
        distribution. Must be positive.

    Returns
    -------
    pit : float or array_like
        The probability integral transform value(s) in the range [0, 1].

    Notes
    -----
    - If PIT values are clustered near 0 and 1 (U-shape), the model is under-dispersed.
    - If PIT values are clustered near 0.5 (hump-shape), the model is over-dispersed.
    '''
    z = (y - mu) / sigma
    return norm.cdf(z) 

# =========================================================
# 1. Mean model fitting (Abstraction layer)
# =========================================================

def fit_mean_model(y, 
                   model_type, 
                   model_params, 
                   exog=None, 
                   start_params=None,
                   verbose=False):
    """
    Fits a mean model and returns a fitted object.
    Supported models: 'sarima', 'ets', 'tbats'
    """

    if verbose:
        logger.info(f"Fitting {model_type.upper()} model")

    if model_type == 'sarima':
        if 'trend' not in model_params:
            raise ValueError("Explicit trend must be specified for SARIMA")
        model = SARIMAX(
            y,
            exog=exog,
            order=model_params['order'],
            seasonal_order=model_params['seasonal_order'],
            trend=model_params.get('trend', 'n'),
            enforce_stationarity=model_params.get('enforce_stationarity', True),
            enforce_invertibility=model_params.get('enforce_invertibility', True)
        )

        results = model.fit(
            start_params=start_params,
            disp=False,
            maxiter=model_params.get('maxiter', 300)
        )

        resid = results.resid.dropna()
        fitted_vals = results.fittedvalues.loc[resid.index]

    elif model_type == 'ets':
        model = AutoETS(**model_params)
        results = model.fit(y)

        resid = results.resid.dropna()
        fitted_vals = results.fittedvalues.loc[resid.index]

    elif model_type == 'tbats':
        model = TBATS(**model_params)
        results = model.fit(y)

        resid = results.predict_residuals().dropna()
        fitted_vals = y.loc[resid.index] - resid

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return {
        "model_type": model_type,
        "results": results,
        "residuals": resid,
        "fitted_values": fitted_vals
    }

# =========================================================
# 2. Forecasting wrapper
# =========================================================

def forecast_mean_model(
    fitted, 
    horizon, 
    exog=None
    ):
    '''
    Generate forecasts and predictive uncertainty.
    Returns dict with mean, sigma, intervals.
    '''

    model_type = fitted['model_type']
    results = fitted['results']

    if model_type == 'sarima':
        fc = results.get_forecast(steps=horizon, exog=exog)
        mean = fc.predicted_mean
        var = fc.var_pred_mean
        sigma = np.sqrt(var)
        intervals = fc.conf_int()

    elif model_type in ['ets', 'tbats']:
        fh = np.arange(1, horizon + 1)
        mean = results.predict(fh)

        try:
            intervals = results.predict_interval(fh, coverage=0.95)
            sigma = (intervals.iloc[:, 1] - intervals.iloc[:, 0]) / (2 * 1.96)
        except Exception:
            sigma = pd.Series(np.nan, index=mean.index)
            intervals = None

    else:
        raise ValueError('Unsupported model type')

    return {
        'mean': pd.Series(mean),
        'sigma': pd.Series(sigma),
        'intervals': intervals
    }

# =========================================================
# 3. Residual diagnostics (in + out-of-sample)
# =========================================================

def residual_diagnostics(
    residuals, 
    title='', 
    plot=True,
    return_results=False
    ):
    """
    Model-agnostic residual diagnostics.
    Works for in-sample or out-of-sample residuals.
    
    Returns
    -------
    dict or None
        If return_results=True, returns a dictionary with keys:
        - 'ljung_box' : DataFrame of Ljung-Box test results
        - 'jb_pvalue' : Jarque-Bera p-value
        - 'adf_pvalue': ADF test p-value (or None if insufficient data)
        - 'arch_pvalue': Engle's ARCH test p-value

    Notes
    -----
    The plots include:
    - Time series of residuals
    - ACF of residuals (40 lags)
    - Q-Q plot against normal distribution
    - Histogram with fitted normal density
    - Squared residuals time series
    - ACF of squared residuals (40 lags)

    The ADF test requires at least 50 observations; otherwise p-value is None.
    """

    print(f"\n=== Residual Diagnostics: {title} ===")

    # --- Summary stats ---
    print(f"Mean    : {residuals.mean():.5f}")
    print(f"Std     : {residuals.std():.5f}")
    print(f"Skew    : {residuals.skew():.3f}")
    print(f"Kurtosis: {residuals.kurtosis():.3f}")

    # --- Normality ---
    jb_stat, jb_p, _, _ = jarque_bera(residuals)
    print(f"\nJarque-Bera p-value: {jb_p:.4f}")

    # --- Autocorrelation ---
    lb = acorr_ljungbox(residuals, 
                        lags=[1, 4, 13, 26, 52], 
                        return_df=True)
    print("\nLjung-Box p-values:")
    for lag in lb.index:
        print(f"  lag {lag}: p = {lb.loc[lag, 'lb_pvalue']:.4f}")
        
    # --- Conditional heteroscedasticity ---   
    arch_stat, arch_p, _, _ = het_arch(residuals, nlags=52)
    print(f"\nEngle's ARCH test (nlags=52) p-value: {arch_p:.4e}")

    # --- Stationarity ---
    adf_p = None
    if len(residuals) > 50:
        adf_p = adfuller(residuals, autolag='AIC')[1]
        print(f"\nADF p-value: {adf_p:.4f}")

    if plot:       
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        plt.suptitle(f'\n{title} Residual Diagnostic Plots\n', fontsize=18)
        
        axes[0,0].plot(residuals)
        axes[0,0].set_title('Residuals')

        plot_acf(residuals, lags=40, ax=axes[0,1])
        axes[0,1].set_title('ACF')

        qqplot(residuals, line='s', ax=axes[1,0])
        axes[1,0].set_title('Q-Q Plot')

        axes[1,1].hist(residuals, bins=30, density=True, alpha=0.7)
        x = np.linspace(residuals.min(), residuals.max(), 200)
        axes[1,1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()))
        axes[1,1].set_title('Histogram')
        
        axes[2,0].plot(residuals**2)
        axes[2,0].set_title('Squared Residuals')

        plot_acf(residuals**2, lags=40, ax=axes[2,1])
        axes[2,1].set_title('Squared Residual ACF')

        plt.tight_layout()
        plt.show()

    results = {
        'ljung_box'  : lb,
        'jb_pvalue'  : jb_p,
        'adf_pvalue' : adf_p,
        'arch_pvalue': arch_p
    }
    
    if return_results:
        return results

# Example notebook usage:
#
# in_sample_resid = fitted.resid
# residual_diagnostics(
#     in_sample_resid,
#     title="SARIMA in-sample residuals"
# )
#
# out_sample_resid = y_test - mean
# residual_diagnostics(
#     out_sample_resid,
#     title="SARIMA forecast residuals"
# )

# =========================================================
# 4. GARCH volatility modeling
# =========================================================

def fit_garch(residuals, 
              p=1, 
              q=1, 
              dist='normal', 
              verbose=False):
    """
    Fits a GARCH(p,q) model to residuals.
    
    Parameters
    ----------
    residuals : array-like
        Residuals from a mean model (e.g., y_train - fitted).
    p : int, default=1
        GARCH lag order for the variance term.
    q : int, default=1
        ARCH lag order for the squared residual term.
    dist : {'normal', 't', 'skewt'}, default='normal'
        Conditional distribution of the standardized residuals.
    verbose : bool, default=False
        If True, log an info message.

    Returns
    -------
    result : ARCHModelResult
        Fitted GARCH model result from `arch` package.
    scale : float
        Standard deviation used to scale residuals before fitting.
        Multiply conditional volatilities by `scale` to return to original units.

    Notes
    -----
    The function scales the residuals by their standard deviation before fitting
    to keep parameters well-behaved. The returned `scale` should be used to
    rescale forecasted variances/volatilities.
    """

    if verbose:
        logger.info(f"Fitting GARCH({p},{q})")
        
    scale = residuals.std()               # keeps sigma_t in the original units
    resids_scaled = residuals / scale

    model = arch_model(resids_scaled,
                       vol='GARCH', 
                       p=p, 
                       q=q, 
                       dist=dist,
                       rescale=False)     # prevents arch from silently re-scaling again
    
    res = model.fit(disp='off')

    return res, scale

# Example notebook usage:
#
# residuals = y_train - fitted.fittedvalues
#
# garch_res = fit_garch(
#     residuals,
#     p=1,
#     q=1,
#     dist="normal",
#     verbose=True
# )
#
# garch_res.summary()

# =========================================================
# 7. Rolling-origins generator
# =========================================================

def rolling_origins(y, exog, start_train_size, H_MAX, step):
    '''
    Generator for expanding-window rolling origins.
    Yields (t, y_train, y_test, exog_train, exog_test)
    Use in rolling_origins_evaluation() and rolling_crps()
    '''

    for t in range(start_train_size, len(y) - H_MAX + 1, step):

        y_train = y.iloc[:t]
        y_test  = y.iloc[t:t + H_MAX]

        if exog is not None:
            exog_train = exog.iloc[:t]
            exog_test  = exog.iloc[t:t + H_MAX]
        else:
            exog_train = None
            exog_test  = None

        yield t, y_train, y_test, exog_train, exog_test
        
# =========================================================
# 8. Rolling-origin (walk-forward) evaluation
# =========================================================

def rolling_origin_evaluation(
    y,
    model_type,
    model_params,
    exog=None,
    start_train_size=156,
    horizons=(1, 13, 26, 52),
    step=13,
    sp=52,
    checkpoint_path=None,
    random_state=None,
    verbose=False
):
    '''
    Rolling-origin evaluation using expanding window (point forecasts).
    Returns one record per (origin, horizon).
    
    y_train: np.ndarray
    '''
    start_time = time.time()
    
    records = []
    H_MAX = max(horizons)
    
    completed_origins = set()  
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
            records = ckpt['records']
            completed_origins = ckpt['completed_origins']
            
        if verbose:
            print(f'Loaded checkpoint with {len(completed_origins)} completed folds')
    
    origins = range(start_train_size, len(y) - H_MAX + 1, step)    # expanding window 
    n_folds_total = len(origins) 
    
    # initialize the rolling_origins() generator 
    gen = rolling_origins(y, exog, start_train_size, H_MAX, step)
    
    pbar = tqdm(gen, total=n_folds_total, desc='Rolling-origin folds', unit='fold')
    
    completed_count = len(completed_origins)

    for t, y_train, y_test, exog_train, exog_test in pbar:
        
        if t in completed_origins:
            continue

        try:
            fitted = fit_mean_model(
                y_train,
                model_type,
                model_params,
                exog=exog_train,
                verbose=verbose
            )
            
            fc = forecast_mean_model(
                fitted,
                horizon=H_MAX,
                exog=exog_test
            )
            
            mean = fc['mean']
            
            for h in horizons:
                y_true = y_test.iloc[h - 1]
                y_pred = mean.iloc[h - 1]

                # seasonal naive
                naive_fcst = seasonal_naive_forecast(
                    y_train=y_train.values,
                    horizon=h,
                    sp=sp
                )
                y_naive = naive_fcst[h - 1]

                err = y_pred - y_true
                naive_err = y_naive - y_true

                # debug – first successful error only
                if verbose and len(records) == 0:
                    print('DEBUG err value:', err)
                    print('DEBUG err type:', type(err))
                    print('DEBUG err ndim:', np.ndim(err))

                records.append({
                    'origin'     : y.index[t],
                    'origin_idx' : t,
                    'horizon'    : h,
                    'model'      : model_type,
                    'y_true'     : y_true,
                    'y_pred'     : y_pred,
                    'y_naive'    : y_naive,
                    'error'      : err,
                    'naive_error': naive_err,
                    'abs_error'  : abs(err),
                    'sq_error'   : err**2,
                    'train_size' : len(y_train)
                })
                
            completed_origins.add(t)
            completed_count += 1
            
            if checkpoint_path:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(
                        {
                            'completed_origins': completed_origins,
                            'records'          : records
                        },
                        f
                    )
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_per_fold = elapsed_time / completed_count
            eta = avg_per_fold * (n_folds_total - completed_count)
                
            pbar.set_postfix(
                elapsed_min=f'{elapsed_time/60:.2f}',
                eta_min=f'{eta/60:.2f}'
            )

        except Exception as e:
            tqdm.write(f'Fold failed at t={t}: {e}')
            if verbose:
                import traceback
                traceback.print_exc()                         # prints full traceback to stderr
                # logger.exception(f"Fold failed at t={t}")   # alternative approach to debugging
            continue

    return pd.DataFrame(records)

# Example notebook usage:
#
# ro_results = rolling_origin_evaluation(
#     y=y,
#     model_type="sarima",
#     model_params={
#         "order": (1, 1, 1),
#         "seasonal_order": (2, 0, 0, 52),
#         "trend": "n"
#     },
#     start_train_size=156,
#     horizons=(1, 13, 26, 52),
#     step=13
# )
#
# df_ro = pd.DataFrame(ro_results)
#
# summary = (
#     df_ro.groupby(["model", "horizon"])
#          .agg(
#              rmse=("sq_error", lambda x: np.sqrt(x.mean())),
#              mae=("abs_error", lambda x: np.mean(np.abs(x))),
#              n_folds=("error", "count")
#          )
#          .reset_index()
# )
#
# summary

# =========================================================
# 9. Rolling CRPS
# =========================================================
# mean model --> mu, sigma_native
# innovation model --> sigma_innov
# post-model calibration --> variance_inflation_alpha

def rolling_crps(
    y,
    model_type,
    model_params,
    variance_type='static',  # 'static' or 'garch'
    variance_params=None,
    exog=None,
    start_train_size=156,
    horizons=(1,13,26,52),
    step=13,
    variance_inflation_alpha=None,
    max_inflation=5.0,
    random_state=None,
    verbose=False
):

    records = []
    H_MAX = max(horizons)

    for t, y_train, y_test, exog_train, exog_test in rolling_origins(
        y, exog, start_train_size, H_MAX, step):

        # fit the mean model once per origin
        fitted = fit_mean_model(
            y_train,
            model_type,
            model_params,
            exog=exog_train
        )
        
        fc = forecast_mean_model(
            fitted,
            horizon=H_MAX,
            exog=exog_test
        )

        mu = fc['mean']
        sigma_native = fc['sigma']
        
        resid = fitted['residuals']
        resid_var = resid.var(ddof=1)
        resid_var = max(resid_var, 1e-10)

        # variance model (fit once per origin)
        if variance_type == 'static':
            # constant innovation variance
            sigma_innov = {h: np.sqrt(resid_var) for h in horizons}

        elif variance_type == 'garch':
            garch_res, scale = fit_garch(resid, **(variance_params or {}))
            var_fcst = garch_res.forecast(horizon=H_MAX).variance.iloc[-1]
            sigma_innov = {h: np.sqrt(var_fcst.iloc[h - 1]) * scale for h in horizons}
            
        else:
            raise ValueError('Unknown variance_type')
        
        # ---- score all horizons ----
        for h in horizons:
            y_true = y_test.iloc[h - 1]
            mu_h = mu.iloc[h - 1]
            sigma_native_h = sigma_native.iloc[h - 1]
            
            # Fallback for NaN sigma (e.g., ETS/TBATS interval failure)
            if pd.isna(sigma_native_h):
                # Use the in-sample residual standard deviation as a constant estimate
                if resid_var > 0:
                    sigma_native_h = np.sqrt(resid_var)
                    if verbose:
                        tqdm.write(f"Warning: NaN sigma_native at origin {t}, horizon {h}. Using residual SD.")
                else:
                    # If resid_var is zero or missing, skip this record
                    if verbose:
                        tqdm.write(f"Warning: NaN sigma_native and zero residual variance at origin {t}, horizon {h}. Skipping.")
                    continue
            
            # total predictive uncertainty (sigma)
            ratio = np.sqrt(sigma_innov[h]**2 / resid_var) if resid_var > 0 else 1.0
            
            inflation = 1.0 if variance_inflation_alpha is None \
                else np.sqrt(1.0 + variance_inflation_alpha * (h - 1))
            
            inflation = min(inflation, max_inflation)
            # Cap variance inflation to avoid extreme interval widening
            # when residual variance is very small or GARCH variance spikes.
            # This prevents numerical instability and unrealistic predictive intervals.
            # A cap of 5.0 allows substantial volatility adjustment while
            # maintaining practical interpretability.
            
            sigma_h = sigma_native_h * ratio * inflation
            
            records.append({
                'origin'        : y.index[t],
                'origin_idx'    : t,
                'horizon'       : h,
                'mu'            : mu_h,
                'sigma'         : sigma_h,
                'sigma_native'  : sigma_native_h,
                'y_true'        : y_true,
                'ratio'         : ratio,
                'inflation'     : inflation,
                'alpha'         : variance_inflation_alpha,
                'crps'          : crps_gaussian(y_true, mu_h, sigma_h),
                'pit'           : pit_gaussian(y_true, mu_h, sigma_h),
                'mean_model'    : model_type,
                'variance_model': variance_type
            })

    return pd.DataFrame(records)

# Example notebook usage
#
# dfs = []
# for h in [1, 13, 26, 52]:
#     dfs.append(
#         rolling_crps(
#             y=y,
#             model_type="tbats",
#             model_params=tbats_params,
#             variance_type="garch",
#             variance_params={"p": 1, "q": 1},
#             horizons=h
#         )
#     )

# df_all = pd.concat(dfs)

# =========================================================
# 11. GARCH-adjusted interval coverage
# =========================================================

def garch_adjusted_coverage(
    y_true,
    garch_model,
    mean_forecast,
    scale=1.0,
    alpha=0.05
):
    """
    Compute forecast intervals using GARCH conditional volatility forecasts.

    Parameters
    ----------
    y_true : array-like
        Actual observed values for the forecast period.
    garch_model : ARCHModelResult
        Fitted GARCH model result (from fit_garch).
    mean_forecast : array-like
        Mean forecasts from the mean model (length = horizon).
    scale : float, default=1.0
        Scaling factor returned by fit_garch.
    alpha : float, default=0.05
        Significance level (1 - nominal coverage).

    Returns
    -------
    dict with keys:
        lower : array
        upper : array
        coverage : float (empirical coverage rate)
    """
    horizon = len(mean_forecast)
    
    # Generate variance forecasts for the entire horizon
    var_forecast = garch_model.forecast(horizon=horizon).variance.iloc[-1].values
    
    # Conditional volatility forecasts (in original units)
    cond_vol = np.sqrt(var_forecast) * scale
    
    z = stats.norm.ppf(1 - alpha / 2)
    
    lower = mean_forecast - z * cond_vol
    upper = mean_forecast + z * cond_vol
    
    y_true = np.asarray(y_true)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    return {
        "lower": lower,
        "upper": upper,
        "coverage": coverage
    }

# Example notebook usage:
#
# garch_adj = garch_adjusted_coverage(
#     y_true=y_test,
#     garch_model=garch_res,
#     mean_forecast=mean,
#     scale=scale,
#     alpha=0.05
# )
# print(garch_adj["coverage"])

# =========================================================
# 12. Diebold–Mariano test
# =========================================================

def diebold_mariano(e1, e2, h=1, loss='mse'):
    """
    Diebold-Mariano test with Newey-West (HAC) variance correction.

    Parameters
    ----------
    e1, e2 : array-like
        Forecast errors from model A and model B (aligned by origin).
    h : int
        Forecast horizon.
    loss : {"mae", "mse"}
        Loss function used in comparison.

    Returns
    -------
    dm_stat : float
        DM test statistic.
    p_value : float
        Two-sided p-value.
    """
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)
    
    if loss == 'mse':
        d = e1**2 - e2**2
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'mse' or 'mae'")

    T = len(d)
    mean_d = np.mean(d)

    # HAC variance (Newey–West)
    max_lag = h - 1
    gamma0 = np.var(d, ddof=1)

    var_d = gamma0
    
    if var_d <= 1e-12:
        return np.nan, np.nan
    
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        var_d += 2 * weight * cov

    dm_stat = mean_d / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value

# Example notebook usage:
#
# errors_A = y_test - mean_model_A
# errors_B = y_test - mean_model_B
#
# dm_stat, p_val = diebold_mariano(
#     errors_A,
#     errors_B,
#     loss="mae"
# )

# =========================================================
# 13. Interval coverage
# =========================================================

def interval_coverage(df, level=0.95):
    '''
    Computes empirical coverage for Gaussian prediction intervals
    
    parameters
    ----------
    df: DataFrame
        output of rolling_crps()
    level: float
        nominal coverage level (e.g. 0.80 or 0.95)
        
    returns
    -------
    coverate: float
        empirical coverage rate
    '''
    z = norm.ppf(0.5 + level / 2)
    
    lower = df['mu'] - z * df['sigma']
    upper = df['mu'] + z * df['sigma'] 
    
    covered = (df['y_true'] >= lower) & (df['y_true'] <= upper)
    return covered.mean() 