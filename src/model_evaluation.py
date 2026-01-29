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
import warnings
import logging
import time
import os
import pickle
from tqdm.auto import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.api import qqplot
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller

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
# Horizon-specific metrics
# =========================================================

def horizon_squared_error(
    y_true,
    y_pred,
    horizons=(1, 13, 26, 52)
):
    """
    Squared error at specific horizons.
    RMSE must be computed after aggregation across origins or folds.
    """
    results = {}
    for h in horizons:
        idx = h - 1
        if idx < len(y_true):
            err = y_true.iloc[idx] - y_pred.iloc[idx]
            results[h] = err ** 2
    return results

# Example notebook usage:
#
# se_by_horizon = horizon_squared_error(
#     y_test,
#     y_pred,
#     horizons=(1, 13, 26, 52)
# )
#
# # NOTE:
# # These squared errors should be aggregated across folds/origins
# # before taking sqrt to compute RMSE_h

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
        burn_in = max(
            model_params['order'][1],
            model_params['order'][0] + model_params['order'][2]
        )
        
        burn_in = max(burn_in, 5)
        
        fitted = results.fittedvalues.iloc[burn_in:]
        resid = results.resid.iloc[burn_in:]
        
    elif model_type == 'ets':
        model = AutoETS(
            error=model_params['error'],
            trend=model_params['trend'],
            seasonal=model_params['seasonal'],
            sp=model_params.get('sp', None),
            initialization_method=model_params.get('initialization_method', 'heuristic')
        )
        results = model.fit(y)

        fitted = results.fittedvalues
        resid = results.resid

    elif model_type == 'tbats':
        model = TBATS(**model_params)
        results = model.fit(y)
        
        # udr sktime's dedicated method for in-sample one-step-ahead innovations
        resid = results.predict_residuals()
        
        # derive fitted values from residuals
        fitted = y - resid
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return {
        "model_type": model_type,
        "fitted_model": results,
        "fitted_values": fitted,
        "residuals": resid
    }

# Example notebook usage:
#
# sarima_params = {
#     "order": (1,1,1),
#     "seasonal_order": (1,1,1,52),
#     "trend": "n"
# }
#
# fitted = fit_mean_model(
#     y=y_train,
#     model_type="sarima",
#     model_params=sarima_params,
#     exog=exog_train,
#     verbose=True
# )

# =========================================================
# 2. Forecasting wrapper
# =========================================================

def forecast_mean_model(fitted_model, 
                        model_type, 
                        horizon, 
                        exog=None):
    """
    Generates forecasts + uncertainty.
    """

    if model_type == 'sarima':
        fc = fitted_model.get_forecast(steps=horizon, exog=exog)
        mean = fc.predicted_mean
        var = fc.var_pred_mean
        sigma = np.sqrt(var)
        return mean, sigma, fc.conf_int()

    elif model_type in ['ets', 'tbats']:
        fh = np.arange(1, horizon + 1)
        mean = fitted_model.predict(fh)

        try:
            intervals = fitted_model.predict_interval(fh, coverage=0.95)
            sigma = (intervals.iloc[:,1] - intervals.iloc[:,0]) / (2 * 1.96)
        except Exception:
            sigma = None
            intervals = None

        return mean, sigma, intervals

# Example notebook usage:
#
# mean, sigma, intervals = forecast_mean_model(
#     fitted_model=fitted,
#     model_type="sarima",
#     horizon=len(y_test),
#     exog=exog_test
# )
#
# plt.plot(y_test.index, y_test, label="Observed")
# plt.plot(mean.index, mean, label="Forecast")
# plt.fill_between(
#     mean.index,
#     intervals.iloc[:,0],
#     intervals.iloc[:,1],
#     alpha=0.3
# )
# plt.legend()

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
    arch_stat, arch_p, _, _, = het_arch(residuals, nlags=52)
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
    """

    if verbose:
        logger.info(f"Fitting GARCH({p},{q})")

    model = arch_model(residuals * 100, 
                       vol='GARCH', 
                       p=p, 
                       q=q, 
                       dist=dist)
    res = model.fit(disp='off')

    return res

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
# 5. TimeSeriesSplit cross-validation (metrics only)
# =========================================================

def evaluate_models_tscv(
    y,
    model_type,
    model_params,
    exog=None,
    sp=None,
    n_splits=5,
    test_size=52,
    gap=13,
    verbose=False
):
    """
    TimeSeriesSplit cross-validation using standardized metrics.
    Use for hyperparameter screening and tuning.
    """
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        gap=gap
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(y)):

        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        exog_train = exog.iloc[train_idx] if exog is not None else None
        exog_test  = exog.iloc[test_idx] if exog is not None else None

        try:
            fitted = fit_mean_model(
                y_train,
                model_type,
                model_params,
                exog=exog_train,
                verbose=verbose
            )

            mean, sigma, intervals = forecast_mean_model(
                fitted,
                model_type,
                horizon=len(y_test),
                exog=exog_test
            )

            metrics = evaluate_forecast(
                y_train=y_train,
                y_test=y_test,
                y_pred=mean,
                sp=sp,
                intervals=intervals,
                sigma=sigma
            )

            metrics["fold"] = fold + 1
            metrics["model"] = model_type

            rows.append(metrics)

        except Exception as e:
            if verbose:
                logger.warning(f"Fold {fold+1} failed: {e}")

    return pd.DataFrame(rows)

# Example notebook usage:
#
# cv_results = evaluate_models_tscv(
#     y=y,
#     model_type="ets",
#     model_params={"sp": 52, "trend": "add"},
#     sp=52,
#     n_splits=5,
#     test_size=52,
#     gap=13
# )
#
# cv_results.groupby("model").mean()

# =========================================================
# 6. Unified model evaluation (standard output)
# =========================================================

def evaluate_forecast(
    y_train,
    y_test,
    y_pred,
    *,
    sp=None,
    intervals=None,
    sigma=None,
    horizons=(1, 13, 26, 52)
):
    """
    Standardized evaluation across all models.
    """

    results = {}

    # --- Core accuracy ---
    results["RMSE"] = rmse(y_test, y_pred)
    results["MAE"]  = mae(y_test, y_pred)
    results["NSE"]  = nash_sutcliffe_efficiency(y_test, y_pred, y_train)

    # --- Horizon-specific ---
    results["Horizon_SE"] = horizon_squared_error(
        y_test,
        y_pred,
        horizons=horizons
    )
    
    # --- CRPS score ---
    if sigma is not None:
        results["CRPS"] = crps_gaussian(y_test, y_pred, sigma)
    else:
        results["CRPS"] = np.nan

    # --- Seasonal naive benchmark ---
    if sp is not None:
        y_naive = seasonal_naive_forecast(
            y_train=y_train,
            horizon=len(y_test),
            sp=sp
        )
        results["Naive_RMSE"] = rmse(y_test, y_naive)
        results["Skill_vs_Naive"] = skill_vs_naive(
            y_test,
            y_pred,
            y_naive
        )

    # --- Interval coverage ---
    if intervals is not None:
        y = y_test.values
        lower = intervals.iloc[:, 0].values
        upper = intervals.iloc[:, 1].values

        results["Interval_Coverage"] = np.mean(
            (y >= lower) & (y <= upper)
        )

    return results

# Example notebook usage:
#
# metrics = evaluate_forecast(
#     y_train=y_train,
#     y_test=y_test,
#     y_pred=mean,
#     sp=52,
#     intervals=intervals,
#     sigma=sigma
# )
#
# metrics

# =========================================================
# 7. Rolling-origin (walk-forward) evaluation
# =========================================================

def rolling_origin_evaluation(
    y,
    model_type,
    model_params,
    exog=None,
    start_train_size=156,
    H_MAX=52,
    eval_horizons=(1, 13, 26, 52),
    step=13,
    sp=52,
    checkpoint_path=None,
    verbose=False
):
    """
    Rolling-origin evaluation using expanding window.
    Returns one record per (origin, horizon).
    """
    start_time = time.time()
    
    results = []
    
    completed_origins = set()  
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
            results = ckpt['results']
            completed_origins = ckpt['completed_origins']
            
        if verbose:
            print(f"Loaded checkpoint with {len(completed_origins)} completed folds")
    
    origins = list(range(start_train_size, len(y) - H_MAX + 1, step)) # expanding window 
    n_folds_total = len(origins) 
    
    pbar = tqdm(origins, desc="Rolling-origin folds", unit="fold")
    
    completed_count = len(completed_origins)

    for t in pbar:
        
        if t in completed_origins:
            pbar.update(0)
            continue

        y_train = y.iloc[:t]
        y_test  = y.iloc[t:t + H_MAX]

        exog_train = exog.iloc[:t] if exog is not None else None
        exog_test  = exog.iloc[t:t + H_MAX] if exog is not None else None

        try:
            fitted = fit_mean_model(
                y_train,
                model_type,
                model_params,
                exog=exog_train,
                verbose=verbose
            )
            
            mean, sigma, intervals = forecast_mean_model(
                fitted,
                model_type,
                horizon=H_MAX,
                exog=exog_test
            )
            
            for h in eval_horizons:
                y_hat = mean.iloc[h - 1]
                y_true = y_test.iloc[h - 1]

                # seasonal naive
                naive_fcst = seasonal_naive_forecast(
                    y_train=y_train.values,
                    horizon=h,
                    sp=52
                )
                y_naive = naive_fcst[h - 1]

                err = y_hat - y_true
                naive_err = y_naive - y_true

                # debug – first successful error only
                if verbose and len(results) == 0:
                    print("DEBUG err value:", err)
                    print("DEBUG err type:", type(err))
                    print("DEBUG err ndim:", np.ndim(err))

                results.append({
                    "origin"     : y.index[t],
                    "model"      : model_type,
                    "horizon"    : h,
                    "y_true"     : y_true,
                    "y_pred"     : y_hat,
                    "y_naive"    : y_naive,
                    "error"      : err,
                    "naive_error": naive_err,
                    "abs_error"  : abs(err),
                    "sq_error"   : err**2,
                    "origin_idx" : t,
                    "train_size" : len(y_train)
                })
                
            completed_origins.add(t)
            completed_count += 1
            
            if checkpoint_path:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(
                        {
                            "completed_origins": completed_origins,
                            "results"          : results
                        },
                        f
                    )
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_per_fold = elapsed_time / completed_count
            eta = avg_per_fold * (n_folds_total - completed_count)
                
            pbar.set_postfix(
                elapsed_min=f"{elapsed_time/60:.2f}",
                eta_min=f"{eta/60:.2f}"
            )

        except Exception as e:
            tqdm.write(f"Fold failed at t={t}: {e}")
            continue

    return results

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
#     H_MAX=52,
#     eval_horizons=(1, 13, 26, 52),
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
# 8. Results aggregation / comparison
# =========================================================

def summarize_evaluations(results_list):
    """
    Converts a list of evaluation dicts into a clean DataFrame.
    Designed for comparing models AFTER optimization.
    """

    rows = []

    for res in results_list:
        row = {
            "model"            : res.get("model"),
            "origin"           : res.get("origin"),
            "RMSE"             : res.get("RMSE"),
            "MAE"              : res.get("MAE"),
            "NSE"              : res.get("NSE"),
            "Skill_vs_Naive"   : res.get("Skill_vs_Naive"),
            "Interval_Coverage": res.get("Interval_Coverage")
        }

        # Flatten horizon RMSE
        for h, se in res.get("Horizon_SE", {}).items():
            row[f"RMSE_h{h}"] = np.sqrt(se)

        rows.append(row)

    df = pd.DataFrame(rows)

    return df

# Example notebook usage:
#
# all_results = []
# all_results.extend(ro_results_sarima)
# all_results.extend(ro_results_ets)
# all_results.extend(ro_results_tbats)
#
# comparison_df = summarize_evaluations(all_results)
# comparison_df.sort_values("RMSE")

# =========================================================
# 9. GARCH-adjusted interval coverage
# =========================================================

def garch_adjusted_coverage(
    y_true,
    garch_model,
    mean_forecast,
    alpha=0.05
):
    """
    Adjust forecast intervals using GARCH conditional volatility.
    """

    cond_vol = garch_model.conditional_volatility[-len(mean_forecast):] / 100.0

    z = stats.norm.ppf(1 - alpha / 2)

    lower = mean_forecast - z * cond_vol
    upper = mean_forecast + z * cond_vol

    y = y_true.values
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    coverage = np.mean((y >= lower) & (y <= upper))

    return {
        "lower"   : lower,
        "upper"   : upper,
        "coverage": coverage
    }

# Example notebook usage:
#
# garch_adj = garch_adjusted_coverage(
#     y_true=y_test,
#     garch_model=garch_res,
#     mean_forecast=mean,
#     alpha=0.05
# )
#
# garch_adj["coverage"]

# =========================================================
# 10. Diebold–Mariano test
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