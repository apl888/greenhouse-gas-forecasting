import numpy as np
from scipy import stats
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

from .model_evaluation import residual_diagnostics 

# ---------------------------------------------------------
# Single-origin forecasting
# ---------------------------------------------------------

def single_origin_forecast(
    fitted_result,
    forecast_index,
    model_name,
    exog=None,
    target_coverage=0.95,
):
    """
    Generate uncalibrated single-origin forecasts and Gaussian prediction intervals over a specified horizon.

    Parameters
    ----------
    fitted_result : fitted statsmodels or sktime model
        Fitted forecasting model.
    forecast_index : pandas Index
        Dates corresponding to the forecast horizon.
    model_name : str
        Name of the fitted model.
    exog : pandas DataFrame or None
        Future exogenous variables, if required.
    target_coverage : float, default=0.95
        Nominal prediction interval coverage.

    Returns
    -------
    pandas DataFrame
        Forecast means, standard deviations, and prediction
        interval limits.
    """

    n_forecast = len(forecast_index)

    if n_forecast == 0:
        raise ValueError("Forecast index cannot be empty.")

    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1.")

    if exog is not None and len(exog) != n_forecast:
        raise ValueError(
            "Exogenous variables must match the forecast horizon."
        )


    # Generate point forecasts and predictive variances

    # Statsmodels-style models
    if hasattr(fitted_result, "get_forecast"):

        forecast_kwargs = {'steps': n_forecast}

        if exog is not None:
            forecast_kwargs['exog'] = exog

        fc = fitted_result.get_forecast(**forecast_kwargs)

        mu = np.asarray(fc.predicted_mean).reshape(-1)

        # Extract predictive variance
        if hasattr(fc, 'var_pred_mean'):
            variance = np.asarray(fc.var_pred_mean).reshape(-1)
        else:
            sigma_native = np.asarray(fc.se_mean).reshape(-1)
            variance = sigma_native ** 2

    # Sktime-style models
    elif hasattr(fitted_result, "predict"):

        fh = np.arange(1, n_forecast + 1)

        predict_kwargs = {'fh': fh}
        if exog is not None:
            predict_kwargs['X'] = exog

        y_pred = fitted_result.predict(**predict_kwargs)
        mu = np.asarray(y_pred).reshape(-1)

        var_kwargs = {'fh': fh}
        if exog is not None:
            var_kwargs['X'] = exog

        pred_var = fitted_result.predict_var(**var_kwargs)
        variance = np.asarray(pred_var).reshape(-1)

    else:
        raise TypeError(
            f"Unsupported forecasting object: "
            f"{type(fitted_result).__name__}"
        )

    # Validate forecast output

    if len(mu) != n_forecast or len(variance) != n_forecast:
        raise ValueError(
            "Forecast output length does not match forecast horizon."
        )

    if not np.all(np.isfinite(mu)):
        raise ValueError("Non-finite point forecasts detected.")

    if not np.all(np.isfinite(variance)):
        raise ValueError("Non-finite predictive variances detected.")

    # Protect against small negative numerical values
    variance = np.maximum(variance, 0)

    sigma = np.sqrt(variance)

    # Calculate Gaussian prediction intervals

    alpha = 1 - target_coverage
    z_value = norm.ppf(1 - alpha / 2)

    lower = mu - z_value * sigma
    upper = mu + z_value * sigma

    # Construct forecast dataframe

    forecast_df = pd.DataFrame({
        'date'    : forecast_index,
        'horizon' : np.arange(1, n_forecast + 1),
        'mu'      : mu,
        'sigma'   : sigma,
        'lower_95': lower,
        'upper_95': upper,
        'model'   : model_name,
    })

    return forecast_df


# ---------------------------------------------------------
# Evaluate forecasts
# ---------------------------------------------------------

def evaluate_forecasts(forecast_df, test_series):
    """
    Evaluate a single-origin forecast against the observed test series.

    Returns
    -------
    evaluated_df : pd.DataFrame
        Individual forecasts with observed values and evaluation metrics.
    summary_df : pd.DataFrame
        Overall metrics and metrics grouped by forecast horizon.
    """

    required_cols = {
        'date', 'horizon', 'mu', 'sigma',
        'lower_95', 'upper_95', 'model'
    }

    missing_cols = required_cols.difference(forecast_df.columns)
    if missing_cols:
        raise ValueError(f"Missing forecast columns: {missing_cols}")

    if forecast_df['date'].duplicated().any():
        raise ValueError("Forecast dates must be unique.")

    if test_series.index.duplicated().any():
        raise ValueError("Test-series dates must be unique.")

    if len(forecast_df) != len(test_series):
        raise ValueError(
            "Forecast and test series must contain the same number of dates."
        )

    # Add observed values to the forecast DataFrame.
    observed = (
        test_series.rename('y_true')
        .rename_axis('date')
        .reset_index()
    )

    evaluated_df = forecast_df.merge(
        observed,
        on='date',
        how='left',
        validate='one_to_one',
        indicator=True
    )

    if (evaluated_df['_merge'] != 'both').any():
        raise ValueError("Some forecast dates do not match test-series dates.")

    evaluated_df = evaluated_df.drop(columns='_merge')

    # Check forecast values before calculating metrics.
    numeric_cols = ['mu', 'sigma', 'lower_95', 'upper_95', 'y_true']

    if not np.isfinite(evaluated_df[numeric_cols].to_numpy()).all():
        raise ValueError("Forecast or observed values contain non-finite values.")

    if (evaluated_df['sigma'] <= 0).any():
        raise ValueError("All predictive standard deviations must be positive.")

    if (evaluated_df['lower_95'] >= evaluated_df['upper_95']).any():
        raise ValueError("Lower interval limits must be below upper limits.")

    if not evaluated_df['horizon'].between(1, 105).all():
        raise ValueError("Forecast horizons must be between 1 and 105.")

    # Point forecast errors: positive values indicate underprediction.
    evaluated_df['error'] = evaluated_df['y_true'] - evaluated_df['mu']
    evaluated_df['absolute_error'] = evaluated_df['error'].abs()
    evaluated_df['squared_error'] = evaluated_df['error'] ** 2

    # Analytical CRPS for Gaussian predictive distributions.
    z = evaluated_df['error'] / evaluated_df['sigma']

    evaluated_df['crps'] = evaluated_df['sigma'] * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )

    # 95% prediction interval metrics.
    alpha = 0.05

    evaluated_df['covered_95'] = (
        (evaluated_df['y_true'] >= evaluated_df['lower_95'])
        & (evaluated_df['y_true'] <= evaluated_df['upper_95'])
    )

    evaluated_df['interval_width'] = (
        evaluated_df['upper_95'] - evaluated_df['lower_95']
    )

    evaluated_df['interval_score'] = (
        evaluated_df['interval_width']
        + (2 / alpha)
        * (evaluated_df['lower_95'] - evaluated_df['y_true']).clip(lower=0)
        + (2 / alpha)
        * (evaluated_df['y_true'] - evaluated_df['upper_95']).clip(lower=0)
    )

    # Group forecasts by lead time.
    evaluated_df['horizon_group'] = pd.cut(
        evaluated_df['horizon'],
        bins=[0, 13, 26, 52, 105],
        labels=['1–13', '14–26', '27–52', '53–105']
    )

    # Calculate summary metrics for a set of forecasts.
    def summarize(group, group_label):
        return {
            'model': group['model'].iloc[0],
            'horizon_group': group_label,
            'n_forecasts': len(group),
            'mae': group['absolute_error'].mean(),
            'rmse': np.sqrt(group['squared_error'].mean()),
            'bias': group['error'].mean(),
            'mean_crps': group['crps'].mean(),
            'coverage_95': group['covered_95'].mean(),
            'mean_interval_width': group['interval_width'].mean(),
            'mean_interval_score': group['interval_score'].mean()
        }

    summary_rows = []

    # Overall performance.
    summary_rows.append(summarize(evaluated_df, 'Overall'))

    # Performance by forecast horizon.
    for group_label in ['1–13', '14–26', '27–52', '53–105']:
        group = evaluated_df[
            evaluated_df['horizon_group'] == group_label
        ]

        if not group.empty:
            summary_rows.append(summarize(group, group_label))

    summary_df = pd.DataFrame(summary_rows)

    return evaluated_df, summary_df

# ---------------------------------------------------------
# sanity check for refitting model on full series
# ---------------------------------------------------------

def sanity_check_refit(result, full_series, model_name, prior_good_params=None):
    print(f"\n{'='*50}\nSanity check: {model_name} refit on {full_series.index[0].date()} "
          f"to {full_series.index[-1].date()}\n{'='*50}")

    issues = []

    # Convergence flag
    if hasattr(result, 'mle_retvals'):
        converged = result.mle_retvals.get('converged', True)
        if not converged:
            issues.append("Optimizer did not converge")

    # standard errors implausibly large relative to coefficients
    se, coef = result.bse, result.params
    if (se > np.abs(coef) * 2).any():
        bad = coef.index[se > np.abs(coef) * 2].tolist()
        issues.append(f"Standard errors implausibly large for: {bad}, likely non-convergence")

    # large jump from last known good fit, if provided
    if prior_good_params is not None:
        common = coef.index.intersection(prior_good_params.index)
        pct_change = np.abs((coef[common] - prior_good_params[common]) / prior_good_params[common]) * 100
        if (pct_change > 100).any():
            bad = pct_change.index[pct_change > 100].tolist()
            issues.append(f"Large jump (>100%) from prior fit for: {bad}, verify convergence")

    # Residual autocorrelation — should still be white noise
    lb = acorr_ljungbox(result.resid.dropna(), lags=[52], return_df=True)
    if lb['lb_pvalue'].iloc[0] < 0.01:
        issues.append(f"Ljung-Box p={lb['lb_pvalue'].iloc[0]:.4f}, residual structure detected")

    # Boundary/frequency sanity — the exact bug you hit earlier!
    gap = full_series.index.to_series().diff().dropna()
    if not (gap == pd.Timedelta('7 days')).all():
        issues.append("Non-weekly gap detected in series index")

    # Plausible value range
    if full_series.iloc[-1] < full_series.iloc[-53] * 0.9:
        issues.append("Most recent value implausibly lower than 1 year ago")

    if issues:
        print("⚠ ISSUES FOUND:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("✓ No issues detected — refit looks consistent with prior years")

    return len(issues) == 0

# example usage:

# prior_good_params = fitted_models['UCM'].params

# ucm_check_refit = sanity_check_refit(
    # model_fits_full_series['UCM'], 
    # full_series, 
    # 'UCM',
    # prior_good_params=prior_good_params
    # )

# ---------------------------------------------------------
# fit model on full series, diagnose, and sanity check
# ---------------------------------------------------------

def fit_final_model(model, start_params=None, prior_good_params=None):
    """
    Fit a single model (e.g. UC, SARIMA) with a two-stage optimizer polish 
    (lbfgs, then powell) and a basic convergence sanity check. Intended 
    for one-off final fits (e.g. the full-series operational refit).  NOT 
    for use inside rolling_crps / rolling_origin_evaluation loops, where the 
    added cost of a two-stage fit across thousands of folds would be prohibitive
    and unnecessary.  
    """
    res1 = model.fit(start_params=start_params, method='lbfgs', maxiter=2000, disp=False)
    res2 = model.fit(start_params=start_params, method='powell', maxiter=1000, disp=False)

    se, coef = res2.bse, res2.params
    if (se > np.abs(coef) * 2).any():
        print("Warning: standard errors implausibly large. Fit may not have converged.")
    if prior_good_params is not None:
        common = coef.index.intersection(prior_good_params.index)
        pct_change = np.abs((coef[common] - prior_good_params[common]) / prior_good_params[common]) * 100
        if (pct_change > 100).any():
            print(f"Warning: large parameter jump from prior fit: {pct_change[pct_change > 100].to_dict()}")

    return res2        

def fit_and_check_full_series(
    name, 
    model_spec_dict, 
    model_class, 
    prior_good_params,
    full_series
):
    model = model_class(
        full_series, 
        **model_spec_dict
    )
    result = fit_final_model(
        model, 
        start_params=prior_good_params, 
        prior_good_params=prior_good_params
    )

    print(f"\n{name} Results Summary Table")
    print(result.summary().tables[1])
    diag = residual_diagnostics(
        result.resid,
        burnin=52,
        title=f'{name} - full series',
        plot=True,
        return_results=True
    )
    ok = sanity_check_refit(
        result,
        full_series,
        name,
        prior_good_params=prior_good_params
    )

    return result, diag, ok


# ---------------------------------------------------------
# final operational forecast following model fit on full series
# ---------------------------------------------------------

def operational_forecast(
    name, 
    model_fits_dict,
    calibration_dict,
    full_series,
    n_forecast=52, 
    target_coverage=0.95
    ):
    result = model_fits_dict[name]
    cal = calibration_dict[name]
    final_alpha = cal['calibrations'][target_coverage]['final_alpha']

    alpha_h = np.array(sorted(final_alpha.index)); alpha_v = final_alpha.loc[alpha_h].values
    sf_h = np.array(sorted(cal['scale_factors'].index)); sf_v = cal['scale_factors'].loc[sf_h].values
    
    fc = result.get_forecast(steps=n_forecast)
    mu, sigma = fc.predicted_mean, fc.se_mean
    last_date = full_series.index[-1]

    rows = []
    for i in range(n_forecast):
        h = i + 1
        alpha_t = np.interp(h, alpha_h, alpha_v)
        sf = np.interp(h, sf_h, sf_v)
        sigma_cal = sigma.iloc[i] * sf
        z_t = stats.norm.ppf(1 - alpha_t / 2)
        rows.append({
            'model'           : name, 
            'horizon'         : h, 
            'date'            : last_date + pd.Timedelta(weeks=h),
            'y_pred'          : mu.iloc[i], 
            'sigma_calibrated': sigma_cal,
            'lower'           : mu.iloc[i] - z_t*sigma_cal,
            'upper'           : mu.iloc[i] + z_t*sigma_cal
        })
    return pd.DataFrame(rows)