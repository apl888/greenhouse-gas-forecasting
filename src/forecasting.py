import numpy as np
from scipy import stats
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.model_evaluation import residual_diagnostics 


# ---------------------------------------------------------
# single-origin forecasting
# ---------------------------------------------------------

def single_origin_forecast(
    fitted_result,
    test_series,
    exog_test,
    scale_factors,
    final_alpha,
    target_coverage=0.95,
):
    """
    Generate single-origin forecast over full test period with
    variance scaling and ACI initialization from training.
    
    Parameters
    ----------
    fitted_result  : fitted statsmodels or sktime model result
    test_series    : pd.Series — test observations with DatetimeIndex
    exog_test      : pd.DataFrame or None — exogenous variables for test period
    scale_factors  : pd.Series indexed by horizon — variance scaling factors
    final_alpha    : pd.Series indexed by horizon — ACI alpha at end of training
    target_coverage: float — nominal coverage level (default 0.95)

    Returns
    -------
    pd.DataFrame with one row per test week
    """
    n_test     = len(test_series)
    
    # --- generate point forecasts and standard errors ----------
    
    # statsmodels-style models
    if hasattr(fitted_result, "get_forecast"):
        fc = fitted_result.get_forecast(
            steps=n_test,
            exog=exog_test
        )
        mu = np.asarray(fc.predicted_mean).flatten()
        sigma = np.asarray(fc.se_mean).flatten()
    
    # sktime-style models, including AutoETS
    elif hasattr(fitted_result, "predict"):
        fh = np.arange(1, n_test + 1)
        
        # Only pass X if exog_test is not None
        predict_kwargs = {'fh': fh}
        if exog_test is not None:
            predict_kwargs['X'] = exog_test
    
        y_pred = fitted_result.predict(**predict_kwargs)
        mu = np.asarray(y_pred).flatten()
    
        var_kwargs = {'fh': fh}
        if exog_test is not None:
            var_kwargs['X'] = exog_test
    
        pred_var = fitted_result.predict_var(**var_kwargs)
        
        # predict_var is available in sktime's AutoETS but its output format can vary.
        # It sometimes returns a DataFrame with a MultiIndex column rather than a simple array. 
        # Add a defensive flatten:
        if hasattr(pred_var, 'values'):
            var = pred_var.values.flatten()
        else:
            var = np.asarray(pred_var).flatten()
    
        # numerical safety
        var = np.maximum(var, 0)
        sigma = np.sqrt(var)
    
    else:
        raise TypeError(
            f"Unsupported forecasting object: "
            f"{type(fitted_result).__name__}"
        )

    # --- pre-compute horizon caps once outside the loop ----------
    sf_horizons    = sorted(scale_factors.index.tolist())
    alpha_horizons = sorted(final_alpha.index.tolist())
    
    # --- build per-horizon results ----------
    rows = []
    for i in range(n_test):
        h = i + 1
        
        # Scale factor: use largest evaluation horizon <= h
        h_sf      = max([x for x in sf_horizons if x <= h], default=sf_horizons[0])
        sf        = scale_factors.loc[h_sf]
        sigma_cal = sigma[i] * sf

        # ACI alpha: use largest evaluation horizon <= h
        h_alpha = max([x for x in alpha_horizons if x <= h], default=alpha_horizons[0])
        alpha_t = final_alpha.loc[h_alpha]
        z_t = stats.norm.ppf(1 - alpha_t / 2)
        
        lower = mu[i] - z_t * sigma_cal
        upper = mu[i] + z_t * sigma_cal
        y_true = test_series.iloc[i]
        
        covered = float(lower <= y_true <= upper)

        # analytical normal CRPS
        z_score = (y_true - mu[i]) / sigma_cal
        phi = stats.norm.pdf(z_score)
        Phi = stats.norm.cdf(z_score)
        crps = sigma_cal * (
            z_score*(2*Phi-1) 
            + 2*phi 
            - 1/np.sqrt(np.pi)
            )

        rows.append({
            'horizon'          : h,
            'date'             : test_series.index[i],
            'y_true'           : y_true,
            'y_pred'           : mu[i],
            'sigma'            : sigma[i],
            'sigma_calibrated' : sigma_cal,
            'lower_aci'        : lower,
            'upper_aci'        : upper,
            'covered_aci'      : covered,
            'alpha_t'          : alpha_t,
            'crps'             : crps,
            'error'            : mu[i] - y_true,
            'abs_error'        : abs(mu[i] - y_true),
        })

    return pd.DataFrame(rows)

# example usage

# Exog for test set per model
# exog_test_map = {
#     'UCMX'  : exog_bp_test,
#     'UCM'   : None,
#     'SARIMAX': exog_bp_test,
#     'SARIMA': None,
#     'ETS'   : None,
# }

# all_test_forecasts = {}
# for name in fitted_models:
#     cal    = calibration[name]        # from run_final_evaluation()
#     result = fitted_models[name]
#     exog_t = exog_test_map[name]

#     all_test_forecasts[name] = single_origin_forecast(
#         fitted_result=result,
#         test_series=test_preprocessed,
#         exog_test=exog_t,
#         scale_factors=cal['scale_factors'],
#         final_alpha=cal['final_alpha'],
#         target_coverage=0.95,
#     )
#     print(f"{name}: forecast complete")

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

def operational_forecast(name, n_forecast=52, target_coverage=0.95):
    result = model_fits_full[name]
    cal = calibration_full[name]
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
            'model': name, 
            'horizon': h, 
            'date': last_date + pd.Timedelta(weeks=h),
            'y_pred': mu.iloc[i], 
            'sigma_calibrated': sigma_cal,
            'lower': mu.iloc[i] - z_t*sigma_cal,
            'upper': mu.iloc[i] + z_t*sigma_cal
        })
    return pd.DataFrame(rows)