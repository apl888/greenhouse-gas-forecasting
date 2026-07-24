import numpy as np
from scipy import stats
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt


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
    max_h_sf = scale_factors.index.max()
    max_h_alpha = final_alpha.index.max()  
    
    # --- build per-horizon results ----------
    rows = []
    for i in range(n_test):
        h = i + 1
        
        # variance scaling - cap to available horizons
        h_sf_capped = min(h, max_h_sf)
        sf = scale_factors.loc[h_sf_capped]
        sigma_cal = sigma[i] * sf
        
        # ACI alpha - cap to available horizons
        h_alpha_capped = min(h, max_h_alpha)
        alpha_t = final_alpha.loc[h_alpha_capped]
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