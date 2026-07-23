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
    """
    n_test     = len(test_series)
    # fc         = fitted_result.get_forecast(steps=n_test, exog=exog_test)
    # mu         = fc.predicted_mean.values
    # sigma      = fc.se_mean.values
    alpha_tgt  = 1 - target_coverage
    
    # generate point forecasts and standard errors
    
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
        
        # point forecasts
        y_pred = fitted_result.predict(
            fh=fh,
            X=exog_test
        )
    
        mu = np.asarray(y_pred).flatten()
        
        # forecast variance
        pred_var = fitted_result.predict_var(
            fh=fh,
            X=exog_test
        )
        
        # AutoETS returns a DataFrame with one column
        var = np.asarray(pred_var).flatten()
    
        # numerical safety
        var = np.maximum(var, 0)
        sigma = np.sqrt(var)
    
    else:
        raise TypeError(
            f"Unsupported forecasting object: "
            f"{type(fitted_result).__name__}"
        )

    # apply variance scaling 
    
    rows = []
    
    for i in range(n_test):
        h         = i + 1
        
        sf        = scale_factors.get(
            min(h, max(scale_factors.index)),
            scale_factors.iloc[-1]
            )
        
        sigma_cal = sigma[i] * sf
        
        alpha_t   = final_alpha.get(
            min(h, max(final_alpha.index)),
            final_alpha.iloc[-1]
            )
        
        z_t       = stats.norm.ppf(1 - alpha_t / 2)
        
        lower     = mu[i] - z_t * sigma_cal
        upper     = mu[i] + z_t * sigma_cal
        
        y_true    = test_series.iloc[i]
        
        covered   = float(lower <= y_true <= upper)

        z_score = (y_true - mu[i]) / sigma_cal
        
        phi     = stats.norm.pdf(z_score)
        Phi     = stats.norm.cdf(z_score)
        
        crps    = sigma_cal * (
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