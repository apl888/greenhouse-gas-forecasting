# src/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.api import qqplot


def evaluate_sarima_model(train, order, seasonal_order, run_hetero=False, plot_residuals=True):
    """
    Fit a SARIMA model and run in-sample diagnostics.
    
    Parameters
    ----------
    train : pd.Series
        Training set (time-indexed).
    order : tuple
        SARIMA (p,d,q).
    seasonal_order : tuple
        SARIMA seasonal (P,D,Q,s).
    run_hetero : bool, default False
        If True, runs Breusch-Pagan and White tests for heteroscedasticity.
    plot_residuals : bool, default True
        If True, generates residual diagnostic plots.
    
    Returns
    -------
    results : dict
        Contains AIC, BIC, Ljung-Box p-values, and optional heteroscedasticity test results.
    """
    
    # --- Fit model ---
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # --- Residual diagnostics ---
    residuals = results.resid

    if plot_residuals:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residual time series
        axes[0,0].plot(residuals)
        axes[0,0].set_title("Residuals over Time")

        # Histogram + KDE
        residuals.plot(kind="hist", bins=30, ax=axes[0,1], density=True, alpha=0.6)
        residuals.plot(kind="kde", ax=axes[0,1])
        axes[0,1].set_title("Residual Distribution")

        # QQ plot
        qqplot(residuals, line='s', ax=axes[1,0])
        axes[1,0].set_title("QQ Plot")

        # ACF plot
        plot_acf(residuals, lags=52, ax=axes[1,1])
        axes[1,1].set_title("ACF of Residuals")

        plt.tight_layout()
        plt.show()

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[1,4,52], return_df=True)

    results_dict = {
        "order": order,
        "seasonal_order": seasonal_order,
        "AIC": round(results.aic, 3),
        "BIC": round(results.bic, 3),
        "LB_pval_lag1": round(lb_test.loc[1, "lb_pvalue"], 4),
        "LB_pval_lag4": round(lb_test.loc[4, "lb_pvalue"], 4),
        "LB_pval_lag52": round(lb_test.loc[52, "lb_pvalue"], 4)
    }

    # Optional: heteroscedasticity tests
    if run_hetero:
        exog = np.column_stack([np.ones(len(residuals)), np.arange(len(residuals))])  
        bp_test = het_breuschpagan(residuals, exog)
        white_test = het_white(residuals, exog)

        results_dict.update({
            "BP_LM": round(bp_test[0], 3),
            "BP_pval": round(bp_test[1], 4),
            "BP_F": round(bp_test[2], 3),
            "BP_F_pval": round(bp_test[3], 4),
            "White_LM": round(white_test[0], 3),
            "White_pval": round(white_test[1], 4),
            "White_F": round(white_test[2], 3),
            "White_F_pval": round(white_test[3], 4),            
        })

    model_eval_df = pd.DataFrame([results_dict]).t
    return model_eval_df