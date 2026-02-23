# The following functions were removed from the model_evaluation.py file at version 9.
# The decision was made to use rolling_origins_evaluation() and rolling_crps(), with the rolling_origins()
# engine.
# These functions no longer fit the rolling_origins pipeline and are being moved here for archival purposes. 

# NOTE: the appropriate libraries to support these functions are not included in this document.  


# =========================================================
# TimeSeriesSplit cross-validation (metrics only)
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

            fc = forecast_mean_model(
                fitted,
                model_type,
                horizon=len(y_test),
                exog=exog_test
            )
            
            mean = fc['mean']
            sigma = fc['sigma']
            intervals = fc['intervals']

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
# Unified model evaluation (standard output)
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
# Results aggregation / comparison
# =========================================================

def summarize_evaluations(results_list):
    """
    Converts a list of evaluation dicts into a clean DataFrame.
    Designed for comparing models AFTER optimization.
    
    SE_h{h} are per-forecast values
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
            row[f'SE_h{h}'] = np.sqrt(se)

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
