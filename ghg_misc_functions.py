# The following are miscellanious functions in the greenhouse-gas-forecasting project


# build regime regressors
# create exogenous regime regressors for both train and test sets
# The test set regressors must be built using the same break dates and the same global t counter 
# — it must continue from where training left off, not restart at 0.
# The issue is that t_break must be the same integer used during training — you can't recompute it 
# from the test index. So refactor to compute and store t_breaks once, then pass them in:

def make_regime_regressors(index, break_dates, t_breaks, t_offset=0, quadratic_regimes=None):
    """
    Creates both step dummies (level shift) and slope change regressors
    for each breakpoint. Slope regressors count weeks elapsed since the
    break, capturing the change in growth rate across regimes.

    parameters
    ----------
    index             : DateTimeIndex for the split (train or test)
    break_dates       : regime change dates found by rupture_model
    t_breaks          : list of integer week numbers for each break,
                        computed once from the training index and reused for test
    t_offset          : 0 for train; len(train_preprocessed) for test,
                        ensuring a continuous global week counter
    quadratic_regimes : set of regime indices (0-based) to include dt^2 term. 
                        e.g. {2} for only the 2020 break onward.
    """
    if quadratic_regimes is None:
        quadratic_regimes = set()
        
    df = pd.DataFrame(index=index)
    t = np.arange(len(index)) + t_offset

    for i, (bd, t_break) in enumerate(zip(break_dates, t_breaks)):
        mask = (index >= bd).astype(int)
        
        dt = np.where(mask, t - t_break, 0)
        dt_scaled = dt / 52  # KEY FIX
        
        df[f'slope_{i+1}'] = dt_scaled

        if i in quadratic_regimes:
            df[f'curve_{i+1}'] = dt_scaled ** 2
    
    return df


# build Fourier regressors

def make_fourier_terms(index, period=52, K=2):
    '''
    index  : pd.DatetimeIndex (weekly)
    period : int (e.g. 52 weeks)
    K      : number of harmonic pairs (1..K)
    returns: DataFrame of shape (len(index), 2*K) with sin_k and cos_k columns
    '''
    # integer time t (0,1,2,...) aligned with index
    t = np.arange(len(index))
    fourier = {}
    for k in range(1, K+1):
        fourier[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        fourier[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(fourier, index=index)

# compute scale factors

def compute_variance_scale_factors(crps_df):
    """
    Compute empirical scaling factors to correct sigma_h at each horizon.
    Target: standardized variance = 1.0
    scale_factor[h] = sqrt(mean((y_true - mu)**2 / sigma**2))
    Apply: sigma_calibrated = sigma_h * scale_factor[h]
    """
    scale_factors = (crps_df
        .assign(z2=lambda d: ((d.y_true - d.mu) / d.sigma) ** 2)
        .groupby('horizon')['z2']
        .mean()
        .apply(np.sqrt)   # sqrt of mean squared z → multiplicative scale
    )
    return scale_factors

# apply scale factors as a post-processing (post-hoc) step

def apply_variance_scaling(crps_df, scale_factors):
    """Apply empirical scale factors to sigma, recompute CRPS and PIT."""
    df = crps_df.copy()

    # initialize columns
    df['sigma_calibrated'] = np.nan
    df['crps_calibrated'] = np.nan
    df['pit_calibrated'] = np.nan    
    
    for h, sf in scale_factors.items():
        mask = df['horizon'] == h
        
        mu        = df.loc[mask, 'mu'].values
        sigma_cal = df.loc[mask, 'sigma'].values * sf
        y         = df.loc[mask, 'y_true'].values
        
        z   = (y - mu) / sigma_cal
        phi = stats.norm.pdf(z)
        Phi = stats.norm.cdf(z)

        df.loc[mask, 'sigma_calibrated'] = sigma_cal
        df.loc[mask, 'crps_calibrated'] = (
            sigma_cal * (z * (2*Phi - 1) + 2*phi - 1/np.sqrt(np.pi))
        )
        df.loc[mask, 'pit_calibrated'] = stats.norm.cdf(y, loc=mu, scale=sigma_cal)
    
    return df

# plot PIT histogram with consistency bands (maybe this belongs in the model_evaluation.py file or leave in notebook?)

def plot_pit_with_bands(pit_values, n_bins=10, ax=None, title=''):
    """
    Plot PIT histogram with 95% consistency bands under the null
    hypothesis of a uniform distribution.
    """
    n = len(pit_values)
    expected = n / n_bins
    # 95% confidence band for binomial counts
    lower = stats.binom.ppf(0.025, n, 1/n_bins)
    upper = stats.binom.ppf(0.975, n, 1/n_bins)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    
    counts, edges, _ = ax.hist(pit_values, bins=n_bins, 
                                range=(0,1), color='steelblue',
                                edgecolor='white')
    ax.axhline(expected, color='black', linewidth=1.5, 
               linestyle='--', label='Expected (uniform)')
    ax.axhspan(lower, upper, alpha=0.15, color='green', 
               label='95% consistency band')
    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(loc='lower left', fontsize=12)
    
    # formal uniformity test
    stat, p = stats.kstest(pit_values, 'uniform')
    ax.text(0.02, 0.95, f'KS p={p:.3f}', transform=ax.transAxes,
            fontsize=9, verticalalignment='top')
    return ax

# Adaptive conformal inference function to adjust coverage level 

def adaptive_conformal_inference(
    rolling_crps_df,
    sigma_col='sigma',
    target_coverage=0.95,
    gamma=0.05,           # learning rate — how fast to adapt
    horizons=(1,13,26,52)
):
    """
    Adaptive Conformal Inference (Gibbs & Candès 2021) applied to
    rolling-origin probabilistic forecasts.
    
    Adjusts coverage level α_t at each origin based on recent miscoverage,
    providing marginal coverage guarantees under distribution shift.
    
    Parameters
    ----------
    rolling_crps_df : pd.DataFrame
        Output from rolling_crps() with columns mu, sigma, y_true, horizon, origin
    target_coverage : float
        Desired coverage level (e.g. 0.95 for 95% intervals)
    gamma : float
        Adaptation rate. Larger = faster adaptation, less stable.
        Recommended range: 0.02 - 0.10
    horizons : tuple
        Forecast horizons to calibrate
        
    Returns
    -------
    pd.DataFrame with added columns:
        alpha_t          : adapted significance level at each origin
        z_aci            : adapted z-score (wider or narrower than 1.96)
        lower_aci        : lower prediction interval
        upper_aci        : upper prediction interval
        covered_aci      : whether y_true fell inside interval
        pit_aci          : recalibrated PIT value
    """
    df = rolling_crps_df.copy().sort_values(['horizon', 'origin'])
    alpha_target = 1 - target_coverage   # e.g. 0.05
    
    results = []
    
    for h in horizons:
        h_df = df[df['horizon'] == h].copy().sort_values('origin')
        
        # initialize alpha at target level
        alpha_t = alpha_target
        alpha_history = []
        
        for idx, row in h_df.iterrows():
            # record current alpha before updating
            alpha_history.append(alpha_t)
            
            # conformal score: was the last interval correct?
            # 1 if y_true outside interval (miscoverage), 0 if inside
            z_t    = stats.norm.ppf(1 - alpha_t / 2)
            sigma  = row[sigma_col]
            lower  = row['mu'] - z_t * sigma
            upper  = row['mu'] + z_t * sigma
            
            missed = float(row['y_true'] < lower or row['y_true'] > upper)
            
            # ACI update rule (Gibbs & Candès 2021, eq. 1):
            # α_{t+1} = α_t + γ · (α_target - 1{missed})
            # missed=1 → α decreases → wider intervals next time
            # missed=0 → α increases → can narrow slightly
            alpha_t = alpha_t + gamma * (alpha_target - missed)
            alpha_t = np.clip(alpha_t, 0.001, 0.999)   # numerical safety
        
        h_df['alpha_t']     = alpha_history
        h_df['z_aci']       = stats.norm.ppf(1 - np.array(alpha_history) / 2)
        sigma_vals          = h_df[sigma_col].values
        h_df['lower_aci']   = h_df['mu'] - h_df['z_aci'] * sigma_vals
        h_df['upper_aci']   = h_df['mu'] + h_df['z_aci'] * sigma_vals
        h_df['covered_aci'] = (
            (h_df['y_true'] >= h_df['lower_aci']) & 
            (h_df['y_true'] <= h_df['upper_aci'])
        ).astype(float)
        
        # recalibrated PIT using adapted alpha
        effective_sigma = h_df['z_aci'].values / 1.96 * sigma_vals
        h_df['pit_aci'] = stats.norm.cdf(
            h_df['y_true'].values,
            loc=h_df['mu'].values,
            scale=effective_sigma
        )
        
        results.append(h_df)
    
    return pd.concat(results).sort_values(['origin', 'horizon'])

# Adaptive conformal inference diagnostics plot function

def plot_aci_diagnostics(aci_df, horizons=(1,13,26,52)):
    
    fig, axes = plt.subplots(2, len(horizons), figsize=(16, 8))
    fig.suptitle('ACI Calibration Diagnostics', fontsize=14)
    
    for j, h in enumerate(horizons):
        h_df = aci_df[aci_df['horizon'] == h].sort_values('origin')
        
        # ---- Panel 1: alpha_t over time ----
        ax = axes[0, j]
        ax.plot(h_df['origin'], h_df['alpha_t'], 
                linewidth=1, color='steelblue')
        ax.axhline(0.05, color='red', linestyle='--', 
                   linewidth=1, label='Target α=0.05')
        ax.set_title(f'h={h}: α_t over time')
        ax.set_ylabel('α_t')
        ax.set_xlabel('Origin')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        
        # ---- Panel 2: rolling empirical coverage ----
        ax = axes[1, j]
        # 26-fold rolling coverage (approx 1 year of origins)
        rolling_coverage = (h_df['covered_aci']
                           .rolling(window=26, min_periods=13)
                           .mean())
        ax.plot(h_df['origin'], rolling_coverage,
                linewidth=1, color='steelblue', label='Rolling coverage')
        ax.axhline(0.95, color='red', linestyle='--',
                   linewidth=1, label='Target 95%')
        ax.axhline(0.908, color='gray', linestyle=':',
                   linewidth=1, label='95% CI lower')
        ax.axhline(0.992, color='gray', linestyle=':',
                   linewidth=1, label='95% CI upper')
        ax.set_title(f'h={h}: Rolling coverage (26-fold window)')
        ax.set_ylabel('Coverage')
        ax.set_xlabel('Origin')
        ax.set_ylim(0.7, 1.0)
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # ---- Summary table ----
    print("\nACI Summary:")
    print(f"{'Horizon':<10} {'Coverage':<12} {'Mean α':<12} "
          f"{'Std α':<12} {'Mean z':<10}")
    print("-" * 56)
    for h in horizons:
        h_df = aci_df[aci_df['horizon'] == h]
        print(f"{h:<10} {h_df['covered_aci'].mean():<12.4f} "
              f"{h_df['alpha_t'].mean():<12.4f} "
              f"{h_df['alpha_t'].std():<12.4f} "
              f"{h_df['z_aci'].mean():<10.4f}")
        
# Adaptive conformal inference t-distribution function for specific horizons

# this function is for h=1 specific t-distribution calculation and application.   

def aci_t_distribution(h1_df, nu, sigma_col='sigma_calibrated',
                        target_coverage=0.95, gamma=0.02):
    """
    ACI using t-distribution quantiles instead of Gaussian.
    Use only for h=1 where heavy tails dominate.
    """
    df = h1_df.copy().sort_values('origin')
    alpha_target  = 1 - target_coverage
    alpha_t       = alpha_target
    alpha_history = []

    for idx, row in df.iterrows():
        alpha_history.append(alpha_t)

        # t-distribution quantile instead of normal
        z_t   = stats.t.ppf(1 - alpha_t / 2, df=nu)
        sigma = row[sigma_col]
        # scale t so std = sigma (not raw t scale)
        scale_t = sigma * np.sqrt((nu - 2) / nu) if nu > 2 else sigma
        lower = row['mu'] - z_t * scale_t
        upper = row['mu'] + z_t * scale_t

        missed  = float(row['y_true'] < lower or row['y_true'] > upper)
        alpha_t = np.clip(
            alpha_t + gamma * (alpha_target - missed),
            0.001, 0.999
        )

    df['alpha_t']     = alpha_history
    df['z_aci']       = [stats.t.ppf(1 - a/2, df=nu) 
                          for a in alpha_history]
    scale_vals        = (df[sigma_col].values * 
                         np.sqrt((nu-2)/nu) if nu > 2 
                         else df[sigma_col].values)
    df['lower_aci']   = df['mu'] - df['z_aci'] * scale_vals
    df['upper_aci']   = df['mu'] + df['z_aci'] * scale_vals
    df['covered_aci'] = (
        (df['y_true'] >= df['lower_aci']) & 
        (df['y_true'] <= df['upper_aci'])
    ).astype(float)
    df['pit_aci']     = stats.t.cdf(
        df['y_true'].values,
        df=nu,
        loc=df['mu'].values,
        scale=scale_vals
    )
    return df

# apply frozen calibration pipeline

def apply_frozen_pipeline(forecast_df, scale_factors, alpha_t_frozen,
                           nu_h1=2.83):
    """
    Apply variance scaling and ACI to test set forecasts.
    Uses t-distribution at h=1, Gaussian at h=13,26,52.
    
    Parameters
    ----------
    forecast_df   : DataFrame with columns mu, sigma_kalman, y_true
                    index is 0-based step number (0 = h=1, 12 = h=13, etc.)
    scale_factors : dict {horizon: scale_factor} from training CV
    alpha_t_frozen: dict {horizon: alpha_t} last value from ACI
    nu_h1         : degrees of freedom for t-distribution at h=1
    """
    df = forecast_df.copy()
    
    # map each forecast step to nearest evaluated horizon
    eval_horizons = np.array(sorted(scale_factors.keys()))
    
    df['horizon_mapped'] = [
        eval_horizons[np.argmin(np.abs(eval_horizons - (i + 1)))]
        for i in range(len(df))
    ]
    
    # apply variance scale factor
    df['sigma_scaled'] = [
        df['sigma_kalman'].iloc[i] * scale_factors[df['horizon_mapped'].iloc[i]]
        for i in range(len(df))
    ]
    
    # apply ACI alpha_t and compute intervals
    lowers, uppers, covereds = [], [], []
    
    for i in range(len(df)):
        h_mapped = df['horizon_mapped'].iloc[i]
        alpha_t  = alpha_t_frozen[h_mapped]
        sigma_c  = df['sigma_scaled'].iloc[i]
        mu       = df['mu'].iloc[i]
        y        = df['y_true'].iloc[i]
        
        # h=1: use t-distribution
        if h_mapped == 1:
            z_t     = stats.t.ppf(1 - alpha_t / 2, df=nu_h1)
            # scale t so std = sigma_scaled
            scale_t = sigma_c * np.sqrt((nu_h1 - 2) / nu_h1)
            lower   = mu - z_t * scale_t
            upper   = mu + z_t * scale_t
            
        # h=13,26,52: use Gaussian
        else:
            z_t   = stats.norm.ppf(1 - alpha_t / 2)
            lower = mu - z_t * sigma_c
            upper = mu + z_t * sigma_c
        
        lowers.append(lower)
        uppers.append(upper)
        covereds.append(float(y >= lower and y <= upper))
    
    df['z_aci']   = [
        stats.t.ppf(1 - alpha_t_frozen[df['horizon_mapped'].iloc[i]] / 2,
                    df=nu_h1)
        if df['horizon_mapped'].iloc[i] == 1
        else stats.norm.ppf(
            1 - alpha_t_frozen[df['horizon_mapped'].iloc[i]] / 2)
        for i in range(len(df))
    ]
    df['lower']   = lowers
    df['upper']   = uppers
    df['covered'] = covereds
    
    return df



# Model diagnostic function (maybe this belongs in the model_evaluation.py file, or just leave it in the notebook?)

def check_variance_params(res, value_floor=1e-4, pvalue_threshold=0.10):
    params  = res.params
    pvalues = res.pvalues

    variance_params = {k: v for k, v in params.items() if 'sigma' in k}

    print("=== Variance Parameter Health Check ===")
    any_problem = False
    for name, val in variance_params.items():
        pval = pvalues.get(name, np.nan)
        
        collapsed  = abs(val) < value_floor
        insig      = pval > pvalue_threshold
        
        if collapsed:
            status = '❌ COLLAPSED (near-zero value)'
        elif insig:
            status = f'⚠️  INSIGNIFICANT (p={pval:.3f})'
        else:
            status = f'✅ ok (p={pval:.3f})'
        
        print(f'  {status}  {name}: {val:.6e}')
        if collapsed or insig:
            any_problem = True

    return any_problem