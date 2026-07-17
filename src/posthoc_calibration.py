import numpy as np
from scipy import stats
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# compute variance scale factors
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# apply scale factors as a post-processing (post-hoc) step
# ---------------------------------------------------------

def apply_variance_scaling(crps_df, scale_factors):
    """
    Apply empirical scale factors to sigma
    Recompute CRPS and PIT
    """
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

# ---------------------------------------------------------
# plot PIT histogram with confidence interval bands 
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# adaptive conformal inference
# ---------------------------------------------------------

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

    Adjusts the significance level alpha_t at each origin based on recent
    miscoverage, providing marginal coverage guarantees under distribution
    shift. Operates independently per forecast horizon.

    Parameters
    ----------
    rolling_crps_df : pd.DataFrame
        Output from rolling_crps() containing columns:
        mu, sigma (or sigma_calibrated), y_true, horizon, origin.
    sigma_col : str, default 'sigma'
        Column to use as the base uncertainty estimate. Pass
        'sigma_calibrated' when applying ACI after variance scaling.
    target_coverage : float, default 0.95
        Desired nominal coverage level (e.g. 0.95 for 95% intervals).
    gamma : float, default 0.05
        Adaptation rate controlling the speed/stability tradeoff.
        - Larger gamma (0.10-0.20): faster adaptation to regime shifts,
          but produces more volatile interval widths.
        - Smaller gamma (0.01-0.02): smoother, more stable intervals,
          but slower to respond to structural changes.
        Recommended range: 0.02-0.10. Select via grid search over
        empirical coverage and alpha stability.
    horizons : tuple of int, default (1, 13, 26, 52)
        Forecast horizons to calibrate. ACI is applied independently
        at each horizon using its own alpha_t sequence.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        - alpha_t     : adapted significance level at each origin
        - z_aci       : normal quantile corresponding to alpha_t
                        (wider than 1.96 when coverage has been low)
        - lower_aci   : lower prediction interval bound
        - upper_aci   : upper prediction interval bound
        - covered_aci : 1.0 if y_true fell inside [lower_aci, upper_aci],
                        0.0 otherwise

    Notes
    -----
    ACI update rule (Gibbs & Candès 2021, Eq. 1):
        alpha_{t+1} = alpha_t + gamma · (alpha_target - 1{miscovered_t})

    When the interval is missed (miscovered=1), alpha decreases, widening
    subsequent intervals. When covered (miscovered=0), alpha increases
    slightly, allowing intervals to narrow. The clip to [0.001, 0.999]
    prevents degenerate intervals.

    pit_aci is not computed because ACI intervals are adaptive and do not
    correspond to a fixed predictive distribution. Use covered_aci and
    rolling coverage plots for calibration assessment instead.
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
        # effective_sigma = h_df['z_aci'].values / 1.96 * sigma_vals
        # h_df['pit_aci'] = stats.norm.cdf(
        #     h_df['y_true'].values,
        #     loc=h_df['mu'].values,
        #     scale=effective_sigma
        # )
        
        results.append(h_df)
    
    return pd.concat(results).sort_values(['origin', 'horizon']) 

# ---------------------------------------------------------
# adaptive conformal inference diagnostics plot
# ---------------------------------------------------------

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