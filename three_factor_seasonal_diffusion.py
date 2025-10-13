import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ThreeFactorParams:
    spot_mean_reversion_days: float  # e.g. 90
    spot_vol: float                  # annualized, e.g. 0.85
    long_term_vol: float             # annualized, e.g. 0.30
    seasonal_vol: float              # annualized, e.g. 0.19

def seasonal_shape_unit_rms(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return g(m_t) with zero mean across months and unit RMS over {1..12}."""
    # Base 12-month cosine; zero-mean, unit-RMS over the 12 months
    base = np.cos(2*np.pi*np.arange(12)/12)
    base -= base.mean()
    base /= np.sqrt((base**2).mean())
    # Map each date to its month index
    return np.array([base[d.month-1] for d in dates], dtype=float)

def exact_ou_step(x_prev, kappa, sigma, dt, z):
    """Exact OU step: x_t = e^{-kappa dt} x_{t-1} + sigma * sqrt((1-e^{-2kappa dt})/(2kappa)) * z"""
    if kappa <= 0:
        # Degenerates to Brownian if kappa==0
        return x_prev + sigma*np.sqrt(dt)*z
    a = np.exp(-kappa*dt)
    var = (1.0 - np.exp(-2.0*kappa*dt)) / (2.0*kappa)
    return a * x_prev + sigma * np.sqrt(var) * z

def simulate_three_factor_seasonal(
    dates: pd.DatetimeIndex,          # nomination dates (ascending)
    forward: pd.Series,               # F(t) aligned to dates
    params: ThreeFactorParams,
    rho: np.ndarray | None = None,    # 3x3 corr matrix for [st, lt, sw]; None -> identity
    n_paths: int = 10_000,
    day_count: int = 365,
    seed: int = 42,
    return_mu_and_var: bool = False,
):
    """
    Returns dict with S (spot paths) and factor states X_st, X_lt, X_sw, plus mu_t, varY_t if requested.
    All vols are annualized. Time step dt is computed from dates / day_count.
    """
    dates = pd.DatetimeIndex(dates)
    F = forward.reindex(dates).to_numpy().astype(float)
    assert np.all(np.isfinite(F)), "Forward curve must be finite on all dates."

    # Time steps in years
    dt_days = np.diff(dates.insert(0, dates[0])).astype('timedelta64[D]').astype(float)
    dt_days[0] = 0.0
    dt = dt_days / day_count
    T = len(dates)

    # Parameters in annualized units
    kappa = (1.0 / params.spot_mean_reversion_days) * day_count
    sig_st = params.spot_vol
    sig_lt = params.long_term_vol
    sig_sw = params.seasonal_vol

    # Seasonal loading g_t
    g = seasonal_shape_unit_rms(dates)  # deterministic, unit-RMS over 12 months

    # Random generator & correlation
    rng = np.random.default_rng(seed)
    if rho is None:
        L = np.eye(3)
    else:
        # Cholesky with small jitter for numerical stability
        eps = 1e-12
        L = np.linalg.cholesky(rho + eps*np.eye(3))

    # Precompute variances of each factor at each t (for mu_t)
    # OU variance to time t from 0 with exact formula:
    # Var[X_st(t)] = sig_st^2 * (1 - exp(-2*kappa t)) / (2*kappa)
    t_years = np.cumsum(dt)
    var_st_t = np.where(kappa > 0.0,
                        (sig_st**2) * (1.0 - np.exp(-2.0*kappa*t_years)) / (2.0*kappa),
                        (sig_st**2) * t_years)
    var_lt_t = (sig_lt**2) * t_years
    var_sw_t = (sig_sw**2) * t_years
    # Total variance of Y_t = X_st + X_lt + g_t * X_sw.
    # If innovations are correlated, strictly we need cross-terms.
    # Here we assume independence for the mu_t calibration; you can include cross-terms if rho != I.
    varY_t = var_st_t + var_lt_t + (g**2) * var_sw_t

    # Drift to match forwards under lognormal mapping
    mu_t = np.log(F) - 0.5 * varY_t

    # Allocate arrays
    X_st = np.zeros((n_paths, T), dtype=float)
    X_lt = np.zeros_like(X_st)
    X_sw = np.zeros_like(X_st)

    # Simulate paths
    for t in range(1, T):
        Z = rng.standard_normal((n_paths, 3)) @ L.T  # correlated N(0,1)
        # exact OU update for short-term
        X_st[:, t] = exact_ou_step(X_st[:, t-1], kappa, sig_st, dt[t], Z[:, 0])
        # Brownian updates for long-term & seasonal
        sqrt_dt = np.sqrt(dt[t])
        X_lt[:, t] = X_lt[:, t-1] + sig_lt * sqrt_dt * Z[:, 1]
        X_sw[:, t] = X_sw[:, t-1] + sig_sw * sqrt_dt * Z[:, 2]

    # Build log spot and spot
    # ln S_t = mu_t + X_st + X_lt + g_t * X_sw
    Y = X_st + X_lt + (g[None, :] * X_sw)
    lnS = mu_t[None, :] + Y
    S = np.exp(lnS)

    out = {"dates": dates, "S": S, "X_st": X_st, "X_lt": X_lt, "X_sw": X_sw}
    if return_mu_and_var:
        out["mu_t"] = mu_t
        out["varY_t"] = varY_t
        out["g_t"] = g
    return out
