import numpy as np
import pandas as pd

# ---------------------------
# Utilities (ratchets, basis)
# ---------------------------

def stepwise_rate(inv_pct, pct_breaks, vol_rates):
    """No-interp ratchet: return allowed rate for inv%."""
    for k in range(len(pct_breaks) - 1):
        if inv_pct < pct_breaks[k + 1]:
            return float(vol_rates[k])
    return float(vol_rates[-1])

def get_ratchets(inv_pct, inj_pct, inj_vol, wd_pct, wd_vol):
    max_inj = stepwise_rate(inv_pct, inj_pct, inj_vol)
    max_wd  = stepwise_rate(inv_pct, wd_pct,  wd_vol)
    return max_inj, max_wd

def features(price, inv):
    """
    Basis for V(t+1, price, inv).
    Keep it small to avoid overfit; you can expand later.
    """
    p  = price
    i  = inv
    return np.column_stack([
        np.ones_like(p),       # 1
        p, i,                  # linear
        p*i,                   # interaction
        p**2, i**2,            # quadratics
    ])  # shape: (n, 6)

# -----------------------------------------
# 1) Build daily timeline & daily base curve
# -----------------------------------------

def expand_monthly_to_daily(forward_prices, start_date=None):
    fwd = np.asarray(forward_prices, dtype=float)
    if fwd.size == 0:
        raise ValueError("forward_prices is empty.")

    if start_date is None:
        today = pd.Timestamp.today().normalize()
        start_month = today.replace(day=1)
    else:
        start_month = pd.Timestamp(start_date).normalize().replace(day=1)

    month_starts = pd.date_range(start=start_month, periods=fwd.size, freq="MS")
    last_day = month_starts[-1] + pd.offsets.MonthEnd(0)
    all_days = pd.date_range(start=month_starts[0], end=last_day, freq="D")

    monthly_series = pd.Series(fwd, index=month_starts)
    daily_base = monthly_series.reindex(all_days, method="ffill").to_numpy()
    return all_days, daily_base

# ----------------------------------------------------
# 2) Price path generator (plug your own logic here)
#    - OU additive noise anchored to base (inflation optional)
#    - multiplicative monthly seasonality AFTER OU
# ----------------------------------------------------

def generate_price_paths(
    forward_prices, monthly_profile, n_sims,
    daily_volatility=0.5, kappa=3.0, start_date=None,
    annual_inflation=0.0  # e.g. 0.02 for 2%/yr; monthly compounding
):
    all_days, daily_base = expand_monthly_to_daily(forward_prices, start_date)
    n_days = daily_base.size

    # monthly inflation (piecewise constant)
    if annual_inflation and annual_inflation != 0.0:
        infl_m = (1.0 + annual_inflation)**(1/12.0) - 1.0
        start_mnum = all_days[0].year * 12 + all_days[0].month
        mnum = (all_days.year * 12 + all_days.month).to_numpy()
        months_elapsed = (mnum - start_mnum).astype(np.int64)
        infl_mult = (1.0 + infl_m) ** months_elapsed
        daily_base = daily_base * infl_mult

    # month-of-year seasonal multiplier
    if not isinstance(monthly_profile, pd.Series):
        monthly_profile = pd.Series(monthly_profile)
    monthly_profile.index = monthly_profile.index.astype(int)
    seasonal_daily = monthly_profile.loc[all_days.month].to_numpy()

    # OU deviations around BASE (not seasonal)
    dt = 1.0 / 365.0
    alpha = np.exp(-kappa * dt)
    sigma_daily = float(daily_volatility)

    shocks = np.random.normal(0.0, 1.0, size=(n_days - 1, n_sims))
    dev = np.zeros((n_days, n_sims), dtype=float)
    for t in range(n_days - 1):
        dev[t + 1, :] = alpha * dev[t, :] + sigma_daily * np.sqrt(dt) * shocks[t, :]

    # Apply seasonality after OU
    sim_prices = (daily_base[:, None] + dev) * seasonal_daily[:, None]
    sim_prices = np.maximum(sim_prices, 0.0)
    return all_days, sim_prices  # shapes: (n_days,), (n_days, n_sims)

# ---------------------------------------------------------
# 3) Forward rollout under a RANDOM feasible policy (explore)
# ---------------------------------------------------------

def rollout_random_policy(sim_prices, min_inv, max_inv, start_inv,
                          inj_pct, inj_vol, wd_pct, wd_vol, rng=None):
    n_days, n_sims = sim_prices.shape
    if rng is None:
        rng = np.random.default_rng()

    inv = np.full(n_sims, float(start_inv))
    inv_paths = np.zeros((n_days, n_sims), dtype=float)
    cf = np.zeros((n_days, n_sims), dtype=float)
    act = np.empty((n_days, n_sims), dtype=object)

    for t in range(n_days):
        price_t = sim_prices[t, :]
        for s in range(n_sims):
            inv_pct = (inv[s] / max_inv) * 100.0 if max_inv > 0 else 0.0
            max_inj, max_wd = get_ratchets(inv_pct, inj_pct, inj_vol, wd_pct, wd_vol)

            feasible = ["Hold"]
            if (inv[s] < max_inv) and (max_inj > 0):
                feasible.append("Inject")
            if (inv[s] > min_inv) and (max_wd > 0):
                feasible.append("Withdraw")

            a = rng.choice(feasible)

            if a == "Inject":
                vol = min(max_inj, max_inv - inv[s])
                if vol > 0:
                    inv[s] += vol
                    cf[t, s] = -price_t[s] * vol
                else:
                    a = "Hold"; vol = 0.0
            elif a == "Withdraw":
                vol = min(max_wd, inv[s] - min_inv)
                if vol > 0:
                    inv[s] -= vol
                    cf[t, s] = +price_t[s] * vol
                else:
                    a = "Hold"; vol = 0.0
            else:
                vol = 0.0

            inv_paths[t, s] = inv[s]
            act[t, s] = a

    return inv_paths, cf, act  # shapes: (n_days,n_sims), (n_days,n_sims), (n_days,n_sims)

# -------------------------------------------------------
# 4) Backward pass: LSMC policy improvement (single pass)
# -------------------------------------------------------

def lsmc_backward(sim_prices, inv_paths, min_inv, max_inv,
                  inj_pct, inj_vol, wd_pct, wd_vol,
                  r_annual=0.0):
    """
    Given price paths and a *forward* rollout of inventory states (from any baseline policy),
    compute an improved policy using LSMC:
      - regress V_{t+1} on features(price_{t+1}, inv_{t+1}) to get Vhat_{t+1}
      - at time t, pick action a in {Inject, Withdraw, Hold} maximizing:
          cashflow_a(t) + disc * Vhat_{t+1}( price_{t+1}, inv' )
    Returns:
      actions_opt (n_days, n_sims), value0 (expected NPV), and a sample schedule DataFrame.
    """
    n_days, n_sims = sim_prices.shape
    dt = 1.0 / 365.0
    disc = np.exp(-r_annual * dt)

    # Terminal "salvage" — value of leftover inventory at T at terminal price (approximation)
    V_next = sim_prices[-1, :] * (inv_paths[-1, :] - min_inv)

    actions_opt = np.empty((n_days, n_sims), dtype=object)

    # Backward induction
    for t in range(n_days - 1, -1, -1):
        p_t = sim_prices[t, :]
        inv_t = inv_paths[t, :]

        if t == n_days - 1:
            # At T, choose among actions using only immediate cashflows (no future)
            V_t = np.zeros(n_sims)
            for s in range(n_sims):
                ip = (inv_t[s] / max_inv) * 100.0 if max_inv > 0 else 0.0
                max_inj, max_wd = get_ratchets(ip, inj_pct, inj_vol, wd_pct, wd_vol)

                # Values for each action
                val_hold = 0.0
                val_inj  = -p_t[s] * min(max_inj, max_inv - inv_t[s])
                val_wd   =  p_t[s] * min(max_wd, inv_t[s] - min_inv)
                # Pick best
                vals = {"Hold": val_hold, "Inject": val_inj, "Withdraw": val_wd}
                a_star = max(vals, key=vals.get)
                actions_opt[t, s] = a_star
                V_t[s] = vals[a_star]
            V_next = V_t  # for the step t-1
            continue

        # Regress V_next on state at t+1 to get Vhat_{t+1}(·)
        p_next = sim_prices[t + 1, :]
        inv_next = inv_paths[t + 1, :]
        X = features(p_next, inv_next)          # (n_sims, k)
        y = V_next                              # realized value-to-go at t+1
        # OLS fit
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        # For each path, evaluate Q-values for actions at time t
        V_t = np.zeros(n_sims)
        for s in range(n_sims):
            ip = (inv_t[s] / max_inv) * 100.0 if max_inv > 0 else 0.0
            max_inj, max_wd = get_ratchets(ip, inj_pct, inj_vol, wd_pct, wd_vol)

            # Candidate controls (bang-bang)
            # Hold
            inv_hold = inv_t[s]
            cont_hold = features(np.array([p_next[s]]), np.array([inv_hold])).dot(beta)[0]
            val_hold = 0.0 + disc * cont_hold

            # Inject
            vol_inj = min(max_inj, max_inv - inv_t[s])
            inv_inj = inv_t[s] + vol_inj
            cont_inj = features(np.array([p_next[s]]), np.array([inv_inj])).dot(beta)[0]
            val_inj = (-p_t[s] * vol_inj) + disc * cont_inj

            # Withdraw
            vol_wd = min(max_wd, inv_t[s] - min_inv)
            inv_wd = inv_t[s] - vol_wd
            cont_wd = features(np.array([p_next[s]]), np.array([inv_wd])).dot(beta)[0]
            val_wd = (+p_t[s] * vol_wd) + disc * cont_wd

            vals = {"Hold": val_hold, "Inject": val_inj, "Withdraw": val_wd}
            a_star = max(vals, key=vals.get)
            actions_opt[t, s] = a_star
            V_t[s] = vals[a_star]

        V_next = V_t  # feed backward

    # Expected NPV across paths under improved policy (at time 0)
    value0 = V_next.mean()

    # Build a sample schedule: replay one path forward under learned policy
    s0 = 0
    df_rows = []
    inv = inv_paths[0, s0]  # start inv from rollout (equals start_inventory typically)
    for t in range(n_days):
        a = actions_opt[t, s0]
        p = float(sim_prices[t, s0])
        ip = (inv / max_inv) * 100.0 if max_inv > 0 else 0.0
        max_inj, max_wd = get_ratchets(ip, inj_pct, inj_vol, wd_pct, wd_vol)
        vol = 0.0
        if a == "Inject":
            vol = min(max_inj, max_inv - inv); inv += vol
        elif a == "Withdraw":
            vol = min(max_wd, inv - min_inv); inv -= vol
        df_rows.append({
            "Date": pd.NaT,  # you can pass dates in from outside if desired
            "Forward Price": round(p, 3),
            "Action": a,
            "Volume": float(vol),
            "End Inventory": float(inv),
        })
    sample_df = pd.DataFrame(df_rows, columns=["Date", "Forward Price", "Action", "Volume", "End Inventory"])
    return actions_opt, float(value0), sample_df

# -------------------------------------------------------
# 5) End-to-end wrapper: price → rollout → LSMC → results
# -------------------------------------------------------

def monte_carlo_storage_lsmc_func(
    forward_prices, monthly_profile,
    min_inventory, max_inventory, start_inventory,
    inj_pct, inj_vol, wd_pct, wd_vol,
    n_sims=2000, daily_volatility=0.5, kappa=3.0,
    start_date=None, annual_inflation=0.0, r_annual=0.0,
    seed=None
):
    # Prices
    all_days, sim_prices = generate_price_paths(
        forward_prices, monthly_profile, n_sims,
        daily_volatility=daily_volatility, kappa=kappa,
        start_date=start_date, annual_inflation=annual_inflation
    )

    # Rollout under random feasible policy to generate state coverage
    rng = np.random.default_rng(seed)
    inv_paths, cf_rollout, _ = rollout_random_policy(
        sim_prices, min_inventory, max_inventory, start_inventory,
        inj_pct, inj_vol, wd_pct, wd_vol, rng=rng
    )

    # LSMC backward improvement
    actions_opt, value0, sample_df = lsmc_backward(
        sim_prices, inv_paths, min_inventory, max_inventory,
        inj_pct, inj_vol, wd_pct, wd_vol, r_annual=r_annual
    )

    # Attach calendar dates into sample_df for readability
    sample_df["Date"] = expand_monthly_to_daily(forward_prices, start_date)[0].date

    return value0, sample_df, all_days, actions_opt
