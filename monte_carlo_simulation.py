import numpy as np
import pandas as pd

def monte_carlo_simulation_func(min_inventory, max_inventory, forward_prices, start_inventory,
                                injection_percentage_ratchet, injection_volume_ratchet,
                                withdraw_percentage_ratchet, withdraw_volume_ratchet,
                                monthly_profile, n_sims, daily_volatility,
                                kappa=2.0):

    start_date=None  

    """
    Monte Carlo storage with month-of-year seasonality + mean-reverting deviations (OU).
    forward_prices can be any length (daily), no need for a DatetimeIndex.
    """

    # --- coerce forward_prices to numpy array and make a daily date index from start_date ---
    # ---------- 0) Dates & monthly -> daily forward curve ----------
    fwd = np.asarray(forward_prices, dtype=float)
    if fwd.size == 0:
        raise ValueError("forward_prices is empty.")

    # Start on the first day of the current month unless provided
    if start_date is None:
        today = pd.Timestamp.today().normalize()
        start_month = today.replace(day=1)
    else:
        start_month = pd.Timestamp(start_date).normalize().replace(day=1)

    # Build month starts for the horizon, length = number of monthly prices
    month_starts = pd.date_range(start=start_month, periods=fwd.size, freq="MS")
    last_day = (month_starts[-1] + pd.offsets.MonthEnd(0))  # end of last month
    all_days = pd.date_range(start=month_starts[0], end=last_day, freq="D")

    # Monthly base -> daily base by forward-fill within each month
    monthly_series = pd.Series(fwd, index=month_starts)
    daily_base = monthly_series.reindex(all_days, method="ffill").to_numpy()
    n_days = daily_base.size

    annual_inflation = 0.02
    inflation_monthly = (1.0 + annual_inflation)**(1/12.0) - 1.0  # ≈ 0.0016515

    # Count months elapsed from the first day of the start month
    # (works regardless of pandas Period quirks)
    start_mnum = all_days[0].year * 12 + all_days[0].month
    mnum       = (all_days.year * 12 + all_days.month).to_numpy()
    months_elapsed = (mnum - start_mnum).astype(np.int64)
    inflation_multiplier = (1.0 + inflation_monthly) ** months_elapsed

    # Inflate the base curve (daily_base is your monthly→daily forward-filled base)
    daily_base_plus_inflation = daily_base * inflation_multiplier
    # ---------- 1) Month-of-year seasonality (repeat annually) ----------
    if not isinstance(monthly_profile, pd.Series):
        monthly_profile = pd.Series(monthly_profile)
    monthly_profile.index = monthly_profile.index.astype(int)
    if set(monthly_profile.index) != set(range(1, 13)):
        raise ValueError("monthly_profile.index must be the month numbers 1..12 (Jan..Dec).")

    seasonal_daily = monthly_profile.loc[all_days.month].to_numpy()  # multiplier per day


    # --- OU deviations around the seasonal mean (additive) ---
    dt = 1.0 / 30
    alpha = np.exp(-kappa * dt)  # daily mean-reversion factor

    # Per-day sigma handling
    if isinstance(daily_volatility, pd.Series):
        if isinstance(daily_volatility.index, pd.DatetimeIndex):
            sigma_daily = daily_volatility.reindex(all_days, method="ffill").to_numpy()
        elif len(daily_volatility) == n_days:
            sigma_daily = daily_volatility.to_numpy()
        else:
            raise ValueError("daily_volatility Series must be DatetimeIndex-aligned to the horizon or length == n_days.")
    else:
        sigma_daily = np.full(n_days, float(daily_volatility))

    # Pre-generate Gaussian shocks for all sims
    shocks = np.random.normal(0.0, 1.0, size=(n_days - 1, n_sims))

    # --- ratchet helper ---
    def get_ratchet_rates(inv_percent: float):
        """Return (max_inject_rate, max_withdraw_rate) for current inventory %, using tiered limits."""
        # injection tiers (no interpolation)
        if inv_percent < injection_percentage_ratchet[1]:
            max_inject = injection_volume_ratchet[0]
        elif inv_percent < injection_percentage_ratchet[2]:
            max_inject = injection_volume_ratchet[1]
        elif inv_percent < injection_percentage_ratchet[3]:
            max_inject = injection_volume_ratchet[2]
        else:
            max_inject = injection_volume_ratchet[3]

        # withdrawal tiers (no interpolation)
        if inv_percent < withdraw_percentage_ratchet[1]:
            max_withdraw = withdraw_volume_ratchet[0]
        else:
            max_withdraw = withdraw_volume_ratchet[1]
        return float(max_inject), float(max_withdraw)

    best_profit = -np.inf
    best_records = None

    # --- Monte Carlo loop ---
    for sim in range(n_sims):
        # OU deviation path
        dev = np.zeros(n_days, dtype=float)
        for t in range(n_days - 1):
            dev[t + 1] = alpha * dev[t] + sigma_daily[t] * np.sqrt(dt) * shocks[t, sim]

        sim_prices = (daily_base_plus_inflation + dev) * seasonal_daily
        sim_prices = np.maximum(sim_prices, 0.0)  # non-negative

        # simulate storage decisions (bang-bang + ratchets), with next-day trend rule
        current_inventory = float(start_inventory)
        cash_flow = 0.0
        records = []

        for i in range(n_days):
            price = sim_prices[i]
            current_date = all_days[i]

            inv_percent = (current_inventory / max_inventory) * 100.0 if max_inventory > 0 else 0.0
            max_inject, max_withdraw = get_ratchet_rates(inv_percent)

            # decide action vs next day's price
            feasible_actions = ["Hold"]  # always feasible
            if (current_inventory < max_inventory) and (max_inject > 0):
                feasible_actions.append("Inject")
            if (current_inventory > min_inventory) and (max_withdraw > 0):
                feasible_actions.append("Withdraw")

            # Sample one action uniformly at random among feasible
            # (If you want custom weights, replace the next line with rng.choice(feasible_actions, p=weights))
            action = np.random.choice(feasible_actions)
            # execute within constraints
            volume = 0.0
            if action == "Inject":
                volume = min(max_inject, max_inventory - current_inventory)
                if volume > 0:
                    current_inventory += volume
                    cash_flow -= price * volume
                else:
                    action = "Hold"
                    volume = 0.0
            elif action == "Withdraw":
                volume = min(max_withdraw, current_inventory - min_inventory)
                if volume > 0:
                    current_inventory -= volume
                    cash_flow += price * volume
                else:
                    action = "Hold"
                    volume = 0.0

            records.append({
                "Date": current_date.date(),
                "Forward Price": round(float(price), 3),
                "Action": action,
                "Volume": float(volume),
                "End Inventory": float(current_inventory),
            })

        # evaluate this path
        profit = cash_flow
        if profit > best_profit:
            best_profit = profit
            best_records = records

    best_df = pd.DataFrame(best_records, columns=["Date", "Forward Price", "Action", "Volume", "End Inventory"])
    return best_df, best_profit


   
