import numpy as np
import pandas as pd
from math import sqrt

def build_constant_maturity_from_delivery(curve_df: pd.DataFrame, maturities=(24,36,48)) -> pd.DataFrame:
    full_cols = pd.period_range(curve_df.columns.min(), curve_df.columns.max(), freq="M")
    df = curve_df.reindex(columns=full_cols)
    tmp = df.T
    tmp.index = tmp.index.to_timestamp("M")
    tmp = tmp.sort_index().interpolate(method="time", limit_direction="both")
    df_full = tmp.T
    base = df_full.index.to_period("M")

    arr = df_full.to_numpy()
    out = pd.DataFrame(index=df_full.index, columns=maturities, dtype=float)
    cols_as_per = df_full.columns.to_period("M")
    for m in maturities:
        targets = (base + m)
        pos = cols_as_per.get_indexer(targets)
        valid = pos >= 0
        col = np.full(len(df_full), np.nan, dtype=float)
        if valid.any():
            col[valid] = arr[np.arange(len(df_full))[valid], pos[valid]]
        out[m] = col
    return out

def estimate_long_term_vol(cm_df: pd.DataFrame, day_count: int = 252, winsor: float | None = 0.01) -> float:
    cm = cm_df.dropna(how="any")
    if cm.shape[0] < 10:
        raise ValueError("Insufficient complete rows after constant-maturity construction.")
    logp = np.log(cm)
    ret = logp.diff().iloc[1:]                       # daily log-returns per constant maturity
    dt = cm.index.to_series().diff().dt.days.iloc[1:].astype(float).values
    w = np.full(ret.shape[1], 1.0/ret.shape[1])      # equal-weight level shock
    level = ret.to_numpy() @ w
    if winsor is not None and 0.0 < winsor < 0.5:    # light tail control
        lo, hi = np.nanquantile(level, [winsor, 1.0 - winsor])
        level = np.clip(level, lo, hi)
    sigma_per_day = float(np.sqrt(np.nansum(level**2) / np.nansum(dt)))
    return sigma_per_day * sqrt(day_count)

def compute_long_term_vol_from_df(
    df,
    maturities=(24,36,48),
    day_count: int = 252,
    winsor: float | None = 0.01
) -> tuple[float, pd.DataFrame]:
    
    if "period" in df.columns:
        df = df[df["period"].astype(str).str.lower().eq("month")]

    df["traded"] = pd.to_datetime(df["traded"], errors="coerce")
    df["delivery"] = pd.to_datetime(df["delivery"], errors="coerce")
    df = df.dropna(subset=["traded", "delivery", "settlement"])
    df["deliv_per"] = df["delivery"].dt.to_period("M")
    grp = df.groupby(["traded", "deliv_per"], as_index=False)["settlement"].last()
    mat = grp.pivot(index="traded", columns="deliv_per", values="settlement").sort_index()
    cm = build_constant_maturity_from_delivery(mat, maturities=maturities)
    vol = estimate_long_term_vol(cm, day_count=day_count, winsor=winsor)
    return vol

# Example on your file:
# vol, cm = compute_long_term_vol_from_excel("/mnt/data/forward_prices_history.xlsx")
# print(vol); cm.to_csv("/mnt/data/constant_maturity_prices.csv")
