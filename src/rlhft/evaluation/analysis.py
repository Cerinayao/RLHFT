from __future__ import annotations

import numpy as np
import pandas as pd


def signal_fwd_corr(
    df: pd.DataFrame,
    signal_col: str,
    price_col: str,
    horizon: int,
    start_date: str = "2024-02-19",
) -> float:
    """Pearson correlation between a signal column and forward return at given horizon."""
    d = df.sort_index().between_time("00:00", "16:00")
    d = d[d.index >= pd.Timestamp(start_date)]

    s = d[signal_col].astype(float)
    fwd = d[price_col].shift(-horizon) - d[price_col]

    both = pd.concat([s.rename("s"), fwd.rename("r")], axis=1).dropna()
    if both.empty:
        return np.nan

    return both["s"].corr(both["r"])


def sweep_signal_horizons(
    df: pd.DataFrame,
    signal_col: str,
    price_col: str,
    horizons: range | list[int],
    start_date: str = "2024-02-19",
) -> pd.DataFrame:
    """Sweep over horizons and return correlation at each."""
    corrs = [signal_fwd_corr(df, signal_col, price_col, h, start_date) for h in horizons]
    return pd.DataFrame({"horizon": list(horizons), "correlation": corrs})


def action_vs_fwd_return(
    df: pd.DataFrame,
    price_col: str,
    action: pd.Series,
    horizon: int,
    start_date: str = "2024-02-19",
) -> pd.DataFrame:
    """Align action series with forward return at given horizon."""
    d = df.sort_index().between_time("00:00", "16:00")
    d = d[d.index >= pd.Timestamp(start_date)]

    fwd = (d[price_col].shift(-horizon) - d[price_col]).rename("fwd_ret")

    a = action.sort_index().astype(float)
    a = a.between_time("00:00", "16:00")
    a = a[a.index >= pd.Timestamp(start_date)]
    a.name = "action"

    return pd.concat([a, fwd], axis=1).dropna()


def summarize_action_vs_fwd_return(
    df: pd.DataFrame,
    price_col: str,
    action: pd.Series,
    horizon: int,
    start_date: str = "2024-02-19",
) -> tuple[pd.DataFrame, float]:
    """Summarize action buckets against forward returns and compute Spearman rho."""
    aligned = action_vs_fwd_return(df, price_col, action, horizon, start_date)
    if aligned.empty:
        return pd.DataFrame(columns=["mean", "std", "count"]), np.nan

    grouped = aligned.groupby("action")["fwd_ret"].agg(
        mean="mean",
        std="std",
        count="count",
    )
    rho = float(aligned["action"].corr(aligned["fwd_ret"], method="spearman"))
    return grouped, rho


def summarize_pnl_by_regime(
    *,
    pnl: pd.Series,
    regime_state: pd.Series,
    start_date: str = "2024-02-19",
    trading_start: str = "00:00",
    trading_end: str = "16:00",
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Break down PnL statistics by regime state (+1 vs -1)."""
    tmp = pd.DataFrame({
        "pnl": pnl.sort_index().astype(float),
        "regime": regime_state.sort_index(),
    })

    tmp = tmp[tmp.index >= pd.Timestamp(start_date)]
    tmp = tmp.between_time(trading_start, trading_end)
    tmp = tmp.dropna()
    tmp = tmp[tmp["regime"].isin([-1, 1])]

    def sharpe(x):
        if len(x) < 2 or x.std(ddof=1) == 0:
            return np.nan
        return np.sqrt(annualization) * x.mean() / x.std(ddof=1)

    results = {}
    for reg, name in [(1, "works (+1)"), (-1, "fails (-1)")]:
        sub = tmp[tmp["regime"] == reg]["pnl"]
        results[name] = {
            "cum_pnl": sub.sum(),
            "sharpe": sharpe(sub),
            "N": len(sub),
        }

    return pd.DataFrame(results).T
