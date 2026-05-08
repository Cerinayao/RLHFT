from __future__ import annotations

import numpy as np
import pandas as pd


def clip_int(x: int, lo: int, hi: int) -> int:
    return int(min(max(int(x), lo), hi))


def mean_daily_pnl(reward: pd.Series) -> float:
    daily = reward.groupby(reward.index.normalize()).sum()
    return float(daily.mean())


def max_daily_drawdown(reward: pd.Series) -> float:
    daily = reward.resample("1D").sum()
    cum = daily.cumsum()
    dd = cum - cum.cummax()
    return float(dd.min())


def daily_sharpe(pnl: pd.Series, annualization: float = 252.0) -> float:
    """Annualized Sharpe computed from daily-aggregated PnL.

    The input is expected at intraday frequency; we sum within each
    calendar day before computing mean/std so the sqrt(annualization)
    factor matches the per-period horizon.
    """
    daily = pnl.groupby(pnl.index.normalize()).sum()
    if len(daily) < 2:
        return float("nan")
    sd = daily.std(ddof=1)
    if not sd or np.isnan(sd):
        return float("nan")
    return float(np.sqrt(annualization) * daily.mean() / sd)


def compute_strategy_metrics(pnl: pd.Series) -> dict[str, float]:
    return {
        "mean_daily_pnl_$": mean_daily_pnl(pnl),
        "max_drawdown_daily_$": max_daily_drawdown(pnl),
        "daily_sharpe": daily_sharpe(pnl),
    }
