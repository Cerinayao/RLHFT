from __future__ import annotations

import numpy as np
import pandas as pd


def clip_int(x: int, lo: int, hi: int) -> int:
    return int(min(max(int(x), lo), hi))


def daily_pnl(pnl: pd.Series) -> pd.Series:
    daily = pnl.dropna().groupby(pnl.dropna().index.normalize()).sum()
    return daily.astype(float)


def sharpe_ratio(pnl: pd.Series, annualization: float = 252.0) -> float:
    series = pnl.dropna().astype(float)
    if len(series) < 2:
        return float("nan")
    std = series.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(np.sqrt(annualization) * series.mean() / std)


def daily_sharpe(pnl: pd.Series, annualization: float = 252.0) -> float:
    return sharpe_ratio(daily_pnl(pnl), annualization=annualization)


def mean_daily_pnl(reward: pd.Series) -> float:
    daily = daily_pnl(reward)
    return float(daily.mean())


def max_daily_drawdown(reward: pd.Series) -> float:
    daily = daily_pnl(reward)
    cum = daily.cumsum()
    dd = cum - cum.cummax()
    return float(dd.min())


def compute_strategy_metrics(pnl: pd.Series) -> dict[str, float]:
    return {
        "cum_pnl_$": float(pnl.dropna().sum()),
        "mean_daily_pnl_$": mean_daily_pnl(pnl),
        "daily_sharpe": daily_sharpe(pnl),
        "max_drawdown_daily_$": max_daily_drawdown(pnl),
    }
