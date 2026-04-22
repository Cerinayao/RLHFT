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


def compute_strategy_metrics(pnl: pd.Series) -> dict[str, float]:
    return {
        "mean_daily_pnl_$": mean_daily_pnl(pnl),
        "max_drawdown_daily_$": max_daily_drawdown(pnl),
    }
