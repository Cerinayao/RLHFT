from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(
    df: pd.DataFrame,
    title: str,
    show: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of feature importance."""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.barh(df["feature"], df["importance"])
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance (Gain)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_xgb_vs_rl_pnl(
    *,
    xgb_bt: pd.DataFrame,
    out_rl: dict,
    trading_start: str = "00:00",
    trading_end: str = "16:00",
    title: str = "Test Cumulative PnL: RL vs XGB Inventory Policy",
    show: bool = True,
) -> tuple[plt.Figure, pd.Series, pd.Series]:
    """Compare XGB inventory backtest cumulative PnL vs RL test cumulative PnL."""
    xgb_pnl = xgb_bt["pnl"].between_time(trading_start, trading_end).dropna()
    rl_pnl = out_rl["test_pnl"].reindex(xgb_pnl.index).dropna()

    common_idx = xgb_pnl.index.intersection(rl_pnl.index)
    xgb_pnl = xgb_pnl.loc[common_idx]
    rl_pnl = rl_pnl.loc[common_idx]

    xgb_cum = xgb_pnl.cumsum()
    rl_cum = rl_pnl.cumsum()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    x = np.arange(len(common_idx))
    ax.plot(x, xgb_cum.values, label="XGB Inventory Policy", linewidth=2)
    ax.plot(x, rl_cum.values, label="RL", linewidth=2)

    tick_pos: list[int] = []
    tick_labels: list[str] = []
    for date, idx_day in pd.Series(x, index=common_idx).groupby(common_idx.date):
        tick_pos.append(idx_day.iloc[0])
        tick_labels.append(str(date))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Trading Time (Compressed)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, xgb_cum, rl_cum
