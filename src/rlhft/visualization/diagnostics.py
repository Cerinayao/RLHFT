from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def plot_pc_and_acf_trading_hours(
    df: pd.DataFrame,
    pc1_col: str = "pc1",
    pc2_col: str = "pc2",
    nlags: int = 30,
    acf_ylim: tuple = (-0.10, 0.10),
    start_time: str = "00:00",
    end_time: str = "16:00",
    show: bool = True,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot PCA components and ACF of their changes during trading hours."""
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    out = out.between_time(start_time, end_time).copy()

    pc1 = out[pc1_col].astype(float)
    pc2 = out[pc2_col].astype(float)

    dpc1 = pc1.diff().dropna()
    dpc2 = pc2.diff().dropna()

    x = np.arange(len(out))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor="white")
    fig.suptitle("PCA Components and ACF of Changes (Trading Hours, Continuous)", fontsize=22, y=0.98)

    axes[0, 0].plot(x, pc1.values, linewidth=1.2, color="tab:blue")
    axes[0, 0].axhline(0, linestyle="--", linewidth=1.0, color="tab:blue", alpha=0.8)
    axes[0, 0].set_title("PC1", fontsize=16)

    axes[0, 1].plot(x, pc2.values, linewidth=1.2, color="tab:blue")
    axes[0, 1].axhline(0, linestyle="--", linewidth=1.0, color="tab:blue", alpha=0.8)
    axes[0, 1].set_title("PC2", fontsize=16)

    day_vals = out.index.normalize().values
    day_start_pos = np.r_[0, np.flatnonzero(day_vals[1:] != day_vals[:-1]) + 1]
    day_start_labels = [out.index[i].strftime("%m-%d") for i in day_start_pos]

    for ax in axes[0]:
        ax.set_xticks(day_start_pos)
        ax.set_xticklabels(day_start_labels, rotation=45, ha="right")
        ax.set_xlabel("Trading Time (00:00-16:00 each day)", fontsize=12)

    plot_acf(dpc1, lags=nlags, ax=axes[1, 0], zero=False)
    axes[1, 0].set_title(r"ACF of $\Delta PC1$", fontsize=16)
    axes[1, 0].set_ylim(acf_ylim)

    plot_acf(dpc2, lags=nlags, ax=axes[1, 1], zero=False)
    axes[1, 1].set_title(r"ACF of $\Delta PC2$", fontsize=16)
    axes[1, 1].set_ylim(acf_ylim)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    if show:
        plt.show()

    result_df = pd.DataFrame(
        {"PC1": pc1, "PC2": pc2, "dPC1": pc1.diff(), "dPC2": pc2.diff()},
        index=out.index,
    )
    return fig, result_df
