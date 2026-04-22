from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_trading_time_cum(
    cum: pd.Series,
    title: str,
    show: bool = True,
) -> plt.Figure | None:
    """Plot cumulative reward with trading-time x-axis."""
    cum = cum.dropna()
    if cum.empty:
        return None

    x = np.arange(len(cum))
    dates = cum.index.normalize()
    unique_dates, first_pos = np.unique(dates.values, return_index=True)
    unique_dates = pd.to_datetime(unique_dates)

    fig = plt.figure(figsize=(11, 6), facecolor="white")
    fig.text(0.05, 0.92, title, fontsize=24, color="#d55e00")
    ax = fig.add_axes([0.07, 0.12, 0.88, 0.74])

    ax.plot(x, cum.values, linewidth=2.6, color="black", alpha=0.95)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.set_ylabel("Cumulative Reward ($)", fontsize=12)
    ax.set_xticks(first_pos)
    ax.set_xticklabels([d.strftime("%m-%d") for d in unique_dates], rotation=45)

    ax.locator_params(axis="y", nbins=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.tick_params(axis="y", labelleft=True, labelsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if show:
        plt.show()
    return fig


def plot_rl_vs_rule_comparison(
    out_rl: dict,
    out_rule: dict,
    train_end: str,
    show: bool = True,
) -> plt.Figure:
    """Plot RL vs Rule cumulative reward on the test period."""
    rule_cum = out_rule["cum"]
    rl_cum = out_rl["test_cum_pnl"]

    rule_test = rule_cum[rule_cum.index > pd.Timestamp(train_end)].copy()
    rule_test = rule_test - rule_test.iloc[0]
    rl_test = rl_cum - rl_cum.iloc[0]

    aligned = pd.concat(
        [rule_test.rename("Rule"), rl_test.rename("RL")], axis=1
    ).dropna(how="all")

    x = np.arange(len(aligned))
    dates = aligned.index.normalize()
    unique_dates, first_pos = np.unique(dates.values, return_index=True)
    unique_dates = pd.to_datetime(unique_dates)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, aligned["Rule"].values, label="Rule", linewidth=2)
    ax.plot(x, aligned["RL"].values, label="RL", linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(first_pos)
    ax.set_xticklabels([d.strftime("%m-%d") for d in unique_dates], rotation=45)
    ax.set_xlabel("Trading Time (00:00-16:00 each day)")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("ES/NQ - Rule vs RL cumulative reward (test period)")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_signal_horizon_sweep(
    horizon_df: pd.DataFrame,
    title: str,
    show: bool = True,
) -> plt.Figure | None:
    """Plot signal/forward-return correlation over candidate horizons."""
    if horizon_df.empty:
        return None

    plot_df = horizon_df.dropna(subset=["horizon", "correlation"]).copy()
    if plot_df.empty:
        return None

    best_idx = plot_df["correlation"].idxmax()
    best_h = int(plot_df.loc[best_idx, "horizon"])

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(plot_df["horizon"], plot_df["correlation"], linewidth=2)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(best_h, linestyle="--", linewidth=1, color="red", label=f"h*={best_h}")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("corr(signal, fwd return)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_action_vs_fwd_return(
    grouped: pd.DataFrame,
    aligned: pd.DataFrame,
    horizon: int,
    price_col: str,
    spearman_rho: float,
    show: bool = True,
) -> plt.Figure | None:
    """Plot mean forward return by action bucket and a jittered scatter."""
    if grouped.empty or aligned.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    yerr = grouped["std"] / np.sqrt(grouped["count"])
    axes[0].bar(
        grouped.index.astype(int),
        grouped["mean"],
        yerr=yerr,
        capsize=4,
        color="steelblue",
        edgecolor="black",
    )
    axes[0].axhline(0, linestyle="--", linewidth=1, color="k")
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel(f"Mean forward return (h={horizon})")
    axes[0].set_title(f"{price_col}: Action vs mean forward return")

    jitter = (np.random.rand(len(aligned)) - 0.5) * 0.25
    axes[1].scatter(
        aligned["action"] + jitter,
        aligned["fwd_ret"],
        s=6,
        alpha=0.3,
    )
    axes[1].axhline(0, linestyle="--", linewidth=1, color="k")
    axes[1].set_xlabel("Action (jittered)")
    axes[1].set_ylabel(f"Forward return (h={horizon})")
    axes[1].set_title(f"{price_col}: Action vs fwd return (Spearman rho={spearman_rho:.3f})")

    fig.tight_layout()
    if show:
        plt.show()
    return fig
