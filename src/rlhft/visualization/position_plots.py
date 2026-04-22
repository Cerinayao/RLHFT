from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _scatter_positions_with_overlap_control(
    ax,
    df_day: pd.DataFrame,
    asset_specs: list[tuple[str, str, str]],
    *,
    marker: str = "o",
    dot_size: int = 18,
    alpha: float = 0.85,
    jitter: float = 0.06,
    rng=None,
) -> list:
    """Plot one day of positions with overlap control via jitter and random order."""
    if rng is None:
        rng = np.random.default_rng(0)

    for ts, row in df_day.iterrows():
        order = rng.permutation(len(asset_specs))
        for idx in order:
            col_name, label, color = asset_specs[idx]
            y = float(row[col_name])
            y_plot = y + rng.uniform(-jitter, jitter)
            ax.scatter(ts, y_plot, s=dot_size, alpha=alpha, marker=marker, color=color)

    handles = []
    for _, label, color in asset_specs:
        h = ax.scatter([], [], s=dot_size, color=color, alpha=alpha, marker=marker, label=label)
        handles.append(h)
    return handles


def plot_rl_vs_rule_side_by_side_2asset(
    out_rl: dict,
    out_rule: dict,
    *,
    col_a: str,
    col_b: str,
    leg: str = "test",
    trading_start: str = "00:00",
    trading_end: str = "16:00",
    dot_size: int = 18,
    alpha: float = 0.85,
    jitter: float = 0.06,
    seed: int = 0,
    show: bool = True,
) -> list[plt.Figure]:
    """Compare RL vs Rule positions side by side for each day."""
    rng = np.random.default_rng(seed)

    n_a_col = f"n_{col_a}"
    n_b_col = f"n_{col_b}"

    rl_key = f"{leg}_n"
    if rl_key not in out_rl:
        raise ValueError(f"Missing key '{rl_key}' in out_rl.")
    if "n" not in out_rule:
        raise ValueError("Missing key 'n' in out_rule.")

    df_rl = out_rl[rl_key][[n_a_col, n_b_col]].dropna().copy()
    df_rule = out_rule["n"][[n_a_col, n_b_col]].dropna().copy()

    t0 = pd.to_datetime(trading_start).time()
    t1 = pd.to_datetime(trading_end).time()

    df_rl = df_rl[(df_rl.index.time >= t0) & (df_rl.index.time <= t1)]
    df_rule = df_rule[(df_rule.index.time >= t0) & (df_rule.index.time <= t1)]

    common_days = sorted(
        set(df_rl.index.normalize().unique()).intersection(
            set(df_rule.index.normalize().unique())
        )
    )

    if not common_days:
        print("No common days to plot.")
        return []

    asset_specs = [
        (n_a_col, col_a, "tab:blue"),
        (n_b_col, col_b, "tab:orange"),
    ]

    figs = []
    for day in common_days:
        g_rl = df_rl[df_rl.index.normalize() == day]
        g_rule = df_rule[df_rule.index.normalize() == day]

        if g_rl.empty and g_rule.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, facecolor="white")
        fig.suptitle(
            f"RL vs Rule Positions - {pd.to_datetime(day).strftime('%Y-%m-%d')}",
            fontsize=18, y=0.98,
        )

        # RL panel
        ax = axes[0]
        handles = _scatter_positions_with_overlap_control(
            ax, g_rl, asset_specs, marker="o", dot_size=dot_size, alpha=alpha, jitter=jitter, rng=rng,
        )
        ax.set_title("RL", fontsize=14)
        ax.axhline(0, linewidth=1, alpha=0.3)
        ax.set_ylabel("Contracts")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=handles, frameon=False)

        # Rule panel
        ax = axes[1]
        handles = _scatter_positions_with_overlap_control(
            ax, g_rule, asset_specs, marker="x", dot_size=dot_size, alpha=alpha, jitter=jitter, rng=rng,
        )
        ax.set_title("Rule", fontsize=14)
        ax.axhline(0, linewidth=1, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=handles, frameon=False)

        plt.tight_layout()
        if show:
            plt.show()
        figs.append(fig)

    return figs
