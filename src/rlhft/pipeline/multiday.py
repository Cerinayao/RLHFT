from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from rlhft.config import PipelineConfig
from rlhft.data.kdb import KDBConnection
from rlhft.data.loaders import query_active_symbols, load_multi_day
from rlhft.features.pca_signal import compute_ewm_pca_signal
from rlhft.visualization.price_plots import plot_asset_trading_time


def run_multiday_pipeline(
    conn: KDBConnection,
    cfg: PipelineConfig,
    *,
    render_plots: bool = True,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """
    Multi-day pipeline:
      1) find active symbols once
      2) pull midprices for all dates
      3) scale by instrument family
      4) optional scatter plot
      5) compute EWM PCA signal
      6) plot trading-time series
    """
    # 1) Active symbols
    sym_active = query_active_symbols(conn, cfg.data)

    # 2-3) Load and scale multi-day data
    df_mid_all = load_multi_day(conn, sym_active, cfg.data, cfg.scaling)

    # Infer required columns
    required_cols = [c for c in cfg.data.preferred_symbols if c in df_mid_all.columns]
    if len(required_cols) < 2:
        raise ValueError(f"Need at least 2 preferred symbols in data, got: {required_cols}")

    print("Required columns used for signal:", required_cols)

    # 4) Optional scatter plot
    if render_plots and cfg.make_plots and len(required_cols) == 2:
        col_x, col_y = required_cols
        if {col_x, col_y}.issubset(df_mid_all.columns):
            df_sc = df_mid_all[["datetime", col_x, col_y]].dropna()
            cvals = mdates.date2num(df_sc["datetime"])
            tick_idx = np.linspace(0, len(df_sc) - 1, min(6, len(df_sc)), dtype=int)
            tick_vals = cvals[tick_idx]
            tick_labels = df_sc["datetime"].dt.strftime("%Y-%m-%d").iloc[tick_idx]

            plt.figure(figsize=(8, 6))
            plt.plot(df_sc[col_x], df_sc[col_y], alpha=0.35, linewidth=0.4)
            plt.scatter(df_sc[col_x], df_sc[col_y], c=cvals, cmap="viridis", alpha=0.75, s=18)
            cb = plt.colorbar()
            cb.set_ticks(tick_vals)
            cb.set_ticklabels(tick_labels)
            cb.set_label("Date")
            plt.grid(which="minor", visible=False)
            plt.title(f"{col_x} vs {col_y} (All Dates)", fontsize=18, fontweight="bold", color="darkorange")
            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.tight_layout()
            plt.show()

    # 5) Compute signal
    prices = df_mid_all[["datetime"] + required_cols].copy()
    prices = prices.set_index("datetime")
    prices = prices.dropna(subset=required_cols, how="any")

    if prices.empty:
        raise ValueError("After dropping missing prices, no rows remain for signal construction.")

    df_state_all = compute_ewm_pca_signal(prices, cfg.signal)

    # Filter rows where required columns are non-finite or zero (notebook cell 8)
    finite_mask = np.isfinite(df_state_all[required_cols]).all(axis=1)
    nonzero_mask = (df_state_all[required_cols] != 0).all(axis=1)
    df_state_all = df_state_all[finite_mask & nonzero_mask].copy()

    # 6) Plot
    if render_plots and cfg.make_plots and len(required_cols) == 2:
        plot_asset_trading_time(df_state_all, required_cols, ["darkorange", "navy"])

    return sym_active, df_mid_all, df_state_all
