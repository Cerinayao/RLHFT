from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rlhft.config import PipelineConfig
from rlhft.data.kdb import KDBConnection
from rlhft.data.loaders import (
    query_active_symbols,
    query_midprices,
    to_pandas,
    scale_for_sym,
)
from rlhft.features.pca_signal import compute_ewm_pca_signal, attach_datetime_from_ns_index
from rlhft.visualization.price_plots import plot_asset


def run_oneday_pipeline(
    conn: KDBConnection,
    cfg: PipelineConfig,
    date_quotes: str | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end single-day pipeline:
      1) query active symbols
      2) pull 1-min midprices
      3) apply symbol-specific price scaling
      4) compute EWM PCA signal
      5) attach datetime index and plot
    """
    if date_quotes is None:
        date_quotes = cfg.data.date_active

    # 1) Active symbols
    sym_active = query_active_symbols(conn, cfg.data)

    # 2) Pull midprices
    df_midprice_all = None
    for sym in sym_active:
        df_sym = query_midprices(conn, sym, date_quotes, cfg.data.time_grid_q)
        df_midprice_all = (
            df_sym if df_midprice_all is None
            else df_midprice_all.merge(df_sym, on="time", how="left")
        )

    # 3) Scaling
    df_midprice_all = df_midprice_all.copy()
    for sym in sym_active:
        if sym in df_midprice_all.columns:
            s = scale_for_sym(sym, cfg.scaling.scales)
            df_midprice_all[sym] = df_midprice_all[sym].astype(float) / s
            df_midprice_all[sym] = df_midprice_all[sym].where(df_midprice_all[sym] > 0.0, np.nan)

    # 4) Optional scatter plot
    if cfg.make_plots:
        required_vis = cfg.data.preferred_symbols
        if set(required_vis).issubset(df_midprice_all.columns) and len(required_vis) == 2:
            df_sc = df_midprice_all[["time"] + required_vis].copy().dropna()
            time_labels = (
                pd.Timestamp("2000-01-01")
                + pd.to_timedelta(df_sc["time"].astype("int64"), unit="ns")
            ).dt.strftime("%H:%M")
            cvals = np.arange(len(df_sc))

            plt.figure(figsize=(8, 6))
            plt.plot(df_sc[required_vis[0]], df_sc[required_vis[1]], alpha=0.35, linewidth=0.4)
            plt.scatter(df_sc[required_vis[0]], df_sc[required_vis[1]], c=cvals, cmap="viridis", alpha=0.75, s=18)
            cb = plt.colorbar()
            tick_idx = np.linspace(0, len(df_sc) - 1, min(6, len(df_sc)), dtype=int)
            cb.set_ticks(tick_idx)
            cb.set_ticklabels(time_labels.iloc[tick_idx])
            cb.set_label("Time")
            plt.grid(which="minor", visible=False)
            plt.title(f"{required_vis[0]} vs {required_vis[1]}", fontsize=18, fontweight="bold", color="darkorange")
            plt.xlabel(required_vis[0])
            plt.ylabel(required_vis[1])
            plt.tight_layout()
            plt.show()

    # 5) Compute signal
    required_cols = cfg.data.preferred_symbols
    if not set(required_cols).issubset(df_midprice_all.columns):
        raise ValueError(f"Missing required columns: {required_cols}")

    data_one = df_midprice_all[["time"] + required_cols].copy().set_index("time")
    df_oneday = compute_ewm_pca_signal(data_one, cfg.signal)

    # 6) Attach datetime + plot
    date_anchor = date_quotes.replace(".", "-")
    df_dt = attach_datetime_from_ns_index(df_oneday, date_anchor)

    if cfg.make_plots:
        plot_asset(df_dt, required_cols, ["darkorange", "navy"])

    return sym_active, df_midprice_all, df_dt, df_oneday
