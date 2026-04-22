from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm


def _filter_trading_hours(
    df: pd.DataFrame,
    *,
    start_time: str = "00:00",
    end_time: str = "16:00",
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    return df.between_time(start_time, end_time).copy()


def plot_multiday_scatter(
    df_mid_all: pd.DataFrame,
    col_x: str,
    col_y: str,
    trading_start: str = "00:00",
    trading_end: str = "16:00",
    show: bool = True,
) -> plt.Figure | None:
    """Scatter plot over all dates, matching the notebook/multiday pipeline style."""
    needed = ["datetime", col_x, col_y]
    if not set(needed).issubset(df_mid_all.columns):
        return None

    df_sc = df_mid_all[needed].dropna().copy()
    df_sc = df_sc.set_index("datetime").between_time(trading_start, trading_end).reset_index()
    if df_sc.empty:
        return None

    cvals = mdates.date2num(df_sc["datetime"])
    tick_idx = np.linspace(0, len(df_sc) - 1, min(6, len(df_sc)), dtype=int)
    tick_vals = cvals[tick_idx]
    tick_labels = df_sc["datetime"].dt.strftime("%Y-%m-%d").iloc[tick_idx]

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = fig.add_subplot(111)
    ax.plot(df_sc[col_x], df_sc[col_y], alpha=0.35, linewidth=0.4)
    sc = ax.scatter(df_sc[col_x], df_sc[col_y], c=cvals, cmap="viridis", alpha=0.75, s=18)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_ticks(tick_vals)
    cb.set_ticklabels(tick_labels)
    cb.set_label("Date")
    ax.grid(which="minor", visible=False)
    ax.set_title(f"{col_x} vs {col_y} (All Dates)", fontsize=18, fontweight="bold", color="darkorange")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_asset(
    df: pd.DataFrame,
    tickers: list[str],
    colors: list[str],
    show: bool = True,
) -> plt.Figure:
    """Plot normalized prices + forecasts and OLS residuals for a single day."""
    if len(tickers) != len(colors):
        raise ValueError("tickers and colors must have the same length")
    if len(tickers) != 2:
        raise ValueError("plot_asset currently expects exactly two tickers")

    t = df.index

    fig = plt.figure(figsize=(11, 9), facecolor="white")
    fig.text(0.05, 0.92, "Normalized time series and forecasts", fontsize=28, color="#d55e00")

    ax = fig.add_axes([0.07, 0.48, 0.88, 0.38])

    for ticker, color in zip(tickers, colors):
        p = df[ticker].astype(float)
        z_hat = (df[ticker] + df[f"s_{ticker}"]).astype(float)

        p0 = p.dropna().iloc[0]
        z0 = z_hat.dropna().iloc[0]
        p_norm = p / p0
        z_hat_norm = z_hat / z0

        ax.step(t, p_norm, where="post", linewidth=2.4, color=color, label=f"{ticker} (normalized price)")
        ax.step(t, z_hat_norm, where="post", linewidth=1.6, color=color, alpha=0.35, label=f"{ticker} forecast")

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.tick_params(axis="x", which="major", length=10, width=1.2, direction="out", labelsize=12)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax.text(0.98, 0.98, f"{t.min().date()}", transform=ax.transAxes, ha="right", va="top", fontsize=14)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=1.0, fancybox=False, borderpad=0.8)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)

    # OLS residuals
    x_ticker, y_ticker = tickers[0], tickers[1]
    reg_df = df[[x_ticker, y_ticker]].astype(float).dropna()
    x = reg_df[x_ticker].values
    y = reg_df[y_ticker].values
    ols = sm.OLS(y, x).fit()
    beta = float(np.asarray(ols.params).squeeze())
    epsilon = reg_df[y_ticker] - beta * reg_df[x_ticker]

    ax_res = fig.add_axes([0.07, 0.12, 0.88, 0.22])
    ax_res.plot(epsilon.index, epsilon, color="black", linewidth=1.6, label=fr"$\epsilon_t = {y_ticker} - {beta:.4f}{x_ticker}$")
    ax_res.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    for sp in ["top", "right"]:
        ax_res.spines[sp].set_visible(False)

    ax_res.tick_params(axis="x", which="major", length=10, width=1.2, direction="out", labelsize=12)
    ax_res.tick_params(axis="y", which="major", length=6, width=1.0, direction="out", labelsize=11)
    ax_res.xaxis.set_major_locator(mdates.HourLocator())
    ax_res.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_res.set_title(f"OLS residuals for {y_ticker} = B x {x_ticker} + epsilon", fontsize=14)
    ax_res.set_ylabel("epsilon")
    ax_res.legend(loc="upper right", frameon=True, framealpha=1.0, fancybox=False)

    if show:
        plt.show()
    return fig


def plot_asset_trading_time(
    df: pd.DataFrame,
    tickers: list[str],
    colors: list[str],
    trading_start: str = "00:00",
    trading_end: str = "16:00",
    show: bool = True,
) -> plt.Figure:
    """Multi-day normalized price plot with trading-time x-axis and OLS residuals."""
    if len(tickers) != len(colors):
        raise ValueError("tickers and colors must have the same length")
    if len(tickers) != 2:
        raise ValueError("plot_asset_trading_time currently expects exactly two tickers")

    df = _filter_trading_hours(df.sort_index(), start_time=trading_start, end_time=trading_end)
    x = np.arange(len(df))

    dates = df.index.normalize()
    unique_dates, date_positions = np.unique(dates, return_index=True)
    unique_dates = pd.to_datetime(unique_dates)

    fig = plt.figure(figsize=(11, 9), facecolor="white")
    fig.text(0.05, 0.92, "Normalized time series and forecasts", fontsize=28, color="#d55e00")

    ax = fig.add_axes([0.07, 0.48, 0.88, 0.38])

    for ticker, color in zip(tickers, colors):
        p = df[ticker].astype(float)
        z_hat = (df[ticker] + df[f"s_{ticker}"]).astype(float)

        p0 = p.dropna().iloc[0]
        z0 = z_hat.dropna().iloc[0]
        p_norm = (p / p0).values
        z_hat_norm = (z_hat / z0).values

        ax.step(x, p_norm, where="post", linewidth=2.4, color=color, label=f"{ticker} (normalized price)")
        ax.step(x, z_hat_norm, where="post", linewidth=1.6, color=color, alpha=0.35, label=f"{ticker} forecast")

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.tick_params(axis="x", which="major", length=10, width=1.2, direction="out", labelsize=12)
    ax.set_xticks(date_positions)
    ax.set_xticklabels([d.strftime("%m-%d") for d in unique_dates], rotation=45)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=1.0, fancybox=False, borderpad=0.8)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)

    ax.text(
        0.98, 0.98, f"{df.index.min().date()} to {df.index.max().date()}",
        transform=ax.transAxes, ha="right", va="top", fontsize=14,
    )

    # OLS residuals
    x_ticker, y_ticker = tickers[0], tickers[1]
    reg_df = df[[x_ticker, y_ticker]].astype(float).dropna()
    x_reg = reg_df[x_ticker].values
    y_reg = reg_df[y_ticker].values
    ols = sm.OLS(y_reg, x_reg).fit()
    beta = float(np.asarray(ols.params).squeeze())
    epsilon = reg_df[y_ticker] - beta * reg_df[x_ticker]

    epsilon_x = np.arange(len(epsilon))
    epsilon_dates = epsilon.index.normalize()
    epsilon_unique_dates, epsilon_date_positions = np.unique(epsilon_dates, return_index=True)
    epsilon_unique_dates = pd.to_datetime(epsilon_unique_dates)

    ax_res = fig.add_axes([0.07, 0.12, 0.88, 0.22])
    ax_res.plot(epsilon_x, epsilon.values, color="black", linewidth=1.6,
                label=fr"$\epsilon_t = {y_ticker} - {beta:.4f}{x_ticker}$")
    ax_res.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    for sp in ["top", "right"]:
        ax_res.spines[sp].set_visible(False)

    ax_res.tick_params(axis="x", which="major", length=10, width=1.2, direction="out", labelsize=12)
    ax_res.tick_params(axis="y", which="major", length=6, width=1.0, direction="out", labelsize=11)
    ax_res.set_xticks(epsilon_date_positions)
    ax_res.set_xticklabels([d.strftime("%m-%d") for d in epsilon_unique_dates], rotation=45)
    ax_res.set_title(f"OLS residuals for {y_ticker} = B x {x_ticker} + epsilon", fontsize=14)
    ax_res.set_ylabel("epsilon")
    ax_res.legend(loc="upper right", frameon=True, framealpha=1.0, fancybox=False)

    if show:
        plt.show()
    return fig
