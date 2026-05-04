from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

from rlhft.config import PipelineConfig


def _series_df(series: pd.Series, value_name: str) -> pd.DataFrame:
    s = series.dropna().copy()
    return pd.DataFrame({
        "datetime": s.index,
        value_name: s.values,
    })


def _trading_time_axis(index: pd.Index) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dt_index = pd.DatetimeIndex(index)
    x = np.arange(len(dt_index))
    dates = dt_index.normalize()
    unique_dates, first_pos = np.unique(dates.values, return_index=True)
    labels = [pd.Timestamp(d).strftime("%m-%d") for d in unique_dates]
    return x, first_pos, labels


def _apply_notebook_style(fig: go.Figure, *, height: int | None = None) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        font=dict(family="Arial, sans-serif", color="black"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="outside")
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="outside")
    if height is not None:
        fig.update_layout(height=height)


def _table_figure(df: pd.DataFrame, title: str) -> go.Figure:
    table_df = df.reset_index().copy()
    table_df.columns = [str(c) for c in table_df.columns]
    for col in table_df.columns:
        if pd.api.types.is_numeric_dtype(table_df[col]):
            table_df[col] = table_df[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )
        else:
            table_df[col] = table_df[col].fillna("").astype(str)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_df.columns),
                    fill_color="#13315c",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[table_df[col].tolist() for col in table_df.columns],
                    fill_color="#f5f7fb",
                    align="left",
                    height=28,
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=50, b=20),
        height=max(220, 80 + 28 * len(table_df)),
    )
    return fig


def _mpl_figure_to_img_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f'<img alt="plot" src="data:image/png;base64,{encoded}" style="width:100%;height:auto;display:block;border-radius:12px;" />'


def _png_path_to_img_html(path) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img alt="plot" src="data:image/png;base64,{encoded}" style="width:100%;height:auto;display:block;border-radius:12px;" />'


def _render_zoomable_png_section(heading: str, path: str | Path) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    body = (
        '<div class="scroll-frame">'
        f'<img alt="{heading}" src="data:image/png;base64,{encoded}" '
        'style="width:auto;max-width:none;height:auto;display:block;border-radius:12px;" />'
        "</div>"
    )
    return f"<section><h2>{heading}</h2>{body}</section>"


def _render_section(heading: str, fig) -> str:
    if isinstance(fig, (str, Path)) and "tree" in heading.lower():
        return _render_zoomable_png_section(heading, fig)
    if isinstance(fig, go.Figure):
        body = to_html(fig, full_html=False, include_plotlyjs=False)
    elif isinstance(fig, (str, Path)):
        body = _png_path_to_img_html(fig)
    else:
        body = _mpl_figure_to_img_html(fig)
    return f"<section><h2>{heading}</h2>{body}</section>"


def _line_figure(
    traces: list[tuple[pd.Series, str, str]],
    title: str,
    yaxis_title: str,
) -> go.Figure:
    fig = go.Figure()
    x_ticks = None
    x_labels = None
    for series, name, color in traces:
        df = _series_df(series, "value")
        x_vals, tick_pos, tick_labels = _trading_time_axis(df["datetime"])
        x_ticks = tick_pos
        x_labels = tick_labels
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df["value"],
                mode="lines",
                name=name,
                line=dict(color=color, width=2.6),
            )
        )
    fig.update_layout(title=dict(text=title, font=dict(size=24, color="#d55e00")), yaxis_title=yaxis_title)
    if x_ticks is not None and x_labels is not None:
        fig.update_xaxes(tickmode="array", tickvals=x_ticks, ticktext=x_labels, tickangle=45)
    fig.update_yaxes(tickprefix="$")
    _apply_notebook_style(fig, height=520)
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        griddash="dot",
    )
    return fig


def _comparison_figure(out_rl: dict, out_rule: dict, train_end: str) -> go.Figure | None:
    rule_cum = out_rule["cum"]
    rl_cum = out_rl["test_cum_pnl"]

    if rule_cum.empty or rl_cum.empty:
        return None

    rule_test = rule_cum[rule_cum.index > pd.Timestamp(train_end)].copy()
    rl_test = rl_cum.copy()
    if rule_test.empty or rl_test.empty:
        return None

    rule_test = rule_test - rule_test.iloc[0]
    rl_test = rl_test - rl_test.iloc[0]
    aligned = pd.concat(
        [rule_test.rename("Rule"), rl_test.rename("RL")],
        axis=1,
    ).dropna(how="all")
    if aligned.empty:
        return None

    x_vals, tick_pos, tick_labels = _trading_time_axis(aligned.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=aligned["Rule"], mode="lines", name="Rule", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x_vals, y=aligned["RL"], mode="lines", name="RL", line=dict(width=2)))
    fig.add_hline(y=0, line_dash="dash", line_width=1)
    fig.update_layout(
        title="Rule vs RL cumulative reward (test period)",
        xaxis_title="Trading Time (00:00-16:00 each day)",
        yaxis_title="Cumulative Reward",
    )
    fig.update_xaxes(tickmode="array", tickvals=tick_pos, ticktext=tick_labels, tickangle=45)
    _apply_notebook_style(fig, height=460)
    return fig


def _scatter_figure(
    df_mid_all: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> go.Figure | None:
    cols = ["datetime", col_a, col_b]
    if not set(cols).issubset(df_mid_all.columns):
        return None

    plot_df = df_mid_all[cols].dropna().copy()
    if plot_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[col_a],
            y=plot_df[col_b],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.25)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df[col_a],
            y=plot_df[col_b],
            mode="markers",
            marker=dict(
                size=6,
                opacity=0.75,
                color=plot_df["datetime"].astype("int64") / 1e9,
                colorscale="Viridis",
                colorbar=dict(title="Date"),
            ),
            text=plot_df["datetime"].dt.strftime("%Y-%m-%d %H:%M"),
            hovertemplate=f"{col_a}: %{{x:.4f}}<br>{col_b}: %{{y:.4f}}<br>%{{text}}<extra></extra>",
            name="Observations",
        )
    )
    fig.update_layout(title=f"{col_a} vs {col_b} (All Dates)", xaxis_title=col_a, yaxis_title=col_b)
    _apply_notebook_style(fig, height=520)
    return fig


def _horizon_figure(horizon_df: pd.DataFrame) -> go.Figure | None:
    plot_df = horizon_df.dropna(subset=["horizon", "correlation"]).copy()
    if plot_df.empty:
        return None

    best_idx = plot_df["correlation"].idxmax()
    best_h = int(plot_df.loc[best_idx, "horizon"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["horizon"],
            y=plot_df["correlation"],
            mode="lines+markers",
            name="corr(signal, fwd return)",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_width=1)
    fig.add_vline(x=best_h, line_dash="dash", line_color="red")
    title = f"Signal horizon sweep (best h = {best_h})"
    fig.update_layout(title=title, xaxis_title="Horizon", yaxis_title="Correlation")
    _apply_notebook_style(fig, height=420)
    return fig


def _action_figure(
    action_summary: pd.DataFrame | None,
    action_aligned: pd.DataFrame | None,
    horizon: int | None,
    price_col: str,
    spearman_rho: float | None,
) -> go.Figure | None:
    if action_summary is None or action_aligned is None:
        return None
    if action_summary.empty or action_aligned.empty or horizon is None:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{price_col}: mean forward return by action",
            f"{price_col}: action vs forward return",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=action_summary.index.astype(int),
            y=action_summary["mean"],
            error_y=dict(
                type="data",
                array=(action_summary["std"] / action_summary["count"].pow(0.5)).fillna(0.0),
            ),
            marker_color="#4682b4",
            name="Mean forward return",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=action_aligned["action"],
            y=action_aligned["fwd_ret"],
            mode="markers",
            marker=dict(size=7, opacity=0.35, color="#1f1f1f"),
            name="Observations",
        ),
        row=1,
        col=2,
    )
    title = f"Action analysis at h = {horizon}"
    if spearman_rho is not None and pd.notna(spearman_rho):
        title += f" (Spearman rho = {spearman_rho:.3f})"
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(title_text="Action", row=1, col=1)
    fig.update_yaxes(title_text="Mean forward return", row=1, col=1)
    fig.update_xaxes(title_text="Action", row=1, col=2)
    fig.update_yaxes(title_text="Forward return", row=1, col=2)
    _apply_notebook_style(fig, height=460)
    return fig


def _positions_figure(
    out_rl: dict,
    out_rule: dict,
    col_a: str,
    col_b: str,
    leg: str = "test",
) -> go.Figure | None:
    rl_key = f"{leg}_n"
    if rl_key not in out_rl or "n" not in out_rule:
        return None

    rl_df = out_rl[rl_key][[f"n_{col_a}", f"n_{col_b}"]].copy()
    rule_df = out_rule["n"][[f"n_{col_a}", f"n_{col_b}"]].copy()
    aligned = pd.concat(
        [
            rl_df.rename(columns={f"n_{col_a}": f"RL {col_a}", f"n_{col_b}": f"RL {col_b}"}),
            rule_df.rename(columns={f"n_{col_a}": f"Rule {col_a}", f"n_{col_b}": f"Rule {col_b}"}),
        ],
        axis=1,
    ).dropna(how="all")
    if aligned.empty:
        return None

    x_vals, tick_pos, tick_labels = _trading_time_axis(aligned.index)
    fig = go.Figure()
    colors = {
        f"RL {col_a}": "#1f77b4",
        f"RL {col_b}": "#ff7f0e",
        f"Rule {col_a}": "#2a9d8f",
        f"Rule {col_b}": "#e76f51",
    }
    for col in aligned.columns:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=aligned[col],
                mode="lines",
                name=col,
                line=dict(width=2, color=colors.get(col)),
            )
        )
    fig.update_layout(
        title=f"Inventory paths ({leg})",
        xaxis_title="Trading Time",
        yaxis_title="Contracts",
    )
    fig.update_xaxes(tickmode="array", tickvals=tick_pos, ticktext=tick_labels, tickangle=45)
    _apply_notebook_style(fig, height=460)
    return fig


def _positions_side_by_side_figure(
    out_rl: dict,
    out_rule: dict,
    col_a: str,
    col_b: str,
    leg: str = "test",
) -> go.Figure | None:
    rl_key = f"{leg}_n"
    if rl_key not in out_rl or "n" not in out_rule:
        return None

    n_a_col = f"n_{col_a}"
    n_b_col = f"n_{col_b}"
    df_rl = out_rl[rl_key][[n_a_col, n_b_col]].dropna().copy()
    df_rule = out_rule["n"][[n_a_col, n_b_col]].dropna().copy()

    common_days = sorted(
        set(df_rl.index.normalize().unique()).intersection(set(df_rule.index.normalize().unique()))
    )
    if not common_days:
        return None

    fig = make_subplots(
        rows=len(common_days),
        cols=2,
        shared_yaxes=True,
        subplot_titles=[
            title
            for day in common_days
            for title in (
                f"RL - {pd.Timestamp(day).strftime('%Y-%m-%d')}",
                f"Rule - {pd.Timestamp(day).strftime('%Y-%m-%d')}",
            )
        ],
        vertical_spacing=0.05,
    )

    color_map = {n_a_col: "#1f77b4", n_b_col: "#ff7f0e"}
    marker_map = {n_a_col: "circle", n_b_col: "x"}
    for row_idx, day in enumerate(common_days, start=1):
        day_rl = df_rl[df_rl.index.normalize() == day]
        day_rule = df_rule[df_rule.index.normalize() == day]
        for col_idx, day_df in enumerate((day_rl, day_rule), start=1):
            for asset_col in (n_a_col, n_b_col):
                fig.add_trace(
                    go.Scatter(
                        x=day_df.index,
                        y=day_df[asset_col],
                        mode="markers",
                        marker=dict(
                            size=6,
                            opacity=0.8,
                            color=color_map[asset_col],
                            symbol=marker_map[asset_col],
                        ),
                        name=asset_col.replace("n_", ""),
                        showlegend=row_idx == 1 and col_idx == 1,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    fig.update_layout(title=f"RL vs Rule Positions ({leg})")
    fig.update_yaxes(title_text="Contracts")
    _apply_notebook_style(fig, height=max(420, 240 * len(common_days)))
    return fig


def _residual_figure(
    df_state_all: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> go.Figure | None:
    plot_df = df_state_all[[col_a, col_b]].astype(float).dropna().copy()
    if plot_df.empty:
        return None

    x_vals = plot_df[col_a].values
    y_vals = plot_df[col_b].values
    ols = sm.OLS(y_vals, x_vals).fit()
    beta = float(ols.params.squeeze())
    epsilon = plot_df[col_b] - beta * plot_df[col_a]

    x_vals, tick_pos, tick_labels = _trading_time_axis(epsilon.index)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=epsilon.values,
            mode="lines",
            line=dict(color="black", width=2),
            name=f"epsilon = {col_b} - {beta:.4f}{col_a}",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"OLS residuals for {col_b} = B x {col_a} + epsilon", yaxis_title="epsilon")
    fig.update_xaxes(tickmode="array", tickvals=tick_pos, ticktext=tick_labels, tickangle=45)
    _apply_notebook_style(fig, height=420)
    return fig


def _normalized_forecast_figure(
    df_state_all: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> go.Figure | None:
    needed = [col_a, col_b, f"s_{col_a}", f"s_{col_b}"]
    if not set(needed).issubset(df_state_all.columns):
        return None

    df = df_state_all.sort_index().copy()
    plot_df = df[needed].dropna(how="all").copy()
    if plot_df.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08,
        subplot_titles=(
            "Normalized time series and forecasts",
            f"OLS residuals for {col_b} = B x {col_a} + epsilon",
        ),
    )

    x_vals, tick_pos, tick_labels = _trading_time_axis(df.index)
    for ticker, color in ((col_a, "darkorange"), (col_b, "navy")):
        p = df[ticker].astype(float)
        z_hat = (df[ticker] + df[f"s_{ticker}"]).astype(float)
        p0 = p.dropna().iloc[0]
        z0 = z_hat.dropna().iloc[0]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=(p / p0),
                mode="lines",
                line=dict(color=color, width=2.4, shape="hv"),
                name=f"{ticker} normalized price",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=(z_hat / z0),
                mode="lines",
                line=dict(color=color, width=1.6, shape="hv"),
                opacity=0.35,
                name=f"{ticker} forecast",
            ),
            row=1,
            col=1,
        )

    reg_df = df[[col_a, col_b]].astype(float).dropna()
    x_vals = reg_df[col_a].values
    y_vals = reg_df[col_b].values
    ols = sm.OLS(y_vals, x_vals).fit()
    beta = float(ols.params.squeeze())
    epsilon = reg_df[col_b] - beta * reg_df[col_a]
    epsilon_x, epsilon_tick_pos, epsilon_tick_labels = _trading_time_axis(epsilon.index)
    fig.add_trace(
        go.Scatter(
            x=epsilon_x,
            y=epsilon.values,
            mode="lines",
            line=dict(color="black", width=1.8),
            name=f"epsilon = {col_b} - {beta:.4f}{col_a}",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_layout(showlegend=True)
    fig.update_yaxes(title_text="Normalized level", row=1, col=1)
    fig.update_yaxes(title_text="epsilon", row=2, col=1)
    fig.update_xaxes(tickmode="array", tickvals=tick_pos, ticktext=tick_labels, tickangle=45, row=1, col=1)
    fig.update_xaxes(title_text="Trading Time", tickmode="array", tickvals=epsilon_tick_pos, ticktext=epsilon_tick_labels, tickangle=45, row=2, col=1)
    _apply_notebook_style(fig, height=760)
    return fig


def _diagnostics_figure(
    df_state_all: pd.DataFrame,
    nlags: int = 30,
) -> go.Figure | None:
    if not {"pc1", "pc2"}.issubset(df_state_all.columns):
        return None

    out = df_state_all.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return None

    out = out.between_time("00:00", "16:00").copy()
    if out.empty:
        return None

    pc1 = out["pc1"].astype(float)
    pc2 = out["pc2"].astype(float)
    dpc1 = pc1.diff().dropna()
    dpc2 = pc2.diff().dropna()
    if dpc1.empty or dpc2.empty:
        return None

    acf1 = acf(dpc1, nlags=nlags, fft=False, adjusted=False)[1:]
    acf2 = acf(dpc2, nlags=nlags, fft=False, adjusted=False)[1:]
    lags = np.arange(1, len(acf1) + 1)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("PC1", "PC2", "ACF of dPC1", "ACF of dPC2"),
        vertical_spacing=0.1,
    )
    fig.add_trace(go.Scatter(x=out.index, y=pc1, mode="lines", line=dict(color="#1f77b4", width=1.4), name="PC1"), row=1, col=1)
    fig.add_trace(go.Scatter(x=out.index, y=pc2, mode="lines", line=dict(color="#1f77b4", width=1.4), name="PC2"), row=1, col=2)
    fig.add_trace(go.Bar(x=lags, y=acf1, marker_color="#1f77b4", name="ACF dPC1"), row=2, col=1)
    fig.add_trace(go.Bar(x=lags, y=acf2, marker_color="#1f77b4", name="ACF dPC2"), row=2, col=2)
    fig.update_layout(title="PCA components and ACF of changes (Trading Hours)", showlegend=False)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="ACF", row=2, col=1, range=[-0.10, 0.10])
    fig.update_yaxes(title_text="ACF", row=2, col=2, range=[-0.10, 0.10])
    fig.update_xaxes(title_text="Datetime", row=1, col=1)
    fig.update_xaxes(title_text="Datetime", row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    _apply_notebook_style(fig, height=760)
    return fig


def build_dashboard(
    *,
    cfg: PipelineConfig,
    df_mid_all: pd.DataFrame,
    df_state_all: pd.DataFrame,
    out_rule: dict,
    out_rl: dict,
    horizon_df: pd.DataFrame,
    action_summary: pd.DataFrame | None,
    action_aligned: pd.DataFrame | None,
    action_rho: float | None,
    regime_summary: pd.DataFrame | None,
    output_path: str | Path,
    matplotlib_sections: list[tuple[str, object]] | None = None,
) -> Path:
    """Write a standalone dashboard that mirrors the notebook plot order when available."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [
            {"strategy": "Rule", **out_rule["metrics"]},
            {"strategy": "RL Train", **out_rl["metrics"] | {
                "mean_daily_pnl_$": out_rl["metrics"]["train_mean_daily_pnl_$"],
                "max_drawdown_daily_$": out_rl["metrics"]["train_max_drawdown_$"],
            }},
            {"strategy": "RL Test", **out_rl["metrics"] | {
                "mean_daily_pnl_$": out_rl["metrics"]["test_mean_daily_pnl_$"],
                "max_drawdown_daily_$": out_rl["metrics"]["test_max_drawdown_$"],
            }},
        ]
    )
    keep_cols = ["strategy", "mean_daily_pnl_$", "max_drawdown_daily_$"]
    metrics_df = metrics_df[keep_cols]

    fallback_plots: list[tuple[str, go.Figure]] = []
    tables: list[tuple[str, go.Figure]] = []

    fallback_plots.append((
        "Cumulative reward overview",
        _line_figure(
            [
                (out_rule["cum"], "Rule cumulative reward", "#111111"),
                (out_rl["train_cum_pnl"], "RL train cumulative PnL", "#d55e00"),
                (out_rl["test_cum_pnl"], "RL test cumulative PnL", "#1d4ed8"),
            ],
            "Cumulative reward and PnL",
            "Value",
        ),
    ))

    scatter_fig = _scatter_figure(df_mid_all, cfg.trading.col_a, cfg.trading.col_b)
    if scatter_fig is not None:
        fallback_plots.append(("All-dates scatter", scatter_fig))

    forecast_fig = _normalized_forecast_figure(df_state_all, cfg.trading.col_a, cfg.trading.col_b)
    if forecast_fig is not None:
        fallback_plots.append(("Normalized series and forecasts", forecast_fig))

    comparison_fig = _comparison_figure(out_rl, out_rule, cfg.qlearning.train_end)
    if comparison_fig is not None:
        fallback_plots.append(("Rule vs RL", comparison_fig))

    horizon_fig = _horizon_figure(horizon_df)
    if horizon_fig is not None:
        fallback_plots.append(("Signal horizon sweep", horizon_fig))

    valid_horizons = horizon_df.dropna(subset=["correlation"])
    best_h = int(valid_horizons.loc[valid_horizons["correlation"].idxmax(), "horizon"]) if not valid_horizons.empty else None
    action_fig = _action_figure(
        action_summary,
        action_aligned,
        best_h,
        cfg.trading.col_a,
        action_rho,
    )
    if action_fig is not None:
        fallback_plots.append(("Action analysis", action_fig))

    positions_fig = _positions_side_by_side_figure(out_rl, out_rule, cfg.trading.col_a, cfg.trading.col_b)
    if positions_fig is not None:
        fallback_plots.append(("RL vs Rule positions", positions_fig))

    inventory_fig = _positions_figure(out_rl, out_rule, cfg.trading.col_a, cfg.trading.col_b)
    if inventory_fig is not None:
        fallback_plots.append(("Inventory paths", inventory_fig))

    residual_fig = _residual_figure(df_state_all, cfg.trading.col_a, cfg.trading.col_b)
    if residual_fig is not None:
        fallback_plots.append(("OLS residuals", residual_fig))

    diagnostics_fig = _diagnostics_figure(df_state_all)
    if diagnostics_fig is not None:
        fallback_plots.append(("PCA diagnostics", diagnostics_fig))

    tables.append(("Metrics", _table_figure(metrics_df, "Strategy metrics")))
    if regime_summary is not None and not regime_summary.empty:
        tables.append(("Regime summary", _table_figure(regime_summary, "PnL by regime")))

    sections: list[str] = []
    excluded_headings = {
        "ES Rule Decision Tree",
        "NQ Rule Decision Tree",
        "Action vs forward return",
    }
    # Prefer the notebook-style Matplotlib/PNG figures collected by the pipeline.
    # These are the closest match to exploration.ipynb. Only fall back to Plotly
    # recreations when those notebook-style figures are unavailable.
    if matplotlib_sections:
        for heading, fig in matplotlib_sections:
            if heading in excluded_headings:
                continue
            sections.append(_render_section(heading, fig))
    else:
        for heading, fig in fallback_plots:
            sections.append(_render_section(heading, fig))
    for heading, fig in tables:
        sections.append(_render_section(heading, fig))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RLHFT Dashboard</title>
  <script>{get_plotlyjs()}</script>
  <style>
    :root {{
      --bg: #f3f6fb;
      --panel: #ffffff;
      --ink: #102542;
      --muted: #5b6b82;
      --accent: #d55e00;
      --border: #d7e0ea;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.08), transparent 28%),
        linear-gradient(180deg, #eef4fb 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    header {{
      padding: 12px 4px 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 3vw, 3.4rem);
      line-height: 1.05;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .meta-card, section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 14px 40px rgba(16, 37, 66, 0.06);
    }}
    .meta-card {{
      padding: 16px 18px;
    }}
    .meta-card strong {{
      display: block;
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .meta-card span {{
      font-size: 1.05rem;
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      margin-top: 24px;
    }}
    section {{
      padding: 12px 12px 4px;
      overflow: hidden;
    }}
    section h2 {{
      margin: 10px 10px 0;
      font-size: 1.15rem;
    }}
    .scroll-frame {{
      overflow: auto;
      max-width: 100%;
      max-height: 78vh;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #fbfdff;
      cursor: grab;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>RLHFT Notebook-Style Dashboard</h1>
      <p>Pipeline output for {cfg.trading.col_a}/{cfg.trading.col_b}, matched to exploration.ipynb</p>
      <div class="meta">
        <div class="meta-card"><strong>Train End</strong><span>{cfg.qlearning.train_end}</span></div>
        <div class="meta-card"><strong>Date Range</strong><span>{cfg.data.start_date} to {cfg.data.end_date}</span></div>
        <div class="meta-card"><strong>Z Window</strong><span>{cfg.zscore.z_window}</span></div>
        <div class="meta-card"><strong>Epochs</strong><span>{cfg.qlearning.epochs}</span></div>
      </div>
    </header>
    <div class="grid">
      {''.join(sections)}
    </div>
  </main>
</body>
</html>
"""
    output.write_text(html)
    return output
