from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    # Allow running this file directly from a src-layout checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlhft.config import PipelineConfig
from rlhft.data.kdb import KDBConnection
from rlhft.evaluation.metrics import mean_daily_pnl, max_daily_drawdown
from rlhft.features.zscore import build_discrete_2asset_input
from rlhft.pipeline.multiday import run_multiday_pipeline
from rlhft.strategies.rule_based import run_rule_2asset_discrete
from rlhft.strategies.rl_strategy import run_rl_strategy
from rlhft.visualization.price_plots import plot_asset_trading_time, plot_multiday_scatter
from rlhft.visualization.pnl_plots import (
    plot_action_vs_fwd_return,
    plot_signal_horizon_sweep,
    plot_trading_time_cum,
    plot_rl_vs_rule_comparison,
)
from rlhft.visualization.position_plots import plot_rl_vs_rule_side_by_side_2asset
from rlhft.visualization.diagnostics import plot_pc_and_acf_trading_hours
from rlhft.visualization.dashboard import build_dashboard
from rlhft.evaluation.analysis import (
    action_vs_fwd_return,
    summarize_action_vs_fwd_return,
    sweep_signal_horizons,
    summarize_pnl_by_regime,
)


def export_debug_data(
    *,
    output_dir: str | Path,
    cfg: PipelineConfig,
    df_input: pd.DataFrame,
    out_rl: dict,
) -> Path:
    """Export core intermediate series for notebook/package comparisons."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_cols = [
        c for c in [cfg.trading.col_a, cfg.trading.col_b, "sig_a", "sig_b"] if c in df_input.columns
    ]
    df_input[input_cols].to_csv(out_dir / "df_input.csv", index=True)

    if "test_pnl" in out_rl:
        out_rl["test_pnl"].rename("test_pnl").to_csv(out_dir / "test_pnl.csv", index=True)
    if "train_pnl" in out_rl:
        out_rl["train_pnl"].rename("train_pnl").to_csv(out_dir / "train_pnl.csv", index=True)
    if "test_actions" in out_rl:
        out_rl["test_actions"].to_csv(out_dir / "test_actions.csv", index=True)
    if "test_n" in out_rl:
        out_rl["test_n"].to_csv(out_dir / "test_inventory.csv", index=True)
    if "regime_state_a" in out_rl:
        out_rl["regime_state_a"].rename("regime_state_a").to_csv(out_dir / "regime_state_a.csv", index=True)
    if "regime_state_b" in out_rl:
        out_rl["regime_state_b"].rename("regime_state_b").to_csv(out_dir / "regime_state_b.csv", index=True)
    if "signal_confidence_a" in out_rl:
        out_rl["signal_confidence_a"].rename("signal_confidence_a").to_csv(
            out_dir / "signal_confidence_a.csv", index=True
        )
    if "signal_confidence_b" in out_rl:
        out_rl["signal_confidence_b"].rename("signal_confidence_b").to_csv(
            out_dir / "signal_confidence_b.csv", index=True
        )

    summary = pd.DataFrame(
        {
            "item": [
                "col_a",
                "col_b",
                "train_end",
                "z_window",
                "z_lag",
                "z_step",
                "z_clip",
                "epochs",
                "lr",
                "eps_start",
                "eps_end",
                "regime_window",
                "q_decay",
                "persistent_exploration",
                "inv_limit",
                "mult_a",
                "mult_b",
                "cost_per_trade",
                "rule_inv_penalty",
                "rl_inv_penalty",
            ],
            "value": [
                cfg.trading.col_a,
                cfg.trading.col_b,
                cfg.qlearning.train_end,
                cfg.zscore.z_window,
                cfg.zscore.z_lag,
                cfg.zscore.z_step,
                cfg.zscore.z_clip,
                cfg.qlearning.epochs,
                cfg.qlearning.lr,
                cfg.qlearning.eps_start,
                cfg.qlearning.eps_end,
                cfg.qlearning.regime_window,
                cfg.qlearning.q_decay,
                cfg.qlearning.persistent_exploration,
                cfg.trading.inv_limit,
                cfg.trading.mult_a,
                cfg.trading.mult_b,
                cfg.trading.cost_per_trade,
                cfg.rule.inv_penalty,
                cfg.qlearning.inv_penalty,
            ],
        }
    )
    summary.to_csv(out_dir / "run_config_summary.csv", index=False)
    return out_dir


def run(
    cfg: PipelineConfig,
    *,
    dashboard_out: str | Path | None = "outputs/rlhft_dashboard.html",
    debug_export_dir: str | Path | None = None,
    render_matplotlib: bool = False,
) -> dict:
    """Run the full pipeline: data -> features -> strategies -> evaluation."""

    # --- Data ---
    print("=" * 60)
    print("Connecting to KDB+ and loading data...")
    print("=" * 60)

    with KDBConnection(cfg.kdb) as conn:
        sym_active, df_mid_all, df_state_all = run_multiday_pipeline(
            conn,
            cfg,
            render_plots=render_matplotlib,
        )

    # --- Features ---
    print("\n" + "=" * 60)
    print("Building discrete 2-asset input...")
    print("=" * 60)

    df_input = build_discrete_2asset_input(
        df_state_all,
        col_a=cfg.trading.col_a,
        col_b=cfg.trading.col_b,
    )

    # --- Rule-based strategy ---
    print("\n" + "=" * 60)
    print("Running rule-based strategy...")
    print("=" * 60)

    out_rule = run_rule_2asset_discrete(
        df_input,
        trading_cfg=cfg.trading,
        rule_cfg=cfg.rule,
        zscore_cfg=cfg.zscore,
    )
    print("Rule metrics:", pd.Series(out_rule["metrics"]))

    # --- RL strategy ---
    print("\n" + "=" * 60)
    print("Training Q-learning agent...")
    print("=" * 60)

    out_rl = run_rl_strategy(
        df_input,
        trading_cfg=cfg.trading,
        zscore_cfg=cfg.zscore,
        ql_cfg=cfg.qlearning,
    )

    print("\nRL metrics:")
    print({
        "train_mean_daily_pnl_$": mean_daily_pnl(out_rl["train_pnl"]),
        "train_max_drawdown_$": max_daily_drawdown(out_rl["train_pnl"]),
        "test_mean_daily_pnl_$": mean_daily_pnl(out_rl["test_pnl"]),
        "test_max_drawdown_$": max_daily_drawdown(out_rl["test_pnl"]),
    })

    matplotlib_sections: list[tuple[str, object]] = []

    # --- Visualization ---
    if cfg.make_plots:
        required_cols = [c for c in cfg.data.preferred_symbols if c in df_mid_all.columns]
        if len(required_cols) == 2:
            scatter_fig = plot_multiday_scatter(
                df_mid_all,
                required_cols[0],
                required_cols[1],
                trading_start="00:00",
                trading_end="16:00",
                show=render_matplotlib,
            )
            if scatter_fig is not None:
                matplotlib_sections.append(("All-dates scatter", scatter_fig))

            forecast_fig = plot_asset_trading_time(
                df_state_all,
                required_cols,
                ["darkorange", "navy"],
                trading_start="00:00",
                trading_end="16:00",
                show=render_matplotlib,
            )
            matplotlib_sections.append(("Normalized time series and forecasts", forecast_fig))

        rule_cum_fig = plot_trading_time_cum(
            out_rule["cum"],
            "Rule cumulative reward",
            show=render_matplotlib,
        )
        if rule_cum_fig is not None:
            matplotlib_sections.append(("Rule cumulative reward", rule_cum_fig))

        train_cum_fig = plot_trading_time_cum(
            out_rl["train_cum_pnl"],
            "Q-Learning Training cumulative PnL",
            show=render_matplotlib,
        )
        if train_cum_fig is not None:
            matplotlib_sections.append(("Q-Learning Training cumulative PnL", train_cum_fig))

        test_cum_fig = plot_trading_time_cum(
            out_rl["test_cum_pnl"],
            "Q-Learning Test cumulative PnL",
            show=render_matplotlib,
        )
        if test_cum_fig is not None:
            matplotlib_sections.append(("Q-Learning Test cumulative PnL", test_cum_fig))

        rl_vs_rule_fig = plot_rl_vs_rule_comparison(
            out_rl,
            out_rule,
            train_end=cfg.qlearning.train_end,
            show=render_matplotlib,
        )
        matplotlib_sections.append(("Rule vs RL cumulative reward", rl_vs_rule_fig))

        position_figs = plot_rl_vs_rule_side_by_side_2asset(
            out_rl, out_rule,
            col_a=cfg.trading.col_a,
            col_b=cfg.trading.col_b,
            leg="test",
            show=render_matplotlib,
        )
        for i, fig in enumerate(position_figs, start=1):
            matplotlib_sections.append((f"RL vs Rule positions {i}", fig))

        diag_fig, _ = plot_pc_and_acf_trading_hours(df_state_all, show=render_matplotlib)
        matplotlib_sections.append(("PCA diagnostics", diag_fig))

    if render_matplotlib and cfg.make_plots:
        print("\n" + "=" * 60)
        print("Generating Matplotlib plots...")
        print("=" * 60)

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("Signal analysis...")
    print("=" * 60)

    horizon_df = sweep_signal_horizons(
        df_input, "sig_a", cfg.trading.col_a, range(1, 31),
        start_date=cfg.qlearning.train_end,
    )
    action_summary = None
    action_rho = None
    action_aligned = None
    valid_horizons = horizon_df.dropna(subset=["correlation"])
    if valid_horizons.empty:
        print("No valid signal/forward-return horizon correlations were available.")
    else:
        best_row = valid_horizons.loc[valid_horizons["correlation"].idxmax()]
        best_h = int(best_row["horizon"])
        print(f"Best horizon h* = {best_h}   corr = {best_row['correlation']:.4f}")

        if cfg.make_plots:
            horizon_fig = plot_signal_horizon_sweep(
                horizon_df,
                f"{cfg.trading.col_a} Signal vs Forward Return - horizon sweep",
                show=render_matplotlib,
            )
            if horizon_fig is not None:
                matplotlib_sections.append(("Signal horizon sweep", horizon_fig))

        action_col = f"a_{cfg.trading.col_a}"
        if "test_actions" in out_rl and action_col in out_rl["test_actions"]:
            action_summary, action_rho = summarize_action_vs_fwd_return(
                df_input,
                cfg.trading.col_a,
                out_rl["test_actions"][action_col],
                best_h,
                start_date=cfg.qlearning.train_end,
            )
            print("\nAction vs forward return:")
            print(action_summary)
            print(f"Spearman rho = {action_rho:.4f}")

            action_aligned = action_vs_fwd_return(
                df_input,
                cfg.trading.col_a,
                out_rl["test_actions"][action_col],
                best_h,
                start_date=cfg.qlearning.train_end,
            )

            if cfg.make_plots:
                action_fig = plot_action_vs_fwd_return(
                    action_summary,
                    action_aligned,
                    best_h,
                    cfg.trading.col_a,
                    action_rho,
                    show=render_matplotlib,
                )
                if action_fig is not None:
                    matplotlib_sections.append(("Action vs forward return", action_fig))

    regime_summary = None
    if "regime_state_a" in out_rl:
        regime_summary = summarize_pnl_by_regime(
            pnl=out_rl["test_pnl"],
            regime_state=out_rl["regime_state_a"],
            start_date=cfg.qlearning.train_end,
        )
        print("\nPnL by regime:")
        print(regime_summary)

    dashboard_path = None
    if dashboard_out is not None:
        dashboard_path = build_dashboard(
            cfg=cfg,
            df_mid_all=df_mid_all,
            df_state_all=df_state_all,
            out_rule=out_rule,
            out_rl=out_rl,
            horizon_df=horizon_df,
            action_summary=action_summary,
            action_aligned=action_aligned,
            action_rho=action_rho,
            regime_summary=regime_summary,
            output_path=dashboard_out,
            matplotlib_sections=matplotlib_sections,
        )
        print(f"\nPlotly dashboard written to: {dashboard_path}")

    debug_export_path = None
    if debug_export_dir is not None:
        debug_export_path = export_debug_data(
            output_dir=debug_export_dir,
            cfg=cfg,
            df_input=df_input,
            out_rl=out_rl,
        )
        print(f"Debug comparison exports written to: {debug_export_path}")

    return {
        "sym_active": sym_active,
        "df_mid_all": df_mid_all,
        "df_state_all": df_state_all,
        "df_input": df_input,
        "out_rule": out_rule,
        "out_rl": out_rl,
        "horizon_df": horizon_df,
        "action_summary": action_summary,
        "action_aligned": action_aligned,
        "regime_summary": regime_summary,
        "dashboard_path": dashboard_path,
        "debug_export_path": debug_export_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RLHFT pair trading pipeline")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--no-plots", action="store_true", help="Disable dashboard and Matplotlib plotting")
    parser.add_argument(
        "--dashboard-out",
        default="outputs/rlhft_dashboard.html",
        help="Path to write the Plotly HTML dashboard. Use --no-dashboard to disable.",
    )
    parser.add_argument("--no-dashboard", action="store_true", help="Disable Plotly dashboard generation")
    parser.add_argument("--matplotlib-plots", action="store_true", help="Also open legacy Matplotlib popup plots")
    parser.add_argument(
        "--debug-export-dir",
        default="outputs/debug_compare",
        help="Directory to write intermediate CSVs for notebook/package comparison. Use --no-debug-export to disable.",
    )
    parser.add_argument("--no-debug-export", action="store_true", help="Disable intermediate CSV export")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.no_plots:
        cfg.make_plots = False

    run(
        cfg,
        dashboard_out=None if args.no_dashboard else args.dashboard_out,
        debug_export_dir=None if args.no_debug_export else args.debug_export_dir,
        render_matplotlib=args.matplotlib_plots and cfg.make_plots,
    )


if __name__ == "__main__":
    main()
