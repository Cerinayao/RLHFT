from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    # Allow running this file directly from a src-layout checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlhft.config import PipelineConfig
from rlhft.data.kdb import KDBConnection
from rlhft.data.loaders import sym_prefix
from rlhft.evaluation.metrics import mean_daily_pnl, max_daily_drawdown
from rlhft.evaluation.xgb_backtest import backtest_predicted_inventory_2asset
from rlhft.evaluation.xrl_analysis import build_xrl_policy_df
from rlhft.features.zscore import build_discrete_2asset_input
from rlhft.models.xgb_inventory import (
    INV_VALUES,
    train_xgb_inventory,
    xgb_feature_importance,
)
from rlhft.models.rule_extraction import extract_and_select_rules
from rlhft.models.rule_extraction import predict_from_partition_tree_rules
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
from rlhft.visualization.xgb_plots import plot_feature_importance, plot_xgb_vs_rl_pnl
from rlhft.visualization.xrl_plots import (
    plot_action_distribution_by_regime,
    plot_policy_heatmap_by_regime,
    plot_rule_agreement_binary_signal,
    plot_xrl_policy_curves,
)
from rlhft.visualization.rule_tree import plot_rule_tree_from_df
from rlhft.visualization.partition_tree import plot_partition_decision_tree
from rlhft.evaluation.analysis import (
    action_vs_fwd_return,
    summarize_action_vs_fwd_return,
    sweep_signal_horizons,
    summarize_pnl_by_regime,
)


def _resolve_display_scale(cfg: PipelineConfig) -> float:
    """Single factor used to convert raw x100/x10000 price units back to display/PnL units."""
    pfx_a = sym_prefix(cfg.trading.col_a)
    pfx_b = sym_prefix(cfg.trading.col_b)
    scale_a = cfg.scaling.display_scales.get(pfx_a, 1.0)
    scale_b = cfg.scaling.display_scales.get(pfx_b, 1.0)
    if scale_a != scale_b:
        print(
            f"Warning: display_scales differ for {pfx_a}={scale_a} and {pfx_b}={scale_b}; "
            f"using {pfx_a} ({scale_a}) as the single rescale factor."
        )
    return float(scale_a)


def rescale_rl_outputs(out_rl: dict, scale: float) -> dict:
    """Rescale all RL PnL/reward series by the raw price multiplier (e.g. 100 for ES/NQ)."""
    if scale == 1.0:
        return out_rl
    keys = [
        "train_pnl", "test_pnl",
        "train_reward", "test_reward",
        "train_cum", "test_cum",
        "train_cum_pnl", "test_cum_pnl",
    ]
    for k in keys:
        if k in out_rl and out_rl[k] is not None:
            out_rl[k] = out_rl[k] / scale
    out_rl["metrics"] = {
        "train_mean_daily_pnl_$": mean_daily_pnl(out_rl["train_pnl"]),
        "train_max_drawdown_$": max_daily_drawdown(out_rl["train_pnl"]),
        "test_mean_daily_pnl_$": mean_daily_pnl(out_rl["test_pnl"]),
        "test_max_drawdown_$": max_daily_drawdown(out_rl["test_pnl"]),
    }
    return out_rl


def rescale_rule_outputs(out_rule: dict, scale: float) -> dict:
    """Rescale all rule-based reward/cumulative series by the raw price multiplier."""
    if scale == 1.0:
        return out_rule
    for k in ("reward", "cum"):
        if k in out_rule and out_rule[k] is not None:
            out_rule[k] = out_rule[k] / scale
    out_rule["metrics"] = {
        "mean_daily_pnl_$": mean_daily_pnl(out_rule["reward"]),
        "max_drawdown_daily_$": max_daily_drawdown(out_rule["reward"]),
    }
    return out_rule


def rescale_price_frame(
    df: pd.DataFrame,
    scale: float,
    price_cols: list[str],
) -> pd.DataFrame:
    if scale == 1.0 or df is None or df.empty:
        return df
    out = df.copy()
    linear_cols: list[str] = []
    for c in price_cols:
        if c in out.columns:
            linear_cols.append(c)
    for c in out.columns:
        if c.startswith(("A_", "s_")) or c in ("pc1", "pc2", "sig_a", "sig_b"):
            linear_cols.append(c)
    for c in linear_cols:
        out[c] = out[c].astype(float) / scale
    for c in out.columns:
        if c.startswith("M_"):
            out[c] = out[c].astype(float) / (scale ** 2)
    return out


def rescale_action_outputs(
    action_summary: pd.DataFrame | None,
    action_aligned: pd.DataFrame | None,
    scale: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if scale == 1.0:
        return action_summary, action_aligned
    if action_summary is not None and not action_summary.empty:
        action_summary = action_summary.copy()
        for c in ("mean", "std"):
            if c in action_summary.columns:
                action_summary[c] = action_summary[c].astype(float) / scale
    if action_aligned is not None and not action_aligned.empty:
        action_aligned = action_aligned.copy()
        if "fwd_ret" in action_aligned.columns:
            action_aligned["fwd_ret"] = action_aligned["fwd_ret"].astype(float) / scale
    return action_summary, action_aligned


def export_debug_data(
    *,
    output_dir: str | Path,
    cfg: PipelineConfig,
    df_input: pd.DataFrame,
    out_rl: dict,
) -> Path:
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
                "col_a", "col_b", "train_end", "z_window", "z_lag", "z_step", "z_clip",
                "epochs", "lr", "eps_start", "eps_end", "regime_window", "q_decay",
                "persistent_exploration", "inv_limit", "mult_a", "mult_b",
                "cost_per_trade", "rule_inv_penalty", "rl_inv_penalty",
            ],
            "value": [
                cfg.trading.col_a, cfg.trading.col_b, cfg.qlearning.train_end,
                cfg.zscore.z_window, cfg.zscore.z_lag, cfg.zscore.z_step, cfg.zscore.z_clip,
                cfg.qlearning.epochs, cfg.qlearning.lr, cfg.qlearning.eps_start, cfg.qlearning.eps_end,
                cfg.qlearning.regime_window, cfg.qlearning.q_decay, cfg.qlearning.persistent_exploration,
                cfg.trading.inv_limit, cfg.trading.mult_a, cfg.trading.mult_b,
                cfg.trading.cost_per_trade, cfg.rule.inv_penalty, cfg.qlearning.inv_penalty,
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
    """Run the full pipeline: data -> features -> strategies -> evaluation -> XGB -> rule extraction."""

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

    display_scale = _resolve_display_scale(cfg)
    if display_scale != 1.0:
        print(
            f"Rescaling all reported PnL/reward outputs by {display_scale:g} "
            f"because raw prices are stored in x{display_scale:g} units."
        )
    out_rule = rescale_rule_outputs(out_rule, display_scale)
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
    out_rl = rescale_rl_outputs(out_rl, display_scale)

    price_cols_for_display = [cfg.trading.col_a, cfg.trading.col_b]
    df_mid_all_display = rescale_price_frame(df_mid_all, display_scale, price_cols_for_display)
    df_state_all_display = rescale_price_frame(df_state_all, display_scale, price_cols_for_display)

    print("\nRL metrics:")
    print({
        "train_mean_daily_pnl_$": mean_daily_pnl(out_rl["train_pnl"]),
        "train_max_drawdown_$": max_daily_drawdown(out_rl["train_pnl"]),
        "test_mean_daily_pnl_$": mean_daily_pnl(out_rl["test_pnl"]),
        "test_max_drawdown_$": max_daily_drawdown(out_rl["test_pnl"]),
    })

    matplotlib_sections: list[tuple[str, object]] = []

    # --- Visualization (notebook order) ---
    if cfg.make_plots:
        required_cols = [c for c in cfg.data.preferred_symbols if c in df_mid_all_display.columns]
        if len(required_cols) == 2:
            scatter_fig = plot_multiday_scatter(
                df_mid_all_display, required_cols[0], required_cols[1],
                trading_start="00:00", trading_end="16:00",
                show=render_matplotlib,
            )
            if scatter_fig is not None:
                matplotlib_sections.append((f"{required_cols[0]} vs {required_cols[1]} (All Dates)", scatter_fig))

            forecast_fig = plot_asset_trading_time(
                df_state_all_display, required_cols, ["darkorange", "navy"],
                trading_start="00:00", trading_end="16:00",
                show=render_matplotlib,
            )
            matplotlib_sections.append(("Normalized time series and forecasts", forecast_fig))

        rule_cum_fig = plot_trading_time_cum(
            out_rule["cum"], "Level 2 Rule cumulative reward", show=render_matplotlib,
        )
        if rule_cum_fig is not None:
            matplotlib_sections.append(("Level 2 Rule cumulative reward", rule_cum_fig))

        train_cum_fig = plot_trading_time_cum(
            out_rl["train_cum_pnl"], "Q-Learning Training cumulative PnL", show=render_matplotlib,
        )
        if train_cum_fig is not None:
            matplotlib_sections.append(("Q-Learning Training cumulative PnL", train_cum_fig))

        test_cum_fig = plot_trading_time_cum(
            out_rl["test_cum_pnl"], "Q-Learning Test cumulative PnL", show=render_matplotlib,
        )
        if test_cum_fig is not None:
            matplotlib_sections.append(("Q-Learning Test cumulative PnL", test_cum_fig))

        rl_vs_rule_fig = plot_rl_vs_rule_comparison(
            out_rl, out_rule, train_end=cfg.qlearning.train_end, show=render_matplotlib,
        )
        matplotlib_sections.append(("ES/NQ - Rule vs RL cumulative reward (test period)", rl_vs_rule_fig))

        position_figs = plot_rl_vs_rule_side_by_side_2asset(
            out_rl, out_rule,
            col_a=cfg.trading.col_a, col_b=cfg.trading.col_b,
            leg="test", show=render_matplotlib,
        )
        for fig in position_figs:
            day_label = fig._suptitle.get_text() if fig._suptitle is not None else "RL vs Rule positions"
            matplotlib_sections.append((day_label, fig))

        diag_fig, _ = plot_pc_and_acf_trading_hours(df_state_all_display, show=render_matplotlib)
        matplotlib_sections.append(("PCA Components and ACF of Changes", diag_fig))

    # --- Signal horizon analysis ---
    print("\n" + "=" * 60)
    print("Signal analysis...")
    print("=" * 60)

    horizons = range(1, cfg.analysis.signal_horizons_max + 1)
    horizon_df = sweep_signal_horizons(
        df_input, "sig_a", cfg.trading.col_a, horizons,
        start_date=cfg.analysis.start_date,
    )
    action_summary = None
    action_rho = None
    action_aligned = None
    best_h = None
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
                df_input, cfg.trading.col_a,
                out_rl["test_actions"][action_col], best_h,
                start_date=cfg.analysis.start_date,
            )
            print("\nAction vs forward return:")
            print(action_summary)
            print(f"Spearman rho = {action_rho:.4f}")

            action_aligned = action_vs_fwd_return(
                df_input, cfg.trading.col_a,
                out_rl["test_actions"][action_col], best_h,
                start_date=cfg.analysis.start_date,
            )

            action_summary, action_aligned = rescale_action_outputs(
                action_summary, action_aligned, display_scale,
            )

            if cfg.make_plots:
                action_fig = plot_action_vs_fwd_return(
                    action_summary, action_aligned, best_h,
                    cfg.trading.col_a, action_rho,
                    show=render_matplotlib,
                )
                if action_fig is not None:
                    matplotlib_sections.append(("Action vs forward return", action_fig))

    # --- XRL policy curves / heatmaps / distribution / agreement ---
    a_action_col = f"a_{cfg.trading.col_a}"
    n_a_col = f"n_{cfg.trading.col_a}"
    df_xrl = df_input.copy()
    if "test_n" in out_rl and n_a_col in out_rl["test_n"]:
        df_xrl[n_a_col] = out_rl["test_n"][n_a_col]

    tmp_xrl = None
    if (
        "test_actions" in out_rl
        and a_action_col in out_rl["test_actions"]
        and "regime_state_a" in out_rl
        and n_a_col in df_xrl.columns
    ):
        try:
            tmp_xrl = build_xrl_policy_df(
                df_xrl,
                signal_col="sig_a",
                inventory_col=n_a_col,
                action=out_rl["test_actions"][a_action_col],
                regime_state=out_rl["regime_state_a"],
                start_date=cfg.analysis.start_date,
                trading_start=cfg.analysis.trading_start,
                trading_end=cfg.analysis.trading_end,
            )
        except Exception as exc:
            print(f"Skipping XRL analysis: {exc}")
            tmp_xrl = None

    if tmp_xrl is not None and not tmp_xrl.empty and cfg.make_plots:
        try:
            xrl_curves_fig, *_ = plot_xrl_policy_curves(
                tmp_xrl, signal_bins=3,
                title_prefix=f"{cfg.trading.col_a} XRL Policy Curves",
                show=render_matplotlib,
            )
            matplotlib_sections.append((f"{cfg.trading.col_a} XRL Policy Curves", xrl_curves_fig))
        except Exception as exc:
            print(f"XRL policy curves skipped: {exc}")

        try:
            heatmap_fig = plot_policy_heatmap_by_regime(
                tmp_xrl,
                title_prefix=f"{cfg.trading.col_a} Mean Action Heatmap by Regime",
                show=render_matplotlib,
            )
            matplotlib_sections.append((f"{cfg.trading.col_a} Mean Action Heatmap by Regime", heatmap_fig))
        except Exception as exc:
            print(f"Policy heatmap skipped: {exc}")

        try:
            dist_fig, _ = plot_action_distribution_by_regime(
                tmp_xrl,
                title_prefix=f"{cfg.trading.col_a} Action Distribution by Regime",
                show=render_matplotlib,
            )
            matplotlib_sections.append((f"{cfg.trading.col_a} Action Distribution by Regime", dist_fig))
        except Exception as exc:
            print(f"Action distribution skipped: {exc}")

        try:
            tmp_xrl_with_rule = tmp_xrl.copy()
            tmp_xrl_with_rule["rule_action"] = np.sign(tmp_xrl_with_rule["signal"])
            agree_fig, _, _ = plot_rule_agreement_binary_signal(
                tmp_xrl_with_rule,
                title_prefix="RL vs Rule Agreement (Binary Signal)",
                show=render_matplotlib,
            )
            matplotlib_sections.append(("RL vs Rule Agreement (Binary Signal)", agree_fig))
        except Exception as exc:
            print(f"Rule agreement plot skipped: {exc}")

    # --- PnL by regime ---
    regime_summary = None
    if "regime_state_a" in out_rl:
        regime_summary = summarize_pnl_by_regime(
            pnl=out_rl["test_pnl"],
            regime_state=out_rl["regime_state_a"],
            start_date=cfg.analysis.start_date,
            trading_start=cfg.analysis.trading_start,
            trading_end=cfg.analysis.trading_end,
        )
        print("\nPnL by regime:")
        print(regime_summary)

    # --- XGB inventory model ---
    print("\n" + "=" * 60)
    print("Training XGBoost inventory model...")
    print("=" * 60)
    xgb_results: dict | None = None
    xgb_val_bt = None
    xgb_test_bt = None
    rule_extraction_results: dict | None = None
    partition_tree_results: dict | None = None
    try:
        xgb_results = train_xgb_inventory(
            df_input,
            out_rl,
            col_a=cfg.trading.col_a,
            col_b=cfg.trading.col_b,
            z_cfg=cfg.zscore,
            xgb_cfg=cfg.xgb,
        )
    except Exception as exc:
        print(f"XGB training skipped: {exc}")

    if xgb_results is not None:
        try:
            xgb_val_bt, xgb_val_summary = backtest_predicted_inventory_2asset(
                df_input,
                xgb_results["pred_n_val"],
                col_a=cfg.trading.col_a,
                col_b=cfg.trading.col_b,
                mult_a=cfg.trading.mult_a / display_scale,
                mult_b=cfg.trading.mult_b / display_scale,
                trading_start=cfg.analysis.trading_start,
                trading_end=cfg.analysis.trading_end,
                cost_per_trade=cfg.trading.cost_per_trade,
                inv_penalty=cfg.xgb.inv_penalty,
            )
            print("\n=== XGB Validation Backtest ===")
            print(xgb_val_summary)

            xgb_test_bt, xgb_test_summary = backtest_predicted_inventory_2asset(
                df_input,
                xgb_results["pred_n_test"],
                col_a=cfg.trading.col_a,
                col_b=cfg.trading.col_b,
                mult_a=cfg.trading.mult_a / display_scale,
                mult_b=cfg.trading.mult_b / display_scale,
                trading_start=cfg.analysis.trading_start,
                trading_end=cfg.analysis.trading_end,
                cost_per_trade=cfg.trading.cost_per_trade,
                inv_penalty=cfg.xgb.inv_penalty,
            )
            print("\n=== XGB Final Test Backtest ===")
            print(xgb_test_summary)
        except Exception as exc:
            print(f"XGB backtest skipped: {exc}")
            xgb_test_bt = None

        if cfg.make_plots and xgb_test_bt is not None:
            try:
                xgb_pnl_fig, _, _ = plot_xgb_vs_rl_pnl(
                    xgb_bt=xgb_test_bt, out_rl=out_rl,
                    trading_start=cfg.analysis.trading_start,
                    trading_end=cfg.analysis.trading_end,
                    title="Test Cumulative PnL: RL vs XGB Inventory Policy",
                    show=render_matplotlib,
                )
                matplotlib_sections.append(("Test Cumulative PnL: RL vs XGB Inventory Policy", xgb_pnl_fig))
            except Exception as exc:
                print(f"XGB-vs-RL plot skipped: {exc}")

        if cfg.make_plots:
            try:
                imp_es = xgb_feature_importance(xgb_results["model_a"], xgb_results["feature_cols"])
                imp_nq = xgb_feature_importance(xgb_results["model_b"], xgb_results["feature_cols"])
                print("\n=== ES Feature Importance (Gain) ===")
                print(imp_es)
                print("\n=== NQ Feature Importance (Gain) ===")
                print(imp_nq)
                imp_es_fig = plot_feature_importance(imp_es, "ES Feature Importance (Gain)", show=render_matplotlib)
                imp_nq_fig = plot_feature_importance(imp_nq, "NQ Feature Importance (Gain)", show=render_matplotlib)
                matplotlib_sections.append(("ES Feature Importance (Gain)", imp_es_fig))
                matplotlib_sections.append(("NQ Feature Importance (Gain)", imp_nq_fig))
            except Exception as exc:
                print(f"Feature importance skipped: {exc}")

        # --- TE2Rules-style rule extraction ---
        try:
            xgb_es_class = xgb_results["model_a"].predict(xgb_results["X_test"])
            xgb_nq_class = xgb_results["model_b"].predict(xgb_results["X_test"])
            xgb_es_inv = INV_VALUES[xgb_es_class]
            xgb_nq_inv = INV_VALUES[xgb_nq_class]

            print("\n" + "=" * 60)
            print("Extracting selected ES rules (TE2Rules-style greedy)...")
            print("=" * 60)
            rules_es_raw, selected_es, pred_es, fid_es = extract_and_select_rules(
                model=xgb_results["model_a"],
                feature_cols=xgb_results["feature_cols"],
                X_test=xgb_results["X_test"],
                xgb_inv_targets=xgb_es_inv,
                target_fidelity=cfg.rule_extraction.target_fidelity,
                max_rules=cfg.rule_extraction.max_rules,
                candidate_top_n=cfg.rule_extraction.candidate_top_n,
            )

            print("\n" + "=" * 60)
            print("Extracting selected NQ rules (TE2Rules-style greedy)...")
            print("=" * 60)
            rules_nq_raw, selected_nq, pred_nq, fid_nq = extract_and_select_rules(
                model=xgb_results["model_b"],
                feature_cols=xgb_results["feature_cols"],
                X_test=xgb_results["X_test"],
                xgb_inv_targets=xgb_nq_inv,
                target_fidelity=cfg.rule_extraction.target_fidelity,
                max_rules=cfg.rule_extraction.max_rules,
                candidate_top_n=cfg.rule_extraction.candidate_top_n,
            )

            print(f"\nES fidelity: {fid_es:.3f} ({len(selected_es)} rules)")
            print(f"NQ fidelity: {fid_nq:.3f} ({len(selected_nq)} rules)")

            pred_n_rule_test = pd.DataFrame(
                {
                    f"xgb_n_{cfg.trading.col_a}": pred_es,
                    f"xgb_n_{cfg.trading.col_b}": pred_nq,
                },
                index=xgb_results["X_test"].index,
            )

            rule_bt, rule_summary = backtest_predicted_inventory_2asset(
                df_input, pred_n_rule_test,
                col_a=cfg.trading.col_a, col_b=cfg.trading.col_b,
                mult_a=cfg.trading.mult_a / display_scale,
                mult_b=cfg.trading.mult_b / display_scale,
                trading_start=cfg.analysis.trading_start,
                trading_end=cfg.analysis.trading_end,
                cost_per_trade=cfg.trading.cost_per_trade,
                inv_penalty=cfg.xgb.inv_penalty,
            )
            print("\n=== Selected TE2Rules Backtest ===")
            print(rule_summary)

            rule_extraction_results = {
                "es_raw": rules_es_raw,
                "es_selected": selected_es,
                "es_fidelity": fid_es,
                "nq_raw": rules_nq_raw,
                "nq_selected": selected_nq,
                "nq_fidelity": fid_nq,
                "rule_bt": rule_bt,
                "rule_summary": rule_summary,
            }

            if cfg.make_plots:
                try:
                    rule_vs_rl_fig, _, _ = plot_xgb_vs_rl_pnl(
                        xgb_bt=rule_bt, out_rl=out_rl,
                        trading_start=cfg.analysis.trading_start,
                        trading_end=cfg.analysis.trading_end,
                        title="Test Cumulative PnL: RL vs Selected TE2Rules",
                        show=render_matplotlib,
                    )
                    matplotlib_sections.append(("Test Cumulative PnL: RL vs Selected TE2Rules", rule_vs_rl_fig))
                except Exception as exc:
                    print(f"Rule-vs-RL plot skipped: {exc}")

                # Decision tree visualization (graphviz)
                try:
                    tmp_dir = Path(tempfile.mkdtemp(prefix="rlhft_tree_"))
                    es_png = plot_rule_tree_from_df(
                        selected_es,
                        title="ES Rule Decision Tree",
                        top_n=cfg.rule_extraction.tree_top_n,
                        filename=str(tmp_dir / "es_tree"),
                    )
                    nq_png = plot_rule_tree_from_df(
                        selected_nq,
                        title="NQ Rule Decision Tree",
                        top_n=cfg.rule_extraction.tree_top_n,
                        filename=str(tmp_dir / "nq_tree"),
                    )
                    if es_png is not None:
                        matplotlib_sections.append(("ES Rule Decision Tree", es_png))
                    if nq_png is not None:
                        matplotlib_sections.append(("NQ Rule Decision Tree", nq_png))
                except Exception as exc:
                    print(f"Rule decision tree skipped: {exc}")

            print("\n=== Selected ES Rules ===")
            for i, r in selected_es.iterrows():
                print(f"[{i+1}] Inventory {r['inventory']} | score={r['score']:.4f} | {r['rule_text']}")
            print("\n=== Selected NQ Rules ===")
            for i, r in selected_nq.iterrows():
                print(f"[{i+1}] Inventory {r['inventory']} | score={r['score']:.4f} | {r['rule_text']}")

            # --- Notebook partition tree + score-vote backtest ---
            pred_es_partition = predict_from_partition_tree_rules(
                xgb_results["X_test"],
                selected_es,
                min_score=0.03,
            )
            pred_nq_partition = predict_from_partition_tree_rules(
                xgb_results["X_test"],
                selected_nq,
                min_score=0.03,
            )
            pred_partition_test = pd.DataFrame(
                {
                    f"xgb_n_{cfg.trading.col_a}": pred_es_partition,
                    f"xgb_n_{cfg.trading.col_b}": pred_nq_partition,
                },
                index=xgb_results["X_test"].index,
            )

            partition_bt, partition_summary = backtest_predicted_inventory_2asset(
                df_input,
                pred_partition_test,
                col_a=cfg.trading.col_a,
                col_b=cfg.trading.col_b,
                mult_a=cfg.trading.mult_a / display_scale,
                mult_b=cfg.trading.mult_b / display_scale,
                trading_start=cfg.analysis.trading_start,
                trading_end=cfg.analysis.trading_end,
                cost_per_trade=cfg.trading.cost_per_trade,
                inv_penalty=cfg.xgb.inv_penalty,
            )
            print("\n=== Partition Tree Rule Backtest ===")
            print(partition_summary)

            partition_pnl = partition_bt["pnl"].between_time(
                cfg.analysis.trading_start, cfg.analysis.trading_end
            ).dropna()
            rl_pnl = out_rl["test_pnl"].reindex(partition_pnl.index).dropna()
            common_idx = partition_pnl.index.intersection(rl_pnl.index)
            partition_pnl = partition_pnl.loc[common_idx]
            rl_pnl = rl_pnl.loc[common_idx]
            compare_summary = pd.DataFrame(
                {
                    "Partition_Tree": {
                        "cum_pnl": partition_pnl.sum(),
                        "mean_pnl": partition_pnl.mean(),
                        "std_pnl": partition_pnl.std(),
                        "sharpe": (
                            np.sqrt(252) * partition_pnl.mean() / partition_pnl.std()
                            if partition_pnl.std() > 0
                            else np.nan
                        ),
                        "max_drawdown": (
                            partition_pnl.cumsum() - partition_pnl.cumsum().cummax()
                        ).min(),
                    },
                    "RL": {
                        "cum_pnl": rl_pnl.sum(),
                        "mean_pnl": rl_pnl.mean(),
                        "std_pnl": rl_pnl.std(),
                        "sharpe": (
                            np.sqrt(252) * rl_pnl.mean() / rl_pnl.std()
                            if rl_pnl.std() > 0
                            else np.nan
                        ),
                        "max_drawdown": (
                            rl_pnl.cumsum() - rl_pnl.cumsum().cummax()
                        ).min(),
                    },
                }
            )
            print("\n=== Partition Tree Rules vs RL Summary ===")
            print(compare_summary)

            partition_tree_results = {
                "pred_test": pred_partition_test,
                "partition_bt": partition_bt,
                "partition_summary": partition_summary,
                "compare_summary": compare_summary,
            }

            if cfg.make_plots:
                try:
                    partition_vs_rl_fig, _, _ = plot_xgb_vs_rl_pnl(
                        xgb_bt=partition_bt,
                        out_rl=out_rl,
                        trading_start=cfg.analysis.trading_start,
                        trading_end=cfg.analysis.trading_end,
                        title="Test Cumulative PnL: RL vs Partition Tree Rules",
                        show=render_matplotlib,
                    )
                    matplotlib_sections.append(("Test Cumulative PnL: RL vs Partition Tree Rules", partition_vs_rl_fig))
                except Exception as exc:
                    print(f"Partition-tree-vs-RL plot skipped: {exc}")

                try:
                    tmp_dir = Path(tempfile.mkdtemp(prefix="rlhft_partition_tree_"))
                    es_partition_png = plot_partition_decision_tree(
                        selected_es,
                        feature_order=["zqa", "regime_a", "n_a_prev", "zqb", "regime_b", "n_b_prev"],
                        title="Partition Decision Tree for ES Inventory",
                        filename=tmp_dir / "partition_tree_es",
                        min_score=0.03,
                        top_n=50,
                        max_depth=3,
                    )
                    nq_partition_png = plot_partition_decision_tree(
                        selected_nq,
                        feature_order=["zqb", "regime_b", "n_b_prev", "zqa", "regime_a", "n_a_prev"],
                        title="Partition Decision Tree for NQ Inventory",
                        filename=tmp_dir / "partition_tree_nq",
                        min_score=0.03,
                        top_n=50,
                        max_depth=3,
                    )
                    if es_partition_png is not None:
                        matplotlib_sections.append(("Partition Decision Tree for ES Inventory", es_partition_png))
                    if nq_partition_png is not None:
                        matplotlib_sections.append(("Partition Decision Tree for NQ Inventory", nq_partition_png))
                except Exception as exc:
                    print(f"Partition decision tree skipped: {exc}")
        except Exception as exc:
            print(f"Rule extraction skipped: {exc}")

    if render_matplotlib and cfg.make_plots:
        print("\n" + "=" * 60)
        print("Matplotlib plots rendered above.")
        print("=" * 60)

    dashboard_path = None
    if dashboard_out is not None:
        dashboard_path = build_dashboard(
            cfg=cfg,
            df_mid_all=df_mid_all_display,
            df_state_all=df_state_all_display,
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
        "xgb_results": xgb_results,
        "xgb_val_bt": xgb_val_bt,
        "xgb_test_bt": xgb_test_bt,
        "rule_extraction": rule_extraction_results,
        "partition_tree": partition_tree_results,
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
