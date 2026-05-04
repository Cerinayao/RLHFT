from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORKS_COLOR = "#a1d99b"
FAILS_COLOR = "#fcbba1"
MAIN_COLOR = "#9ecae1"


def plot_xrl_policy_curves(
    tmp: pd.DataFrame,
    *,
    signal_bins: int = 50,
    title_prefix: str = "ESH4 XRL Policy Curves",
    show: bool = True,
) -> tuple[plt.Figure, pd.Series, pd.DataFrame, pd.Series]:
    """E[a|signal], E[a|signal,regime], E[a|inventory] curves."""
    tmp = tmp.copy()
    tmp["signal_bin"] = pd.qcut(tmp["signal"], q=signal_bins, duplicates="drop")

    sig_curve = tmp.groupby("signal_bin", observed=True)["action"].mean()
    sig_x = [interval.mid for interval in sig_curve.index]

    sig_regime_curve = (
        tmp.groupby(["signal_bin", "regime_label"], observed=True)["action"]
        .mean()
        .unstack("regime_label")
    )
    sig_regime_x = [interval.mid for interval in sig_regime_curve.index]

    inv_curve = tmp.groupby("inventory")["action"].mean().sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    axes[0].plot(
        sig_x, sig_curve.values, marker="o", linewidth=2.2,
        color=MAIN_COLOR, markeredgecolor="black", alpha=0.9,
    )
    axes[0].axhline(0, linestyle="--", linewidth=1, color="gray")
    axes[0].set_title(r"$E[a_t \mid signal_t]$")
    axes[0].set_xlabel("Signal bin midpoint")
    axes[0].set_ylabel("Mean action")
    axes[0].grid(alpha=0.2)

    color_map = {"works (+1)": WORKS_COLOR, "fails (-1)": FAILS_COLOR}
    for col in sig_regime_curve.columns:
        axes[1].plot(
            sig_regime_x, sig_regime_curve[col].values,
            marker="o", linewidth=2.2, label=col,
            color=color_map.get(col, MAIN_COLOR),
            markeredgecolor="black", alpha=0.9,
        )
    axes[1].axhline(0, linestyle="--", linewidth=1, color="gray")
    axes[1].set_title(r"$E[a_t \mid signal_t, regime_t]$")
    axes[1].set_xlabel("Signal bin midpoint")
    axes[1].set_ylabel("Mean action")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(
        inv_curve.index, inv_curve.values,
        marker="o", linewidth=2.2,
        color=MAIN_COLOR, markeredgecolor="black", alpha=0.9,
    )
    axes[2].axhline(0, linestyle="--", linewidth=1, color="gray")
    axes[2].set_title(r"$E[a_t \mid inventory_t]$")
    axes[2].set_xlabel("Inventory")
    axes[2].set_ylabel("Mean action")
    axes[2].set_xticks(inv_curve.index)
    axes[2].grid(alpha=0.2)

    fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if show:
        plt.show()
    return fig, sig_curve, sig_regime_curve, inv_curve


def plot_policy_heatmap_by_regime(
    tmp: pd.DataFrame,
    *,
    title_prefix: str = "ESH4 Mean Action Heatmap by Regime",
    show: bool = True,
) -> plt.Figure:
    tmp = tmp.copy()
    tmp["signal_sign"] = np.where(tmp["signal"] > 0, "s > 0", "s < 0")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    last_im = None
    for ax, regime in zip(axes, ["works (+1)", "fails (-1)"]):
        sub = tmp[tmp["regime_label"] == regime].copy()
        if sub.empty:
            ax.set_title(f"{regime} (no data)", fontweight="bold")
            ax.axis("off")
            continue

        heat = (
            sub.groupby(["inventory", "signal_sign"])["action"]
            .mean()
            .unstack("signal_sign")
            .reindex(index=sorted(sub["inventory"].unique()), columns=["s > 0", "s < 0"])
        )
        counts = (
            sub.groupby(["inventory", "signal_sign"])["action"]
            .size()
            .unstack("signal_sign")
            .reindex(index=heat.index, columns=heat.columns)
            .fillna(0)
            .astype(int)
        )

        im = ax.imshow(heat.values, aspect="auto", vmin=-2, vmax=2, cmap="coolwarm", alpha=0.85)
        last_im = im
        ax.set_title(regime, fontweight="bold")
        ax.set_xlabel("Signal")
        ax.set_ylabel("Inventory")
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index.astype(int))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["s > 0", "s < 0"])

        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                v = heat.values[i, j]
                n = counts.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}\n(n={n})", ha="center", va="center", fontsize=9)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.035, pad=0.08)
        cbar.set_label("Mean action")

    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.18, wspace=0.25)
    fig.suptitle("Mean Action by Signal Sign and Regime", fontsize=14, fontweight="bold")
    if show:
        plt.show()
    return fig


def plot_action_distribution_by_regime(
    tmp: pd.DataFrame,
    *,
    title_prefix: str = "ESH4 Action Distribution by Regime",
    show: bool = True,
) -> tuple[plt.Figure, dict]:
    tmp = tmp.copy()

    counts_signed = (
        tmp.groupby(["regime", "action"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[1, -1], columns=[-2, -1, 0, 1, 2], fill_value=0)
    )
    probs_signed = counts_signed.div(counts_signed.sum(axis=1).replace(0, 1), axis=0)

    counts_abs = (
        tmp.groupby(["regime", "abs_action"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[1, -1], columns=[0, 1, 2], fill_value=0)
    )
    probs_abs = counts_abs.div(counts_abs.sum(axis=1).replace(0, 1), axis=0)

    mean_signed = tmp.groupby("regime")["action"].mean().reindex([1, -1])
    mean_abs = tmp.groupby("regime")["abs_action"].mean().reindex([1, -1])

    print("=== Mean Action (signed) ===")
    print(mean_signed.rename(index={1: "works (+1)", -1: "fails (-1)"}))
    print("\n=== Mean |Action| ===")
    print(mean_abs.rename(index={1: "works (+1)", -1: "fails (-1)"}))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.35

    x_signed = np.arange(len(probs_signed.columns))
    axes[0, 0].bar(x_signed - width / 2, probs_signed.loc[1].values if 1 in probs_signed.index else 0,
                   width, label="works (+1)", color=WORKS_COLOR, edgecolor="black", alpha=0.9)
    axes[0, 0].bar(x_signed + width / 2, probs_signed.loc[-1].values if -1 in probs_signed.index else 0,
                   width, label="fails (-1)", color=FAILS_COLOR, edgecolor="black", alpha=0.9)
    axes[0, 0].set_xticks(x_signed)
    axes[0, 0].set_xticklabels(probs_signed.columns.astype(int))
    axes[0, 0].set_title("Signed Action Distribution")
    axes[0, 0].set_xlabel("Action")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.2)

    x_abs = np.arange(len(probs_abs.columns))
    axes[0, 1].bar(x_abs - width / 2, probs_abs.loc[1].values if 1 in probs_abs.index else 0,
                   width, label="works (+1)", color=WORKS_COLOR, edgecolor="black", alpha=0.9)
    axes[0, 1].bar(x_abs + width / 2, probs_abs.loc[-1].values if -1 in probs_abs.index else 0,
                   width, label="fails (-1)", color=FAILS_COLOR, edgecolor="black", alpha=0.9)
    axes[0, 1].set_xticks(x_abs)
    axes[0, 1].set_xticklabels(probs_abs.columns.astype(int))
    axes[0, 1].set_title("|Action| Distribution")
    axes[0, 1].set_xlabel("|Action|")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[1, 0].bar(["works (+1)", "fails (-1)"], mean_signed.values,
                   color=[WORKS_COLOR, FAILS_COLOR], edgecolor="black", alpha=0.9)
    axes[1, 0].axhline(0, linestyle="--", linewidth=1, color="gray")
    axes[1, 0].set_title("Mean Signed Action")
    axes[1, 0].set_ylabel(r"$E[a_t]$")
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].bar(["works (+1)", "fails (-1)"], mean_abs.values,
                   color=[WORKS_COLOR, FAILS_COLOR], edgecolor="black", alpha=0.9)
    axes[1, 1].set_title("Mean |Action|")
    axes[1, 1].set_ylabel(r"$E[|a_t|]$")
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if show:
        plt.show()

    return fig, {
        "counts_signed": counts_signed,
        "probs_signed": probs_signed,
        "counts_abs": counts_abs,
        "probs_abs": probs_abs,
        "mean_signed": mean_signed,
        "mean_abs": mean_abs,
    }


def plot_rule_agreement_binary_signal(
    tmp_xrl: pd.DataFrame,
    *,
    rule_action_col: str = "rule_action",
    rl_action_col: str = "action",
    regime_col: str = "regime",
    signal_col: str = "signal",
    title_prefix: str = "RL vs Rule Agreement (Binary Signal)",
    show: bool = True,
) -> tuple[plt.Figure, pd.Series, pd.DataFrame]:
    tmp = tmp_xrl.copy()

    tmp["agree"] = (tmp[rl_action_col].astype(int) == tmp[rule_action_col].astype(int)).astype(int)
    tmp["regime_label"] = np.where(tmp[regime_col] > 0, "works (+1)", "fails (-1)")
    tmp["signal_sign"] = np.where(tmp[signal_col] > 0, "signal > 0", "signal < 0")

    agree_regime = tmp.groupby("regime_label")["agree"].mean().reindex(["works (+1)", "fails (-1)"])
    agree_heat = (
        tmp.groupby(["regime_label", "signal_sign"])["agree"]
        .mean()
        .unstack("signal_sign")
        .reindex(["works (+1)", "fails (-1)"])
    )

    print("=== Agreement rate by regime ===")
    print(agree_regime)
    print("\n=== Agreement by signal sign x regime ===")
    print(agree_heat)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(agree_regime.index, agree_regime.values,
                color=[WORKS_COLOR, FAILS_COLOR], edgecolor="black", alpha=0.9)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Agreement rate")
    axes[0].set_title("Agreement by Regime")
    axes[0].grid(axis="y", alpha=0.2)

    im = axes[1].imshow(agree_heat.values, aspect="auto", vmin=0, vmax=1, cmap="Blues", alpha=0.85)
    axes[1].set_yticks(np.arange(len(agree_heat.index)))
    axes[1].set_yticklabels(agree_heat.index)
    axes[1].set_xticks(np.arange(len(agree_heat.columns)))
    axes[1].set_xticklabels(agree_heat.columns)
    axes[1].set_title("Agreement by Signal x Regime")

    for i in range(agree_heat.shape[0]):
        for j in range(agree_heat.shape[1]):
            v = agree_heat.values[i, j]
            if np.isfinite(v):
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11)

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.04)
    cbar.set_label("Agreement")

    fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if show:
        plt.show()
    return fig, agree_regime, agree_heat
