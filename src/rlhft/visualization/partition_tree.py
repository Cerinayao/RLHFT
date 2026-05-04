from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_RENAME = {
    "zqa": "Signal_ES",
    "zqb": "Signal_NQ",
    "n_a_prev": "Prev_Position_ES",
    "n_b_prev": "Prev_Position_NQ",
    "regime_a": "Regime_ES",
    "regime_b": "Regime_NQ",
}


def _ensure_graphviz_path() -> None:
    if shutil.which("dot") is None:
        os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")


def _leaf_color(action: int) -> str:
    if action > 0:
        return "lightblue"
    if action < 0:
        return "lightcoral"
    return "lightgray"


def _rule_bounds(path: list) -> dict[str, list[float]]:
    bounds: dict[str, list[float]] = {}
    for feature, op, split in path:
        if feature not in bounds:
            bounds[feature] = [-math.inf, math.inf]
        if op == "<":
            bounds[feature][1] = min(bounds[feature][1], float(split))
        else:
            bounds[feature][0] = max(bounds[feature][0], float(split))
    return bounds


def _extract_thresholds(rules: pd.DataFrame, feature: str) -> list[float]:
    vals = set()
    for _, rule in rules.iterrows():
        for feat, _, split in rule["path"]:
            if feat == feature:
                vals.add(float(split))
    return sorted(vals)


def _make_intervals(thresholds: list[float]) -> list[tuple[float, float]]:
    bounds = [-math.inf] + thresholds + [math.inf]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _interval_label(feature: str, lo: float, hi: float, rename_map: dict[str, str]) -> str:
    name = rename_map.get(feature, feature)
    if lo == -math.inf:
        return f"{name} < {hi:.3f}"
    if hi == math.inf:
        return f"{name} >= {lo:.3f}"
    return f"{lo:.3f} <= {name} < {hi:.3f}"


def _rule_applies_to_interval(rule_bound: dict[str, list[float]], feature: str, lo: float, hi: float) -> bool:
    if feature not in rule_bound:
        return True
    r_lo, r_hi = rule_bound[feature]
    return lo >= r_lo and hi <= r_hi


def _prepare_rules(rules: pd.DataFrame, *, min_score: float = 0.03, top_n: int = 50) -> pd.DataFrame:
    df = rules.copy()
    df = df[df["score"] > min_score].copy()
    df = df.sort_values("score", ascending=False).head(top_n).copy()
    df["bounds"] = df["path"].apply(_rule_bounds)
    return df.reset_index(drop=True)


def _leaf_action_from_rules(rules: pd.DataFrame) -> tuple[int, float]:
    scores = {-2: 0.0, -1: 0.0, 0: 0.0, 1: 0.0, 2: 0.0}
    for _, rule in rules.iterrows():
        scores[int(rule["inventory"])] += float(rule["score"])
    if all(v == 0 for v in scores.values()):
        return 0, 0.0
    action = max(scores, key=scores.get)
    return action, scores[action]


def plot_partition_decision_tree(
    rules: pd.DataFrame,
    feature_order: list[str],
    *,
    title: str = "Partition Decision Tree",
    filename: str | Path = "partition_tree",
    min_score: float = 0.03,
    top_n: int = 50,
    rename_map: dict[str, str] | None = None,
    max_depth: int | None = 3,
) -> Path | None:
    """Render the notebook-style partition tree and return the PNG path."""
    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz Python package not installed; skipping partition tree.")
        return None

    _ensure_graphviz_path()
    if shutil.which("dot") is None:
        print("Graphviz `dot` binary not found on PATH; skipping partition tree.")
        return None

    if rename_map is None:
        rename_map = DEFAULT_RENAME

    df = _prepare_rules(rules, min_score=min_score, top_n=top_n)
    if max_depth is not None:
        feature_order = feature_order[:max_depth]

    dot = Digraph(comment=title)
    dot.attr(rankdir="LR", splines="polyline")
    dot.attr("node", shape="box", style="rounded,filled", fontsize="10")
    dot.attr("edge", fontsize="9")

    node_counter = [0]

    def new_id(prefix: str = "n") -> str:
        node_counter[0] += 1
        return f"{prefix}_{node_counter[0]}"

    root = "root"
    dot.node(root, title, fillcolor="lightgray")

    def recurse(parent_id: str, cur_rules: pd.DataFrame, depth: int) -> None:
        if depth >= len(feature_order) or len(cur_rules) == 0:
            action, strength = _leaf_action_from_rules(cur_rules)
            leaf_id = new_id("leaf")
            dot.node(
                leaf_id,
                f"a = {action}\nscore = {strength:.3f}\nN rules = {len(cur_rules)}",
                fillcolor=_leaf_color(action),
            )
            dot.edge(parent_id, leaf_id)
            return

        feature = feature_order[depth]
        thresholds = _extract_thresholds(cur_rules, feature)
        if len(thresholds) == 0:
            recurse(parent_id, cur_rules, depth + 1)
            return

        feature_name = rename_map.get(feature, feature)
        split_node = new_id("split")
        dot.node(split_node, feature_name, fillcolor="white")
        dot.edge(parent_id, split_node)

        for lo, hi in _make_intervals(thresholds):
            child_rules = []
            for _, rule in cur_rules.iterrows():
                if _rule_applies_to_interval(rule["bounds"], feature, lo, hi):
                    child_rules.append(rule)
            if not child_rules:
                continue

            child_df = pd.DataFrame(child_rules)
            branch_id = new_id("branch")
            dot.node(branch_id, _interval_label(feature, lo, hi, rename_map), fillcolor="white")
            dot.edge(split_node, branch_id)
            recurse(branch_id, child_df, depth + 1)

    recurse(root, df, 0)

    out_path = Path(filename)
    rendered = dot.render(str(out_path), format="png", cleanup=True)
    return Path(rendered)

