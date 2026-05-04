from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_RENAME = {
    "Signal_ES": "Signal ES",
    "Signal_NQ": "Signal NQ",
    "Regime_ES": "Regime ES",
    "Regime_NQ": "Regime NQ",
    "Prev_Position_ES": "Prev Pos ES",
    "Prev_Position_NQ": "Prev Pos NQ",
}


def _ensure_graphviz_path() -> None:
    """Make sure the homebrew `dot` binary is on PATH (matches notebook cell 0)."""
    if shutil.which("dot") is None:
        os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")


def _copy_rule(rule: dict) -> dict:
    new_rule = dict(rule)
    new_rule["path"] = list(rule["path"])
    return new_rule


def _majority_leaf(rules: list[dict]) -> dict:
    if not rules:
        return {"type": "leaf", "inventory": 0, "score": 0.0, "count": 0}

    by_inventory: dict[int, dict[str, float]] = {}
    for r in rules:
        inv = int(r["inventory"])
        by_inventory.setdefault(inv, {"score": 0.0, "count": 0})
        by_inventory[inv]["score"] += float(r["score"])
        by_inventory[inv]["count"] += 1

    inventory, stats = max(
        by_inventory.items(),
        key=lambda kv: (kv[1]["count"], abs(kv[1]["score"]), kv[0]),
    )
    avg_score = stats["score"] / max(stats["count"], 1)
    return {"type": "leaf", "inventory": inventory, "score": avg_score, "count": len(rules)}


def _choose_split(rules: list[dict]) -> tuple[str, float] | None:
    counter: Counter = Counter()
    for r in rules:
        seen: set = set()
        for feature, op, split in r["path"]:
            key = (feature, split)
            if key in seen:
                continue
            seen.add(key)
            counter[key] += 1

    best = None
    best_score = (-1, -1)
    for feature, split in counter:
        yes_count = 0
        no_count = 0
        for r in rules:
            ops = [op for f, op, s in r["path"] if f == feature and s == split]
            if "<" in ops:
                yes_count += 1
            if any(op != "<" for op in ops):
                no_count += 1
        score = (min(yes_count, no_count), yes_count + no_count)
        if score > best_score:
            best = (feature, split)
            best_score = score

    if best is None or best_score[0] == 0:
        return None
    return best


def _build_binary_tree(rules: list[dict]) -> dict | None:
    if not rules:
        return None

    split = _choose_split(rules)
    if split is None:
        return _majority_leaf(rules)

    feature, split_value = split
    yes_rules: list[dict] = []
    no_rules: list[dict] = []

    for rule in rules:
        remaining = []
        matches_yes = False
        matches_no = False

        for f, op, s in rule["path"]:
            if f == feature and s == split_value:
                if op == "<":
                    matches_yes = True
                else:
                    matches_no = True
            else:
                remaining.append((f, op, s))

        if matches_yes:
            new_rule = _copy_rule(rule)
            new_rule["path"] = remaining
            yes_rules.append(new_rule)
        if matches_no:
            new_rule = _copy_rule(rule)
            new_rule["path"] = remaining
            no_rules.append(new_rule)

    if not yes_rules or not no_rules:
        return _majority_leaf(rules)

    return {
        "type": "split",
        "feature": feature,
        "split": split_value,
        "yes": _build_binary_tree(yes_rules),
        "no": _build_binary_tree(no_rules),
    }


def _leaf_color(action: int) -> str:
    if action > 0:
        return "lightblue"
    if action < 0:
        return "lightcoral"
    return "lightgray"


def _render_tree(node, dot, parent_id: str, edge_label: str, counter: list[int], rename_map: dict[str, str]) -> None:
    if node is None:
        return
    counter[0] += 1
    node_id = f"n{counter[0]}"

    if node["type"] == "leaf":
        dot.node(
            node_id,
            f"a = {node['inventory']}\\nscore = {node['score']:.3f}\\nn = {node['count']}",
            fillcolor=_leaf_color(node["inventory"]),
        )
        dot.edge(parent_id, node_id, label=edge_label)
        return

    feature_name = rename_map.get(node["feature"], node["feature"])
    dot.node(node_id, f"{feature_name} < {node['split']:.3f}?", fillcolor="white")
    dot.edge(parent_id, node_id, label=edge_label)
    _render_tree(node["yes"], dot, node_id, "yes", counter, rename_map)
    _render_tree(node["no"], dot, node_id, "no", counter, rename_map)


def plot_rule_tree_from_df(
    df: pd.DataFrame,
    title: str = "Rule Tree",
    top_n: int = 15,
    filename: str | Path = "rule_tree",
    rename_map: dict[str, str] | None = None,
) -> Path | None:
    """Build a binary decision tree visualization from a rules dataframe.

    Returns the rendered PNG path on disk, or None if graphviz is unavailable.
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz Python package not installed; skipping rule tree.")
        return None

    _ensure_graphviz_path()
    if shutil.which("dot") is None:
        print("Graphviz `dot` binary not found on PATH; skipping rule tree.")
        return None

    if rename_map is None:
        rename_map = DEFAULT_RENAME

    df_plot = df.copy()
    df_plot["abs_score"] = df_plot["score"].abs()
    df_plot = df_plot.sort_values("abs_score", ascending=False).head(top_n)

    rules = df_plot.to_dict(orient="records")
    tree = _build_binary_tree(rules)

    dot = Digraph(comment=title)
    dot.attr(rankdir="TB", splines="polyline")
    dot.attr("node", shape="box", style="rounded,filled", fontsize="10")
    dot.attr("edge", fontsize="9")

    dot.node("root", title, fillcolor="lightgray")
    _render_tree(tree, dot, "root", "", [0], rename_map)

    out_path = Path(filename)
    rendered = dot.render(str(out_path), format="png", cleanup=True)
    return Path(rendered)
