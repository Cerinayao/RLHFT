from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from rlhft.models.xgb_inventory import (
    CLASS_TO_INVENTORY,
    FEATURE_RENAME_MAP,
    INV_VALUES,
)


INV_TO_COL = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
COL_TO_INV = {v: k for k, v in INV_TO_COL.items()}


def extract_xgb_leaf_paths(
    model,
    feature_cols: list[str],
    class_to_inventory: dict[int, int] = CLASS_TO_INVENTORY,
) -> pd.DataFrame:
    """Walk the booster trees and emit one row per leaf with its path."""
    booster = model.get_booster()
    booster.feature_names = feature_cols

    tree_df = booster.trees_to_dataframe()
    n_classes = len(class_to_inventory)

    tree_df["class"] = tree_df["Tree"] % n_classes
    tree_df["inventory"] = tree_df["class"].map(class_to_inventory)

    children = defaultdict(list)
    nodes = {}

    for _, row in tree_df.iterrows():
        tree_id = int(row["Tree"])
        node_id = int(row["Node"])
        nodes[(tree_id, node_id)] = row

        if row["Feature"] != "Leaf":
            yes_node = int(str(row["Yes"]).split("-")[-1])
            no_node = int(str(row["No"]).split("-")[-1])
            children[(tree_id, node_id)].append((yes_node, "<", row["Feature"], float(row["Split"])))
            children[(tree_id, node_id)].append((no_node, ">=", row["Feature"], float(row["Split"])))

    rules: list[dict] = []

    def dfs(tree_id: int, node_id: int, path: list) -> None:
        row = nodes[(tree_id, node_id)]
        if row["Feature"] == "Leaf":
            rules.append({
                "tree": tree_id,
                "class": int(row["class"]),
                "inventory": int(row["inventory"]),
                "leaf_value": float(row["Gain"]),
                "cover": float(row["Cover"]),
                "path": path.copy(),
            })
            return
        for child_id, op, feature, split in children[(tree_id, node_id)]:
            dfs(tree_id, child_id, path + [(feature, op, split)])

    for tree_id in sorted(tree_df["Tree"].unique()):
        dfs(int(tree_id), 0, [])

    rules_df = pd.DataFrame(rules)
    rules_df["score"] = rules_df["leaf_value"] * np.log1p(rules_df["cover"])
    rules_df["abs_score"] = rules_df["score"].abs()
    return rules_df


def rule_applies_to_row(row: pd.Series, path: list) -> bool:
    for feature, op, split in path:
        if op == "<":
            if not row[feature] < split:
                return False
        else:
            if not row[feature] >= split:
                return False
    return True


def build_rule_matrix(X: pd.DataFrame, rules_df: pd.DataFrame) -> np.ndarray:
    masks = []
    for _, rule in rules_df.iterrows():
        mask = X.apply(lambda r: rule_applies_to_row(r, rule["path"]), axis=1).values
        masks.append(mask)
    return np.asarray(masks, dtype=bool)


def path_to_text(path: list, rename_map: dict[str, str] = FEATURE_RENAME_MAP) -> str:
    parts = []
    for feature, op, split in path:
        feature_name = rename_map.get(feature, feature)
        parts.append(f"{feature_name} {op} {split:.3f}")
    return " AND ".join(parts)


def predict_from_selected_rules(X: pd.DataFrame, selected_rules: pd.DataFrame) -> pd.Series:
    preds = []
    for _, row in X.iterrows():
        scores = {inv: 0.0 for inv in [-2, -1, 0, 1, 2]}
        for _, rule in selected_rules.iterrows():
            if rule_applies_to_row(row, rule["path"]):
                scores[int(rule["inventory"])] += float(rule["score"])
        if all(v == 0 for v in scores.values()):
            pred_inv = 0
        else:
            pred_inv = max(scores, key=scores.get)
        preds.append(pred_inv)
    return pd.Series(preds, index=X.index)


def greedy_select_rules_for_fidelity(
    X: pd.DataFrame,
    rules_df: pd.DataFrame,
    y_xgb_inventory: np.ndarray,
    *,
    target_fidelity: float = 0.90,
    max_rules: int = 500,
    candidate_top_n: int = 5000,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Greedy add rules; stop when reaching target fidelity vs XGB labels."""
    rules = (
        rules_df.sort_values("abs_score", ascending=False)
        .head(candidate_top_n)
        .reset_index(drop=True)
    )

    rule_matrix = build_rule_matrix(X, rules)

    selected_indices: list[int] = []
    current_scores = np.zeros((len(X), 5))

    best_fidelity = 0.0
    best_pred = np.zeros(len(X), dtype=int)
    remaining = set(range(len(rules)))

    for step in range(max_rules):
        best_candidate = None
        best_candidate_fidelity = best_fidelity
        best_candidate_scores = None
        best_candidate_pred = None

        for i in list(remaining):
            rule = rules.iloc[i]
            inv = int(rule["inventory"])
            col = INV_TO_COL[inv]
            score = float(rule["score"])
            mask = rule_matrix[i]

            trial_scores = current_scores.copy()
            trial_scores[mask, col] += score

            pred_cols = np.argmax(trial_scores, axis=1)
            pred_inv = np.array([COL_TO_INV[c] for c in pred_cols])

            zero_mask = np.all(trial_scores == 0, axis=1)
            pred_inv[zero_mask] = 0

            fid = accuracy_score(y_xgb_inventory, pred_inv)
            if fid > best_candidate_fidelity:
                best_candidate = i
                best_candidate_fidelity = fid
                best_candidate_scores = trial_scores
                best_candidate_pred = pred_inv

        if best_candidate is None:
            print(f"No more improvement at step {step}. Best fidelity={best_fidelity:.3f}")
            break

        selected_indices.append(best_candidate)
        remaining.remove(best_candidate)
        current_scores = best_candidate_scores
        best_fidelity = best_candidate_fidelity
        best_pred = best_candidate_pred

        print(f"Step {step + 1}: rules={len(selected_indices)}, fidelity={best_fidelity:.3f}")

        if best_fidelity >= target_fidelity:
            print(f"Reached target fidelity {best_fidelity:.3f}")
            break

    selected_rules = rules.iloc[selected_indices].copy().reset_index(drop=True)
    selected_rules["rule_text"] = selected_rules["path"].apply(path_to_text)
    return selected_rules, pd.Series(best_pred, index=X.index), best_fidelity


def extract_and_select_rules(
    *,
    model,
    feature_cols: list[str],
    X_test: pd.DataFrame,
    xgb_inv_targets: np.ndarray,
    target_fidelity: float = 0.90,
    max_rules: int = 600,
    candidate_top_n: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, float]:
    """Convenience: extract leaf paths and greedy-select for fidelity."""
    raw = extract_xgb_leaf_paths(model, feature_cols)
    selected, pred, fidelity = greedy_select_rules_for_fidelity(
        X_test,
        raw,
        xgb_inv_targets,
        target_fidelity=target_fidelity,
        max_rules=max_rules,
        candidate_top_n=candidate_top_n,
    )
    return raw, selected, pred, fidelity


__all__ = [
    "INV_VALUES",
    "extract_xgb_leaf_paths",
    "build_rule_matrix",
    "path_to_text",
    "predict_from_selected_rules",
    "greedy_select_rules_for_fidelity",
    "extract_and_select_rules",
]
