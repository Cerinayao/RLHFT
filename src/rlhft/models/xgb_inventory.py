from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

from rlhft.config import XGBConfig, ZScoreConfig


INV_VALUES = np.array([-2, -1, 0, 1, 2])
INV_TO_CLASS = {v: i for i, v in enumerate(INV_VALUES)}
CLASS_TO_INVENTORY = {i: int(v) for i, v in enumerate(INV_VALUES)}

FEATURE_RENAME_MAP = {
    "zqa": "Signal_ES",
    "zqb": "Signal_NQ",
    "n_a_prev": "Prev_Position_ES",
    "n_b_prev": "Prev_Position_NQ",
    "regime_a": "Regime_ES",
    "regime_b": "Regime_NQ",
}


def build_xgb_state_like_rl(
    df: pd.DataFrame,
    out_rl: dict,
    *,
    col_a: str = "ESH4",
    col_b: str = "NQH4",
    sig_a: str = "sig_a",
    sig_b: str = "sig_b",
    n_a_col: str | None = None,
    n_b_col: str | None = None,
    z_cfg: ZScoreConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build XGB feature frame mirroring the RL state."""
    if n_a_col is None:
        n_a_col = f"n_{col_a}"
    if n_b_col is None:
        n_b_col = f"n_{col_b}"

    df = df.sort_index().copy()

    z_window = z_cfg.z_window
    z_lag = z_cfg.z_lag
    z_step = z_cfg.z_step
    z_clip = z_cfg.z_clip

    def walk_z(x: pd.Series) -> pd.Series:
        mean = x.rolling(z_window, min_periods=30).mean().shift(z_lag)
        std = x.rolling(z_window, min_periods=30).std().shift(z_lag)
        z = (x - mean) / (std + 1e-12)
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def quantize(z: pd.Series) -> pd.Series:
        if z_clip is not None:
            z = z.clip(-z_clip, z_clip)
        return z_step * np.round(z / z_step)

    zqa = quantize(walk_z(df[sig_a]))
    zqb = quantize(walk_z(df[sig_b]))

    regime_a = out_rl["regime_state_a"].rename("regime_a")
    regime_b = out_rl["regime_state_b"].rename("regime_b")

    n_all = pd.concat(
        [
            out_rl["train_n"][[n_a_col, n_b_col]],
            out_rl["test_n"][[n_a_col, n_b_col]],
        ],
        axis=0,
    ).sort_index()
    n_all = n_all[~n_all.index.duplicated(keep="last")]

    tmp = pd.concat(
        [
            df[[col_a, col_b]],
            zqa.rename("zqa"),
            zqb.rename("zqb"),
            regime_a,
            regime_b,
            n_all.rename(columns={n_a_col: "n_a", n_b_col: "n_b"}),
        ],
        axis=1,
    ).dropna()

    tmp["n_a_prev"] = tmp["n_a"].shift(1).fillna(0).astype(int)
    tmp["n_b_prev"] = tmp["n_b"].shift(1).fillna(0).astype(int)

    tmp["target_n_a"] = tmp["n_a"].astype(int)
    tmp["target_n_b"] = tmp["n_b"].astype(int)

    feature_cols = ["zqa", "zqb", "regime_a", "regime_b", "n_a_prev", "n_b_prev"]
    tmp = tmp.dropna(subset=feature_cols + ["target_n_a", "target_n_b"])
    X = tmp[feature_cols].astype(float)

    return tmp, X, feature_cols


def train_xgb_inventory(
    df: pd.DataFrame,
    out_rl: dict,
    *,
    col_a: str,
    col_b: str,
    z_cfg: ZScoreConfig,
    xgb_cfg: XGBConfig,
) -> dict:
    """Train XGBoost classifiers for inventory targets (train/val/final-test split)."""
    tmp, X, feature_cols = build_xgb_state_like_rl(
        df, out_rl, col_a=col_a, col_b=col_b, z_cfg=z_cfg,
    )

    y_a = tmp["target_n_a"].map(INV_TO_CLASS).astype(int)
    y_b = tmp["target_n_b"].map(INV_TO_CLASS).astype(int)

    train_end = pd.Timestamp(xgb_cfg.train_end)
    val_end = pd.Timestamp(xgb_cfg.val_end)

    train_mask = tmp.index <= train_end
    val_mask = (tmp.index > train_end) & (tmp.index <= val_end)
    test_mask = tmp.index > val_end

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    X_test = X.loc[test_mask]

    y_a_train = y_a.loc[train_mask]
    y_a_val = y_a.loc[val_mask]
    y_a_test = y_a.loc[test_mask]

    y_b_train = y_b.loc[train_mask]
    y_b_val = y_b.loc[val_mask]
    y_b_test = y_b.loc[test_mask]

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_a_train_val = pd.concat([y_a_train, y_a_val], axis=0)
    y_b_train_val = pd.concat([y_b_train, y_b_val], axis=0)

    print("Train N:", len(X_train))
    print("Val N:  ", len(X_val))
    print("Train+Val N:", len(X_train_val))
    print("Test N: ", len(X_test))

    params = dict(
        n_estimators=xgb_cfg.n_estimators,
        max_depth=xgb_cfg.max_depth,
        learning_rate=xgb_cfg.learning_rate,
        subsample=xgb_cfg.subsample,
        colsample_bytree=xgb_cfg.colsample_bytree,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=xgb_cfg.random_state,
        reg_lambda=xgb_cfg.reg_lambda,
        reg_alpha=xgb_cfg.reg_alpha,
        min_child_weight=xgb_cfg.min_child_weight,
        tree_method="hist",
    )

    val_model_a = xgb.XGBClassifier(**params)
    val_model_b = xgb.XGBClassifier(**params)
    val_model_a.fit(X_train, y_a_train, verbose=False)
    val_model_b.fit(X_train, y_b_train, verbose=False)

    pred_val_a = INV_VALUES[val_model_a.predict(X_val)]
    pred_val_b = INV_VALUES[val_model_b.predict(X_val)]
    true_val_a = INV_VALUES[y_a_val.values]
    true_val_b = INV_VALUES[y_b_val.values]

    print("\n=== Validation Inventory Accuracy ===")
    print(f"{col_a}: {accuracy_score(true_val_a, pred_val_a):.3f}")
    print(f"{col_b}: {accuracy_score(true_val_b, pred_val_b):.3f}")

    final_model_a = xgb.XGBClassifier(**params)
    final_model_b = xgb.XGBClassifier(**params)
    final_model_a.fit(X_train_val, y_a_train_val, verbose=False)
    final_model_b.fit(X_train_val, y_b_train_val, verbose=False)

    pred_test_a = INV_VALUES[final_model_a.predict(X_test)]
    pred_test_b = INV_VALUES[final_model_b.predict(X_test)]
    true_test_a = INV_VALUES[y_a_test.values]
    true_test_b = INV_VALUES[y_b_test.values]

    print("\n=== Final Test Inventory Accuracy ===")
    print(f"{col_a}: {accuracy_score(true_test_a, pred_test_a):.3f}")
    print(f"{col_b}: {accuracy_score(true_test_b, pred_test_b):.3f}")

    print(f"\n=== {col_a} Test Classification Report ===")
    print(classification_report(true_test_a, pred_test_a, labels=INV_VALUES, zero_division=0))
    print(f"\n=== {col_b} Test Classification Report ===")
    print(classification_report(true_test_b, pred_test_b, labels=INV_VALUES, zero_division=0))

    pred_n_val = pd.DataFrame(
        {f"xgb_n_{col_a}": pred_val_a, f"xgb_n_{col_b}": pred_val_b},
        index=X_val.index,
    )
    pred_n_test = pd.DataFrame(
        {f"xgb_n_{col_a}": pred_test_a, f"xgb_n_{col_b}": pred_test_b},
        index=X_test.index,
    )

    val_acc_a = accuracy_score(true_val_a, pred_val_a)
    val_acc_b = accuracy_score(true_val_b, pred_val_b)
    test_acc_a = accuracy_score(true_test_a, pred_test_a)
    test_acc_b = accuracy_score(true_test_b, pred_test_b)

    return {
        "val_model_a": val_model_a,
        "val_model_b": val_model_b,
        "model_a": final_model_a,
        "model_b": final_model_b,
        "tmp": tmp,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_val": X_val,
        "X_train_val": X_train_val,
        "X_test": X_test,
        "y_a_train": y_a_train,
        "y_a_val": y_a_val,
        "y_a_train_val": y_a_train_val,
        "y_a_test": y_a_test,
        "y_b_train": y_b_train,
        "y_b_val": y_b_val,
        "y_b_train_val": y_b_train_val,
        "y_b_test": y_b_test,
        "pred_n_val": pred_n_val,
        "pred_n_test": pred_n_test,
        "params": params,
        "accuracies": {
            "val_a": val_acc_a,
            "val_b": val_acc_b,
            "test_a": test_acc_a,
            "test_b": test_acc_b,
        },
    }


def xgb_feature_importance(
    model,
    feature_cols: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Get feature importance dataframe with display names."""
    booster = model.get_booster()
    score = booster.get_score(importance_type=importance_type)

    imp = pd.DataFrame({
        "feature_raw": feature_cols,
        "importance": [score.get(f, 0.0) for f in feature_cols],
    })
    imp["feature"] = imp["feature_raw"].map(lambda x: FEATURE_RENAME_MAP.get(x, x))
    return (
        imp[["feature", "importance"]]
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
