from __future__ import annotations

import numpy as np
import pandas as pd

from rlhft.config import ZScoreConfig


def make_walkforward_zscore(
    signal: pd.Series,
    cfg: ZScoreConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling z-score with lag to prevent lookahead."""
    s = signal.astype(float)
    min_periods = cfg.z_min_periods
    if min_periods is None:
        min_periods = max(20, cfg.z_window // 5)

    mu = s.rolling(cfg.z_window, min_periods=min_periods).mean().shift(cfg.z_lag)
    sd = s.rolling(cfg.z_window, min_periods=min_periods).std(ddof=0).shift(cfg.z_lag)
    sd = sd.replace(0.0, np.nan)

    z = (s - mu) / sd
    return z, mu, sd


def build_discrete_2asset_input(
    df_state_all: pd.DataFrame,
    *,
    col_a: str,
    col_b: str,
    use_price_fallback: bool = False,
) -> pd.DataFrame:
    """Build discrete 2-asset input from df_state_all using s_* signal columns."""
    df = df_state_all.copy().sort_index()

    s_a = f"s_{col_a}"
    s_b = f"s_{col_b}"

    needed_s = [s_a, s_b]
    if all(c in df.columns for c in needed_s):
        df["sig_a"] = df[s_a].astype(float)
        df["sig_b"] = df[s_b].astype(float)
    else:
        if not use_price_fallback:
            raise ValueError(f"Missing residual columns among {needed_s}.")
        df["sig_a"] = df[col_a].astype(float)
        df["sig_b"] = df[col_b].astype(float)

    out = df[[col_a, col_b, "sig_a", "sig_b"]].dropna().copy()
    return out


def quantize_z(
    zs: pd.Series,
    idx: pd.Index,
    z_step: float,
    z_clip: float | None,
) -> pd.Series:
    """Discretize z-score values by rounding to nearest z_step."""
    vals = zs.iloc[:-1].astype(float).values
    out = np.empty(len(vals), dtype=float)
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            out[i] = 0.0
            continue
        if z_clip is not None:
            v = float(np.clip(v, -z_clip, z_clip))
        out[i] = float(z_step * np.round(v / z_step))
    return pd.Series(out, index=idx, dtype=float)


def quantize_conf_sign(x: pd.Series) -> pd.Series:
    """Compress continuous confidence score to {-1, 0, 1}."""
    vals = x.astype(float).values
    out = np.zeros(len(vals), dtype=int)
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            out[i] = 0
        elif v > 0:
            out[i] = 1
        elif v < 0:
            out[i] = -1
        else:
            out[i] = 0
    return pd.Series(out, index=x.index, dtype=int)


def compute_regime_confidence(
    z_signal: pd.Series,
    dP_next: pd.Series,
    regime_window: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute rolling signal-confidence score and its quantized sign.

    c_t = rolling mean of sign(signal_{t-1}) * sign(return_t)
    Returns (continuous_confidence, quantized_sign).
    """
    eff = np.sign(z_signal.shift(1).iloc[:-1]) * np.sign(dP_next)
    c = eff.rolling(regime_window, min_periods=1).mean().fillna(0.0)
    cq = quantize_conf_sign(c)
    return c, cq
