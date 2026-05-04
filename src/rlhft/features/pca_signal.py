from __future__ import annotations

import numpy as np
import pandas as pd

from rlhft.config import SignalConfig


def compute_ewm_pca_signal(
    df_prices: pd.DataFrame,
    cfg: SignalConfig,
    col_prefix: str = "",
) -> pd.DataFrame:
    """Compute exponentially-weighted PCA-based signal (A/B/M statistics, pc1/pc2, per-asset signal s_*)."""
    df = df_prices.copy()
    tickers = list(df.columns)
    N = len(tickers)
    alpha = float(np.exp(-1.0 / cfg.window))
    z = df.values.astype(float)
    T = z.shape[0]

    finite_abs = np.abs(z[np.isfinite(z)])
    if finite_abs.size:
        median_abs = float(np.nanmedian(finite_abs))
        max_abs = float(np.nanmax(finite_abs))
    else:
        median_abs = 1.0
        max_abs = 1.0

    max_safe_abs = np.sqrt(np.finfo(float).max) / 4.0
    scale_floor = max_abs / max_safe_abs if max_abs > 0.0 else 1.0
    value_scale = max(median_abs, scale_floor, cfg.eps)
    value_scale_sq = float(np.square(np.float64(min(value_scale, np.sqrt(np.finfo(float).max)))))
    z_scaled = z / value_scale

    A = np.zeros(N, dtype=float)
    B = np.zeros(N, dtype=float)
    M = np.zeros((N, N), dtype=float)

    A_hist = np.zeros((T, N))
    B_hist = np.zeros((T, N))
    M_hist = np.zeros((T, N, N))
    s_hist = np.zeros((T, N))

    pc1_hist = np.full(T, np.nan, dtype=float)
    pc2_hist = np.full(T, np.nan, dtype=float)

    prev_v1 = None
    prev_v2 = None

    for t in range(T):
        zt = z_scaled[t]
        mask = np.isfinite(zt).astype(float)

        A = alpha * A + mask * np.where(np.isfinite(zt), zt, 0.0)
        B = alpha * B + mask

        z_filled = np.where(np.isfinite(zt), zt, 0.0)
        M = alpha * M + np.outer(z_filled, z_filled)

        mu = A / np.maximum(B, cfg.eps)

        Cov = M / np.maximum(np.mean(B), cfg.eps) - np.outer(mu, mu)
        Cov = np.nan_to_num(Cov, nan=0.0, posinf=0.0, neginf=0.0)
        Cov = 0.5 * (Cov + Cov.T)

        diag = np.diag(Cov)
        diag = np.where(np.isfinite(diag), diag, cfg.eps)
        sigma = np.sqrt(np.maximum(diag, cfg.eps))

        vals, vecs = np.linalg.eigh(Cov)
        order = np.argsort(vals)[::-1]
        vecs = vecs[:, order]

        v1 = vecs[:, 0]
        v2 = vecs[:, 1] if N > 1 else np.full(N, np.nan)

        if prev_v1 is not None and np.dot(v1, prev_v1) < 0:
            v1 = -v1
        prev_v1 = v1.copy()

        if N > 1 and prev_v2 is not None and np.dot(v2, prev_v2) < 0:
            v2 = -v2
        if N > 1:
            prev_v2 = v2.copy()

        z_std = (np.where(np.isfinite(zt), zt, mu) - mu) / sigma

        x_centered = np.where(np.isfinite(zt), zt, mu) - mu
        pc1_hist[t] = float(v1.T @ x_centered)
        if N > 1:
            pc2_hist[t] = float(v2.T @ x_centered)

        xi = float(v1.T @ z_std)
        z_hat = mu + (v1 * sigma) * xi
        s = z_hat - np.where(np.isfinite(zt), zt, z_hat)

        A_hist[t] = A * value_scale
        B_hist[t] = B
        M_hist[t] = M * value_scale_sq
        s_hist[t] = s * value_scale

    out = df.copy()

    for i, tk in enumerate(tickers):
        out[f"{col_prefix}A_{tk}"] = A_hist[:, i]
        out[f"{col_prefix}s_{tk}"] = s_hist[:, i]
        out[f"{col_prefix}B_{tk}"] = B_hist[:, i]

    for i, tki in enumerate(tickers):
        for j, tkj in enumerate(tickers):
            out[f"{col_prefix}M_{tki}_{tkj}"] = M_hist[:, i, j]

    out[f"{col_prefix}pc1"] = pc1_hist * value_scale
    out[f"{col_prefix}pc2"] = pc2_hist * value_scale

    return out


def attach_datetime_from_ns_index(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Index is ns since midnight; anchor to date_str."""
    td = pd.to_timedelta(df.index.astype("int64"), unit="ns")
    base = pd.to_datetime(date_str)
    out = df.copy()
    out.index = base + td
    out.index.name = "datetime"
    return out
