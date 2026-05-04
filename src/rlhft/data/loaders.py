from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rlhft.config import DataConfig, ScalingConfig
from rlhft.data.kdb import KDBConnection


def to_pandas(x: Any) -> pd.DataFrame:
    """Convert common kdb/qpython outputs to a pandas DataFrame."""
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, (np.recarray, np.ndarray)) and getattr(x, "dtype", None) is not None and x.dtype.names:
        return pd.DataFrame.from_records(x)
    if isinstance(x, dict):
        return pd.DataFrame(x)
    return pd.DataFrame(x)


def decode_syms(lst: list) -> list[str]:
    return [s.decode() if isinstance(s, (bytes, bytearray)) else str(s) for s in lst]


def sym_prefix(sym: str) -> str:
    s = sym.decode() if isinstance(sym, (bytes, bytearray)) else str(sym)
    return "".join(ch for ch in s if ch.isalpha())[:2].upper()


def scale_for_sym(sym: str, scales: dict[str, float]) -> float:
    pfx = sym_prefix(sym)
    return scales.get(pfx, 100.0)


def sanitize_midprice_series(series: pd.Series) -> pd.Series:
    """Coerce malformed midprices to NaN before downstream signal construction."""
    out = pd.to_numeric(series, errors="coerce")
    out = out.where(np.isfinite(out) & (out > 0.0), np.nan)

    positive = out.dropna()
    if positive.empty:
        return out

    median = float(positive.median())
    if not np.isfinite(median) or median <= 0.0:
        return out

    # Reject values that are implausibly far from the day's typical level,
    # e.g. tiny denormals like 2e-312 from q/python conversion glitches.
    lower = median * 0.5
    upper = median * 1.5
    return out.where((out >= lower) & (out <= upper), np.nan)


def query_active_symbols(
    conn: KDBConnection,
    cfg: DataConfig,
) -> list[str]:
    inst_list = "; ".join([f"`{x}" for x in cfg.instruments])
    q_active = f"""
    8 sublist
    select v, sym
    from `v xdesc
      (select v: sum siz by sym
       from trade
       where date = {cfg.date_active},
             sym2inst[sym] in ({inst_list}))
    """
    res = conn.execute(q_active)
    df_res = to_pandas(res)

    if "sym" not in df_res.columns:
        raise ValueError("Active-symbol query did not return a 'sym' column.")

    sym_active = decode_syms(df_res["sym"].tolist())
    print("Active symbols:", sym_active)

    # Prefer specific symbols if present
    chosen = [s for s in cfg.preferred_symbols if s in sym_active]
    if len(chosen) == len(cfg.preferred_symbols):
        sym_active = chosen
        print("Using symbols:", sym_active)

    return sym_active


def query_midprices(
    conn: KDBConnection,
    sym: str,
    date_quotes: str,
    time_grid_q: str,
) -> pd.DataFrame:
    q_mid = f"""
    aj[`time;
        ([] time: {time_grid_q});
        select midprice1: (last bid + last ask) % 2 by time
        from quote where date={date_quotes}, sym=`{sym}
    ]
    """
    raw = conn.execute(q_mid)
    df_sym = to_pandas(raw)

    if "time" not in df_sym.columns:
        raise ValueError(f"Midprice query for {sym} on {date_quotes} did not return 'time' column.")

    if "midprice1" in df_sym.columns:
        df_sym = df_sym.rename(columns={"midprice1": sym})
    elif sym not in df_sym.columns:
        raise ValueError(f"Midprice query for {sym} on {date_quotes} did not return 'midprice1' or '{sym}'.")

    # Missing quote buckets can come back as 0.0 from kdb/q conversions; treat them as missing.
    df_sym[sym] = sanitize_midprice_series(df_sym[sym])

    return df_sym


def load_single_day(
    conn: KDBConnection,
    sym_active: list[str],
    date_quotes: str,
    cfg: DataConfig,
    cfg_scale: ScalingConfig,
) -> pd.DataFrame | None:
    df_midprice_all = None

    for sym in sym_active:
        df_sym = query_midprices(conn, sym, date_quotes, cfg.time_grid_q)
        if df_sym is None or len(df_sym) == 0:
            continue
        df_midprice_all = (
            df_sym if df_midprice_all is None
            else df_midprice_all.merge(df_sym, on="time", how="left")
        )

    if df_midprice_all is None or df_midprice_all.empty:
        return None

    df_midprice_all = df_midprice_all.copy()
    for sym in sym_active:
        if sym in df_midprice_all.columns:
            s = scale_for_sym(sym, cfg_scale.scales)
            df_midprice_all[sym] = df_midprice_all[sym].astype(float) / s

    return df_midprice_all


def load_multi_day(
    conn: KDBConnection,
    sym_active: list[str],
    cfg: DataConfig,
    cfg_scale: ScalingConfig,
) -> pd.DataFrame:
    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="B")
    dates_q = [d.strftime("%Y.%m.%d") for d in dates]

    all_days: list[pd.DataFrame] = []

    for date_quotes in dates_q:
        df_day = load_single_day(conn, sym_active, date_quotes, cfg, cfg_scale)
        if df_day is None:
            continue

        if cfg.drop_all_nan_rows:
            price_cols = [c for c in df_day.columns if c != "time"]
            df_day = df_day.dropna(subset=price_cols, how="all")

        if df_day.empty:
            continue

        date_anchor = date_quotes.replace(".", "-")
        df_day["date"] = date_quotes
        df_day["datetime"] = (
            pd.to_datetime(date_anchor)
            + pd.to_timedelta(df_day["time"].astype("int64"), unit="ns")
        )
        all_days.append(df_day)

    if len(all_days) == 0:
        raise ValueError("No days returned data in the requested range.")

    df_mid_all = pd.concat(all_days, ignore_index=True)
    df_mid_all = df_mid_all.sort_values("datetime").reset_index(drop=True)
    return df_mid_all
