from __future__ import annotations

import numpy as np
import pandas as pd


def build_xrl_policy_df(
    df: pd.DataFrame,
    *,
    signal_col: str,
    inventory_col: str,
    action: pd.Series,
    regime_state: pd.Series,
    start_date: str = "2024-02-19",
    trading_start: str = "00:00",
    trading_end: str = "16:00",
) -> pd.DataFrame:
    """Build the XRL analysis frame: signal, inventory, action, regime."""
    tmp = pd.DataFrame({
        "signal": df[signal_col].sort_index().astype(float),
        "inventory": df[inventory_col].sort_index().astype(float),
        "action": action.sort_index().astype(float),
        "regime": regime_state.sort_index().astype(float),
    })

    tmp = tmp[tmp.index >= pd.Timestamp(start_date)]
    tmp = tmp.between_time(trading_start, trading_end)
    tmp = tmp.dropna()
    tmp = tmp[tmp["regime"].isin([-1, 1])].copy()

    tmp["regime_label"] = np.where(tmp["regime"] > 0, "works (+1)", "fails (-1)")
    tmp["abs_action"] = tmp["action"].abs()
    return tmp
