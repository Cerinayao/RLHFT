from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_predicted_inventory_2asset(
    df: pd.DataFrame,
    pred_n: pd.DataFrame,
    *,
    col_a: str = "ESH4",
    col_b: str = "NQH4",
    mult_a: float = 50.0,
    mult_b: float = 20.0,
    trading_start: str = "00:00",
    trading_end: str = "16:00",
    cost_per_trade: float = 0.0,
    inv_penalty: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Backtest a predicted-inventory time series for two assets."""
    tmp = pd.concat([df[[col_a, col_b]], pred_n], axis=1).dropna()
    tmp = tmp.between_time(trading_start, trading_end).copy()

    n_a_col = f"xgb_n_{col_a}"
    n_b_col = f"xgb_n_{col_b}"

    tmp["dPa_next"] = tmp[col_a].shift(-1) - tmp[col_a]
    tmp["dPb_next"] = tmp[col_b].shift(-1) - tmp[col_b]

    tmp["dn_a"] = tmp[n_a_col].diff().fillna(tmp[n_a_col]).abs()
    tmp["dn_b"] = tmp[n_b_col].diff().fillna(tmp[n_b_col]).abs()

    tmp["gross_pnl"] = (
        tmp[n_a_col] * tmp["dPa_next"] * mult_a
        + tmp[n_b_col] * tmp["dPb_next"] * mult_b
    )
    tmp["trade_cost"] = cost_per_trade * (tmp["dn_a"] + tmp["dn_b"])
    tmp["inventory_cost"] = inv_penalty * (
        tmp[n_a_col] ** 2 + tmp[n_b_col] ** 2
    )

    tmp["pnl"] = tmp["gross_pnl"] - tmp["trade_cost"]
    tmp["reward"] = tmp["pnl"] - tmp["inventory_cost"]

    tmp["cum_pnl"] = tmp["pnl"].fillna(0).cumsum()
    tmp["cum_reward"] = tmp["reward"].fillna(0).cumsum()

    pnl = tmp["pnl"].dropna()
    reward = tmp["reward"].dropna()
    drawdown = tmp["cum_pnl"] - tmp["cum_pnl"].cummax()

    summary = pd.Series(
        {
            "cum_pnl": pnl.sum(),
            "cum_reward": reward.sum(),
            "mean_pnl": pnl.mean(),
            "mean_reward": reward.mean(),
            "std_pnl": pnl.std(),
            "std_reward": reward.std(),
            "pnl_sharpe": np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 0 else np.nan,
            "reward_sharpe": np.sqrt(252) * reward.mean() / reward.std() if reward.std() > 0 else np.nan,
            "max_drawdown": drawdown.min(),
            "avg_abs_inv_a": tmp[n_a_col].abs().mean(),
            "avg_abs_inv_b": tmp[n_b_col].abs().mean(),
            "avg_turnover": (tmp["dn_a"] + tmp["dn_b"]).mean(),
            "N": len(pnl),
        }
    )
    return tmp, summary
