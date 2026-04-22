from __future__ import annotations

import numpy as np
import pandas as pd

from rlhft.config import RuleConfig, TradingConfig, ZScoreConfig
from rlhft.evaluation.metrics import mean_daily_pnl, max_daily_drawdown
from rlhft.features.zscore import make_walkforward_zscore


def rule_action_from_z(
    z_t: float,
    cfg: RuleConfig,
) -> int:
    """Map z-score to discrete position target in {-2,-1,0,1,2}."""
    if not np.isfinite(z_t):
        return 0
    if abs(z_t) <= cfg.z_flat:
        return 0
    elif z_t >= cfg.z_strong:
        return 2
    elif z_t >= cfg.z_entry:
        return 1
    elif z_t <= -cfg.z_strong:
        return -2
    elif z_t <= -cfg.z_entry:
        return -1
    else:
        return 0


def run_rule_2asset_discrete(
    df: pd.DataFrame,
    *,
    trading_cfg: TradingConfig,
    rule_cfg: RuleConfig,
    zscore_cfg: ZScoreConfig,
    sig_a: str = "sig_a",
    sig_b: str = "sig_b",
) -> dict:
    """Run rule-based discrete 2-asset strategy. Each asset trades on its own z-score."""
    col_a = trading_cfg.col_a
    col_b = trading_cfg.col_b
    df = df.sort_index().copy()

    Pa = df[col_a].astype(float)
    Pb = df[col_b].astype(float)

    za, _, _ = make_walkforward_zscore(df[sig_a], zscore_cfg)
    zb, _, _ = make_walkforward_zscore(df[sig_b], zscore_cfg)

    dPa_next = (Pa.shift(-1) - Pa).iloc[:-1]
    dPb_next = (Pb.shift(-1) - Pb).iloc[:-1]

    idx = df.index[:-1]

    na_prev, nb_prev = 0, 0

    dates = []
    aa_list, ab_list = [], []
    na_list, nb_list = [], []
    reward_list = []
    za_used, zb_used = [], []

    for k in range(len(idx)):
        t = idx[k]

        z_a_t = float(za.iloc[k]) if np.isfinite(za.iloc[k]) else np.nan
        z_b_t = float(zb.iloc[k]) if np.isfinite(zb.iloc[k]) else np.nan

        n_a = rule_action_from_z(z_a_t, rule_cfg)
        n_b = rule_action_from_z(z_b_t, rule_cfg)

        n_a = int(np.clip(n_a, -trading_cfg.inv_limit, trading_cfg.inv_limit))
        n_b = int(np.clip(n_b, -trading_cfg.inv_limit, trading_cfg.inv_limit))

        dn_a = n_a - na_prev
        dn_b = n_b - nb_prev

        dPa = float(dPa_next.iloc[k])
        dPb = float(dPb_next.iloc[k])

        gross = (
            n_a * dPa * trading_cfg.mult_a
            + n_b * dPb * trading_cfg.mult_b
        )
        trade_cost = trading_cfg.cost_per_trade * (abs(dn_a) + abs(dn_b))
        inventory_cost = rule_cfg.inv_penalty * (abs(n_a) + abs(n_b))
        R = gross - trade_cost - inventory_cost

        dates.append(t)
        aa_list.append(dn_a)
        ab_list.append(dn_b)
        na_list.append(n_a)
        nb_list.append(n_b)
        reward_list.append(R)
        za_used.append(z_a_t)
        zb_used.append(z_b_t)

        na_prev, nb_prev = n_a, n_b

    dt = pd.DatetimeIndex(dates)
    reward = pd.Series(reward_list, index=dt, name="reward_$")
    cum = reward.cumsum().rename("cum_reward_$")

    actions_df = pd.DataFrame({
        f"a_{col_a}": aa_list,
        f"a_{col_b}": ab_list,
    }, index=dt)

    n_df = pd.DataFrame({
        f"n_{col_a}": na_list,
        f"n_{col_b}": nb_list,
    }, index=dt)

    z_df = pd.DataFrame({
        f"z_{col_a}": za_used,
        f"z_{col_b}": zb_used,
    }, index=dt)

    return {
        "strategy": "rule_2asset_discrete",
        "reward": reward,
        "cum": cum,
        "actions": actions_df,
        "n": n_df,
        "z_used": z_df,
        "metrics": {
            "mean_daily_pnl_$": mean_daily_pnl(reward),
            "max_drawdown_daily_$": max_daily_drawdown(reward),
        },
    }
