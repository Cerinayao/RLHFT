from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from rlhft.config import QLearningConfig, TradingConfig, ZScoreConfig
from rlhft.evaluation.metrics import clip_int, mean_daily_pnl, max_daily_drawdown
from rlhft.features.zscore import (
    make_walkforward_zscore,
    quantize_z,
    compute_regime_confidence,
)


def train_q_learning_2asset_discrete_adaptive(
    df: pd.DataFrame,
    *,
    trading_cfg: TradingConfig,
    zscore_cfg: ZScoreConfig,
    ql_cfg: QLearningConfig,
    sig_a: str = "sig_a",
    sig_b: str = "sig_b",
) -> dict:
    """
    Regime-aware tabular Q-learning for 2-asset discrete pair trading.

    State: (quantized_z_a, quantized_z_b, regime_a, regime_b, inv_a, inv_b)
    Actions: joint (delta_a, delta_b) from action_values grid.
    """
    col_a = trading_cfg.col_a
    col_b = trading_cfg.col_b
    inv_limit = trading_cfg.inv_limit

    rng = np.random.default_rng(ql_cfg.seed)
    df = df.sort_index().copy()

    train_mask = df.index <= pd.Timestamp(ql_cfg.train_end)
    test_mask = ~train_mask
    if train_mask.sum() < 200:
        raise ValueError(
            "Training set too small. "
            f"train_end={ql_cfg.train_end}, train_rows={int(train_mask.sum())}, "
            f"test_rows={int(test_mask.sum())}, "
            f"data_range=[{df.index.min()}, {df.index.max()}]."
        )
    if test_mask.sum() < 200:
        raise ValueError(
            "Test set too small. "
            f"train_end={ql_cfg.train_end}, train_rows={int(train_mask.sum())}, "
            f"test_rows={int(test_mask.sum())}, "
            f"data_range=[{df.index.min()}, {df.index.max()}]."
        )

    Pa = df[col_a].astype(float)
    Pb = df[col_b].astype(float)

    za, _, _ = make_walkforward_zscore(df[sig_a], zscore_cfg)
    zb, _, _ = make_walkforward_zscore(df[sig_b], zscore_cfg)

    dPa_next = (Pa.shift(-1) - Pa).iloc[:-1]
    dPb_next = (Pb.shift(-1) - Pb).iloc[:-1]
    idx = df.index[:-1]

    zqa = quantize_z(za, idx, zscore_cfg.z_step, zscore_cfg.z_clip)
    zqb = quantize_z(zb, idx, zscore_cfg.z_step, zscore_cfg.z_clip)

    c_a, cqa = compute_regime_confidence(za, dPa_next, ql_cfg.regime_window)
    c_b, cqb = compute_regime_confidence(zb, dPb_next, ql_cfg.regime_window)

    one_dim = np.array(sorted(set(int(a) for a in trading_cfg.action_values)), dtype=int)
    joint_actions = [(a, b) for a in one_dim for b in one_dim]
    n_actions = len(joint_actions)

    Q: dict = {}
    N: dict = defaultdict(int)

    train_mask_rl = train_mask[:-1]
    train_idx = np.where(train_mask_rl)[0]
    train_start, train_end_i = int(train_idx[0]), int(train_idx[-1])

    for ep in range(ql_cfg.epochs):
        if ql_cfg.reset_every is not None and ep > 0 and (ep % ql_cfg.reset_every == 0):
            Q = {}
            N = defaultdict(int)

        eps_sched = ql_cfg.eps_start + (ql_cfg.eps_end - ql_cfg.eps_start) * (ep / max(1, ql_cfg.epochs - 1))
        eps = max(ql_cfg.eps_end, eps_sched) + ql_cfg.persistent_exploration

        na_prev, nb_prev = 0, 0

        for k in range(train_start, train_end_i + 1):
            s_t = (
                float(zqa.iloc[k]),
                float(zqb.iloc[k]),
                int(cqa.iloc[k]),
                int(cqb.iloc[k]),
                int(na_prev),
                int(nb_prev),
            )

            if rng.random() < eps:
                a_idx = int(rng.integers(0, n_actions))
            else:
                q_vals = np.array([Q.get((s_t, j), 0.0) for j in range(n_actions)])
                a_idx = int(np.argmax(q_vals))

            a_a, a_b = joint_actions[a_idx]
            n_a = clip_int(na_prev + a_a, -inv_limit, inv_limit)
            n_b = clip_int(nb_prev + a_b, -inv_limit, inv_limit)

            dn_a = n_a - na_prev
            dn_b = n_b - nb_prev

            dPa = float(dPa_next.iloc[k])
            dPb = float(dPb_next.iloc[k])

            gross = n_a * dPa * trading_cfg.mult_a + n_b * dPb * trading_cfg.mult_b
            trade_cost = trading_cfg.cost_per_trade * (abs(dn_a) + abs(dn_b))
            inventory_cost = ql_cfg.inv_penalty * (n_a**2 + n_b**2)
            pnl = gross - trade_cost
            reward = pnl - inventory_cost

            if k == train_end_i:
                td_target = reward
            else:
                s_next = (
                    float(zqa.iloc[k + 1]),
                    float(zqb.iloc[k + 1]),
                    int(cqa.iloc[k + 1]),
                    int(cqb.iloc[k + 1]),
                    int(n_a),
                    int(n_b),
                )
                q_next = np.array([Q.get((s_next, j), 0.0) for j in range(n_actions)])
                td_target = reward + ql_cfg.gamma * float(np.max(q_next))

            q_old = Q.get((s_t, a_idx), 0.0)
            N[(s_t, a_idx)] += 1
            lr_eff = (
                ql_cfg.lr / (1.0 + ql_cfg.lr_visit_scale * N[(s_t, a_idx)])
                if ql_cfg.use_adaptive_lr
                else ql_cfg.lr
            )
            Q[(s_t, a_idx)] = ql_cfg.q_decay * q_old + lr_eff * (td_target - q_old)

            na_prev, nb_prev = n_a, n_b

    def greedy_run(mask_np):
        ks = np.where(mask_np[:-1])[0]
        na_prev, nb_prev = 0, 0

        dates, pnl_list, reward_list = [], [], []
        n_a_list, n_b_list = [], []
        act_a, act_b = [], []

        for k in ks:
            s_t = (
                float(zqa.iloc[k]),
                float(zqb.iloc[k]),
                int(cqa.iloc[k]),
                int(cqb.iloc[k]),
                int(na_prev),
                int(nb_prev),
            )
            q_vals = np.array([Q.get((s_t, j), 0.0) for j in range(n_actions)])
            a_a, a_b = joint_actions[int(np.argmax(q_vals))]

            n_a = clip_int(na_prev + a_a, -inv_limit, inv_limit)
            n_b = clip_int(nb_prev + a_b, -inv_limit, inv_limit)

            dn_a = n_a - na_prev
            dn_b = n_b - nb_prev

            dPa = float(dPa_next.iloc[k])
            dPb = float(dPb_next.iloc[k])

            gross = n_a * dPa * trading_cfg.mult_a + n_b * dPb * trading_cfg.mult_b
            trade_cost = trading_cfg.cost_per_trade * (abs(dn_a) + abs(dn_b))
            inventory_cost = ql_cfg.inv_penalty * (n_a**2 + n_b**2)
            pnl = gross - trade_cost
            reward = pnl - inventory_cost

            dates.append(idx[k])
            pnl_list.append(pnl)
            reward_list.append(reward)
            n_a_list.append(n_a)
            n_b_list.append(n_b)
            act_a.append(a_a)
            act_b.append(a_b)

            na_prev, nb_prev = n_a, n_b

        dt = pd.DatetimeIndex(dates)
        pnl_s = pd.Series(pnl_list, index=dt, name="pnl_$")
        reward_s = pd.Series(reward_list, index=dt, name="reward_$")
        n_df = pd.DataFrame({f"n_{col_a}": n_a_list, f"n_{col_b}": n_b_list}, index=dt)
        actions_df = pd.DataFrame({f"a_{col_a}": act_a, f"a_{col_b}": act_b}, index=dt)
        return pnl_s, reward_s, pnl_s.cumsum(), reward_s.cumsum(), n_df, actions_df

    tr_pnl, tr_reward, tr_cum_pnl, tr_cum_reward, tr_n, tr_act = greedy_run(train_mask)
    te_pnl, te_reward, te_cum_pnl, te_cum_reward, te_n, te_act = greedy_run(test_mask)

    return {
        "Q": Q,
        "joint_actions": joint_actions,
        "train_pnl": tr_pnl,
        "train_reward": tr_reward,
        "train_cum": tr_cum_reward,
        "train_cum_pnl": tr_cum_pnl,
        "train_n": tr_n,
        "train_actions": tr_act,
        "test_pnl": te_pnl,
        "test_reward": te_reward,
        "test_cum": te_cum_reward,
        "test_cum_pnl": te_cum_pnl,
        "test_n": te_n,
        "test_actions": te_act,
        "signal_confidence_a": c_a,
        "signal_confidence_b": c_b,
        "regime_state_a": cqa,
        "regime_state_b": cqb,
        "state_info": {
            "state_definition": "(zqa, zqb, regime_a, regime_b, na_prev, nb_prev)",
            "z_values": "quantized by z_step and clipped by z_clip",
            "regime_values": [-1, 0, 1],
            "inventory_values": list(range(-inv_limit, inv_limit + 1)),
        },
        "metrics": {
            "train_mean_daily_pnl_$": mean_daily_pnl(tr_pnl),
            "train_max_drawdown_$": max_daily_drawdown(tr_pnl),
            "test_mean_daily_pnl_$": mean_daily_pnl(te_pnl),
            "test_max_drawdown_$": max_daily_drawdown(te_pnl),
        },
    }
