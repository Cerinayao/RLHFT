from __future__ import annotations

import pandas as pd

from rlhft.config import QLearningConfig, TradingConfig, ZScoreConfig
from rlhft.models.q_learning import train_q_learning_2asset_discrete_adaptive


def run_rl_strategy(
    df: pd.DataFrame,
    *,
    trading_cfg: TradingConfig,
    zscore_cfg: ZScoreConfig,
    ql_cfg: QLearningConfig,
    sig_a: str = "sig_a",
    sig_b: str = "sig_b",
) -> dict:
    """Orchestrate feature computation + Q-learning training + greedy evaluation."""
    return train_q_learning_2asset_discrete_adaptive(
        df,
        trading_cfg=trading_cfg,
        zscore_cfg=zscore_cfg,
        ql_cfg=ql_cfg,
        sig_a=sig_a,
        sig_b=sig_b,
    )
