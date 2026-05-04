from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class KDBConfig(BaseModel):
    host: str = "hfm.princeton.edu"
    port: int = 6007
    retry_port: int = 6009
    retry_interval: int = 5


class DataConfig(BaseModel):
    start_date: str = "2024-02-05"
    end_date: str = "2024-02-23"
    date_active: str = "2024.02.05"
    instruments: tuple[str, ...] = ("ES", "NQ")
    preferred_symbols: list[str] = ["ESH4", "NQH4"]
    time_grid_q: str = "(0D00:00:00 + 0D00:01:00 * til 961)"
    drop_all_nan_rows: bool = True


class ScalingConfig(BaseModel):
    scales: dict[str, float] = {
        "ES": 1.0,
        "NQ": 1.0,
        "CL": 1.0,
        "RB": 1.0,
        "HO": 1.0,
    }
    display_scales: dict[str, float] = {
        "ES": 100.0,
        "NQ": 100.0,
        "CL": 100.0,
        "RB": 10000.0,
        "HO": 10000.0,
    }


class SignalConfig(BaseModel):
    window: int = 60
    eps: float = 1e-8


class ZScoreConfig(BaseModel):
    z_window: int = 300
    z_min_periods: int | None = None
    z_lag: int = 1
    z_step: float = 0.5
    z_clip: float | None = 5.0


class TradingConfig(BaseModel):
    col_a: str = "ESH4"
    col_b: str = "NQH4"
    inv_limit: int = 2
    action_values: list[int] = [-2, -1, 0, 1, 2]
    mult_a: float = 50.0
    mult_b: float = 20.0
    cost_per_trade: float = 0.0


class RuleConfig(BaseModel):
    z_flat: float = 0.5
    z_entry: float = 1.0
    z_strong: float = 2.0
    inv_penalty: float = 0.0


class QLearningConfig(BaseModel):
    train_end: str = "2024-02-18"
    gamma: float = 0.99
    lr: float = 0.001
    epochs: int = 100
    eps_start: float = 0.5
    eps_end: float = 0.01
    seed: int = 0
    regime_window: int = 60
    q_decay: float = 0.998
    use_adaptive_lr: bool = False
    lr_visit_scale: float = 0.01
    reset_every: int | None = None
    persistent_exploration: float = 0.02
    inv_penalty: float = 1.0


class XGBConfig(BaseModel):
    train_end: str = "2024-02-14"
    val_end: str = "2024-02-18"
    n_estimators: int = 500
    max_depth: int = 4
    learning_rate: float = 0.02
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 5.0
    reg_alpha: float = 0.3
    min_child_weight: int = 3
    random_state: int = 42
    inv_penalty: float = 1.0


class RuleExtractionConfig(BaseModel):
    target_fidelity: float = 0.9
    max_rules: int = 600
    candidate_top_n: int = 5000
    tree_top_n: int = 12


class AnalysisConfig(BaseModel):
    start_date: str = "2024-02-19"
    trading_start: str = "00:00"
    trading_end: str = "16:00"
    signal_horizons_max: int = 30


class PipelineConfig(BaseModel):
    kdb: KDBConfig = KDBConfig()
    data: DataConfig = DataConfig()
    scaling: ScalingConfig = ScalingConfig()
    signal: SignalConfig = SignalConfig()
    zscore: ZScoreConfig = ZScoreConfig()
    trading: TradingConfig = TradingConfig()
    rule: RuleConfig = RuleConfig()
    qlearning: QLearningConfig = QLearningConfig()
    xgb: XGBConfig = XGBConfig()
    rule_extraction: RuleExtractionConfig = RuleExtractionConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    make_plots: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
