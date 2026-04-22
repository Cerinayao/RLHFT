# 📈 RLHFT — Reinforcement Learning for High-Frequency Trading

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/status-research-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A modular research framework for **regime-aware reinforcement learning in high-frequency trading**, combining:

- 📊 EWMA PCA signals
- 📉 Statistical arbitrage (z-score)
- 🤖 Q-learning trading agents
- 📈 PnL + regime-based evaluation

Designed for **intraday futures trading (ES/NQ)** with **KDB+ integration**.

---

## 🚀 Highlights

- End-to-end pipeline: **data → signal → strategy → evaluation**
- **Regime-aware RL** with signal, inventory, and regime-confidence state
- Rule-based baseline vs. RL comparison
- Modular research architecture under `src/rlhft/`
- Dashboard + debug export workflow for notebook/package comparison
- Built for **research + extensibility**

---

## 🧱 Project Structure

```text
rlhft/
├── pyproject.toml
├── README.md
├── configs/
│   └── es_nq.yaml
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── rlhft_dashboard.html
│   └── debug_compare/
│       ├── df_input.csv
│       ├── regime_state_a.csv
│       ├── regime_state_b.csv
│       ├── run_config_summary.csv
│       ├── signal_confidence_a.csv
│       ├── signal_confidence_b.csv
│       ├── test_actions.csv
│       ├── test_inventory.csv
│       ├── test_pnl.csv
│       └── train_pnl.csv
├── scripts/
│   └── compare_debug_exports.py
├── rlhft/
│   └── __init__.py
└── src/rlhft/
    ├── __init__.py
    ├── config.py
    ├── data/
    │   ├── __init__.py
    │   ├── kdb.py
    │   └── loaders.py
    ├── features/
    │   ├── __init__.py
    │   ├── pca_signal.py
    │   └── zscore.py
    ├── models/
    │   ├── __init__.py
    │   └── q_learning.py
    ├── strategies/
    │   ├── __init__.py
    │   ├── rule_based.py
    │   └── rl_strategy.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py
    │   └── analysis.py
    ├── visualization/
    │   ├── __init__.py
    │   ├── dashboard.py
    │   ├── diagnostics.py
    │   ├── pnl_plots.py
    │   ├── position_plots.py
    │   └── price_plots.py
    └── pipeline/
        ├── __init__.py
        ├── oneday.py
        ├── multiday.py
        └── runner.py
```

---

## ⚙️ Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

Edit `configs/es_nq.yaml`. Key parameters:

| Parameter | Description |
|---|---|
| `data.start_date`, `data.end_date` | Date range for the multi-day pipeline |
| `data.preferred_symbols` | Target contracts, e.g. `ESH4`, `NQH4` |
| `signal.window` | EWMA / PCA signal lookback |
| `zscore.*` | Rolling z-score controls |
| `trading.*` | Inventory bounds, multipliers, costs |
| `rule.*` | Rule-based threshold settings |
| `qlearning.*` | RL hyperparameters and regime settings |

### 3. Run Pipeline

```bash
python -m rlhft.pipeline.runner configs/es_nq.yaml
```

### 4. View Dashboard

This writes:

- `outputs/rlhft_dashboard.html`
- `outputs/debug_compare/`

Open the dashboard with:

```bash
open outputs/rlhft_dashboard.html
```

If you also want the legacy Matplotlib popup plots:

```bash
python -m rlhft.pipeline.runner configs/es_nq.yaml --matplotlib-plots
```

---

## 🧠 Methodology

### Signal Construction

EWMA covariance matrix → PCA decomposition. PC1 captures the common market factor; the residual serves as the mean-reversion signal:

```text
signal = projected_price − actual_price
```

### Rule-Based Strategy

Z-score thresholds drive mean-reversion entries and exits with fixed discrete target positions.

### RL Strategy (Q-Learning)

**State:** `(quantized signal_a, quantized signal_b, regime_a, regime_b, inventory_a, inventory_b)`

**Action space:** joint discrete inventory adjustments built from:

```text
{-2, -1, 0, +1, +2}
```

The key idea is dynamic position adjustment conditioned on both signal level and recent signal effectiveness, rather than static threshold rules.

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Cumulative PnL | Total strategy return |
| Max Drawdown | Peak-to-trough equity decline |
| Mean Daily PnL | Average daily profit and loss |
| Regime PnL | Performance decomposed by working vs failing regimes |
| Signal Horizon Sweep | Correlation of signal with forward returns |

---

## 📌 Results Workflow

- Run the package pipeline through `runner.py`
- Review the dashboard HTML for plots/tables
- Compare package exports vs notebook outputs with:

```bash
python scripts/compare_debug_exports.py notebook_exports outputs/debug_compare
```

This helps verify that package outputs match `notebooks/exploration.ipynb`.

---

## 🔬 Pipeline

```text
KDB+ Data
    ↓
Feature Engineering  (EWMA PCA + Z-score)
    ↓
Strategy  (Rule-Based  |  Q-Learning)
    ↓
Backtest
    ↓
Evaluation + Dashboard + Debug Exports
```

---

## 📊 Visualization

- All-dates ES/NQ scatter
- Price series with PCA projections
- OLS residual plots
- RL vs. rule-based cumulative PnL
- Position dynamics over time
- Signal diagnostics (PC1/PC2, ACF, regime-based summaries)
- Dashboard HTML output for consolidated review

---

## 🧩 Extensions

- Deep RL agents (PPO, SAC)
- CNN / Transformer feature extractors
- Multi-asset portfolio optimization
- Richer transaction cost modeling
- Live trading integration
- Alternative regime state definitions

---

## 📓 Notebook

`notebooks/exploration.ipynb` — signal validation, research prototyping, strategy diagnostics, and visualization experiments.

---

## 🛠️ Tech Stack

- Python 3.10+
- NumPy / Pandas
- KDB+ / q
- Matplotlib
- Plotly
- Tabular Q-Learning
- Pydantic + YAML configuration

---

## 📜 License

MIT License

---

## 👤 Author

**Cerina Yao**  
Princeton MFin · Quant Research · RL + Trading Systems
