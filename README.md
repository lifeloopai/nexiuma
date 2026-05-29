<div align="center">

# Nexiuma

### Quantitative Research & Systematic Trading Platform

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-47%20Passing-brightgreen)](tests/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](dashboard/streamlit_app.py)
[![Quant Research](https://img.shields.io/badge/Focus-Quant%20Research-8B5CF6)](research/)

**Current Version:** `v1.0` &nbsp;·&nbsp; **Status:** Active Development

</div>

---

**Nexiuma** is an open-source quantitative research platform for systematic trading strategy development, validation, and reporting. It combines institutional-style software architecture with a research-first workflow: backtest strategies, optimize parameters, compare outcomes across assets, and stress-test robustness using walk-forward and universe-level analysis.

**Mission:** Build tools that help researchers and developers evaluate *whether* a strategy works—not just how high its backtest return looks.

| Capability | Description |
|------------|-------------|
| **Backtesting** | Historical simulation with commission, slippage, and risk controls |
| **Strategy comparison** | Side-by-side evaluation of multiple signals on one ticker |
| **Parameter optimization** | Grid search over strategy parameters with HTML reports |
| **Universe analysis** | Cross-asset optimization and walk-forward robustness scoring |
| **Reporting & dashboard** | Jinja2 tearsheets, Plotly charts, and a Streamlit UI |

> **Disclaimer:** For research and education only. Not investment advice.

---

## Screenshots

> Place image files in [`docs/`](docs/). Links below resolve once assets are added.

### Dashboard

![Dashboard](docs/dashboard.png)

### Strategy Optimization

![Optimization](docs/optimization.png)

### Walk-Forward Analysis

![Walk Forward](docs/walkforward.png)

### Universe Robustness Report

![Universe](docs/universe.png)

---

## Why Nexiuma?

Most trading projects focus on **maximizing historical returns**. A strategy that looks excellent on a full-sample backtest often fails when parameters are chosen on the same data used to evaluate performance—a form of **in-sample overfitting**.

Nexiuma is designed around a different goal: **robustness and methodological rigor**.

| Focus area | What Nexiuma provides |
|------------|------------------------|
| **Robustness** | Walk-forward and universe-level scoring, not single-number backtest hype |
| **Out-of-sample validation** | Train windows for optimization; test windows that were never used for selection |
| **Cross-asset testing** | Universe optimization and walk-forward-universe reports across mega-cap equities |
| **Research methodology** | Reproducible CLI, typed Python modules, CSV/HTML artifacts, and 47 automated tests |

**Why walk-forward testing matters:** Optimizing on 2020–2024 and reporting the best result uses information from the entire period—including future bars relative to each training decision. Walk-forward analysis mimics live deployment: parameters are selected on past data only, then evaluated on genuinely unseen periods. The gap between **train Sharpe** and **test Sharpe** is one of the most direct measures of whether a strategy is research-grade or curve-fit.

---

## Research Findings

The following summarizes representative results from Nexiuma’s research workflows on U.S. large-cap equities (2020–2024), using the default moving-average crossover strategy and standard parameter grids (`10/30`, `10/50`, `20/50`, `20/100`, `50/200`).

**Universe:** AAPL, AMZN, MSFT, META, NVDA, GOOGL

**Key observations:**

1. **Parameter selection** — The `10/50` SMA configuration was frequently selected during in-sample optimization across assets and windows, suggesting short fast periods paired with medium slow periods dominated the Sharpe-based objective on historical train segments.

2. **Universe walk-forward robustness** — A full walk-forward-universe run (`moving_average`, 3-year train / 1-year test) produced a composite **robustness score of approximately 24/100** (grade: *Poor* on the platform’s 0–100 scale). This score weights out-of-sample Sharpe, train-to-test degradation, win rate, and parameter stability.

3. **Train vs. test degradation** — In-sample (train) Sharpe ratios consistently exceeded out-of-sample (test) Sharpe across assets and windows, indicating **material performance decay** when parameters selected on past data were applied forward.

4. **Implication** — Strong full-period or train-only optimization results did **not** translate into reliable cross-asset out-of-sample performance. These findings reinforce the platform’s design emphasis: **reporting and scoring robustness is as important as reporting return.**

*Figures and HTML reports are generated under `reports/optimization/`, `reports/walkforward/`, and `reports/walkforward_universe/` after each run.*

---

## Key Features

| Feature | Status |
|---------|--------|
| Backtesting | ✅ |
| Strategy Comparison | ✅ |
| Parameter Optimization | ✅ |
| Universe Optimization | ✅ |
| Walk-Forward Testing | ✅ |
| Walk-Forward Universe Analysis | ✅ |
| Streamlit Dashboard | ✅ |
| Benchmark Comparison | 🚧 |
| Portfolio Engine | 🚧 |
| Paper Trading | 🚧 |
| Machine Learning Models | 🚧 |

---

## Features

- **Market Data** — yfinance OHLCV with parquet cache, validation, retry, and auto-refresh
- **Strategy Engine** — Abstract signal API (`generate_signal`, `position_size`, `risk_parameters`)
- **Core Layer** — Protocol-driven broker, portfolio, execution, and engine abstractions
- **Backtesting** — backtrader with commission, slippage, stop-loss, take-profit, vol sizing
- **Analytics** — CAGR, Sharpe, Sortino, Calmar, VaR, CVaR, benchmark comparison
- **Visualization** — Equity, drawdown, price/trades, rolling returns, return distribution
- **Reporting** — Jinja2 HTML reports and tearsheets
- **Walk-Forward Testing** — Rolling train/test optimization with out-of-sample validation
- **Walk-Forward Universe** — Cross-asset robustness scoring and research reports
- **Dashboard** — Modern Streamlit UI

---

## Requirements

- Python **3.12+** (3.9+ supported for development)
- Dependencies in [`requirements.txt`](requirements.txt)

---

## Installation

```bash
cd Nexiuma
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Quick Start

```bash
# List strategies
python main.py strategies

# Download data
python main.py download --ticker AAPL --start 2020-01-01 --end 2024-12-31

# Run backtest + report
python main.py backtest --ticker AAPL --strategy moving_average

# Backtest with custom MA periods
python main.py backtest --ticker AAPL --strategy moving_average --fast-period 20 --slow-period 50

# Optimize moving-average periods
python main.py optimize --ticker AAPL --strategy moving_average

# Optimize parameters across the default universe
python main.py optimize-universe --strategy moving_average

# Compare all strategies on one ticker
python main.py compare --ticker AAPL

# Run one strategy across the default universe
python main.py compare-universe --strategy moving_average

# Walk-forward: optimize on train, validate on unseen test windows
python main.py walkforward --ticker AAPL --strategy moving_average

python main.py walkforward \
  --ticker NVDA \
  --strategy moving_average \
  --train-years 3 \
  --test-years 1

# Walk-forward across the default universe (robustness report)
python main.py walkforward-universe --strategy moving_average

python main.py walkforward-universe \
  --strategy moving_average \
  --tickers AAPL,MSFT,NVDA \
  --train-years 3 \
  --test-years 1

# Dashboard
streamlit run dashboard/streamlit_app.py
```

---

## Architecture

Nexiuma follows a layered architecture that separates data ingestion, execution simulation, strategy logic, analytics, and presentation.

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of concerns** | `data/`, `core/`, `strategies/`, `analytics/`, `research/`, `reports/` |
| **Protocol-driven architecture** | `core/interfaces.py` defines broker, execution, and signal contracts |
| **Extensibility** | Strategy registry, pluggable settings, CLI subcommands |
| **Reproducibility** | Cached OHLCV, versioned configs, CSV/HTML run artifacts |
| **Research-first design** | Optimization and walk-forward modules prioritize OOS validation over headline returns |

```mermaid
flowchart TB
    subgraph UI["Presentation"]
        CLI[main.py CLI]
        DASH[Streamlit Dashboard]
    end

    subgraph Core["core/"]
        ENG[BacktestEngine]
        BRK[SimulatedBroker]
        EXE[ExecutionSimulator]
        PF[Portfolio]
        IF[interfaces.py Protocols]
    end

    subgraph Data["data/"]
        DL[downloader.py]
        VAL[validator.py]
        PROV[providers/]
    end

    subgraph Strat["strategies/"]
        REG[registry.py]
        BASE[base_strategy.py]
    end

    subgraph Analytics["analytics/"]
        MET[metrics.py]
        RISK[risk.py]
        VIS[visualization.py]
        TEAR[tearsheet.py]
    end

    CLI --> ENG
    DASH --> ENG
    ENG --> DL
    ENG --> REG
    ENG --> BRK
    ENG --> EXE
    DL --> VAL
    DL --> PROV
    REG --> BASE
    ENG --> MET
    ENG --> VIS
    ENG --> TEAR
```

### Data Flow

1. **Config** loads `.env` + CLI overrides → `NexiumaSettings`
2. **Downloader** fetches/caches OHLCV → **Validator** cleans and validates
3. **Engine** builds backtrader cerebro with **Broker** + **ExecutionSimulator**
4. **Strategy** emits `Signal` each bar → orders sized via `position_size()`
5. **EquityCurveAnalyzer** records portfolio value
6. **PerformanceAnalyzer** computes metrics + benchmark alpha
7. **ChartGenerator** + **ReportGenerator** persist artifacts

### Walk-Forward Testing

Walk-forward analysis prevents **in-sample overfitting** by never testing on data used for parameter selection.

```mermaid
flowchart LR
    subgraph W1["Window 1"]
        T1["Train 2020–2022\nOptimize grid → best params"]
        O1["Test 2023\nRun best params OOS"]
        T1 --> O1
    end
    subgraph W2["Window 2"]
        T2["Train 2021–2023\nOptimize grid → best params"]
        O2["Test 2024\nRun best params OOS"]
        T2 --> O2
    end
    W1 --> W2
    O1 --> EQ["Stitched OOS equity curve"]
    O2 --> EQ
    EQ --> MET["Robustness metrics\nAvg Sharpe · Stability · Win rate"]
```

**Why walk-forward beats simple optimization:** Optimizing on the full 2020–2024 dataset selects parameters that fit *all* history—including future bars the optimizer shouldn't see. Walk-forward mimics live deployment: parameters are chosen on past data only, then evaluated on genuinely unseen periods. Degradation from train to test Sharpe is a direct measure of overfit risk.

| Phase | Action |
|-------|--------|
| **Train** | Grid-search MA periods on training window; select best by Sharpe |
| **Test** | Run selected params on next out-of-sample window |
| **Roll** | Advance both windows; repeat until data exhausted |
| **Aggregate** | Stitch OOS equity, compute robustness metrics, generate report |

**Outputs** (`reports/walkforward/{TICKER}_{strategy}_{timestamp}/`):

| File | Description |
|------|-------------|
| `walkforward_results.csv` | Per-window train/test metrics |
| `parameter_history.csv` | Selected params per window |
| `equity_curve.png` | Stitched out-of-sample equity |
| `performance_chart.png` | Train vs test Sharpe/return bars |
| `interactive_charts.html` | Plotly parameter history & comparisons |
| `summary.html` | Professional HTML report with conclusions |

### Walk-Forward Universe Analysis

Extends single-asset walk-forward to an entire basket, producing a **cross-asset robustness report** with a composite 0–100 score.

```mermaid
flowchart TB
    subgraph Universe["Default Universe"]
        AAPL & AMZN & MSFT & META & NVDA & GOOGL
    end
    Universe --> WF["WalkForwardEngine per asset"]
    WF --> AGG["Aggregate windows + assets"]
    AGG --> SCORE["Robustness Score 0–100"]
    AGG --> RPT["HTML + CSV + Plotly heatmaps"]
```

**Workflow:**

1. For each asset, run full walk-forward (optimize train → test OOS)
2. Flatten all asset-window results into universe tables
3. Compute aggregate metrics, parameter frequency, best/worst assets
4. Calculate robustness score and generate research report

**Robustness Score (0–100):**

| Range | Grade | Interpretation |
|-------|-------|----------------|
| 90–100 | Exceptional | Strong OOS performance, low degradation |
| 70–89 | Strong | Acceptable cross-asset robustness |
| 50–69 | Moderate | Mixed results, parameter instability |
| 30–49 | Weak | Significant overfitting signals |
| 0–29 | Poor | Unreliable out-of-sample performance |

Components: Test Sharpe (30 pts) · Low degradation (25 pts) · Win rate (25 pts) · Parameter stability (20 pts)

**Outputs** (`reports/walkforward_universe/{strategy}_{timestamp}/`):

| File | Description |
|------|-------------|
| `asset_results.csv` | Per-asset aggregated walk-forward metrics |
| `summary.csv` | Universe-level statistics |
| `parameter_frequency.csv` | How often each param set was selected |
| `window_results.csv` | Every asset × window detail row |
| `sharpe_heatmap.html` | OOS Sharpe heatmap (asset × window) |
| `degradation_heatmap.html` | Train−test Sharpe degradation heatmap |
| `robustness_chart.html` | Interactive Plotly dashboard |
| `index.html` | Full HTML research report |

---

## Folder Structure

| Folder | Purpose |
|--------|---------|
| `config/` | Centralized dataclass settings (.env + CLI) |
| `core/` | Engine, broker, portfolio, execution, Protocol interfaces |
| `data/` | Download, validate, cache, provider adapters |
| `strategies/` | Signal logic + backtrader bridge |
| `analytics/` | Metrics, risk, tearsheet, charts |
| `research/` | Comparison, optimization, walk-forward engines |
| `backtests/` | Runner alias + custom analyzers |
| `reports/` | HTML generator + Jinja2 templates |
| `dashboard/` | Streamlit application |
| `tests/` | pytest unit & integration tests |
| `notebooks/` | Research notebooks |
| `logs/` | Rotating loguru file logs |
| `docs/` | README screenshots and documentation assets |

---

## Configuration

| Variable | Description |
|----------|-------------|
| `TICKER` | Symbol (e.g. `AAPL`) |
| `STRATEGY` | `moving_average`, `rsi`, `momentum` |
| `INITIAL_CAPITAL` | Starting cash |
| `STOP_LOSS_PCT` / `TAKE_PROFIT_PCT` | Risk exits |
| `COMMISSION_PCT` / `SLIPPAGE_PCT` | Transaction costs |
| `USE_VOLATILITY_SIZING` | Vol-target position sizing |

---

## Adding a Strategy

1. Create `strategies/my_strategy.py`:

```python
from core.interfaces import Signal, SignalAction
from strategies.base_strategy import NexiumaStrategy
import backtrader as bt

class MyStrategy(NexiumaStrategy):
    strategy_name = "my_strategy"
    strategy_description = "Custom logic"

    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SMA(self.data.close, period=10)

    def generate_signal(self) -> Signal:
        if self.sma[0] > self.data.close[0]:
            return Signal(SignalAction.BUY)
        if self.position:
            return Signal(SignalAction.EXIT)
        return Signal(SignalAction.HOLD)
```

2. Register in `strategies/registry.py`
3. Run: `python main.py backtest --strategy my_strategy`

---

## Deploying Streamlit

```bash
streamlit run dashboard/streamlit_app.py --server.port 8501
```

**Docker:**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.address=0.0.0.0"]
```

---

## Testing

```bash
pytest
pytest tests/test_strategy_signals.py -v
```

The test suite includes **47 passing tests** covering metrics, strategies, optimization, walk-forward analysis, and universe robustness workflows.

---

## Lessons Learned

Building Nexiuma surfaced several lessons that apply beyond this codebase:

1. **Optimization can be misleading** — The parameter set with the highest in-sample Sharpe is not necessarily the best strategy; it is often the best fit to noise in the training window.

2. **Walk-forward testing exposes overfitting** — Separating train and test periods consistently revealed large gaps between train and out-of-sample Sharpe—gaps that full-sample backtests concealed.

3. **Cross-asset testing is critical** — A configuration that appears robust on one ticker may underperform on others; universe-level walk-forward scoring helps surface that heterogeneity.

4. **Robustness matters more than raw returns** — For research and production readiness, a moderate return with stable out-of-sample behavior is more informative than a spectacular backtest that fails forward validation.

These principles shaped Nexiuma’s CLI design, reporting outputs, and the emphasis on walk-forward-universe scoring in **v1.0**.

---

## Roadmap

### v1.0 (Completed)

- Market data pipeline (yfinance, cache, validation)
- Protocol-driven backtest engine (backtrader)
- Moving-average, RSI, and momentum strategies
- Strategy comparison (`compare`, `compare-universe`)
- Parameter optimization (`optimize`, `optimize-universe`)
- Walk-forward testing (`walkforward`)
- Walk-forward universe analysis with robustness score (`walkforward-universe`)
- HTML/CSV/Plotly reporting
- Streamlit dashboard (backtest, walk-forward, walk-forward universe)
- 47 automated unit tests

### v2.0

- Benchmark comparison
- Portfolio backtesting
- Portfolio optimization
- Portfolio walk-forward testing

### v3.0

- Alpaca integration
- Interactive Brokers integration
- Paper trading
- Multi-asset portfolio support

### v4.0

- Feature engineering pipeline
- Machine learning signal models
- Sentiment analysis
- Factor models

---

## Author

**Henry Lin**

Built as an independent quantitative research project exploring systematic trading, strategy validation, and financial data analysis.

Questions, feedback, and contributions are welcome via GitHub issues and pull requests.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
