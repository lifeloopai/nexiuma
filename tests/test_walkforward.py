"""Tests for walk-forward testing framework."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config.settings import DataSettings, NexiumaSettings, RiskSettings
from core.engine import BacktestEngine, BacktestRunResult
from research.optimizer import StrategyOptimizer
from research.walkforward import (
    WalkForwardEngine,
    generate_window_specs,
    slice_ohlcv,
)
from research.walkforward_report import (
    WalkForwardReportGenerator,
    format_walkforward_terminal,
)
from research.walkforward_result import WalkForwardWindowSpec


def _extended_ohlcv(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Stretch synthetic data across 2020–2024 for window generation."""
    frames = []
    for year in range(2020, 2025):
        chunk = sample_ohlcv.copy()
        chunk.index = pd.date_range(f"{year}-01-01", periods=len(chunk), freq="B")
        frames.append(chunk)
    return pd.concat(frames)


def _fake_run(
    ticker: str,
    fast: int,
    slow: int,
    ohlcv: pd.DataFrame,
    equity_base: pd.Series,
    sharpe_offset: float = 0.0,
) -> BacktestRunResult:
    from analytics.metrics import compute_metrics
    from analytics.performance import PerformanceAnalyzer

    mult = 1.0 + fast / 1000.0 + sharpe_offset
    equity = equity_base.reindex(ohlcv.index, method="nearest") * mult
    if len(equity) != len(ohlcv):
        equity = pd.Series(
            equity_base.values[: len(ohlcv)] * mult,
            index=ohlcv.index,
            name="equity",
        )
    report = PerformanceAnalyzer().build_report(
        equity_curve=equity,
        trade_pnls=[100.0, -50.0, 80.0],
        initial_capital=100_000.0,
    )
    return BacktestRunResult(
        strategy_name="moving_average",
        ticker=ticker,
        start_date=str(ohlcv.index.min().date()),
        end_date=str(ohlcv.index.max().date()),
        ohlcv=ohlcv,
        equity_curve=equity,
        performance=report,
        strategy_params={"fast_period": fast, "slow_period": slow},
    )


@pytest.fixture
def wf_settings(tmp_path: Path) -> NexiumaSettings:
    settings = NexiumaSettings(
        strategy="moving_average",
        data=DataSettings(
            ticker="TEST",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        ),
        risk=RiskSettings(initial_capital=100_000.0),
        walkforward_dir=tmp_path / "walkforward",
    )
    settings.ensure_directories()
    return settings


def test_generate_window_specs() -> None:
    specs = generate_window_specs(
        date(2020, 1, 1),
        date(2024, 12, 31),
        train_years=3,
        test_years=1,
    )
    assert len(specs) == 2
    assert specs[0].train_start == date(2020, 1, 1)
    assert specs[0].train_end == date(2022, 12, 31)
    assert specs[0].test_start == date(2023, 1, 1)
    assert specs[0].test_end == date(2023, 12, 31)
    assert specs[1].test_start == date(2024, 1, 1)


def test_slice_ohlcv(sample_ohlcv: pd.DataFrame) -> None:
    ohlcv = _extended_ohlcv(sample_ohlcv)
    sliced = slice_ohlcv(ohlcv, date(2020, 1, 1), date(2020, 6, 30))
    assert not sliced.empty
    assert sliced.index.min().year == 2020


def test_select_best_params() -> None:
    df = pd.DataFrame(
        [
            {"params": "10/30", "fast_period": 10, "slow_period": 30, "sharpe_ratio": 0.5},
            {"params": "20/50", "fast_period": 20, "slow_period": 50, "sharpe_ratio": 0.9},
        ]
    )
    best = StrategyOptimizer.select_best_params(df)
    assert best["params"] == "20/50"


def test_walkforward_engine(
    wf_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    ohlcv = _extended_ohlcv(sample_ohlcv)
    engine = BacktestEngine(wf_settings)
    grid = ((10, 30), (20, 50))

    def fake_run_ma_grid(ticker, ohlcv_df, grid_arg):
        rows = []
        runs = {}
        for fast, slow in grid_arg:
            label = f"{fast}/{slow}"
            run = _fake_run(ticker, fast, slow, ohlcv_df, sample_equity)
            m = run.performance.metrics
            rows.append(
                {
                    "ticker": ticker,
                    "fast_period": fast,
                    "slow_period": slow,
                    "params": label,
                    "total_return": m.total_return,
                    "cagr": m.cagr,
                    "sharpe_ratio": m.sharpe_ratio + (0.1 if slow == 50 else 0),
                    "max_drawdown": m.max_drawdown,
                    "num_trades": m.num_trades,
                }
            )
            runs[label] = run
        return pd.DataFrame(rows), runs

    def fake_engine_run(**kwargs):
        sp = kwargs.get("strategy_params") or {}
        data = kwargs.get("ohlcv")
        return _fake_run(
            kwargs.get("ticker") or "TEST",
            int(sp["fast_period"]),
            int(sp["slow_period"]),
            data,
            sample_equity,
            sharpe_offset=-0.05,
        )

    engine.run = fake_engine_run  # type: ignore[method-assign]
    optimizer = StrategyOptimizer(wf_settings, engine=engine)
    optimizer.run_ma_grid = fake_run_ma_grid  # type: ignore[method-assign]

    wf_engine = WalkForwardEngine(wf_settings, engine=engine, optimizer=optimizer)

    with patch(
        "research.walkforward.MarketDataDownloader.get_data",
        return_value=ohlcv,
    ):
        result = wf_engine.run(
            ticker="TEST",
            train_years=3,
            test_years=1,
            grid=grid,
        )

    assert result.robustness.num_windows == 2
    assert len(result.results_df) == 2
    assert not result.combined_equity.empty
    assert result.windows[0].best_params in ("10/30", "20/50")


def test_walkforward_report(
    wf_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    ohlcv = _extended_ohlcv(sample_ohlcv)
    engine = BacktestEngine(wf_settings)
    grid = ((10, 30), (20, 50))

    def fake_run_ma_grid(ticker, ohlcv_df, grid_arg):
        rows, runs = [], {}
        for fast, slow in grid_arg:
            label = f"{fast}/{slow}"
            run = _fake_run(ticker, fast, slow, ohlcv_df, sample_equity)
            m = run.performance.metrics
            rows.append(
                {
                    "ticker": ticker,
                    "fast_period": fast,
                    "slow_period": slow,
                    "params": label,
                    "total_return": m.total_return,
                    "cagr": m.cagr,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown": m.max_drawdown,
                    "num_trades": m.num_trades,
                }
            )
            runs[label] = run
        return pd.DataFrame(rows), runs

    def fake_engine_run(**kwargs):
        sp = kwargs.get("strategy_params") or {}
        return _fake_run(
            kwargs.get("ticker") or "TEST",
            int(sp["fast_period"]),
            int(sp["slow_period"]),
            kwargs.get("ohlcv"),
            sample_equity,
        )

    engine.run = fake_engine_run  # type: ignore[method-assign]
    optimizer = StrategyOptimizer(wf_settings, engine=engine)
    optimizer.run_ma_grid = fake_run_ma_grid  # type: ignore[method-assign]
    wf_engine = WalkForwardEngine(wf_settings, engine=engine, optimizer=optimizer)

    with patch(
        "research.walkforward.MarketDataDownloader.get_data",
        return_value=ohlcv,
    ):
        result = wf_engine.run(ticker="TEST", train_years=3, test_years=1, grid=grid)

    paths = WalkForwardReportGenerator(wf_settings).publish(result)
    assert paths["results"].exists()
    assert paths["parameters"].exists()
    assert paths["summary"].exists()
    assert paths["equity_curve"].exists()
    assert paths["performance_chart"].exists()

    terminal = format_walkforward_terminal(result)
    assert "Robustness Metrics" in terminal
    assert "Window 1" in terminal


def test_robustness_aggregation(wf_settings: NexiumaSettings) -> None:
    from research.walkforward import WalkForwardEngine
    from research.walkforward_result import WindowResult

    windows = [
        WindowResult(
            window_id=1,
            train_start="2020-01-01",
            train_end="2022-12-31",
            test_start="2023-01-01",
            test_end="2023-12-31",
            best_params="10/50",
            fast_period=10,
            slow_period=50,
            train_sharpe=0.9,
            train_return=0.5,
            test_sharpe=0.4,
            test_return=0.2,
            test_cagr=0.2,
            test_sortino=0.5,
            test_max_drawdown=-0.15,
            test_volatility=0.2,
            test_num_trades=10,
        ),
        WindowResult(
            window_id=2,
            train_start="2021-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
            best_params="20/50",
            fast_period=20,
            slow_period=50,
            train_sharpe=0.8,
            train_return=0.4,
            test_sharpe=0.2,
            test_return=0.1,
            test_cagr=0.1,
            test_sortino=0.3,
            test_max_drawdown=-0.25,
            test_volatility=0.22,
            test_num_trades=8,
        ),
    ]
    metrics = WalkForwardEngine._compute_robustness(windows)
    assert metrics.num_windows == 2
    assert metrics.avg_test_sharpe == pytest.approx(0.3)
    assert metrics.profitable_windows_pct == 1.0
    assert metrics.worst_test_drawdown == pytest.approx(-0.25)
