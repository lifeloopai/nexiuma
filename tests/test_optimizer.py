"""Tests for strategy parameter optimization."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config.settings import DataSettings, NexiumaSettings, RiskSettings
from core.engine import BacktestEngine, BacktestRunResult
from reports.optimization_report import OptimizationReportGenerator, format_optimization_terminal
from research.optimizer import StrategyOptimizer


def _fake_run_result(
    fast: int,
    slow: int,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> BacktestRunResult:
    from analytics.performance import PerformanceAnalyzer

    mult = 1.0 + (fast / 1000.0)
    equity = sample_equity * mult
    report = PerformanceAnalyzer().build_report(
        equity_curve=equity,
        trade_pnls=[100.0, -50.0],
        initial_capital=100_000.0,
    )
    return BacktestRunResult(
        strategy_name="moving_average",
        ticker="TEST",
        start_date="2020-01-01",
        end_date="2020-12-31",
        ohlcv=sample_ohlcv,
        equity_curve=equity,
        performance=report,
        strategy_params={"fast_period": fast, "slow_period": slow},
    )


@pytest.fixture
def opt_settings(tmp_path: Path) -> NexiumaSettings:
    settings = NexiumaSettings(
        strategy="moving_average",
        data=DataSettings(
            ticker="TEST",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 10, 1),
        ),
        risk=RiskSettings(initial_capital=100_000.0),
        optimization_dir=tmp_path / "optimization",
    )
    settings.ensure_directories()
    return settings


def test_optimizer_grid(
    opt_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    engine = BacktestEngine(opt_settings)
    grid = ((10, 30), (20, 50))

    def fake_run(
        strategy_name=None,
        ticker=None,
        ohlcv=None,
        save_charts=True,
        strategy_params=None,
    ):
        assert strategy_params is not None
        fast = int(strategy_params["fast_period"])
        slow = int(strategy_params["slow_period"])
        return _fake_run_result(fast, slow, sample_ohlcv, sample_equity)

    engine.run = fake_run  # type: ignore[method-assign]
    optimizer = StrategyOptimizer(opt_settings, engine=engine)

    with patch.object(
        StrategyOptimizer,
        "optimize_moving_average",
        wraps=optimizer.optimize_moving_average,
    ):
        with patch(
            "research.optimizer.MarketDataDownloader.get_data",
            return_value=sample_ohlcv,
        ):
            result = optimizer.optimize_moving_average(
                ticker="TEST",
                grid=grid,
            )

    assert len(result.results_df) == 2
    assert "10/30" in result.results_df.index
    assert result.results_df.loc["10/30", "num_trades"] >= 0


def test_optimization_report(
    opt_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    engine = BacktestEngine(opt_settings)

    def fake_run(**kwargs):
        sp = kwargs.get("strategy_params") or {}
        return _fake_run_result(
            int(sp.get("fast_period", 20)),
            int(sp.get("slow_period", 50)),
            sample_ohlcv,
            sample_equity,
        )

    engine.run = fake_run  # type: ignore[method-assign]
    optimizer = StrategyOptimizer(opt_settings, engine=engine)

    with patch(
        "research.optimizer.MarketDataDownloader.get_data",
        return_value=sample_ohlcv,
    ):
        result = optimizer.optimize_moving_average(
            ticker="TEST",
            grid=((10, 30), (20, 50)),
        )

    paths = OptimizationReportGenerator(opt_settings).publish(result)
    assert paths["csv"].exists()
    assert paths["heatmap"].exists()
    assert paths["html"].exists()
    terminal = format_optimization_terminal(result)
    assert "10/30" in terminal
