"""Tests for cross-asset parameter optimization."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config.settings import DataSettings, NexiumaSettings, RiskSettings
from core.engine import BacktestEngine, BacktestRunResult
from reports.universe_optimization_report import (
    UniverseOptimizationReportGenerator,
    format_universe_optimization_terminal,
)
from research.optimizer import StrategyOptimizer
from research.universe_optimizer import UniverseOptimizer


def _fake_run_result(
    ticker: str,
    fast: int,
    slow: int,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> BacktestRunResult:
    from analytics.performance import PerformanceAnalyzer

    mult = 1.0 + (fast / 1000.0) + (hash(ticker) % 10) / 100.0
    equity = sample_equity * mult
    report = PerformanceAnalyzer().build_report(
        equity_curve=equity,
        trade_pnls=[100.0, -50.0],
        initial_capital=100_000.0,
    )
    return BacktestRunResult(
        strategy_name="moving_average",
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2020-12-31",
        ohlcv=sample_ohlcv,
        equity_curve=equity,
        performance=report,
        strategy_params={"fast_period": fast, "slow_period": slow},
    )


@pytest.fixture
def uni_opt_settings(tmp_path: Path) -> NexiumaSettings:
    settings = NexiumaSettings(
        strategy="moving_average",
        data=DataSettings(
            ticker="TEST",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 10, 1),
        ),
        risk=RiskSettings(initial_capital=100_000.0),
        universe_optimization_dir=tmp_path / "universe_optimization",
    )
    settings.ensure_directories()
    return settings


def test_universe_optimizer_aggregates(
    uni_opt_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    engine = BacktestEngine(uni_opt_settings)
    grid = ((10, 30), (20, 50))

    def fake_run(**kwargs):
        sp = kwargs.get("strategy_params") or {}
        ticker = kwargs.get("ticker") or "AAA"
        return _fake_run_result(
            ticker,
            int(sp["fast_period"]),
            int(sp["slow_period"]),
            sample_ohlcv,
            sample_equity,
        )

    engine.run = fake_run  # type: ignore[method-assign]
    strategy_optimizer = StrategyOptimizer(uni_opt_settings, engine=engine)
    universe_optimizer = UniverseOptimizer(uni_opt_settings, optimizer=strategy_optimizer)

    with patch(
        "research.universe_optimizer.MarketDataDownloader.get_data",
        return_value=sample_ohlcv,
    ):
        result = universe_optimizer.optimize_moving_average_universe(
            tickers=["AAA", "BBB"],
            grid=grid,
        )

    assert len(result.full_results_df) == 4
    assert len(result.averages_df) == 2
    assert result.ranking_df.iloc[0]["rank"] == 1
    assert result.robust_params in ("10/30", "20/50")


def test_universe_optimization_report(
    uni_opt_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    engine = BacktestEngine(uni_opt_settings)

    def fake_run(**kwargs):
        sp = kwargs.get("strategy_params") or {}
        ticker = kwargs.get("ticker") or "AAA"
        return _fake_run_result(
            ticker,
            int(sp["fast_period"]),
            int(sp["slow_period"]),
            sample_ohlcv,
            sample_equity,
        )

    engine.run = fake_run  # type: ignore[method-assign]
    optimizer = UniverseOptimizer(
        uni_opt_settings,
        optimizer=StrategyOptimizer(uni_opt_settings, engine=engine),
    )

    with patch(
        "research.universe_optimizer.MarketDataDownloader.get_data",
        return_value=sample_ohlcv,
    ):
        result = optimizer.optimize_moving_average_universe(
            tickers=["AAA", "BBB"],
            grid=((10, 30), (20, 50)),
        )

    paths = UniverseOptimizationReportGenerator(uni_opt_settings).publish(result)
    assert paths["full_results"].exists()
    assert paths["averages"].exists()
    assert paths["ranking"].exists()
    assert paths["heatmaps"].exists()
    assert paths["html"].exists()

    terminal = format_universe_optimization_terminal(result)
    assert "Avg Return" in terminal
    assert result.robust_params in terminal


def test_ranking_prefers_higher_avg_sharpe() -> None:
    from research.universe_optimizer import UniverseOptimizer

    full = pd.DataFrame(
        [
            {"params": "10/50", "fast_period": 10, "slow_period": 50, "sharpe_ratio": 0.5, "total_return": 0.6, "cagr": 0.1, "max_drawdown": -0.2, "num_trades": 10},
            {"params": "10/50", "fast_period": 10, "slow_period": 50, "sharpe_ratio": 0.3, "total_return": 0.5, "cagr": 0.1, "max_drawdown": -0.25, "num_trades": 8},
            {"params": "20/100", "fast_period": 20, "slow_period": 100, "sharpe_ratio": -0.2, "total_return": 0.1, "cagr": 0.02, "max_drawdown": -0.3, "num_trades": 5},
            {"params": "20/100", "fast_period": 20, "slow_period": 100, "sharpe_ratio": -0.1, "total_return": 0.2, "cagr": 0.03, "max_drawdown": -0.28, "num_trades": 6},
        ]
    )
    averages = UniverseOptimizer._aggregate_by_params(full)
    ranking = UniverseOptimizer._build_ranking(averages)
    assert ranking.iloc[0]["params"] == "10/50"
