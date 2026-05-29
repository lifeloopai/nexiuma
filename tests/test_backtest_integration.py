"""Integration test for backtest runner with synthetic data."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from core.engine import BacktestEngine
from config.settings import DataSettings, NexiumaSettings, RiskSettings


@pytest.mark.integration
def test_backtest_runner_with_mock_data(sample_ohlcv) -> None:
    settings = NexiumaSettings(
        strategy="moving_average",
        data=DataSettings(
            ticker="TEST",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 10, 1),
        ),
        risk=RiskSettings(initial_capital=50_000.0),
    )
    settings.ensure_directories()
    runner = BacktestEngine(settings)

    with patch.object(runner._downloader, "get_data", return_value=sample_ohlcv):
        result = runner.run(
            strategy_name="moving_average",
            ticker="TEST",
            save_charts=False,
        )

    assert result.performance.metrics.num_trades >= 0
    assert len(result.equity_curve) > 0
    assert result.ticker == "TEST"
