"""Tests for moving-average parameter validation and engine wiring."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import backtrader as bt
import pandas as pd
import pytest

from config.settings import DataSettings, NexiumaSettings, RiskSettings, StrategySettings
from core.engine import BacktestEngine
from strategies.moving_average import MovingAverageCrossover, MovingAverageStrategy
from strategies.parameters import (
    ParameterValidationError,
    MovingAverageParams,
    ma_grid_from_strings,
    parse_ma_periods,
    validate_ma_periods,
)


def test_validate_ma_periods_success() -> None:
    validate_ma_periods(20, 50)
    params = MovingAverageParams(fast_period=10, slow_period=30)
    assert params.label == "10/30"


@pytest.mark.parametrize(
    "fast,slow",
    [(1, 50), (0, 50), (20, 1), (50, 20), (20, 20)],
)
def test_validate_ma_periods_rejects_invalid(fast: int, slow: int) -> None:
    with pytest.raises(ParameterValidationError):
        validate_ma_periods(fast, slow)


def test_parse_ma_periods_defaults() -> None:
    params = parse_ma_periods(None, None)
    assert params.fast_period == 20
    assert params.slow_period == 50


def test_ma_grid_from_strings() -> None:
    grid = ma_grid_from_strings(["10/30", "20/50"])
    assert grid == ((10, 30), (20, 50))


def test_moving_average_strategy_alias() -> None:
    assert MovingAverageStrategy is MovingAverageCrossover


def test_engine_passes_ma_params(sample_ohlcv: pd.DataFrame) -> None:
    settings = NexiumaSettings(
        strategy="moving_average",
        strategy_params=StrategySettings(fast_period=10, slow_period=30),
        data=DataSettings(ticker="TEST", start_date=date(2020, 1, 1), end_date=date(2020, 6, 1)),
        risk=RiskSettings(initial_capital=50_000.0),
    )
    settings.ensure_directories()
    engine = BacktestEngine(settings)

    captured: dict[str, int] = {}

    original_build = engine._build_cerebro

    def spy_build(ohlcv, strategy_cls, strategy_kwargs=None):
        captured.update(strategy_kwargs or {})
        return original_build(ohlcv, strategy_cls, strategy_kwargs)

    engine._build_cerebro = spy_build  # type: ignore[method-assign]

    with patch.object(engine._downloader, "get_data", return_value=sample_ohlcv):
        result = engine.run(save_charts=False)

    assert captured["fast_period"] == 10
    assert captured["slow_period"] == 30
    assert result.strategy_params["fast_period"] == 10
    assert result.strategy_params["slow_period"] == 30


def test_ma_periods_affect_signals(sample_ohlcv: pd.DataFrame) -> None:
    """Different periods should produce valid backtrader runs."""
    cerebro = bt.Cerebro()
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=sample_ohlcv,
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
    )
    cerebro.addstrategy(
        MovingAverageCrossover,
        fast_period=5,
        slow_period=15,
        risk=RiskSettings(),
        printlog=False,
    )
    cerebro.broker.setcash(100_000)
    cerebro.run()
