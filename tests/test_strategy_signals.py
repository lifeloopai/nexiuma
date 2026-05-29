"""Tests for strategy signal generation."""

from __future__ import annotations

import backtrader as bt
import pandas as pd

from config.settings import RiskSettings
from core.interfaces import SignalAction
from strategies.moving_average import MovingAverageCrossover
from strategies.momentum_strategy import MomentumStrategy
from strategies.rsi_strategy import RSIStrategy


def _run_strategy_signals(strategy_cls, ohlcv: pd.DataFrame) -> list:
    """Collect signals by running cerebro briefly."""
    signals: list = []

    class Collector(strategy_cls):
        def generate_signal(self):
            sig = super().generate_signal()
            signals.append(sig.action)
            return sig

    cerebro = bt.Cerebro()
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=ohlcv,
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
    )
    cerebro.addstrategy(Collector, risk=RiskSettings(), printlog=False)
    cerebro.broker.setcash(100_000)
    cerebro.run()
    return signals


def test_moving_average_generates_signals(sample_ohlcv: pd.DataFrame) -> None:
    signals = _run_strategy_signals(MovingAverageCrossover, sample_ohlcv)
    assert SignalAction.BUY in signals or SignalAction.EXIT in signals


def test_rsi_strategy_has_risk_parameters() -> None:
    assert RSIStrategy.strategy_name == "rsi"
    assert "RSI" in RSIStrategy.strategy_description


def test_momentum_strategy_name() -> None:
    assert MomentumStrategy.strategy_name == "momentum"
