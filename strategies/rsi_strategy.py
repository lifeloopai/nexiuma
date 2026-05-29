"""RSI mean-reversion strategy."""

from __future__ import annotations

import backtrader as bt

from core.interfaces import Signal, SignalAction
from strategies.base_strategy import NexiumaStrategy


class RSIStrategy(NexiumaStrategy):
    """RSI(14): buy oversold (<30), sell overbought (>70)."""

    strategy_name = "rsi"
    strategy_description = "RSI(14) mean reversion"

    params = (
        ("rsi_period", 14),
        ("oversold", 30),
        ("overbought", 70),
        ("risk", None),
        ("execution", None),
        ("printlog", False),
    )

    def __init__(self) -> None:
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def generate_signal(self) -> Signal:
        if self.rsi[0] < self.p.oversold:
            return Signal(SignalAction.BUY)
        if self.rsi[0] > self.p.overbought:
            return Signal(SignalAction.EXIT)
        return Signal(SignalAction.HOLD)
