"""90-day momentum trend-following strategy."""

from __future__ import annotations

import backtrader as bt

from core.interfaces import Signal, SignalAction
from strategies.base_strategy import NexiumaStrategy


class MomentumStrategy(NexiumaStrategy):
    """Long positive 90-day ROC; exit when momentum turns negative."""

    strategy_name = "momentum"
    strategy_description = "90-day momentum trend following"

    params = (
        ("momentum_period", 90),
        ("momentum_threshold", 0.0),
        ("risk", None),
        ("execution", None),
        ("printlog", False),
    )

    def __init__(self) -> None:
        super().__init__()
        self.roc = bt.indicators.ROC(self.data.close, period=self.p.momentum_period)

    def generate_signal(self) -> Signal:
        if self.roc[0] > self.p.momentum_threshold:
            return Signal(SignalAction.BUY)
        if self.roc[0] <= self.p.momentum_threshold:
            return Signal(SignalAction.EXIT)
        return Signal(SignalAction.HOLD)
