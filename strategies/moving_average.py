"""Moving average crossover strategy with configurable SMA periods."""

from __future__ import annotations

import backtrader as bt

from core.interfaces import Signal, SignalAction
from strategies.base_strategy import NexiumaStrategy
from strategies.parameters import MovingAverageParams


class MovingAverageCrossover(NexiumaStrategy):
    """Golden cross / death cross using configurable fast and slow SMAs."""

    strategy_name = "moving_average"
    strategy_description = "SMA crossover (configurable fast/slow periods)"

    params = (
        ("fast_period", 20),
        ("slow_period", 50),
        ("risk", None),
        ("execution", None),
        ("printlog", False),
    )

    def __init__(self) -> None:
        super().__init__()
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def generate_signal(self) -> Signal:
        if self.crossover[0] > 0:
            return Signal(SignalAction.BUY, strength=1.0)
        if self.crossover[0] < 0:
            return Signal(SignalAction.EXIT, strength=1.0)
        return Signal(SignalAction.HOLD)

    @classmethod
    def params_label(cls, fast_period: int, slow_period: int) -> str:
        return MovingAverageParams(fast_period=fast_period, slow_period=slow_period).label


# Public alias matching strategy documentation
MovingAverageStrategy = MovingAverageCrossover
