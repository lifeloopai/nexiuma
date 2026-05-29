"""Backtrader analyzers used by the Nexiuma engine."""

from __future__ import annotations

from typing import Any

import backtrader as bt
import pandas as pd


class EquityCurveAnalyzer(bt.Analyzer):
    """Record portfolio value at each bar for post-backtest analytics."""

    def __init__(self) -> None:
        self.history: list[tuple[pd.Timestamp, float]] = []

    def next(self) -> None:
        dt = self.strategy.datas[0].datetime.datetime(0)
        value = self.strategy.broker.getvalue()
        self.history.append((pd.Timestamp(dt), value))

    def get_analysis(self) -> dict[str, Any]:
        if not self.history:
            return {"equity_curve": pd.Series(dtype=float)}
        dates, values = zip(*self.history)
        series = pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")
        return {"equity_curve": series}
