"""Portfolio state tracking for backtests and future live trading."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    """Open position in a single instrument."""

    ticker: str
    size: int
    entry_price: float
    entry_time: pd.Timestamp | None = None

    @property
    def market_value(self) -> float:
        return self.size * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.size


@dataclass
class Portfolio:
    """Tracks cash, positions, and equity history."""

    cash: float
    initial_capital: float
    positions: dict[str, Position] = field(default_factory=dict)
    equity_history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    trade_history: list[dict[str, float]] = field(default_factory=list)

    def record_equity(self, timestamp: pd.Timestamp, total_value: float) -> None:
        self.equity_history.append((timestamp, total_value))

    def to_equity_series(self) -> pd.Series:
        if not self.equity_history:
            return pd.Series(dtype=float, name="equity")
        dates, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")

    def total_value(self, mark_prices: dict[str, float] | None = None) -> float:
        value = self.cash
        for ticker, pos in self.positions.items():
            price = mark_prices.get(ticker, pos.entry_price) if mark_prices else pos.entry_price
            value += pos.size * price
        return value
