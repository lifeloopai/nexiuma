"""Protocol interfaces and shared enums for extensible Nexiuma components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import pandas as pd


class SignalAction(Enum):
    """Discrete trading signal actions."""

    HOLD = auto()
    BUY = auto()
    SELL = auto()
    EXIT = auto()


class OrderSide(Enum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Signal:
    """Strategy signal at a point in time."""

    action: SignalAction
    strength: float = 1.0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RiskParameters:
    """Strategy-level risk configuration."""

    stop_loss_pct: float
    take_profit_pct: float
    max_position_pct: float
    use_volatility_sizing: bool = False
    target_volatility: float = 0.15


@runtime_checkable
class DataProvider(Protocol):
    """Market data provider contract (yfinance, Alpaca, IBKR, etc.)."""

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame: ...

    def validate_ticker(self, ticker: str) -> bool: ...


@runtime_checkable
class Strategy(Protocol):
    """Research strategy contract independent of execution backend."""

    strategy_name: str
    strategy_description: str

    def generate_signal(self, bar_index: int) -> Signal: ...

    def position_size(
        self,
        cash: float,
        price: float,
        volatility: float | None = None,
    ) -> int: ...

    def risk_parameters(self) -> RiskParameters: ...


@runtime_checkable
class Broker(Protocol):
    """Broker interface for simulated and live execution."""

    def get_cash(self) -> float: ...

    def get_value(self) -> float: ...

    def set_cash(self, amount: float) -> None: ...

    def set_commission(self, rate: float) -> None: ...


@runtime_checkable
class ExecutionHandler(Protocol):
    """Order execution with cost models."""

    def apply_slippage(self, price: float, side: OrderSide) -> float: ...

    def commission(self, value: float) -> float: ...


@runtime_checkable
class BacktestEngineProtocol(Protocol):
    """Backtest orchestration contract."""

    def run(
        self,
        strategy_name: str,
        ticker: str,
        ohlcv: pd.DataFrame | None = None,
        save_charts: bool = True,
    ) -> Any: ...
