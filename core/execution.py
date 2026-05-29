"""Execution simulation with commission and slippage."""

from __future__ import annotations

from config.settings import RiskSettings
from core.interfaces import ExecutionHandler, OrderSide


class ExecutionSimulator:
    """Applies transaction costs for realistic backtest fills."""

    def __init__(self, risk: RiskSettings) -> None:
        self._commission_pct = risk.commission_pct
        self._slippage_pct = risk.slippage_pct

    def apply_slippage(self, price: float, side: OrderSide) -> float:
        """Adjust fill price for slippage."""
        if side == OrderSide.BUY:
            return price * (1 + self._slippage_pct)
        return price * (1 - self._slippage_pct)

    def commission(self, notional: float) -> float:
        """Compute commission on trade notional."""
        return abs(notional) * self._commission_pct
