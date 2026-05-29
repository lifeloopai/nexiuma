"""Simulated broker wrapping backtrader for research backtests."""

from __future__ import annotations

import backtrader as bt

from core.interfaces import Broker


class SimulatedBroker:
    """Adapter over backtrader's broker for interface-driven design.

    Future live brokers (Alpaca, IBKR) will implement the same Broker protocol.
    """

    def __init__(self, cerebro: bt.Cerebro) -> None:
        self._cerebro = cerebro

    @property
    def _bt_broker(self) -> bt.brokers.BackBroker:
        return self._cerebro.broker  # type: ignore[return-value]

    def get_cash(self) -> float:
        return float(self._bt_broker.getcash())

    def get_value(self) -> float:
        return float(self._bt_broker.getvalue())

    def set_cash(self, amount: float) -> None:
        self._bt_broker.setcash(amount)

    def set_commission(self, rate: float) -> None:
        self._bt_broker.setcommission(commission=rate)

    def configure_from_settings(self, initial_capital: float, commission_pct: float) -> None:
        self.set_cash(initial_capital)
        self.set_commission(commission_pct)
