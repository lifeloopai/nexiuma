"""Abstract strategy base with signal API and backtrader execution bridge."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import backtrader as bt
import numpy as np

from config.settings import RiskSettings
from core.execution import ExecutionSimulator
from core.interfaces import OrderSide, RiskParameters, Signal, SignalAction


class NexiumaStrategy(bt.Strategy):
    """Base class for all Nexiuma strategies.

    Subclasses implement ``generate_signal()`` for research-grade signal logic.
    Execution is handled via backtrader with configurable risk and sizing.
    """

    params: tuple[tuple[str, Any], ...] = (
        ("risk", None),
        ("execution", None),
        ("printlog", False),
    )

    strategy_name: ClassVar[str] = "base"
    strategy_description: ClassVar[str] = "Abstract base strategy"

    def __init__(self) -> None:
        self.order: bt.Order | None = None
        self.entry_price: float | None = None
        self.trade_log: list[dict[str, Any]] = []
        self.buy_signals: list[Any] = []
        self.sell_signals: list[Any] = []
        self._risk: RiskSettings = self.p.risk  # type: ignore[assignment]
        self._execution: ExecutionSimulator | None = self.p.execution
        self._vol_window = 20

    @abstractmethod
    def generate_signal(self) -> Signal:
        """Return current bar signal (BUY, SELL/EXIT, or HOLD)."""

    def risk_parameters(self) -> RiskParameters:
        """Expose strategy risk configuration."""
        risk = self._risk
        return RiskParameters(
            stop_loss_pct=risk.stop_loss_pct if risk else 0.05,
            take_profit_pct=risk.take_profit_pct if risk else 0.15,
            max_position_pct=risk.max_position_size_pct if risk else 1.0,
            use_volatility_sizing=getattr(risk, "use_volatility_sizing", False) if risk else False,
            target_volatility=getattr(risk, "target_volatility", 0.15) if risk else 0.15,
        )

    def position_size(self, cash: float | None = None, price: float | None = None) -> int:
        """Compute position size with optional volatility targeting."""
        if self._risk is None:
            return 0
        cash = cash if cash is not None else self.broker.getcash()
        price = price if price is not None else self.data.close[0]
        if price <= 0:
            return 0

        params = self.risk_parameters()
        alloc_pct = min(self._risk.position_size_pct, params.max_position_pct)

        if params.use_volatility_sizing and len(self.data) > self._vol_window:
            vol = self._realized_volatility(self._vol_window)
            if vol > 1e-6:
                vol_scalar = min(params.target_volatility / vol, 2.0)
                alloc_pct = min(alloc_pct * vol_scalar, params.max_position_pct)

        size = int(cash * alloc_pct / price)
        return max(size, 0)

    def _realized_volatility(self, window: int) -> float:
        closes = np.array([self.data.close[-i] for i in range(window, 0, -1)])
        if len(closes) < 2:
            return 0.15
        returns = np.diff(closes) / closes[:-1]
        return float(np.std(returns) * np.sqrt(252))

    def log(self, txt: str) -> None:
        if self.p.printlog:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status == order.Completed:
            dt = self.datas[0].datetime.datetime(0)
            if order.isbuy():
                self.entry_price = order.executed.price
                self.buy_signals.append(dt)
            else:
                self.entry_price = None
                self.sell_signals.append(dt)
        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self.trade_log.append(
                {
                    "pnl": trade.pnl,
                    "pnl_comm": trade.pnlcomm,
                    "barlen": trade.barlen,
                    "is_win": trade.pnlcomm > 0,
                }
            )

    def next(self) -> None:
        if self.order:
            return
        self._check_risk_exits()
        signal = self.generate_signal()

        if not self.position:
            if signal.action == SignalAction.BUY:
                self._enter_long()
        elif signal.action in (SignalAction.SELL, SignalAction.EXIT):
            self._exit_long()

    def _check_risk_exits(self) -> None:
        if not self.position or self.entry_price is None:
            return
        params = self.risk_parameters()
        price = self.data.close[0]
        stop = self.entry_price * (1 - params.stop_loss_pct)
        target = self.entry_price * (1 + params.take_profit_pct)
        if price <= stop or price >= target:
            self._exit_long()

    def _enter_long(self) -> None:
        size = self.position_size()
        if size > 0:
            self.order = self.buy(size=size)

    def _exit_long(self) -> None:
        if self.position:
            self.order = self.close()
