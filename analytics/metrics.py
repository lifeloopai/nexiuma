"""Core performance metrics for backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    """Container for standard backtest performance statistics."""

    total_return: float
    annualized_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    num_trades: int
    final_value: float
    initial_capital: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    initial_capital: float,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Compute performance metrics from equity curve and trade list.

    Args:
        equity_curve: Portfolio value over time (DatetimeIndex).
        trade_pnls: List of closed-trade PnL values.
        initial_capital: Starting portfolio value.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino.
        periods_per_year: Trading days per year.

    Returns:
        BacktestMetrics with all standard statistics.
    """
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty")

    values = equity_curve.astype(float)
    returns = values.pct_change().dropna()

    final_value = float(values.iloc[-1])
    total_return = (final_value / initial_capital) - 1.0

    n_periods = len(returns)
    years = max(n_periods / periods_per_year, 1e-9)
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    vol = float(returns.std() * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(periods_per_year))
        if excess.std() > 1e-12
        else 0.0
    )

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    sortino = (
        float((returns.mean() * periods_per_year - risk_free_rate) / downside_std)
        if downside_std > 1e-12
        else 0.0
    )

    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    max_drawdown = float(drawdown.min())

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    num_trades = len(trade_pnls)
    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else float("inf") if gross_profit > 0 else 0.0

    cagr = annualized_return  # CAGR equivalent for periodic equity curve

    return BacktestMetrics(
        total_return=round(total_return, 6),
        annualized_return=round(annualized_return, 6),
        cagr=round(cagr, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_drawdown, 6),
        volatility=round(vol, 6),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        num_trades=num_trades,
        final_value=round(final_value, 2),
        initial_capital=round(initial_capital, 2),
    )
