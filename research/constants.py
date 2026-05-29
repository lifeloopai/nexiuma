"""Shared constants for research comparison workflows."""

from __future__ import annotations

DEFAULT_UNIVERSE: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "NVDA",
    "AMZN",
)

UNIVERSE_OPTIMIZATION_TICKERS: tuple[str, ...] = (
    "AAPL",
    "AMZN",
    "MSFT",
    "META",
    "NVDA",
    "GOOGL",
)

WALKFORWARD_UNIVERSE_TICKERS: tuple[str, ...] = UNIVERSE_OPTIMIZATION_TICKERS

COMPARISON_METRIC_COLUMNS: tuple[str, ...] = (
    "total_return",
    "cagr",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "volatility",
    "win_rate",
    "profit_factor",
    "num_trades",
)

COMPARISON_DISPLAY_NAMES: dict[str, str] = {
    "total_return": "Return",
    "cagr": "CAGR",
    "sharpe_ratio": "Sharpe",
    "sortino_ratio": "Sortino",
    "max_drawdown": "Max DD",
    "volatility": "Volatility",
    "win_rate": "Win Rate",
    "profit_factor": "Profit Factor",
    "num_trades": "Trades",
}
