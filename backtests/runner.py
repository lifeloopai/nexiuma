"""Backward-compatible aliases for the core backtest engine."""

from __future__ import annotations

from core.engine import BacktestEngine, BacktestRunResult

BacktestResult = BacktestRunResult
BacktestRunner = BacktestEngine

__all__ = ["BacktestResult", "BacktestRunner", "BacktestEngine", "BacktestRunResult"]
