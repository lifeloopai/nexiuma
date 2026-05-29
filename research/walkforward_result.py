"""Data structures for walk-forward analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.engine import BacktestRunResult


@dataclass(frozen=True)
class WalkForwardWindowSpec:
    """Calendar specification for one train/test split."""

    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def train_label(self) -> str:
        return f"{self.train_start.isoformat()} → {self.train_end.isoformat()}"

    @property
    def test_label(self) -> str:
        return f"{self.test_start.isoformat()} → {self.test_end.isoformat()}"


@dataclass
class WindowResult:
    """Optimization + out-of-sample test outcome for one window."""

    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: str
    fast_period: int
    slow_period: int
    train_sharpe: float
    train_return: float
    test_sharpe: float
    test_return: float
    test_cagr: float
    test_sortino: float
    test_max_drawdown: float
    test_volatility: float
    test_num_trades: int
    train_run: BacktestRunResult | None = None
    test_run: BacktestRunResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "best_params": self.best_params,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "train_sharpe": self.train_sharpe,
            "train_return": self.train_return,
            "test_sharpe": self.test_sharpe,
            "test_return": self.test_return,
            "test_cagr": self.test_cagr,
            "test_sortino": self.test_sortino,
            "test_max_drawdown": self.test_max_drawdown,
            "test_volatility": self.test_volatility,
            "test_num_trades": self.test_num_trades,
        }


@dataclass(frozen=True)
class RobustnessMetrics:
    """Aggregate out-of-sample robustness statistics."""

    avg_test_sharpe: float
    avg_test_return: float
    worst_test_drawdown: float
    parameter_stability: float
    profitable_windows_pct: float
    num_windows: int
    avg_train_sharpe: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "avg_test_sharpe": self.avg_test_sharpe,
            "avg_test_return": self.avg_test_return,
            "worst_test_drawdown": self.worst_test_drawdown,
            "parameter_stability": self.parameter_stability,
            "profitable_windows_pct": self.profitable_windows_pct,
            "num_windows": self.num_windows,
            "avg_train_sharpe": self.avg_train_sharpe,
        }


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis output."""

    ticker: str
    strategy_name: str
    train_years: int
    test_years: int
    start_date: str
    end_date: str
    windows: list[WindowResult]
    results_df: pd.DataFrame
    parameter_history_df: pd.DataFrame
    combined_equity: pd.Series
    robustness: RobustnessMetrics
    output_dir: Path
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
