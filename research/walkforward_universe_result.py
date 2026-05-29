"""Data structures for walk-forward universe analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from research.walkforward_result import WalkForwardResult


ROBUSTNESS_GRADES: tuple[tuple[int, str], ...] = (
    (90, "Exceptional"),
    (70, "Strong"),
    (50, "Moderate"),
    (30, "Weak"),
    (0, "Poor"),
)


@dataclass(frozen=True)
class UniverseRobustnessScore:
    """Composite 0–100 robustness score with interpretable grade."""

    score: float
    grade: str
    test_sharpe_component: float
    degradation_component: float
    win_rate_component: float
    stability_component: float

    @staticmethod
    def grade_from_score(score: float) -> str:
        for threshold, label in ROBUSTNESS_GRADES:
            if score >= threshold:
                return label
        return "Poor"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "score": self.score,
            "grade": self.grade,
            "test_sharpe_component": self.test_sharpe_component,
            "degradation_component": self.degradation_component,
            "win_rate_component": self.win_rate_component,
            "stability_component": self.stability_component,
        }


@dataclass(frozen=True)
class UniverseSummaryMetrics:
    """Aggregated walk-forward statistics across all assets and windows."""

    avg_train_sharpe: float
    avg_test_sharpe: float
    sharpe_degradation: float
    avg_return: float
    avg_cagr: float
    avg_drawdown: float
    win_rate: float
    parameter_stability: float
    most_frequent_params: str
    best_asset: str
    worst_asset: str
    pct_positive_test_sharpe: float
    pct_positive_return: float
    num_assets: int
    num_windows_total: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "avg_train_sharpe": self.avg_train_sharpe,
            "avg_test_sharpe": self.avg_test_sharpe,
            "sharpe_degradation": self.sharpe_degradation,
            "avg_return": self.avg_return,
            "avg_cagr": self.avg_cagr,
            "avg_drawdown": self.avg_drawdown,
            "win_rate": self.win_rate,
            "parameter_stability": self.parameter_stability,
            "most_frequent_params": self.most_frequent_params,
            "best_asset": self.best_asset,
            "worst_asset": self.worst_asset,
            "pct_positive_test_sharpe": self.pct_positive_test_sharpe,
            "pct_positive_return": self.pct_positive_return,
            "num_assets": self.num_assets,
            "num_windows_total": self.num_windows_total,
        }


@dataclass
class WalkForwardUniverseResult:
    """Complete walk-forward universe analysis output."""

    strategy_name: str
    tickers: tuple[str, ...]
    train_years: int
    test_years: int
    asset_results_df: pd.DataFrame
    window_results_df: pd.DataFrame
    summary_df: pd.DataFrame
    parameter_frequency_df: pd.DataFrame
    summary_metrics: UniverseSummaryMetrics
    robustness_score: UniverseRobustnessScore
    per_asset_results: dict[str, WalkForwardResult]
    output_dir: Path
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    executive_summary: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "walkforward_universe",
            "strategy": self.strategy_name,
            "tickers": list(self.tickers),
            "run_id": self.run_id,
            "train_years": self.train_years,
            "test_years": self.test_years,
            "summary": self.summary_metrics.to_dict(),
            "robustness_score": self.robustness_score.to_dict(),
            "executive_summary": self.executive_summary,
        }
