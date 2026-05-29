"""Quantitative research workflows: strategy and universe comparison."""

from research.optimizer import OptimizationResult, StrategyOptimizer
from research.strategy_comparison import StrategyComparator, StrategyComparisonResult
from research.universe_comparison import UniverseComparator, UniverseComparisonResult
from research.universe_optimizer import UniverseOptimizationResult, UniverseOptimizer
from research.walkforward import WalkForwardEngine, generate_window_specs
from research.walkforward_result import WalkForwardResult, WalkForwardWindowSpec
from research.walkforward_report import WalkForwardReportGenerator
from research.walkforward_universe import WalkForwardUniverseEngine
from research.walkforward_universe_result import (
    UniverseRobustnessScore,
    WalkForwardUniverseResult,
)
from research.walkforward_universe_report import WalkForwardUniverseReportGenerator

__all__ = [
    "OptimizationResult",
    "StrategyComparator",
    "StrategyComparisonResult",
    "StrategyOptimizer",
    "UniverseComparator",
    "UniverseComparisonResult",
    "UniverseOptimizationResult",
    "UniverseOptimizer",
    "UniverseRobustnessScore",
    "WalkForwardEngine",
    "WalkForwardReportGenerator",
    "WalkForwardResult",
    "WalkForwardUniverseEngine",
    "WalkForwardUniverseReportGenerator",
    "WalkForwardUniverseResult",
    "WalkForwardWindowSpec",
    "generate_window_specs",
]
