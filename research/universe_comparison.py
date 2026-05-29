"""Run one strategy across a ticker universe and aggregate results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from analytics.performance import PerformanceAnalyzer
from config.settings import NexiumaSettings, get_settings
from core.engine import BacktestEngine, BacktestRunResult
from research.constants import COMPARISON_METRIC_COLUMNS, DEFAULT_UNIVERSE


@dataclass
class UniverseComparisonResult:
    """Output from a universe-wide strategy comparison."""

    strategy_name: str
    tickers: tuple[str, ...]
    results: dict[str, BacktestRunResult]
    metrics_df: pd.DataFrame
    averages: dict[str, float]
    ranking_df: pd.DataFrame
    output_dir: Path
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


class UniverseComparator:
    """Backtest one strategy across multiple tickers and rank outcomes."""

    def __init__(
        self,
        settings: NexiumaSettings | None = None,
        engine: BacktestEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._engine = engine or BacktestEngine(self._settings)
        self._analyzer = PerformanceAnalyzer(
            risk_free_rate=self._settings.risk.risk_free_rate
        )

    def run(
        self,
        strategy_name: str | None = None,
        tickers: Sequence[str] | None = None,
        save_individual_charts: bool = False,
    ) -> UniverseComparisonResult:
        """Run strategy on each ticker and build ranking / heatmap inputs."""
        strategy = (strategy_name or self._settings.strategy).lower()
        universe = tuple(t.upper() for t in (tickers or DEFAULT_UNIVERSE))

        logger.info("Universe scan: {} on {} tickers", strategy, len(universe))
        results: dict[str, BacktestRunResult] = {}
        for ticker in universe:
            logger.info("Backtesting {} / {}", strategy, ticker)
            results[ticker] = self._engine.run(
                strategy_name=strategy,
                ticker=ticker,
                save_charts=save_individual_charts,
            )

        rows: list[dict[str, float | int | str]] = []
        for ticker, result in results.items():
            m = result.performance.metrics
            rows.append(
                {
                    "ticker": ticker,
                    "total_return": m.total_return,
                    "cagr": m.cagr,
                    "sharpe_ratio": m.sharpe_ratio,
                    "sortino_ratio": m.sortino_ratio,
                    "max_drawdown": m.max_drawdown,
                    "volatility": m.volatility,
                    "win_rate": m.win_rate,
                    "profit_factor": m.profit_factor,
                    "num_trades": m.num_trades,
                }
            )

        metrics_df = pd.DataFrame(rows).set_index("ticker")
        metrics_df = metrics_df[list(COMPARISON_METRIC_COLUMNS)]

        averages = {
            "avg_return": float(metrics_df["total_return"].mean()),
            "avg_sharpe": float(metrics_df["sharpe_ratio"].mean()),
            "avg_drawdown": float(metrics_df["max_drawdown"].mean()),
            "avg_cagr": float(metrics_df["cagr"].mean()),
            "avg_sortino": float(metrics_df["sortino_ratio"].mean()),
            "avg_volatility": float(metrics_df["volatility"].mean()),
        }

        ranking_df = metrics_df.sort_values("total_return", ascending=False).copy()
        ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._settings.comparisons_dir / f"{strategy}_universe_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return UniverseComparisonResult(
            strategy_name=strategy,
            tickers=universe,
            results=results,
            metrics_df=metrics_df,
            averages=averages,
            ranking_df=ranking_df,
            output_dir=output_dir,
            run_id=run_id,
        )
