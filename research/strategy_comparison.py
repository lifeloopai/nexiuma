"""Run and compare multiple strategies on a single ticker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from analytics.performance import PerformanceAnalyzer, PerformanceReport
from config.settings import NexiumaSettings, get_settings
from core.engine import BacktestEngine, BacktestRunResult
from data.downloader import MarketDataDownloader
from research.constants import COMPARISON_METRIC_COLUMNS
from strategies.registry import list_strategies


@dataclass
class StrategyComparisonResult:
    """Output from a multi-strategy comparison run."""

    ticker: str
    start_date: str
    end_date: str
    results: dict[str, BacktestRunResult]
    comparison_df: pd.DataFrame
    output_dir: Path
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    @property
    def reports(self) -> dict[str, PerformanceReport]:
        return {name: r.performance for name, r in self.results.items()}


class StrategyComparator:
    """Execute all (or selected) strategies on one ticker and compare metrics."""

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
        ticker: str | None = None,
        strategies: Sequence[str] | None = None,
        save_individual_charts: bool = False,
    ) -> StrategyComparisonResult:
        """Run backtests and build comparison dataframe."""
        ticker = (ticker or self._settings.data.ticker).upper()
        strategy_names = list(strategies) if strategies else self._default_strategy_names()

        logger.info("Comparing {} strategies on {}", len(strategy_names), ticker)
        ohlcv = MarketDataDownloader(self._settings).get_data(ticker=ticker)

        results: dict[str, BacktestRunResult] = {}
        for name in strategy_names:
            logger.info("Running strategy: {}", name)
            results[name] = self._engine.run(
                strategy_name=name,
                ticker=ticker,
                ohlcv=ohlcv,
                save_charts=save_individual_charts,
            )

        comparison_df = self._analyzer.comparison_metrics_frame(
            {n: r.performance for n, r in results.items()}
        )
        comparison_df = comparison_df[list(COMPARISON_METRIC_COLUMNS)]

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._settings.comparisons_dir / f"{ticker}_strategies_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return StrategyComparisonResult(
            ticker=ticker,
            start_date=str(ohlcv.index.min().date()),
            end_date=str(ohlcv.index.max().date()),
            results=results,
            comparison_df=comparison_df,
            output_dir=output_dir,
            run_id=run_id,
        )

    @staticmethod
    def _default_strategy_names() -> list[str]:
        return [s["name"] for s in list_strategies()]
