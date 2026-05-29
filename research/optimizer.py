"""Grid search optimization for strategy parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from config.settings import NexiumaSettings, get_settings
from core.engine import BacktestEngine, BacktestRunResult
from data.downloader import MarketDataDownloader
from strategies.parameters import (
    DEFAULT_MA_OPTIMIZATION_GRID,
    MovingAverageParams,
    ma_grid_from_strings,
)


OPTIMIZATION_COLUMNS: tuple[str, ...] = (
    "fast_period",
    "slow_period",
    "total_return",
    "cagr",
    "sharpe_ratio",
    "max_drawdown",
    "num_trades",
)


@dataclass
class OptimizationResult:
    """Output from a parameter grid search."""

    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    results_df: pd.DataFrame
    runs: dict[str, BacktestRunResult]
    output_dir: Path
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


class StrategyOptimizer:
    """Run backtests across a parameter grid and collect metrics."""

    def __init__(
        self,
        settings: NexiumaSettings | None = None,
        engine: BacktestEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._engine = engine or BacktestEngine(self._settings)

    def run_ma_grid(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        grid: Sequence[tuple[int, int]],
    ) -> tuple[pd.DataFrame, dict[str, BacktestRunResult]]:
        """Run moving-average grid on one ticker; return metrics and run objects."""
        rows: list[dict[str, float | int | str]] = []
        runs: dict[str, BacktestRunResult] = {}

        for fast, slow in grid:
            params = MovingAverageParams(fast_period=fast, slow_period=slow)
            label = params.label
            logger.info("{} — testing MA periods {}", ticker, label)

            run = self._engine.run(
                strategy_name="moving_average",
                ticker=ticker,
                ohlcv=ohlcv,
                save_charts=False,
                strategy_params=params.to_backtrader_kwargs(),
            )
            runs[label] = run
            m = run.performance.metrics
            rows.append(
                {
                    "ticker": ticker,
                    "fast_period": fast,
                    "slow_period": slow,
                    "params": label,
                    "total_return": m.total_return,
                    "cagr": m.cagr,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown": m.max_drawdown,
                    "num_trades": m.num_trades,
                }
            )

        return pd.DataFrame(rows), runs

    def optimize_moving_average(
        self,
        ticker: str | None = None,
        grid: Sequence[tuple[int, int]] | None = None,
    ) -> OptimizationResult:
        """Test moving-average period combinations on one ticker."""
        ticker = (ticker or self._settings.data.ticker).upper()
        combinations = tuple(grid) if grid else DEFAULT_MA_OPTIMIZATION_GRID

        logger.info(
            "Optimizing moving_average on {} ({} combinations)",
            ticker,
            len(combinations),
        )
        ohlcv = MarketDataDownloader(self._settings).get_data(ticker=ticker)
        full_df, runs = self.run_ma_grid(ticker, ohlcv, combinations)
        results_df = full_df.set_index("params").drop(columns=["ticker"])
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            self._settings.optimization_dir
            / f"{ticker}_moving_average_{run_id}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        return OptimizationResult(
            strategy_name="moving_average",
            ticker=ticker,
            start_date=str(ohlcv.index.min().date()),
            end_date=str(ohlcv.index.max().date()),
            results_df=results_df,
            runs=runs,
            output_dir=output_dir,
            run_id=run_id,
        )

    @staticmethod
    def parse_grid_arg(grid_str: str | None) -> tuple[tuple[int, int], ...]:
        if not grid_str:
            return DEFAULT_MA_OPTIMIZATION_GRID
        items = [g.strip() for g in grid_str.split(",") if g.strip()]
        return ma_grid_from_strings(items)

    @staticmethod
    def select_best_params(
        results_df: pd.DataFrame,
        metric: str = "sharpe_ratio",
    ) -> pd.Series:
        """Return the best parameter row by metric (default: Sharpe)."""
        if results_df.empty:
            raise ValueError("Cannot select best params from empty results")
        return results_df.sort_values(metric, ascending=False).iloc[0]
