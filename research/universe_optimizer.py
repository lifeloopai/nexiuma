"""Parameter optimization across a multi-asset universe."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from config.settings import NexiumaSettings, get_settings
from core.engine import BacktestRunResult
from data.downloader import MarketDataDownloader
from research.constants import UNIVERSE_OPTIMIZATION_TICKERS
from research.optimizer import StrategyOptimizer
from strategies.parameters import DEFAULT_MA_OPTIMIZATION_GRID


AGGREGATE_COLUMNS: tuple[str, ...] = (
    "avg_return",
    "avg_cagr",
    "avg_sharpe",
    "avg_drawdown",
    "avg_trades",
    "sharpe_std",
    "positive_sharpe_pct",
)


@dataclass
class UniverseOptimizationResult:
    """Cross-asset parameter optimization output."""

    strategy_name: str
    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    full_results_df: pd.DataFrame
    averages_df: pd.DataFrame
    ranking_df: pd.DataFrame
    robust_params: str
    output_dir: Path
    runs: dict[str, BacktestRunResult] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


class UniverseOptimizer:
    """Grid-search parameters across many tickers and rank by robustness."""

    def __init__(
        self,
        settings: NexiumaSettings | None = None,
        optimizer: StrategyOptimizer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._optimizer = optimizer or StrategyOptimizer(self._settings)

    def optimize_moving_average_universe(
        self,
        tickers: Sequence[str] | None = None,
        grid: Sequence[tuple[int, int]] | None = None,
    ) -> UniverseOptimizationResult:
        """Run MA parameter grid on each ticker and aggregate by parameter set."""
        universe = tuple(t.upper() for t in (tickers or UNIVERSE_OPTIMIZATION_TICKERS))
        combinations = tuple(grid) if grid else DEFAULT_MA_OPTIMIZATION_GRID
        downloader = MarketDataDownloader(self._settings)

        logger.info(
            "Universe optimization: {} tickers × {} parameter sets",
            len(universe),
            len(combinations),
        )

        all_rows: list[pd.DataFrame] = []
        all_runs: dict[str, BacktestRunResult] = {}
        start_date = ""
        end_date = ""

        for ticker in universe:
            ohlcv = downloader.get_data(ticker=ticker)
            if not start_date:
                start_date = str(ohlcv.index.min().date())
                end_date = str(ohlcv.index.max().date())

            ticker_df, runs = self._optimizer.run_ma_grid(ticker, ohlcv, combinations)
            all_rows.append(ticker_df)
            for label, run in runs.items():
                all_runs[f"{ticker}_{label}"] = run

        full_results_df = pd.concat(all_rows, ignore_index=True)
        averages_df = self._aggregate_by_params(full_results_df)
        ranking_df = self._build_ranking(averages_df)
        robust_params = str(ranking_df.iloc[0]["params"])

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._settings.universe_optimization_dir / f"moving_average_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return UniverseOptimizationResult(
            strategy_name="moving_average",
            tickers=universe,
            start_date=start_date,
            end_date=end_date,
            full_results_df=full_results_df,
            averages_df=averages_df,
            ranking_df=ranking_df,
            robust_params=robust_params,
            output_dir=output_dir,
            runs=all_runs,
            run_id=run_id,
        )

    @staticmethod
    def _aggregate_by_params(full_df: pd.DataFrame) -> pd.DataFrame:
        """Compute average metrics and consistency stats per parameter set."""
        grouped = full_df.groupby("params", sort=False)
        rows: list[dict[str, float | int | str]] = []

        for params_label, group in grouped:
            sharpe = group["sharpe_ratio"].astype(float)
            rows.append(
                {
                    "params": params_label,
                    "fast_period": int(group["fast_period"].iloc[0]),
                    "slow_period": int(group["slow_period"].iloc[0]),
                    "avg_return": float(group["total_return"].mean()),
                    "avg_cagr": float(group["cagr"].mean()),
                    "avg_sharpe": float(sharpe.mean()),
                    "avg_drawdown": float(group["max_drawdown"].mean()),
                    "avg_trades": float(group["num_trades"].mean()),
                    "sharpe_std": float(sharpe.std(ddof=0)) if len(sharpe) > 1 else 0.0,
                    "positive_sharpe_pct": float((sharpe > 0).mean()),
                }
            )

        return pd.DataFrame(rows).set_index("params")

    @staticmethod
    def _build_ranking(averages_df: pd.DataFrame) -> pd.DataFrame:
        """Rank parameter sets by average Sharpe, then consistency (low std)."""
        ranked = averages_df.copy()
        ranked["robust_score"] = ranked["avg_sharpe"] - 0.5 * ranked["sharpe_std"]
        ranked = ranked.sort_values(
            ["avg_sharpe", "sharpe_std", "positive_sharpe_pct"],
            ascending=[False, True, False],
        )
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        return ranked.reset_index()
