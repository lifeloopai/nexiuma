"""Walk-forward analysis orchestrated across a multi-asset universe."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import NexiumaSettings, get_settings
from research.constants import WALKFORWARD_UNIVERSE_TICKERS
from research.walkforward import WalkForwardEngine
from research.walkforward_result import WalkForwardResult
from research.walkforward_universe_result import (
    UniverseRobustnessScore,
    UniverseSummaryMetrics,
    WalkForwardUniverseResult,
)
from strategies.parameters import DEFAULT_MA_OPTIMIZATION_GRID


class WalkForwardUniverseEngine:
    """Run walk-forward testing on each asset and aggregate universe robustness."""

    def __init__(
        self,
        settings: NexiumaSettings | None = None,
        wf_engine: WalkForwardEngine | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._wf_engine = wf_engine or WalkForwardEngine(self._settings)

    def run(
        self,
        strategy_name: str | None = None,
        tickers: Sequence[str] | None = None,
        train_years: int = 3,
        test_years: int = 1,
        grid: Sequence[tuple[int, int]] | None = None,
    ) -> WalkForwardUniverseResult:
        """Execute walk-forward on each ticker and build universe aggregates."""
        strategy = (strategy_name or self._settings.strategy).lower()
        if strategy not in ("moving_average", "ma"):
            raise ValueError(
                f"Walk-forward universe supports moving_average (got '{strategy}')"
            )

        universe = tuple(t.upper() for t in (tickers or WALKFORWARD_UNIVERSE_TICKERS))
        combinations = tuple(grid) if grid else DEFAULT_MA_OPTIMIZATION_GRID

        logger.info(
            "Walk-forward universe: {} assets, strategy={}, train={}y test={}y",
            len(universe),
            strategy,
            train_years,
            test_years,
        )

        per_asset: dict[str, WalkForwardResult] = {}
        window_frames: list[pd.DataFrame] = []
        asset_rows: list[dict[str, float | int | str]] = []

        for ticker in universe:
            logger.info("Walk-forward starting for {}", ticker)
            wf_result = self._wf_engine.run(
                ticker=ticker,
                strategy_name=strategy,
                train_years=train_years,
                test_years=test_years,
                grid=combinations,
                prepare_output_dir=False,
            )
            per_asset[ticker] = wf_result

            wf_df = wf_result.results_df.copy()
            wf_df.insert(0, "ticker", ticker)
            wf_df["sharpe_degradation"] = wf_df["train_sharpe"] - wf_df["test_sharpe"]
            window_frames.append(wf_df)

            rob = wf_result.robustness
            asset_rows.append(
                {
                    "ticker": ticker,
                    "num_windows": rob.num_windows,
                    "avg_train_sharpe": rob.avg_train_sharpe,
                    "avg_test_sharpe": rob.avg_test_sharpe,
                    "sharpe_degradation": rob.avg_train_sharpe - rob.avg_test_sharpe,
                    "avg_return": rob.avg_test_return,
                    "avg_cagr": float(wf_df["test_cagr"].mean()),
                    "avg_drawdown": rob.worst_test_drawdown,
                    "win_rate": rob.profitable_windows_pct,
                    "parameter_stability": rob.parameter_stability,
                    "pct_positive_test_sharpe": float((wf_df["test_sharpe"] > 0).mean()),
                    "pct_positive_return": float((wf_df["test_return"] > 0).mean()),
                }
            )

        window_results_df = pd.concat(window_frames, ignore_index=True)
        asset_results_df = pd.DataFrame(asset_rows).set_index("ticker")
        summary_metrics = self._compute_summary_metrics(
            asset_results_df, window_results_df, universe
        )
        parameter_frequency_df = self._compute_parameter_frequency(window_results_df)
        robustness_score = self._compute_robustness_score(
            summary_metrics, window_results_df
        )
        summary_df = pd.DataFrame([summary_metrics.to_dict()]).T
        summary_df.columns = ["value"]

        executive = self._build_executive_summary(
            strategy, summary_metrics, robustness_score, parameter_frequency_df
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._settings.walkforward_universe_dir / f"{strategy}_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        return WalkForwardUniverseResult(
            strategy_name="moving_average",
            tickers=universe,
            train_years=train_years,
            test_years=test_years,
            asset_results_df=asset_results_df,
            window_results_df=window_results_df,
            summary_df=summary_df,
            parameter_frequency_df=parameter_frequency_df,
            summary_metrics=summary_metrics,
            robustness_score=robustness_score,
            per_asset_results=per_asset,
            output_dir=output_dir,
            run_id=run_id,
            executive_summary=executive,
        )

    @staticmethod
    def _compute_summary_metrics(
        asset_df: pd.DataFrame,
        window_df: pd.DataFrame,
        universe: tuple[str, ...],
    ) -> UniverseSummaryMetrics:
        """Aggregate per-asset and per-window statistics."""
        best = asset_df["avg_test_sharpe"].idxmax()
        worst = asset_df["avg_test_sharpe"].idxmin()

        param_counts = window_df["best_params"].value_counts()
        most_frequent = str(param_counts.index[0]) if not param_counts.empty else "N/A"
        unique_params = window_df["best_params"].nunique()
        total_selections = len(window_df)
        stability = 1.0 - (unique_params - 1) / max(total_selections - 1, 1)

        return UniverseSummaryMetrics(
            avg_train_sharpe=float(window_df["train_sharpe"].mean()),
            avg_test_sharpe=float(window_df["test_sharpe"].mean()),
            sharpe_degradation=float(
                window_df["train_sharpe"].mean() - window_df["test_sharpe"].mean()
            ),
            avg_return=float(window_df["test_return"].mean()),
            avg_cagr=float(window_df["test_cagr"].mean()),
            avg_drawdown=float(window_df["test_max_drawdown"].mean()),
            win_rate=float((window_df["test_return"] > 0).mean()),
            parameter_stability=max(0.0, min(1.0, stability)),
            most_frequent_params=most_frequent,
            best_asset=str(best),
            worst_asset=str(worst),
            pct_positive_test_sharpe=float((window_df["test_sharpe"] > 0).mean()),
            pct_positive_return=float((window_df["test_return"] > 0).mean()),
            num_assets=len(universe),
            num_windows_total=len(window_df),
        )

    @staticmethod
    def _compute_parameter_frequency(window_df: pd.DataFrame) -> pd.DataFrame:
        """Count how often each parameter set was selected."""
        counts = window_df["best_params"].value_counts().reset_index()
        counts.columns = ["params", "count"]
        total = counts["count"].sum()
        counts["frequency_pct"] = counts["count"] / total if total else 0.0
        counts["frequency_pct"] = counts["frequency_pct"].round(4)
        return counts

    @staticmethod
    def _compute_robustness_score(
        summary: UniverseSummaryMetrics,
        window_df: pd.DataFrame,
    ) -> UniverseRobustnessScore:
        """Composite 0–100 score from OOS Sharpe, degradation, win rate, stability."""
        sharpe_norm = float(np.clip(summary.avg_test_sharpe / 1.5, 0.0, 1.0))
        sharpe_component = sharpe_norm * 30.0

        train_mean = float(window_df["train_sharpe"].mean())
        degradation_ratio = summary.sharpe_degradation / max(abs(train_mean), 0.01)
        degradation_norm = float(np.clip(1.0 - degradation_ratio, 0.0, 1.0))
        degradation_component = degradation_norm * 25.0

        win_rate_component = summary.win_rate * 25.0
        stability_component = summary.parameter_stability * 20.0

        total = sharpe_component + degradation_component + win_rate_component + stability_component
        score = float(np.clip(total, 0.0, 100.0))

        return UniverseRobustnessScore(
            score=round(score, 1),
            grade=UniverseRobustnessScore.grade_from_score(score),
            test_sharpe_component=round(sharpe_component, 2),
            degradation_component=round(degradation_component, 2),
            win_rate_component=round(win_rate_component, 2),
            stability_component=round(stability_component, 2),
        )

    @staticmethod
    def _build_executive_summary(
        strategy: str,
        summary: UniverseSummaryMetrics,
        score: UniverseRobustnessScore,
        param_freq: pd.DataFrame,
    ) -> str:
        top_param = summary.most_frequent_params
        top_freq = 0.0
        if not param_freq.empty and top_param in param_freq["params"].values:
            top_freq = float(
                param_freq.loc[param_freq["params"] == top_param, "frequency_pct"].iloc[0]
            )

        return (
            f"The {strategy} strategy achieved an average out-of-sample Sharpe of "
            f"{summary.avg_test_sharpe:.2f} across {summary.num_assets} assets "
            f"({summary.num_windows_total} total windows). "
            f"{summary.pct_positive_test_sharpe:.0%} of asset-windows produced positive "
            f"out-of-sample Sharpe and {summary.pct_positive_return:.0%} were profitable. "
            f"Parameter {top_param} was selected in {top_freq:.0%} of windows. "
            f"Best asset: {summary.best_asset}; worst: {summary.worst_asset}. "
            f"Universe robustness score: {score.score:.0f}/100 ({score.grade})."
        )
