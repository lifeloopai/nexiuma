"""Walk-forward testing: optimize on train, validate on unseen test windows."""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import NexiumaSettings, get_settings
from core.engine import BacktestEngine, BacktestRunResult
from data.downloader import MarketDataDownloader
from research.optimizer import StrategyOptimizer
from research.walkforward_result import (
    RobustnessMetrics,
    WalkForwardResult,
    WalkForwardWindowSpec,
    WindowResult,
)
from strategies.parameters import DEFAULT_MA_OPTIMIZATION_GRID, MovingAverageParams


def slice_ohlcv(ohlcv: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Return OHLCV rows within inclusive calendar bounds."""
    mask = (ohlcv.index >= pd.Timestamp(start)) & (ohlcv.index <= pd.Timestamp(end))
    return ohlcv.loc[mask].copy()


def generate_window_specs(
    data_start: date,
    data_end: date,
    train_years: int = 3,
    test_years: int = 1,
) -> list[WalkForwardWindowSpec]:
    """Build rolling train/test window calendar specs.

    Example (2020–2024, train=3, test=1):
        Window 1: train 2020–2022, test 2023
        Window 2: train 2021–2023, test 2024
    """
    if train_years < 1 or test_years < 1:
        raise ValueError("train_years and test_years must be at least 1")

    windows: list[WalkForwardWindowSpec] = []
    window_id = 1
    train_start = date(data_start.year, 1, 1)

    while True:
        train_end_year = train_start.year + train_years - 1
        train_end = date(train_end_year, 12, 31)
        test_start = date(train_end_year + 1, 1, 1)
        test_end_year = test_start.year + test_years - 1
        test_end = date(test_end_year, 12, 31)

        if train_end >= data_end or test_start > data_end:
            break
        if test_end > data_end:
            test_end = data_end

        if train_start >= data_start and test_end <= data_end:
            windows.append(
                WalkForwardWindowSpec(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            window_id += 1

        train_start = date(train_start.year + test_years, 1, 1)
        if train_start.year + train_years - 1 > data_end.year:
            break

    return windows


class WalkForwardEngine:
    """Run walk-forward optimization and out-of-sample validation."""

    def __init__(
        self,
        settings: NexiumaSettings | None = None,
        engine: BacktestEngine | None = None,
        optimizer: StrategyOptimizer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._engine = engine or BacktestEngine(self._settings)
        self._optimizer = optimizer or StrategyOptimizer(self._settings, self._engine)

    def run(
        self,
        ticker: str | None = None,
        strategy_name: str | None = None,
        train_years: int = 3,
        test_years: int = 1,
        grid: Sequence[tuple[int, int]] | None = None,
        prepare_output_dir: bool = True,
    ) -> WalkForwardResult:
        """Execute full walk-forward pipeline for moving-average strategy."""
        ticker = (ticker or self._settings.data.ticker).upper()
        strategy = (strategy_name or self._settings.strategy).lower()
        if strategy not in ("moving_average", "ma"):
            raise ValueError(
                f"Walk-forward currently supports moving_average (got '{strategy}')"
            )

        combinations = tuple(grid) if grid else DEFAULT_MA_OPTIMIZATION_GRID
        ohlcv = MarketDataDownloader(self._settings).get_data(ticker=ticker)
        data_start = ohlcv.index.min().date()
        data_end = ohlcv.index.max().date()

        specs = generate_window_specs(data_start, data_end, train_years, test_years)
        if not specs:
            raise ValueError(
                f"Insufficient data ({data_start} → {data_end}) for "
                f"train={train_years}y / test={test_years}y walk-forward"
            )

        logger.info(
            "Walk-forward {} / {}: {} windows, train={}y test={}y",
            ticker,
            strategy,
            len(specs),
            train_years,
            test_years,
        )

        window_results: list[WindowResult] = []
        equity_segments: list[pd.Series] = []
        capital = self._settings.risk.initial_capital

        for spec in specs:
            wr = self._run_window(ticker, ohlcv, spec, combinations, capital)
            window_results.append(wr)
            if wr.test_run is not None and not wr.test_run.equity_curve.empty:
                eq = wr.test_run.equity_curve.astype(float)
                scaled = capital * (eq / eq.iloc[0])
                equity_segments.append(scaled)
                capital = float(scaled.iloc[-1])

        combined_equity = (
            pd.concat(equity_segments).sort_index()
            if equity_segments
            else pd.Series(dtype=float)
        )

        results_df = pd.DataFrame([w.to_dict() for w in window_results])
        parameter_history_df = results_df[
            [
                "window_id",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "best_params",
                "fast_period",
                "slow_period",
                "train_sharpe",
                "test_sharpe",
            ]
        ].copy()

        robustness = self._compute_robustness(window_results)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            self._settings.walkforward_dir / f"{ticker}_{strategy}_{run_id}"
        )
        if prepare_output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return WalkForwardResult(
            ticker=ticker,
            strategy_name="moving_average",
            train_years=train_years,
            test_years=test_years,
            start_date=str(data_start),
            end_date=str(data_end),
            windows=window_results,
            results_df=results_df,
            parameter_history_df=parameter_history_df,
            combined_equity=combined_equity,
            robustness=robustness,
            output_dir=output_dir,
            run_id=run_id,
        )

    def _run_window(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        spec: WalkForwardWindowSpec,
        grid: Sequence[tuple[int, int]],
        capital: float,
    ) -> WindowResult:
        """Optimize on train slice, backtest best params on test slice."""
        train_ohlcv = slice_ohlcv(ohlcv, spec.train_start, spec.train_end)
        test_ohlcv = slice_ohlcv(ohlcv, spec.test_start, spec.test_end)

        if len(train_ohlcv) < 60 or len(test_ohlcv) < 20:
            raise ValueError(
                f"Window {spec.window_id}: insufficient bars "
                f"(train={len(train_ohlcv)}, test={len(test_ohlcv)})"
            )

        logger.info(
            "Window {} — train {} test {}",
            spec.window_id,
            spec.train_label,
            spec.test_label,
        )

        train_df, train_runs = self._optimizer.run_ma_grid(ticker, train_ohlcv, grid)
        best_row = StrategyOptimizer.select_best_params(train_df)
        best_params = str(best_row["params"])
        params = MovingAverageParams(
            fast_period=int(best_row["fast_period"]),
            slow_period=int(best_row["slow_period"]),
        )

        original_capital = self._settings.risk.initial_capital
        window_settings = self._replace_capital(self._settings, capital)
        window_engine = BacktestEngine(window_settings)
        test_run = window_engine.run(
            strategy_name="moving_average",
            ticker=ticker,
            ohlcv=test_ohlcv,
            save_charts=False,
            strategy_params=params.to_backtrader_kwargs(),
        )

        train_run = train_runs.get(best_params)
        tm = test_run.performance.metrics

        return WindowResult(
            window_id=spec.window_id,
            train_start=str(spec.train_start),
            train_end=str(spec.train_end),
            test_start=str(spec.test_start),
            test_end=str(spec.test_end),
            best_params=best_params,
            fast_period=params.fast_period,
            slow_period=params.slow_period,
            train_sharpe=float(best_row["sharpe_ratio"]),
            train_return=float(best_row["total_return"]),
            test_sharpe=tm.sharpe_ratio,
            test_return=tm.total_return,
            test_cagr=tm.cagr,
            test_sortino=tm.sortino_ratio,
            test_max_drawdown=tm.max_drawdown,
            test_volatility=tm.volatility,
            test_num_trades=tm.num_trades,
            train_run=train_run,
            test_run=test_run,
        )

    @staticmethod
    def _replace_capital(settings: NexiumaSettings, capital: float) -> NexiumaSettings:
        from dataclasses import replace

        from config.settings import RiskSettings

        return replace(
            settings,
            risk=RiskSettings(
                initial_capital=capital,
                position_size_pct=settings.risk.position_size_pct,
                stop_loss_pct=settings.risk.stop_loss_pct,
                take_profit_pct=settings.risk.take_profit_pct,
                max_position_size_pct=settings.risk.max_position_size_pct,
                commission_pct=settings.risk.commission_pct,
                slippage_pct=settings.risk.slippage_pct,
                risk_free_rate=settings.risk.risk_free_rate,
                use_volatility_sizing=settings.risk.use_volatility_sizing,
                target_volatility=settings.risk.target_volatility,
            ),
        )

    @staticmethod
    def _compute_robustness(windows: list[WindowResult]) -> RobustnessMetrics:
        if not windows:
            return RobustnessMetrics(0, 0, 0, 0, 0, 0, 0)

        test_sharpes = np.array([w.test_sharpe for w in windows], dtype=float)
        test_returns = np.array([w.test_return for w in windows], dtype=float)
        test_dds = np.array([w.test_max_drawdown for w in windows], dtype=float)
        train_sharpes = np.array([w.train_sharpe for w in windows], dtype=float)
        fast = np.array([w.fast_period for w in windows], dtype=float)
        slow = np.array([w.slow_period for w in windows], dtype=float)

        unique_params = len({w.best_params for w in windows})
        stability = 1.0 - (unique_params - 1) / max(len(windows) - 1, 1)
        param_cv = float(np.std(fast) / np.mean(fast) + np.std(slow) / np.mean(slow)) / 2
        stability = max(0.0, min(1.0, stability * (1.0 - param_cv)))

        return RobustnessMetrics(
            avg_test_sharpe=float(test_sharpes.mean()),
            avg_test_return=float(test_returns.mean()),
            worst_test_drawdown=float(test_dds.min()),
            parameter_stability=stability,
            profitable_windows_pct=float((test_returns > 0).mean()),
            num_windows=len(windows),
            avg_train_sharpe=float(train_sharpes.mean()),
        )
