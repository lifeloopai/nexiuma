"""Central backtest engine orchestrating data, execution, and analytics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Type

import backtrader as bt
import pandas as pd
from loguru import logger

from analytics.performance import PerformanceAnalyzer, PerformanceReport
from analytics.tearsheet import TearsheetGenerator
from analytics.visualization import ChartGenerator
from core.analyzers import EquityCurveAnalyzer
from config.settings import NexiumaSettings, get_settings
from core.broker import SimulatedBroker
from core.execution import ExecutionSimulator
from data.downloader import MarketDataDownloader
from strategies.base_strategy import NexiumaStrategy
from strategies.parameters import parse_ma_periods
from strategies.registry import get_strategy_class


@dataclass
class BacktestRunResult:
    """Complete output from a backtest run."""

    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    ohlcv: pd.DataFrame
    equity_curve: pd.Series
    performance: PerformanceReport
    tearsheet_path: Path | None = None
    buy_signals: list[pd.Timestamp] = field(default_factory=list)
    sell_signals: list[pd.Timestamp] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir: Path | None = None
    strategy_params: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "run_id": self.run_id,
            "strategy": self.strategy_name,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "performance": self.performance.summary(),
        }
        if self.strategy_params:
            meta["strategy_params"] = self.strategy_params
        return meta


class BacktestEngine:
    """Production backtest engine with dependency-injected components."""

    def __init__(
        self,
        settings: NexiumaSettings | None = None,
        downloader: MarketDataDownloader | None = None,
        analyzer: PerformanceAnalyzer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._downloader = downloader or MarketDataDownloader(self._settings)
        self._analyzer = analyzer or PerformanceAnalyzer(
            risk_free_rate=self._settings.risk.risk_free_rate
        )
        self._execution = ExecutionSimulator(self._settings.risk)
        self._charts = ChartGenerator()
        self._tearsheet = TearsheetGenerator()

    def run(
        self,
        strategy_name: str | None = None,
        ticker: str | None = None,
        ohlcv: pd.DataFrame | None = None,
        save_charts: bool = True,
        strategy_params: dict[str, Any] | None = None,
    ) -> BacktestRunResult:
        """Execute full backtest pipeline."""
        strategy_name = strategy_name or self._settings.strategy
        ticker = (ticker or self._settings.data.ticker).upper()
        kwargs = self._resolve_strategy_kwargs(strategy_name, strategy_params)
        if strategy_name == "moving_average":
            logger.info(
                "Engine starting: {} ({}/{}) on {}",
                strategy_name,
                kwargs.get("fast_period"),
                kwargs.get("slow_period"),
                ticker,
            )
        else:
            logger.info("Engine starting: {} on {}", strategy_name, ticker)

        if ohlcv is None:
            ohlcv = self._downloader.get_data(ticker=ticker)

        strategy_cls = get_strategy_class(strategy_name)
        cerebro = self._build_cerebro(ohlcv, strategy_cls, kwargs)
        broker = SimulatedBroker(cerebro)
        broker.configure_from_settings(
            self._settings.risk.initial_capital,
            self._settings.risk.commission_pct,
        )

        run_result = cerebro.run()
        strategy_instance: NexiumaStrategy = run_result[0]
        equity_curve = self._extract_equity(strategy_instance, cerebro, ohlcv)
        trade_pnls = [float(t["pnl_comm"]) for t in strategy_instance.trade_log]

        performance = self._analyzer.build_report(
            equity_curve=equity_curve,
            trade_pnls=trade_pnls,
            initial_capital=self._settings.risk.initial_capital,
            price_series=ohlcv["close"],
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            self._settings.backtest_results_dir / f"{ticker}_{strategy_name}_{run_id}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        result = BacktestRunResult(
            strategy_name=strategy_name,
            ticker=ticker,
            start_date=str(ohlcv.index.min().date()),
            end_date=str(ohlcv.index.max().date()),
            ohlcv=ohlcv,
            equity_curve=equity_curve,
            performance=performance,
            buy_signals=[pd.Timestamp(t) for t in strategy_instance.buy_signals],
            sell_signals=[pd.Timestamp(t) for t in strategy_instance.sell_signals],
            run_id=run_id,
            output_dir=output_dir,
            strategy_params=kwargs,
        )

        self._persist_results(result, save_charts)
        logger.info(
            "Engine complete: CAGR={:.2%}, Sharpe={:.2f}, trades={}",
            performance.metrics.cagr,
            performance.metrics.sharpe_ratio,
            performance.metrics.num_trades,
        )
        return result

    def _resolve_strategy_kwargs(
        self,
        strategy_name: str,
        strategy_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge settings and runtime strategy parameters."""
        if strategy_name != "moving_average":
            return dict(strategy_params or {})

        sp = self._settings.strategy_params
        fast = sp.fast_period
        slow = sp.slow_period
        if strategy_params:
            fast = int(strategy_params.get("fast_period", fast))
            slow = int(strategy_params.get("slow_period", slow))
        ma = parse_ma_periods(fast, slow, default_fast=fast, default_slow=slow)
        return ma.to_backtrader_kwargs()

    def _build_cerebro(
        self,
        ohlcv: pd.DataFrame,
        strategy_cls: Type[NexiumaStrategy],
        strategy_kwargs: dict[str, Any] | None = None,
    ) -> bt.Cerebro:
        cerebro = bt.Cerebro()

        data = bt.feeds.PandasData(
            dataname=ohlcv,
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        cerebro.adddata(data)
        cerebro.addstrategy(
            strategy_cls,
            risk=self._settings.risk,
            execution=self._execution,
            printlog=False,
            **(strategy_kwargs or {}),
        )
        cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        return cerebro

    def _extract_equity(
        self,
        strategy: NexiumaStrategy,
        cerebro: bt.Cerebro,
        ohlcv: pd.DataFrame,
    ) -> pd.Series:
        analysis = strategy.analyzers.equity.get_analysis()
        equity = analysis.get("equity_curve")
        if isinstance(equity, pd.Series) and not equity.empty:
            return equity
        initial = self._settings.risk.initial_capital
        final = cerebro.broker.getvalue()
        close = ohlcv["close"]
        ret = close.pct_change().fillna(0)
        cum = (1 + ret).cumprod()
        scaled = initial * cum * (final / max(initial * cum.iloc[-1], 1e-9))
        return pd.Series(scaled.values, index=ohlcv.index, name="equity")

    def _persist_results(self, result: BacktestRunResult, save_charts: bool) -> None:
        if result.output_dir is None:
            return
        meta_path = result.output_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(result.to_metadata(), indent=2, default=str),
            encoding="utf-8",
        )
        result.equity_curve.to_csv(result.output_dir / "equity_curve.csv", header=True)

        if save_charts:
            self._charts.save_all(result)
            result.tearsheet_path = self._tearsheet.generate(
                result, result.output_dir / "tearsheet.html"
            )
