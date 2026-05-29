"""Performance analysis utilities and strategy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analytics.metrics import BacktestMetrics, compute_metrics
from analytics.risk import RiskAnalyzer, RiskMetrics


@dataclass
class PerformanceReport:
    """Combined performance and risk report for a single backtest."""

    metrics: BacktestMetrics
    risk: RiskMetrics
    equity_curve: pd.Series
    benchmark_return: float | None = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {**self.metrics.to_dict(), **self.risk.to_dict()}
        if self.benchmark_return is not None:
            out["benchmark_return"] = self.benchmark_return
            out["alpha_vs_benchmark"] = self.metrics.total_return - self.benchmark_return
        return out


class PerformanceAnalyzer:
    """Build performance reports and compare multiple strategy runs."""

    def __init__(self, risk_free_rate: float = 0.04) -> None:
        self.risk_free_rate = risk_free_rate
        self._risk_analyzer = RiskAnalyzer()

    def build_report(
        self,
        equity_curve: pd.Series,
        trade_pnls: list[float],
        initial_capital: float,
        price_series: pd.Series | None = None,
    ) -> PerformanceReport:
        """Create full performance report."""
        metrics = compute_metrics(
            equity_curve,
            trade_pnls,
            initial_capital,
            risk_free_rate=self.risk_free_rate,
        )
        risk = self._risk_analyzer.analyze(
            equity_curve, metrics.annualized_return
        )
        benchmark_return = None
        if price_series is not None and len(price_series) > 1:
            benchmark_return = float(price_series.iloc[-1] / price_series.iloc[0] - 1)

        return PerformanceReport(
            metrics=metrics,
            risk=risk,
            equity_curve=equity_curve,
            benchmark_return=benchmark_return,
        )

    @staticmethod
    def compare_strategies(reports: dict[str, PerformanceReport]) -> pd.DataFrame:
        """Tabular comparison of multiple strategy backtests."""
        rows = []
        for name, report in reports.items():
            row = {"strategy": name, **report.summary()}
            rows.append(row)
        return pd.DataFrame(rows).set_index("strategy")

    @staticmethod
    def comparison_metrics_frame(reports: dict[str, PerformanceReport]) -> pd.DataFrame:
        """Focused metrics table for strategy comparison output."""
        rows: list[dict[str, float | int | str]] = []
        for name, report in reports.items():
            m = report.metrics
            rows.append(
                {
                    "strategy": name,
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
        return pd.DataFrame(rows).set_index("strategy")
