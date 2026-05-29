"""Chart generation for backtest artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from core.engine import BacktestRunResult


class ChartGenerator:
    """Generate and save standard backtest visualizations."""

    def save_all(self, result: BacktestRunResult) -> None:
        if result.output_dir is None:
            return
        self.plot_equity(result)
        self.plot_drawdown(result)
        self.plot_price_signals(result)
        self.plot_rolling_returns(result)
        self.plot_return_distribution(result)

    def plot_equity(self, result: BacktestRunResult) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        result.equity_curve.plot(ax=ax, color="#2563eb", linewidth=1.5)
        ax.set_title(f"Equity Curve — {result.strategy_name} / {result.ticker}")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        self._save(fig, result.output_dir / "equity_curve.png")

    def plot_drawdown(self, result: BacktestRunResult) -> None:
        values = result.equity_curve.astype(float)
        dd = (values - values.cummax()) / values.cummax()
        fig, ax = plt.subplots(figsize=(12, 4))
        dd.plot(ax=ax, color="#dc2626", linewidth=1.2)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color="#dc2626")
        ax.set_title(f"Drawdown — {result.strategy_name}")
        ax.grid(True, alpha=0.3)
        self._save(fig, result.output_dir / "drawdown.png")

    def plot_price_signals(self, result: BacktestRunResult) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        close = result.ohlcv["close"]
        close.plot(ax=ax, color="#374151", linewidth=1.2, label="Close")
        for ts, color, marker in [
            (result.buy_signals, "#16a34a", "^"),
            (result.sell_signals, "#dc2626", "v"),
        ]:
            for t in ts:
                idx = close.index.get_indexer([t], method="nearest")[0]
                if idx >= 0:
                    ax.scatter(close.index[idx], close.iloc[idx], marker=marker, color=color, s=80)
        ax.set_title(f"Price & Trades — {result.ticker}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, result.output_dir / "price_chart.png")

    def plot_rolling_returns(self, result: BacktestRunResult, window: int = 63) -> None:
        rets = result.equity_curve.pct_change().dropna()
        rolling = rets.rolling(window).mean() * 252
        fig, ax = plt.subplots(figsize=(12, 4))
        rolling.plot(ax=ax, color="#7c3aed")
        ax.set_title(f"Rolling {window}-Day Annualized Return")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        self._save(fig, result.output_dir / "rolling_returns.png")

    def plot_return_distribution(self, result: BacktestRunResult) -> None:
        rets = result.equity_curve.pct_change().dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(rets.values, bins=50, color="#0ea5e9", alpha=0.75, edgecolor="white")
        ax.axvline(rets.mean(), color="#dc2626", linestyle="--", label=f"Mean: {rets.mean():.4f}")
        ax.set_title("Daily Return Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save(fig, result.output_dir / "return_distribution.png")

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.debug("Chart saved: {}", path)
