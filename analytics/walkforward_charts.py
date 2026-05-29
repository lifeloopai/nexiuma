"""Plotly charts for walk-forward analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from research.walkforward_result import WalkForwardResult


class WalkForwardChartGenerator:
    """Interactive walk-forward visualizations."""

    def save_interactive_charts(
        self,
        result: WalkForwardResult,
        path: Path,
    ) -> Path:
        """Parameter history, train vs test, and rolling Sharpe."""
        df = result.results_df
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Parameter Selection History",
                "Train vs Test Sharpe",
                "Train vs Test Return",
                "Rolling Test Sharpe",
            ),
            specs=[[{"colspan": 2}, None], [{}, {}]],
            vertical_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(
                x=df["window_id"],
                y=df["fast_period"],
                mode="lines+markers",
                name="Fast Period",
                line=dict(color="#2563eb"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["window_id"],
                y=df["slow_period"],
                mode="lines+markers",
                name="Slow Period",
                line=dict(color="#16a34a"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=[f"W{int(w)}" for w in df["window_id"]],
                y=df["train_sharpe"],
                name="Train Sharpe",
                marker_color="#93c5fd",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[f"W{int(w)}" for w in df["window_id"]],
                y=df["test_sharpe"],
                name="Test Sharpe",
                marker_color="#2563eb",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=[f"W{int(w)}" for w in df["window_id"]],
                y=df["train_return"],
                name="Train Return",
                marker_color="#fca5a5",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=[f"W{int(w)}" for w in df["window_id"]],
                y=df["test_return"],
                name="Test Return",
                marker_color="#dc2626",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        if not result.combined_equity.empty:
            rets = result.combined_equity.pct_change().dropna()
            rolling = rets.rolling(min(63, max(len(rets) // 2, 5))).mean() * (252 ** 0.5)
            fig.add_trace(
                go.Scatter(
                    x=rolling.index,
                    y=rolling.values,
                    mode="lines",
                    name="Rolling Sharpe",
                    line=dict(color="#7c3aed"),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title=f"Walk-Forward Analysis — {result.ticker}",
            height=780,
            template="plotly_white",
            barmode="group",
        )
        fig.update_yaxes(tickformat=".0%", row=2, col=2)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        return path

    @staticmethod
    def rolling_sharpe_series(equity: pd.Series, window: int = 63) -> pd.Series:
        rets = equity.astype(float).pct_change().dropna()
        if len(rets) < window:
            window = max(len(rets) // 2, 5)
        excess = rets - 0.04 / 252
        rolling = excess.rolling(window).mean() / excess.rolling(window).std()
        return (rolling * (252 ** 0.5)).dropna()
