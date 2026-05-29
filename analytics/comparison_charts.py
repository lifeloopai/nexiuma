"""Plotly charts for strategy and universe comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from research.strategy_comparison import StrategyComparisonResult
    from research.universe_comparison import UniverseComparisonResult


class ComparisonChartGenerator:
    """Build interactive Plotly comparison visualizations."""

    def save_strategy_comparison(
        self,
        result: StrategyComparisonResult,
        path: Path,
    ) -> Path:
        """Overlay equity curves and bar charts for key metrics."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Normalized Equity Curves",
                "Total Return",
                "Sharpe Ratio",
                "Max Drawdown",
            ),
            vertical_spacing=0.14,
        )

        for name, run in result.results.items():
            equity = run.equity_curve.astype(float)
            normalized = equity / equity.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    mode="lines",
                    name=name,
                ),
                row=1,
                col=1,
            )

        df = result.comparison_df
        fig.add_trace(
            go.Bar(
                x=df.index.tolist(),
                y=df["total_return"],
                showlegend=False,
                marker_color="#2563eb",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=df.index.tolist(),
                y=df["sharpe_ratio"],
                showlegend=False,
                marker_color="#16a34a",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df.index.tolist(),
                y=df["max_drawdown"],
                showlegend=False,
                marker_color="#dc2626",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"Strategy Comparison — {result.ticker}",
            height=720,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(tickformat=".0%", row=1, col=2)
        fig.update_yaxes(tickformat=".2f", row=2, col=1)
        fig.update_yaxes(tickformat=".0%", row=2, col=2)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        return path

    def save_universe_heatmaps(
        self,
        result: UniverseComparisonResult,
        path: Path,
    ) -> Path:
        """Heatmaps for return, Sharpe, and drawdown across tickers."""
        df = result.metrics_df
        metrics = [
            ("total_return", "Total Return"),
            ("sharpe_ratio", "Sharpe Ratio"),
            ("max_drawdown", "Max Drawdown"),
        ]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[m[1] for m in metrics],
            horizontal_spacing=0.06,
        )

        tickers = df.index.tolist()
        for col_idx, (col_name, _) in enumerate(metrics, start=1):
            values = df[col_name].values.reshape(1, -1)
            colorscale = "RdYlGn" if col_name != "max_drawdown" else "RdYlGn_r"
            text_row = []
            for v in values[0]:
                if col_name == "sharpe_ratio":
                    text_row.append(f"{v:.2f}")
                else:
                    text_row.append(f"{v:.2%}")
            fig.add_trace(
                go.Heatmap(
                    z=values,
                    x=tickers,
                    y=[result.strategy_name],
                    colorscale=colorscale,
                    showscale=col_idx == 3,
                    text=[text_row],
                    texttemplate="%{text}",
                    hoverongaps=False,
                ),
                row=1,
                col=col_idx,
            )

        fig.update_layout(
            title=f"Universe Heatmaps — {result.strategy_name}",
            height=380,
            template="plotly_white",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        return path
