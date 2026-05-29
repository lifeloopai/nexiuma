"""Plotly charts for parameter optimization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from research.optimizer import OptimizationResult
    from research.universe_optimizer import UniverseOptimizationResult


class OptimizationChartGenerator:
    """Generate optimization heatmaps and summary charts."""

    def save_heatmaps(self, result: OptimizationResult, path: Path) -> Path:
        """Plot heatmaps for return, Sharpe, and max drawdown."""
        df = result.results_df.reset_index()
        fast_vals = sorted(df["fast_period"].unique())
        slow_vals = sorted(df["slow_period"].unique())

        metrics = [
            ("total_return", "Total Return", ".1%"),
            ("sharpe_ratio", "Sharpe Ratio", ".2f"),
            ("max_drawdown", "Max Drawdown", ".1%"),
        ]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[m[1] for m in metrics],
            horizontal_spacing=0.08,
        )

        for col_idx, (col_name, _, _tick_fmt) in enumerate(metrics, start=1):
            z = self._build_matrix(df, fast_vals, slow_vals, col_name)
            text = [
                [
                    ""
                    if v is None
                    else (f"{v:.2f}" if col_name == "sharpe_ratio" else f"{v:.2%}")
                    for v in row
                ]
                for row in z
            ]
            colorscale = "RdYlGn" if col_name != "max_drawdown" else "RdYlGn_r"
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=[str(s) for s in slow_vals],
                    y=[str(f) for f in fast_vals],
                    colorscale=colorscale,
                    text=text,
                    texttemplate="%{text}",
                    showscale=col_idx == 3,
                    hoverongaps=False,
                ),
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(title_text="Slow Period", row=1, col=col_idx)
            fig.update_yaxes(title_text="Fast Period", row=1, col=col_idx)

        fig.update_layout(
            title=f"MA Optimization — {result.ticker}",
            height=420,
            template="plotly_white",
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        return path

    @staticmethod
    def _build_matrix(
        df,
        fast_vals: list[int],
        slow_vals: list[int],
        column: str,
    ) -> list[list[float | None]]:
        lookup = {
            (int(row["fast_period"]), int(row["slow_period"])): float(row[column])
            for _, row in df.iterrows()
        }
        matrix: list[list[float | None]] = []
        for fast in fast_vals:
            row_vals: list[float | None] = []
            for slow in slow_vals:
                row_vals.append(lookup.get((fast, slow)))
            matrix.append(row_vals)
        return matrix

    def save_universe_heatmaps(
        self,
        result: UniverseOptimizationResult,
        path: Path,
    ) -> Path:
        """Heatmaps of Sharpe and return across tickers × parameter sets."""
        full = result.full_results_df
        tickers = list(result.tickers)
        params = list(full["params"].unique())

        metrics = [
            ("sharpe_ratio", "Sharpe Ratio"),
            ("total_return", "Total Return"),
            ("max_drawdown", "Max Drawdown"),
        ]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[m[1] for m in metrics],
            horizontal_spacing=0.06,
        )

        for col_idx, (col_name, _) in enumerate(metrics, start=1):
            z: list[list[float | None]] = []
            text: list[list[str]] = []
            for param in params:
                row_vals: list[float | None] = []
                row_text: list[str] = []
                subset = full[full["params"] == param]
                lookup = {
                    str(r["ticker"]): float(r[col_name])
                    for _, r in subset.iterrows()
                }
                for ticker in tickers:
                    val = lookup.get(ticker)
                    row_vals.append(val)
                    if val is None:
                        row_text.append("")
                    elif col_name == "sharpe_ratio":
                        row_text.append(f"{val:.2f}")
                    else:
                        row_text.append(f"{val:.2%}")
                z.append(row_vals)
                text.append(row_text)

            colorscale = "RdYlGn" if col_name != "max_drawdown" else "RdYlGn_r"
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=tickers,
                    y=params,
                    colorscale=colorscale,
                    text=text,
                    texttemplate="%{text}",
                    showscale=col_idx == 3,
                    hoverongaps=False,
                ),
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(title_text="Ticker", row=1, col=col_idx)

        fig.update_layout(
            title="Universe Parameter Heatmaps",
            height=460,
            template="plotly_white",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        return path
