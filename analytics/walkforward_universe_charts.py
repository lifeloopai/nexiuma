"""Plotly visualizations for walk-forward universe analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from research.walkforward_universe_result import WalkForwardUniverseResult


class WalkForwardUniverseChartGenerator:
    """Generate interactive charts for universe walk-forward reports."""

    def save_sharpe_heatmap(
        self,
        result: WalkForwardUniverseResult,
        path: Path,
    ) -> Path:
        """Heatmap of test Sharpe by asset and window."""
        return self._save_metric_heatmap(
            result,
            path,
            metric="test_sharpe",
            title="Out-of-Sample Sharpe by Asset & Window",
            fmt=".2f",
            colorscale="RdYlGn",
        )

    def save_degradation_heatmap(
        self,
        result: WalkForwardUniverseResult,
        path: Path,
    ) -> Path:
        """Heatmap of Sharpe degradation (train − test) by asset and window."""
        return self._save_metric_heatmap(
            result,
            path,
            metric="sharpe_degradation",
            title="Sharpe Degradation (Train − Test)",
            fmt=".2f",
            colorscale="RdYlGn_r",
        )

    def save_robustness_dashboard(
        self,
        result: WalkForwardUniverseResult,
        path: Path,
    ) -> Path:
        """Multi-panel interactive dashboard."""
        asset_df = result.asset_results_df.reset_index()
        window_df = result.window_results_df
        param_df = result.parameter_frequency_df

        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Train vs Test Sharpe by Asset",
                "Sharpe Degradation by Asset",
                "Parameter Selection Frequency",
                "Asset Ranking (Avg Test Sharpe)",
                "Positive vs Negative Outcomes",
                "Robustness Score Breakdown",
            ),
            specs=[
                [{"colspan": 2}, None],
                [{}, {}],
                [{}, {}],
            ],
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )

        tickers = asset_df["ticker"].tolist()
        fig.add_trace(
            go.Bar(
                name="Train Sharpe",
                x=tickers,
                y=asset_df["avg_train_sharpe"],
                marker_color="#93c5fd",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name="Test Sharpe",
                x=tickers,
                y=asset_df["avg_test_sharpe"],
                marker_color="#2563eb",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=tickers,
                y=asset_df["sharpe_degradation"],
                marker_color="#dc2626",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=param_df["params"],
                y=param_df["frequency_pct"],
                marker_color="#16a34a",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(tickformat=".0%", row=2, col=2)

        ranked = asset_df.sort_values("avg_test_sharpe", ascending=True)
        fig.add_trace(
            go.Bar(
                x=ranked["avg_test_sharpe"],
                y=ranked["ticker"],
                orientation="h",
                marker_color="#7c3aed",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        pos_sharpe = int((window_df["test_sharpe"] > 0).sum())
        neg_sharpe = len(window_df) - pos_sharpe
        pos_ret = int((window_df["test_return"] > 0).sum())
        neg_ret = len(window_df) - pos_ret
        fig.add_trace(
            go.Bar(
                x=["Pos Sharpe", "Neg Sharpe", "Pos Return", "Neg Return"],
                y=[pos_sharpe, neg_sharpe, pos_ret, neg_ret],
                marker_color=["#16a34a", "#dc2626", "#2563eb", "#f97316"],
                showlegend=False,
            ),
            row=3,
            col=2,
        )

        score = result.robustness_score
        fig.add_trace(
            go.Bar(
                x=["Test Sharpe", "Low Degradation", "Win Rate", "Stability"],
                y=[
                    score.test_sharpe_component,
                    score.degradation_component,
                    score.win_rate_component,
                    score.stability_component,
                ],
                marker_color="#0ea5e9",
                showlegend=False,
            ),
            row=3,
            col=2,
        )

        fig.update_layout(
            title=(
                f"Walk-Forward Universe — {result.strategy_name} "
                f"(Score: {score.score:.0f}/100 {score.grade})"
            ),
            height=1100,
            template="plotly_white",
            barmode="group",
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        return path

    @staticmethod
    def _save_metric_heatmap(
        result: WalkForwardUniverseResult,
        path: Path,
        metric: str,
        title: str,
        fmt: str,
        colorscale: str,
    ) -> Path:
        df = result.window_results_df
        tickers = sorted(df["ticker"].unique())
        windows = sorted(df["window_id"].unique())

        z: list[list[float | None]] = []
        text: list[list[str]] = []
        for ticker in tickers:
            row_vals: list[float | None] = []
            row_text: list[str] = []
            subset = df[df["ticker"] == ticker]
            lookup = {int(r["window_id"]): float(r[metric]) for _, r in subset.iterrows()}
            for w in windows:
                val = lookup.get(int(w))
                row_vals.append(val)
                row_text.append("" if val is None else f"{val:{fmt}}")
            z.append(row_vals)
            text.append(row_text)

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=[f"W{w}" for w in windows],
                y=tickers,
                colorscale=colorscale,
                text=text,
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        fig.update_layout(title=title, height=420, template="plotly_white")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        return path
