"""Terminal, CSV, Plotly, and HTML outputs for comparison workflows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from analytics.comparison_charts import ComparisonChartGenerator
from config.settings import PROJECT_ROOT, NexiumaSettings, get_settings
from research.constants import COMPARISON_DISPLAY_NAMES, COMPARISON_METRIC_COLUMNS

if TYPE_CHECKING:
    from research.strategy_comparison import StrategyComparisonResult
    from research.universe_comparison import UniverseComparisonResult

TEMPLATE_DIR = PROJECT_ROOT / "reports" / "templates"


class ComparisonReportGenerator:
    """Persist strategy and universe comparison artifacts."""

    def __init__(self, settings: NexiumaSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._charts = ComparisonChartGenerator()
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def publish_strategy_comparison(
        self, result: StrategyComparisonResult
    ) -> dict[str, Path]:
        """Write all strategy comparison outputs to result.output_dir."""
        out = result.output_dir
        csv_path = out / "comparison.csv"
        chart_path = out / "comparison_chart.html"
        html_path = out / "index.html"

        result.comparison_df.to_csv(csv_path)
        self._charts.save_strategy_comparison(result, chart_path)
        html_path.write_text(self._render_strategy_html(result, chart_path.name), encoding="utf-8")

        meta = {
            "type": "strategy_comparison",
            "ticker": result.ticker,
            "run_id": result.run_id,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "strategies": list(result.results.keys()),
            "paths": {
                "csv": str(csv_path),
                "chart": str(chart_path),
                "html": str(html_path),
            },
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Strategy comparison saved: {}", out)
        return {"csv": csv_path, "chart": chart_path, "html": html_path}

    def publish_universe_comparison(
        self, result: UniverseComparisonResult
    ) -> dict[str, Path]:
        """Write universe comparison outputs."""
        out = result.output_dir
        metrics_path = out / "universe_metrics.csv"
        ranking_path = out / "ranking.csv"
        heatmap_path = out / "heatmaps.html"
        html_path = out / "index.html"

        result.metrics_df.to_csv(metrics_path)
        result.ranking_df.to_csv(ranking_path)
        self._charts.save_universe_heatmaps(result, heatmap_path)
        html_path.write_text(
            self._render_universe_html(result, heatmap_path.name),
            encoding="utf-8",
        )

        meta: dict[str, Any] = {
            "type": "universe_comparison",
            "strategy": result.strategy_name,
            "tickers": list(result.tickers),
            "run_id": result.run_id,
            "averages": result.averages,
            "paths": {
                "metrics": str(metrics_path),
                "ranking": str(ranking_path),
                "heatmaps": str(heatmap_path),
                "html": str(html_path),
            },
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        logger.info("Universe comparison saved: {}", out)
        return {
            "metrics": metrics_path,
            "ranking": ranking_path,
            "heatmaps": heatmap_path,
            "html": html_path,
        }

    def _render_strategy_html(
        self, result: StrategyComparisonResult, chart_filename: str
    ) -> str:
        rows = self._table_rows_from_df(result.comparison_df)
        template = self._env.get_template("comparison.html.j2")
        return template.render(
            title=f"Strategy Comparison — {result.ticker}",
            subtitle=f"{result.start_date} → {result.end_date}",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            columns=[COMPARISON_DISPLAY_NAMES[c] for c in COMPARISON_METRIC_COLUMNS],
            rows=rows,
            chart_path=chart_filename,
            summary_cards=[
                ("Ticker", result.ticker),
                ("Strategies", str(len(result.results))),
                ("Best Return", self._best_strategy(result.comparison_df, "total_return")),
                ("Best Sharpe", self._best_strategy(result.comparison_df, "sharpe_ratio")),
            ],
        )

    def _render_universe_html(
        self, result: UniverseComparisonResult, heatmap_filename: str
    ) -> str:
        rows = self._table_rows_from_df(result.metrics_df)
        ranking_rows = []
        for _, row in result.ranking_df.iterrows():
            ranking_rows.append(
                [
                    str(int(row["rank"])),
                    row.name,
                    f"{row['total_return']:.2%}",
                    f"{row['sharpe_ratio']:.2f}",
                    f"{row['max_drawdown']:.2%}",
                ]
            )

        template = self._env.get_template("universe_comparison.html.j2")
        return template.render(
            title=f"Universe Comparison — {result.strategy_name}",
            subtitle=", ".join(result.tickers),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            columns=[COMPARISON_DISPLAY_NAMES[c] for c in COMPARISON_METRIC_COLUMNS],
            rows=rows,
            heatmap_path=heatmap_filename,
            averages=[
                ("Average Return", f"{result.averages['avg_return']:.2%}"),
                ("Average Sharpe", f"{result.averages['avg_sharpe']:.2f}"),
                ("Average Drawdown", f"{result.averages['avg_drawdown']:.2%}"),
                ("Average CAGR", f"{result.averages['avg_cagr']:.2%}"),
            ],
            ranking_headers=["Rank", "Ticker", "Return", "Sharpe", "Max DD"],
            ranking_rows=ranking_rows,
        )

    @staticmethod
    def _table_rows_from_df(df: pd.DataFrame) -> list[list[str]]:
        rows: list[list[str]] = []
        for name, row in df.iterrows():
            cells = [str(name)]
            for col in COMPARISON_METRIC_COLUMNS:
                val = row[col]
                if col == "num_trades":
                    cells.append(str(int(val)))
                elif col in ("sharpe_ratio", "sortino_ratio", "profit_factor"):
                    cells.append(f"{float(val):.2f}")
                elif col == "win_rate":
                    cells.append(f"{float(val):.1%}")
                else:
                    cells.append(f"{float(val):.2%}")
            rows.append(cells)
        return rows

    @staticmethod
    def _best_strategy(df: pd.DataFrame, column: str) -> str:
        if column == "max_drawdown":
            idx = df[column].idxmax()
        else:
            idx = df[column].idxmax()
        return str(idx)


def format_strategy_terminal_table(df: pd.DataFrame) -> str:
    """Format a compact terminal table for strategy comparison."""
    headers = ["Strategy", "Return", "Sharpe", "Max DD"]
    lines = [
        "",
        "## Strategy         Return    Sharpe    Max DD",
        "",
    ]
    for name, row in df.iterrows():
        lines.append(
            f"{str(name):16}   {row['total_return']:7.2%}   "
            f"{row['sharpe_ratio']:5.2f}      {row['max_drawdown']:7.2%}"
        )
    lines.append("")
    return "\n".join(lines)


def format_universe_terminal_summary(result: UniverseComparisonResult) -> str:
    """Format universe averages and ranking for terminal output."""
    lines = [
        "",
        f"=== Universe Comparison: {result.strategy_name} ===",
        "",
        f"Average Return:    {result.averages['avg_return']:.2%}",
        f"Average Sharpe:    {result.averages['avg_sharpe']:.2f}",
        f"Average Drawdown:  {result.averages['avg_drawdown']:.2%}",
        "",
        "## Ranking (by Return)",
        "",
        f"{'Rank':<6}{'Ticker':<8}{'Return':>10}{'Sharpe':>10}{'Max DD':>12}",
    ]
    for _, row in result.ranking_df.iterrows():
        lines.append(
            f"{int(row['rank']):<6}{row.name:<8}"
            f"{row['total_return']:>10.2%}"
            f"{row['sharpe_ratio']:>10.2f}"
            f"{row['max_drawdown']:>12.2%}"
        )
    lines.append("")
    return "\n".join(lines)
