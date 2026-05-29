"""CSV, HTML, and Plotly outputs for strategy optimization."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from analytics.optimization_charts import OptimizationChartGenerator
from config.settings import PROJECT_ROOT, NexiumaSettings, get_settings

if TYPE_CHECKING:
    from research.optimizer import OptimizationResult

TEMPLATE_DIR = PROJECT_ROOT / "reports" / "templates"


class OptimizationReportGenerator:
    """Persist optimization grid-search artifacts."""

    def __init__(self, settings: NexiumaSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._charts = OptimizationChartGenerator()
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def publish(self, result: OptimizationResult) -> dict[str, Path]:
        out = result.output_dir
        csv_path = out / "optimization_results.csv"
        heatmap_path = out / "heatmap.html"
        html_path = out / "index.html"

        result.results_df.to_csv(csv_path)
        self._charts.save_heatmaps(result, heatmap_path)
        html_path.write_text(self._render_html(result, heatmap_path.name), encoding="utf-8")

        best = result.results_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
        meta = {
            "type": "optimization",
            "strategy": result.strategy_name,
            "ticker": result.ticker,
            "run_id": result.run_id,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "best_params": str(best.name),
            "best_sharpe": float(best["sharpe_ratio"]),
            "paths": {
                "csv": str(csv_path),
                "heatmap": str(heatmap_path),
                "html": str(html_path),
            },
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Optimization report saved: {}", out)
        return {"csv": csv_path, "heatmap": heatmap_path, "html": html_path}

    def _render_html(self, result: OptimizationResult, heatmap_filename: str) -> str:
        rows: list[list[str]] = []
        for label, row in result.results_df.iterrows():
            rows.append(
                [
                    str(label),
                    str(int(row["fast_period"])),
                    str(int(row["slow_period"])),
                    f"{row['total_return']:.2%}",
                    f"{row['cagr']:.2%}",
                    f"{row['sharpe_ratio']:.2f}",
                    f"{row['max_drawdown']:.2%}",
                    str(int(row["num_trades"])),
                ]
            )

        best = result.results_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
        template = self._env.get_template("optimization.html.j2")
        return template.render(
            title=f"Optimization — {result.strategy_name} / {result.ticker}",
            subtitle=f"{result.start_date} → {result.end_date}",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            best_params=str(best.name),
            best_sharpe=f"{best['sharpe_ratio']:.2f}",
            best_return=f"{best['total_return']:.2%}",
            headers=[
                "Params",
                "Fast",
                "Slow",
                "Return",
                "CAGR",
                "Sharpe",
                "Max DD",
                "Trades",
            ],
            rows=rows,
            heatmap_path=heatmap_filename,
        )


def format_optimization_terminal(result: OptimizationResult) -> str:
    """Format optimization results for CLI output."""
    lines = [
        "",
        f"=== Optimization: {result.strategy_name} / {result.ticker} ===",
        "",
        f"{'Params':<10}{'Return':>10}{'CAGR':>10}{'Sharpe':>10}{'Max DD':>12}{'Trades':>8}",
    ]
    for label, row in result.results_df.iterrows():
        lines.append(
            f"{str(label):<10}"
            f"{row['total_return']:>10.2%}"
            f"{row['cagr']:>10.2%}"
            f"{row['sharpe_ratio']:>10.2f}"
            f"{row['max_drawdown']:>12.2%}"
            f"{int(row['num_trades']):>8}"
        )
    best = result.results_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            f"Best (Sharpe): {best.name} — Sharpe {best['sharpe_ratio']:.2f}, "
            f"Return {best['total_return']:.2%}",
            "",
        ]
    )
    return "\n".join(lines)
