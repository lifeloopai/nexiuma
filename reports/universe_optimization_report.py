"""Reports for cross-asset parameter optimization."""

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
    from research.universe_optimizer import UniverseOptimizationResult

TEMPLATE_DIR = PROJECT_ROOT / "reports" / "templates"


class UniverseOptimizationReportGenerator:
    """Persist universe-wide optimization artifacts."""

    def __init__(self, settings: NexiumaSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._charts = OptimizationChartGenerator()
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def publish(self, result: UniverseOptimizationResult) -> dict[str, Path]:
        out = result.output_dir
        full_csv = out / "full_results.csv"
        averages_csv = out / "averages.csv"
        ranking_csv = out / "ranking.csv"
        heatmap_path = out / "heatmaps.html"
        html_path = out / "index.html"

        result.full_results_df.to_csv(full_csv, index=False)
        result.averages_df.to_csv(averages_csv)
        result.ranking_df.to_csv(ranking_csv, index=False)
        self._charts.save_universe_heatmaps(result, heatmap_path)
        html_path.write_text(self._render_html(result, heatmap_path.name), encoding="utf-8")

        best = result.ranking_df.iloc[0]
        meta = {
            "type": "universe_optimization",
            "strategy": result.strategy_name,
            "tickers": list(result.tickers),
            "run_id": result.run_id,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "robust_params": result.robust_params,
            "best_avg_sharpe": float(best["avg_sharpe"]),
            "paths": {
                "full_results": str(full_csv),
                "averages": str(averages_csv),
                "ranking": str(ranking_csv),
                "heatmaps": str(heatmap_path),
                "html": str(html_path),
            },
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Universe optimization report saved: {}", out)
        return {
            "full_results": full_csv,
            "averages": averages_csv,
            "ranking": ranking_csv,
            "heatmaps": heatmap_path,
            "html": html_path,
        }

    def _render_html(self, result: UniverseOptimizationResult, heatmap_filename: str) -> str:
        ranking_rows: list[list[str]] = []
        for _, row in result.ranking_df.iterrows():
            ranking_rows.append(
                [
                    str(int(row["rank"])),
                    str(row["params"]),
                    f"{row['avg_return']:.2%}",
                    f"{row['avg_sharpe']:.2f}",
                    f"{row['avg_drawdown']:.2%}",
                    f"{row['sharpe_std']:.2f}",
                    f"{row['positive_sharpe_pct']:.0%}",
                ]
            )

        detail_rows: list[list[str]] = []
        for _, row in result.full_results_df.iterrows():
            detail_rows.append(
                [
                    str(row["ticker"]),
                    str(row["params"]),
                    f"{row['total_return']:.2%}",
                    f"{row['cagr']:.2%}",
                    f"{row['sharpe_ratio']:.2f}",
                    f"{row['max_drawdown']:.2%}",
                    str(int(row["num_trades"])),
                ]
            )

        best = result.ranking_df.iloc[0]
        template = self._env.get_template("universe_optimization.html.j2")
        return template.render(
            title=f"Universe Optimization — {result.strategy_name}",
            subtitle=f"{', '.join(result.tickers)} · {result.start_date} → {result.end_date}",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            robust_params=result.robust_params,
            robust_sharpe=f"{best['avg_sharpe']:.2f}",
            robust_return=f"{best['avg_return']:.2%}",
            robust_consistency=f"{best['sharpe_std']:.2f} σ",
            ranking_headers=[
                "Rank",
                "Params",
                "Avg Return",
                "Avg Sharpe",
                "Avg DD",
                "Sharpe σ",
                "% Pos Sharpe",
            ],
            ranking_rows=ranking_rows,
            detail_headers=[
                "Ticker",
                "Params",
                "Return",
                "CAGR",
                "Sharpe",
                "Max DD",
                "Trades",
            ],
            detail_rows=detail_rows,
            heatmap_path=heatmap_filename,
        )


def format_universe_optimization_terminal(result: UniverseOptimizationResult) -> str:
    """Format universe optimization summary for CLI."""
    lines = [
        "",
        f"=== Universe Optimization: {result.strategy_name} ===",
        f"Tickers: {', '.join(result.tickers)}",
        "",
        "## Params   Avg Return   Avg Sharpe   Avg DD",
        "",
    ]
    for params_label, row in result.averages_df.iterrows():
        lines.append(
            f"{str(params_label):<10}"
            f"{row['avg_return']:>10.1%}"
            f"{row['avg_sharpe']:>13.2f}"
            f"{row['avg_drawdown']:>10.1%}"
        )

    best = result.ranking_df.iloc[0]
    lines.extend(
        [
            "",
            "Most robust parameter set (avg Sharpe + consistency):",
            f"  {result.robust_params} — Avg Sharpe {best['avg_sharpe']:.2f}, "
            f"Avg Return {best['avg_return']:.2%}, Sharpe σ {best['sharpe_std']:.2f}",
            "",
        ]
    )
    return "\n".join(lines)
