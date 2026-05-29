"""Report generation for walk-forward analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from analytics.walkforward_charts import WalkForwardChartGenerator
from config.settings import PROJECT_ROOT, NexiumaSettings, get_settings

if TYPE_CHECKING:
    from research.walkforward_result import WalkForwardResult

TEMPLATE_DIR = PROJECT_ROOT / "reports" / "templates"


class WalkForwardReportGenerator:
    """Persist walk-forward CSV, charts, and HTML summary."""

    def __init__(self, settings: NexiumaSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._charts = WalkForwardChartGenerator()
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def publish(self, result: WalkForwardResult) -> dict[str, Path]:
        out = result.output_dir
        results_csv = out / "walkforward_results.csv"
        params_csv = out / "parameter_history.csv"
        equity_png = out / "equity_curve.png"
        performance_png = out / "performance_chart.png"
        interactive_html = out / "interactive_charts.html"
        summary_html = out / "summary.html"

        result.results_df.to_csv(results_csv, index=False)
        result.parameter_history_df.to_csv(params_csv, index=False)
        self._save_equity_curve_png(result, equity_png)
        self._save_performance_chart_png(result, performance_png)
        self._charts.save_interactive_charts(result, interactive_html)
        summary_html.write_text(
            self._render_summary(result, interactive_html.name),
            encoding="utf-8",
        )

        meta = {
            "type": "walkforward",
            "ticker": result.ticker,
            "strategy": result.strategy_name,
            "run_id": result.run_id,
            "train_years": result.train_years,
            "test_years": result.test_years,
            "robustness": result.robustness.to_dict(),
            "paths": {
                "results": str(results_csv),
                "parameters": str(params_csv),
                "equity_curve": str(equity_png),
                "performance_chart": str(performance_png),
                "interactive": str(interactive_html),
                "summary": str(summary_html),
            },
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Walk-forward report saved: {}", out)
        return {
            "results": results_csv,
            "parameters": params_csv,
            "equity_curve": equity_png,
            "performance_chart": performance_png,
            "interactive": interactive_html,
            "summary": summary_html,
        }

    def _render_summary(
        self,
        result: WalkForwardResult,
        interactive_filename: str,
    ) -> str:
        r = result.robustness
        window_sections: list[dict[str, str]] = []
        for w in result.windows:
            window_sections.append(
                {
                    "title": f"Window {w.window_id}",
                    "train": f"{w.train_start} → {w.train_end}",
                    "test": f"{w.test_start} → {w.test_end}",
                    "params": w.best_params,
                    "train_sharpe": f"{w.train_sharpe:.2f}",
                    "test_sharpe": f"{w.test_sharpe:.2f}",
                    "test_return": f"{w.test_return:.2%}",
                    "test_drawdown": f"{w.test_max_drawdown:.2%}",
                }
            )

        conclusion = self._build_conclusion(result)
        template = self._env.get_template("walkforward_summary.html.j2")
        return template.render(
            title=f"Walk-Forward — {result.ticker} / {result.strategy_name}",
            subtitle=f"{result.start_date} → {result.end_date} · "
            f"Train {result.train_years}y / Test {result.test_years}y",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            num_windows=r.num_windows,
            avg_test_sharpe=f"{r.avg_test_sharpe:.2f}",
            avg_test_return=f"{r.avg_test_return:.2%}",
            worst_drawdown=f"{r.worst_test_drawdown:.2%}",
            param_stability=f"{r.parameter_stability:.0%}",
            profitable_pct=f"{r.profitable_windows_pct:.0%}",
            avg_train_sharpe=f"{r.avg_train_sharpe:.2f}",
            window_sections=window_sections,
            conclusion=conclusion,
            equity_chart="equity_curve.png",
            performance_chart="performance_chart.png",
            interactive_chart=interactive_filename,
        )

    @staticmethod
    def _build_conclusion(result: WalkForwardResult) -> str:
        r = result.robustness
        if r.num_windows == 0:
            return "No walk-forward windows were completed."
        if r.avg_test_sharpe > 0.5 and r.profitable_windows_pct >= 0.6:
            verdict = "Parameters show reasonable out-of-sample robustness."
        elif r.avg_test_sharpe > 0 and r.profitable_windows_pct >= 0.5:
            verdict = "Mixed out-of-sample results — proceed with caution."
        else:
            verdict = "Poor out-of-sample performance — likely overfit to training data."
        degradation = r.avg_train_sharpe - r.avg_test_sharpe
        return (
            f"{verdict} Average test Sharpe ({r.avg_test_sharpe:.2f}) vs train "
            f"({r.avg_train_sharpe:.2f}) implies {degradation:.2f} Sharpe degradation. "
            f"Parameter stability: {r.parameter_stability:.0%}. "
            f"{r.profitable_windows_pct:.0%} of test windows were profitable."
        )

    @staticmethod
    def _save_equity_curve_png(result: WalkForwardResult, path: Path) -> None:
        if result.combined_equity.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 5))
        result.combined_equity.plot(ax=ax, color="#2563eb", linewidth=1.5)
        ax.set_title(f"Out-of-Sample Equity — {result.ticker}")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _save_performance_chart_png(result: WalkForwardResult, path: Path) -> None:
        df = result.results_df
        if df.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = [f"W{int(w)}" for w in df["window_id"]]
        width = 0.35
        idx = range(len(x))
        axes[0].bar([i - width / 2 for i in idx], df["train_sharpe"], width, label="Train", color="#93c5fd")
        axes[0].bar([i + width / 2 for i in idx], df["test_sharpe"], width, label="Test", color="#2563eb")
        axes[0].set_xticks(list(idx))
        axes[0].set_xticklabels(x)
        axes[0].set_title("Sharpe Ratio")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar([i - width / 2 for i in idx], df["train_return"], width, label="Train", color="#fca5a5")
        axes[1].bar([i + width / 2 for i in idx], df["test_return"], width, label="Test", color="#dc2626")
        axes[1].set_xticks(list(idx))
        axes[1].set_xticklabels(x)
        axes[1].set_title("Total Return")
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Train vs Test Performance")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def format_walkforward_terminal(result: WalkForwardResult) -> str:
    """Format walk-forward summary for CLI output."""
    lines = [
        "",
        f"=== Walk-Forward: {result.ticker} / {result.strategy_name} ===",
        f"Windows: {result.robustness.num_windows} "
        f"(train {result.train_years}y / test {result.test_years}y)",
        "",
    ]
    for w in result.windows:
        lines.extend(
            [
                f"Window {w.window_id}",
                f"  Train: {w.train_start} → {w.train_end}  "
                f"Best Params: {w.best_params}  Train Sharpe: {w.train_sharpe:.2f}",
                f"  Test:  {w.test_start} → {w.test_end}  "
                f"Test Sharpe: {w.test_sharpe:.2f}  Return: {w.test_return:.2%}",
                "",
            ]
        )
    r = result.robustness
    lines.extend(
        [
            "Robustness Metrics:",
            f"  Avg Test Sharpe:     {r.avg_test_sharpe:.2f}",
            f"  Avg Test Return:     {r.avg_test_return:.2%}",
            f"  Worst Test Drawdown: {r.worst_test_drawdown:.2%}",
            f"  Parameter Stability: {r.parameter_stability:.0%}",
            f"  Profitable Windows:  {r.profitable_windows_pct:.0%}",
            "",
        ]
    )
    return "\n".join(lines)
