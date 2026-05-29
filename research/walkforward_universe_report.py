"""Report generation for walk-forward universe analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from analytics.walkforward_universe_charts import WalkForwardUniverseChartGenerator
from config.settings import PROJECT_ROOT, NexiumaSettings, get_settings

if TYPE_CHECKING:
    from research.walkforward_universe_result import WalkForwardUniverseResult

TEMPLATE_DIR = PROJECT_ROOT / "reports" / "templates"


class WalkForwardUniverseReportGenerator:
    """Persist CSV exports, Plotly charts, and HTML research report."""

    def __init__(self, settings: NexiumaSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._charts = WalkForwardUniverseChartGenerator()
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def publish(self, result: WalkForwardUniverseResult) -> dict[str, Path]:
        """Write all walk-forward universe artifacts."""
        out = result.output_dir
        asset_csv = out / "asset_results.csv"
        summary_csv = out / "summary.csv"
        param_csv = out / "parameter_frequency.csv"
        window_csv = out / "window_results.csv"
        sharpe_heatmap = out / "sharpe_heatmap.html"
        degradation_heatmap = out / "degradation_heatmap.html"
        robustness_chart = out / "robustness_chart.html"
        index_html = out / "index.html"

        result.asset_results_df.to_csv(asset_csv)
        result.summary_df.to_csv(summary_csv)
        result.parameter_frequency_df.to_csv(param_csv, index=False)
        result.window_results_df.to_csv(window_csv, index=False)

        self._charts.save_sharpe_heatmap(result, sharpe_heatmap)
        self._charts.save_degradation_heatmap(result, degradation_heatmap)
        self._charts.save_robustness_dashboard(result, robustness_chart)

        index_html.write_text(
            self._render_html(
                result,
                sharpe_heatmap.name,
                degradation_heatmap.name,
                robustness_chart.name,
            ),
            encoding="utf-8",
        )

        meta = result.to_metadata()
        meta["paths"] = {
            "asset_results": str(asset_csv),
            "summary": str(summary_csv),
            "parameter_frequency": str(param_csv),
            "window_results": str(window_csv),
            "sharpe_heatmap": str(sharpe_heatmap),
            "degradation_heatmap": str(degradation_heatmap),
            "robustness_chart": str(robustness_chart),
            "index": str(index_html),
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Walk-forward universe report saved: {}", out)

        return {
            "asset_results": asset_csv,
            "summary": summary_csv,
            "parameter_frequency": param_csv,
            "window_results": window_csv,
            "sharpe_heatmap": sharpe_heatmap,
            "degradation_heatmap": degradation_heatmap,
            "robustness_chart": robustness_chart,
            "index": index_html,
        }

    def _render_html(
        self,
        result: WalkForwardUniverseResult,
        sharpe_file: str,
        degradation_file: str,
        robustness_file: str,
    ) -> str:
        sm = result.summary_metrics
        rs = result.robustness_score

        asset_rows: list[list[str]] = []
        for ticker, row in result.asset_results_df.iterrows():
            asset_rows.append(
                [
                    str(ticker),
                    f"{row['avg_train_sharpe']:.2f}",
                    f"{row['avg_test_sharpe']:.2f}",
                    f"{row['sharpe_degradation']:.2f}",
                    f"{row['avg_return']:.2%}",
                    f"{row['avg_cagr']:.2%}",
                    f"{row['avg_drawdown']:.2%}",
                    f"{row['win_rate']:.0%}",
                ]
            )

        window_rows: list[list[str]] = []
        for _, row in result.window_results_df.iterrows():
            window_rows.append(
                [
                    str(row["ticker"]),
                    str(int(row["window_id"])),
                    str(row["best_params"]),
                    f"{row['train_sharpe']:.2f}",
                    f"{row['test_sharpe']:.2f}",
                    f"{row['test_return']:.2%}",
                    f"{row['test_max_drawdown']:.2%}",
                    str(int(row["test_num_trades"])),
                ]
            )

        param_rows = [
            [str(r["params"]), str(int(r["count"])), f"{r['frequency_pct']:.1%}"]
            for _, r in result.parameter_frequency_df.iterrows()
        ]

        conclusions = self._research_conclusions(result)
        template = self._env.get_template("walkforward_universe.html.j2")
        return template.render(
            title=f"Walk-Forward Universe — {result.strategy_name}",
            subtitle=(
                f"{', '.join(result.tickers)} · "
                f"Train {result.train_years}y / Test {result.test_years}y"
            ),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            executive_summary=result.executive_summary,
            robustness_score=f"{rs.score:.0f}",
            robustness_grade=rs.grade,
            key_findings=[
                f"Average OOS Sharpe: {sm.avg_test_sharpe:.2f} (train: {sm.avg_train_sharpe:.2f})",
                f"Sharpe degradation: {sm.sharpe_degradation:.2f}",
                f"{sm.pct_positive_test_sharpe:.0%} of windows with positive OOS Sharpe",
                f"Most selected params: {sm.most_frequent_params}",
                f"Best asset: {sm.best_asset} · Worst: {sm.worst_asset}",
            ],
            stats=[
                ("Avg Test Return", f"{sm.avg_return:.2%}"),
                ("Avg CAGR", f"{sm.avg_cagr:.2%}"),
                ("Avg Drawdown", f"{sm.avg_drawdown:.2%}"),
                ("Win Rate", f"{sm.win_rate:.0%}"),
                ("Param Stability", f"{sm.parameter_stability:.0%}"),
                ("Assets", str(sm.num_assets)),
                ("Total Windows", str(sm.num_windows_total)),
            ],
            asset_headers=[
                "Ticker", "Train Sharpe", "Test Sharpe", "Degradation",
                "Return", "CAGR", "Drawdown", "Win Rate",
            ],
            asset_rows=asset_rows,
            window_headers=[
                "Ticker", "Window", "Params", "Train Sharpe", "Test Sharpe",
                "Return", "Max DD", "Trades",
            ],
            window_rows=window_rows,
            param_headers=["Params", "Count", "Frequency"],
            param_rows=param_rows,
            score_breakdown=[
                ("Test Sharpe", f"{rs.test_sharpe_component:.1f}"),
                ("Low Degradation", f"{rs.degradation_component:.1f}"),
                ("Win Rate", f"{rs.win_rate_component:.1f}"),
                ("Stability", f"{rs.stability_component:.1f}"),
            ],
            sharpe_heatmap=sharpe_file,
            degradation_heatmap=degradation_file,
            robustness_chart=robustness_file,
            conclusions=conclusions,
        )

    @staticmethod
    def _research_conclusions(result: WalkForwardUniverseResult) -> str:
        sm = result.summary_metrics
        rs = result.robustness_score
        if rs.score >= 70:
            verdict = (
                "The strategy demonstrates cross-asset robustness with acceptable "
                "out-of-sample performance degradation."
            )
        elif rs.score >= 50:
            verdict = (
                "Results are mixed across the universe. Parameter selection varies "
                "significantly and OOS performance is inconsistent."
            )
        else:
            verdict = (
                "The strategy shows signs of overfitting. Out-of-sample Sharpe "
                "degrades materially from in-sample optimization."
            )
        return (
            f"{verdict} With a robustness score of {rs.score:.0f}/100 ({rs.grade}), "
            f"average OOS Sharpe of {sm.avg_test_sharpe:.2f}, and "
            f"{sm.pct_positive_test_sharpe:.0%} positive-Sharpe windows, "
            f"further research should focus on parameter regularization or "
            f"alternative signal definitions before live deployment."
        )


def format_walkforward_universe_terminal(result: WalkForwardUniverseResult) -> str:
    """Format universe walk-forward summary for CLI."""
    sm = result.summary_metrics
    rs = result.robustness_score
    lines = [
        "",
        f"=== Walk-Forward Universe: {result.strategy_name} ===",
        f"Assets: {', '.join(result.tickers)}",
        f"Windows: {sm.num_windows_total} total ({sm.num_assets} assets)",
        "",
        f"Robustness Score: {rs.score:.0f}/100 ({rs.grade})",
        "",
        f"{'Ticker':<8}{'Train':>8}{'Test':>8}{'Degrad':>8}{'Return':>10}{'Win%':>8}",
    ]
    for ticker, row in result.asset_results_df.iterrows():
        lines.append(
            f"{str(ticker):<8}"
            f"{row['avg_train_sharpe']:>8.2f}"
            f"{row['avg_test_sharpe']:>8.2f}"
            f"{row['sharpe_degradation']:>8.2f}"
            f"{row['avg_return']:>10.2%}"
            f"{row['win_rate']:>8.0%}"
        )
    lines.extend(
        [
            "",
            "Universe Summary:",
            f"  Avg Test Sharpe:       {sm.avg_test_sharpe:.2f}",
            f"  Sharpe Degradation:    {sm.sharpe_degradation:.2f}",
            f"  Positive OOS Sharpe:   {sm.pct_positive_test_sharpe:.0%}",
            f"  Most Frequent Params:  {sm.most_frequent_params}",
            f"  Best Asset:            {sm.best_asset}",
            f"  Worst Asset:           {sm.worst_asset}",
            "",
            result.executive_summary,
            "",
        ]
    )
    return "\n".join(lines)
