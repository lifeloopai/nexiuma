"""Professional HTML report generation from backtest results."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from config.settings import NexiumaSettings, PROJECT_ROOT, get_settings
from core.engine import BacktestRunResult


TEMPLATE_DIR = PROJECT_ROOT / "reports" / "templates"


class ReportGenerator:
    """Generate HTML reports with metrics, risk analysis, and charts."""

    def __init__(self, settings: NexiumaSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._reports_dir = self._settings.reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate(self, result: BacktestRunResult) -> Path:
        """Create report directory with HTML and copied charts."""
        report_dir = self._reports_dir / f"{result.ticker}_{result.strategy_name}_{result.run_id}"
        report_dir.mkdir(parents=True, exist_ok=True)

        charts = self._copy_charts(result, report_dir)
        m = result.performance.metrics
        r = result.performance.risk
        benchmark = result.performance.benchmark_return

        performance_cards = [
            ("Total Return", f"{m.total_return:.2%}"),
            ("CAGR", f"{m.cagr:.2%}"),
            ("Sharpe", f"{m.sharpe_ratio:.2f}"),
            ("Sortino", f"{m.sortino_ratio:.2f}"),
            ("Max Drawdown", f"{m.max_drawdown:.2%}"),
            ("Volatility", f"{m.volatility:.2%}"),
            ("Win Rate", f"{m.win_rate:.1%}"),
            ("Profit Factor", f"{m.profit_factor:.2f}"),
            ("Trades", str(m.num_trades)),
        ]

        risk_rows = [
            ("VaR (95%)", f"{r.var_95:.2%}"),
            ("CVaR (95%)", f"{r.cvar_95:.2%}"),
            ("Calmar Ratio", f"{r.calmar_ratio:.2f}"),
            ("Ulcer Index", f"{r.ulcer_index:.4f}"),
            ("Avg Drawdown", f"{r.avg_drawdown:.2%}"),
        ]

        strategy_params_label = self._format_strategy_params(result)

        template = self._env.get_template("report.html.j2")
        html = template.render(
            strategy=result.strategy_name,
            ticker=result.ticker,
            start_date=result.start_date,
            end_date=result.end_date,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            strategy_params=strategy_params_label,
            performance_cards=performance_cards,
            risk_rows=risk_rows,
            charts=charts,
            benchmark_return=f"{benchmark:.2%}" if benchmark is not None else "N/A",
            alpha=f"{m.total_return - benchmark:.2%}" if benchmark is not None else "N/A",
        )

        report_path = report_dir / "index.html"
        report_path.write_text(html, encoding="utf-8")

        meta = result.to_metadata()
        meta["report_path"] = str(report_path)
        (report_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Report saved: {}", report_path)
        return report_path

    @staticmethod
    def _format_strategy_params(result: BacktestRunResult) -> str | None:
        params = result.strategy_params
        if not params:
            return None
        if result.strategy_name == "moving_average":
            fast = params.get("fast_period")
            slow = params.get("slow_period")
            if fast is not None and slow is not None:
                return f"Fast SMA {fast} / Slow SMA {slow}"
        return ", ".join(f"{k}={v}" for k, v in params.items())

    def _copy_charts(self, result: BacktestRunResult, report_dir: Path) -> list[dict[str, str]]:
        charts: list[dict[str, str]] = []
        chart_files = [
            ("equity_curve.png", "Equity Curve"),
            ("drawdown.png", "Drawdown"),
            ("price_chart.png", "Price & Trades"),
            ("rolling_returns.png", "Rolling Returns"),
            ("return_distribution.png", "Return Distribution"),
        ]
        if result.output_dir:
            for filename, title in chart_files:
                src = result.output_dir / filename
                if src.exists():
                    dest = report_dir / filename
                    shutil.copy2(src, dest)
                    charts.append({"title": title, "path": filename})
        return charts

    def list_reports(self) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for meta_path in self._reports_dir.glob("*/metadata.json"):
            try:
                data = json.loads(meta_path.read_text())
                data["path"] = str(meta_path.parent / "index.html")
                reports.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(reports, key=lambda r: r.get("run_id", ""), reverse=True)
