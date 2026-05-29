"""Quantitative tearsheet generation for backtest results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config.settings import PROJECT_ROOT

if TYPE_CHECKING:
    from core.engine import BacktestRunResult

TEARSHEET_DIR = PROJECT_ROOT / "reports" / "templates"


class TearsheetGenerator:
    """Render HTML tearsheet summarizing a backtest run."""

    def __init__(self, template_dir: Path | None = None) -> None:
        self._template_dir = template_dir or TEARSHEET_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate(self, result: BacktestRunResult, output_path: Path) -> Path:
        """Write tearsheet HTML to disk."""
        template = self._env.get_template("tearsheet.html.j2")
        m = result.performance.metrics
        r = result.performance.risk
        benchmark = result.performance.benchmark_return

        html = template.render(
            strategy=result.strategy_name,
            ticker=result.ticker,
            start_date=result.start_date,
            end_date=result.end_date,
            metrics={
                "total_return": f"{m.total_return:.2%}",
                "cagr": f"{m.cagr:.2%}",
                "sharpe": f"{m.sharpe_ratio:.2f}",
                "sortino": f"{m.sortino_ratio:.2f}",
                "calmar": f"{r.calmar_ratio:.2f}",
                "max_drawdown": f"{m.max_drawdown:.2%}",
                "volatility": f"{m.volatility:.2%}",
                "win_rate": f"{m.win_rate:.1%}",
                "profit_factor": f"{m.profit_factor:.2f}",
                "trades": m.num_trades,
            },
            benchmark=f"{benchmark:.2%}" if benchmark is not None else "N/A",
            alpha=(
                f"{m.total_return - benchmark:.2%}"
                if benchmark is not None
                else "N/A"
            ),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path
