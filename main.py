#!/usr/bin/env python3
"""Nexiuma CLI — run backtests, download data, and generate reports."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from loguru import logger

# Ensure project root is on path when run as script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.engine import BacktestEngine
from config.settings import configure_logging, load_settings
from data.downloader import MarketDataDownloader
from reports.comparison_report import (
    ComparisonReportGenerator,
    format_strategy_terminal_table,
    format_universe_terminal_summary,
)
from reports.generator import ReportGenerator
from research.optimizer import StrategyOptimizer
from research.strategy_comparison import StrategyComparator
from research.universe_comparison import UniverseComparator
from research.universe_optimizer import UniverseOptimizer
from reports.optimization_report import (
    OptimizationReportGenerator,
    format_optimization_terminal,
)
from reports.universe_optimization_report import (
    UniverseOptimizationReportGenerator,
    format_universe_optimization_terminal,
)
from research.walkforward import WalkForwardEngine
from research.walkforward_report import (
    WalkForwardReportGenerator,
    format_walkforward_terminal,
)
from research.walkforward_universe import WalkForwardUniverseEngine
from research.walkforward_universe_report import (
    WalkForwardUniverseReportGenerator,
    format_walkforward_universe_terminal,
)
from strategies.parameters import ParameterValidationError, parse_ma_periods
from strategies.registry import list_strategies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nexiuma — Quantitative Research & Systematic Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared arguments
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ticker", type=str, help="Stock ticker symbol")
        p.add_argument("--strategy", type=str, help="Strategy name (moving_average, rsi, momentum)")
        p.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
        p.add_argument("--end", type=str, help="End date YYYY-MM-DD")
        p.add_argument("--capital", type=float, help="Initial capital")
        p.add_argument("--env", type=Path, help="Path to .env file")

    bt_parser = sub.add_parser("backtest", help="Run a strategy backtest")
    add_common(bt_parser)
    bt_parser.add_argument("--fast-period", type=int, help="Fast SMA period (moving_average)")
    bt_parser.add_argument("--slow-period", type=int, help="Slow SMA period (moving_average)")
    bt_parser.add_argument("--no-report", action="store_true", help="Skip HTML report")
    bt_parser.add_argument("--refresh", action="store_true", help="Force refresh market data")

    opt_parser = sub.add_parser(
        "optimize",
        help="Grid-search strategy parameters (moving_average)",
    )
    add_common(opt_parser)
    opt_parser.add_argument(
        "--grid",
        type=str,
        help="Comma-separated fast/slow pairs (e.g. 10/30,20/50)",
    )

    opt_uni_parser = sub.add_parser(
        "optimize-universe",
        help="Optimize parameters across a ticker universe",
    )
    add_common(opt_uni_parser)
    opt_uni_parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated tickers (default: mega-cap basket)",
    )
    opt_uni_parser.add_argument(
        "--grid",
        type=str,
        help="Comma-separated fast/slow pairs (e.g. 10/30,20/50)",
    )

    wf_parser = sub.add_parser(
        "walkforward",
        help="Walk-forward optimization and out-of-sample testing",
    )
    add_common(wf_parser)
    wf_parser.add_argument(
        "--train-years",
        type=int,
        default=3,
        help="Training window length in years (default: 3)",
    )
    wf_parser.add_argument(
        "--test-years",
        type=int,
        default=1,
        help="Test window length in years (default: 1)",
    )
    wf_parser.add_argument(
        "--grid",
        type=str,
        help="Comma-separated fast/slow pairs (e.g. 10/30,20/50)",
    )

    wf_uni_parser = sub.add_parser(
        "walkforward-universe",
        help="Walk-forward analysis across a ticker universe",
    )
    add_common(wf_uni_parser)
    wf_uni_parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated tickers (default: mega-cap basket)",
    )
    wf_uni_parser.add_argument(
        "--train-years",
        type=int,
        default=3,
        help="Training window length in years (default: 3)",
    )
    wf_uni_parser.add_argument(
        "--test-years",
        type=int,
        default=1,
        help="Test window length in years (default: 1)",
    )
    wf_uni_parser.add_argument(
        "--grid",
        type=str,
        help="Comma-separated fast/slow pairs (e.g. 10/30,20/50)",
    )

    dl_parser = sub.add_parser("download", help="Download and cache market data")
    add_common(dl_parser)
    dl_parser.add_argument("--refresh", action="store_true", help="Force re-download")

    sub.add_parser("strategies", help="List available strategies")

    list_parser = sub.add_parser("reports", help="List generated reports")

    compare_parser = sub.add_parser(
        "compare",
        help="Compare multiple strategies on the same ticker",
    )
    add_common(compare_parser)
    compare_parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated strategy names (default: all registered)",
    )
    compare_parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip saving individual backtest charts",
    )

    universe_parser = sub.add_parser(
        "compare-universe",
        help="Run one strategy across a ticker universe",
    )
    add_common(universe_parser)
    universe_parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated tickers (default: mega-cap tech basket)",
    )
    universe_parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip saving individual backtest charts",
    )

    return parser


def _cli_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if getattr(args, "ticker", None):
        overrides["ticker"] = args.ticker
    if getattr(args, "strategy", None):
        overrides["strategy"] = args.strategy
    if getattr(args, "start", None):
        overrides["start_date"] = date.fromisoformat(args.start)
    if getattr(args, "end", None):
        overrides["end_date"] = date.fromisoformat(args.end)
    if getattr(args, "capital", None):
        overrides["initial_capital"] = args.capital
    if getattr(args, "refresh", False):
        overrides["auto_refresh"] = True
    if getattr(args, "fast_period", None) is not None:
        overrides["fast_period"] = args.fast_period
    if getattr(args, "slow_period", None) is not None:
        overrides["slow_period"] = args.slow_period
    return overrides


def _validate_ma_cli(strategy: str, fast: int | None, slow: int | None) -> None:
    """Validate MA periods when provided via CLI."""
    if fast is None and slow is None:
        return
    if strategy not in ("moving_average", "ma"):
        raise SystemExit(
            "--fast-period and --slow-period apply only to moving_average strategy"
        )
    try:
        parse_ma_periods(fast, slow)
    except ParameterValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _format_params_line(result) -> str | None:
    params = result.strategy_params
    if not params or result.strategy_name != "moving_average":
        return None
    fast = params.get("fast_period")
    slow = params.get("slow_period")
    if fast is not None and slow is not None:
        return f"MA Periods:  {fast}/{slow} (fast/slow)"
    return None


def cmd_backtest(args: argparse.Namespace) -> int:
    if getattr(args, "fast_period", None) is not None or getattr(args, "slow_period", None) is not None:
        strategy_hint = getattr(args, "strategy", None) or "moving_average"
        _validate_ma_cli(strategy_hint, args.fast_period, args.slow_period)

    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)

    try:
        parse_ma_periods(
            settings.strategy_params.fast_period,
            settings.strategy_params.slow_period,
        )
    except ParameterValidationError as exc:
        raise SystemExit(str(exc)) from exc

    if args.refresh:
        MarketDataDownloader(settings).get_data(force_refresh=True)

    runner = BacktestEngine(settings)
    result = runner.run(
        strategy_name=settings.strategy,
        ticker=settings.data.ticker,
    )

    if not args.no_report:
        report_path = ReportGenerator(settings).generate(result)
        logger.info("Open report: file://{}", report_path.resolve())

    m = result.performance.metrics
    print("\n=== Backtest Results ===")
    print(f"Strategy:  {result.strategy_name}")
    params_line = _format_params_line(result)
    if params_line:
        print(params_line)
    print(f"Ticker:    {result.ticker}")
    print(f"Period:    {result.start_date} → {result.end_date}")
    print(f"Return:    {m.total_return:.2%}")
    print(f"Sharpe:    {m.sharpe_ratio:.2f}")
    print(f"Max DD:    {m.max_drawdown:.2%}")
    print(f"Trades:    {m.num_trades}")
    return 0


def cmd_walkforward_universe(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)

    strategy = settings.strategy
    if strategy not in ("moving_average", "ma"):
        print(
            f"Walk-forward universe not supported for '{strategy}'. "
            "Use moving_average."
        )
        return 1

    tickers = _parse_csv_list(getattr(args, "tickers", None))
    grid = StrategyOptimizer.parse_grid_arg(getattr(args, "grid", None))
    engine = WalkForwardUniverseEngine(settings)
    result = engine.run(
        strategy_name=strategy,
        tickers=tickers,
        train_years=args.train_years,
        test_years=args.test_years,
        grid=grid,
    )
    paths = WalkForwardUniverseReportGenerator(settings).publish(result)
    print(format_walkforward_universe_terminal(result))
    print(f"Assets:      {paths['asset_results']}")
    print(f"Summary:     {paths['summary']}")
    print(f"Parameters:  {paths['parameter_frequency']}")
    print(f"Heatmaps:    {paths['sharpe_heatmap']}")
    print(f"Dashboard:   {paths['robustness_chart']}")
    print(f"Report:      {paths['index']}")
    logger.info("Open report: file://{}", paths["index"].resolve())
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)

    strategy = settings.strategy
    if strategy not in ("moving_average", "ma"):
        print(
            f"Walk-forward not supported for '{strategy}'. Use moving_average."
        )
        return 1

    grid = StrategyOptimizer.parse_grid_arg(getattr(args, "grid", None))
    engine = WalkForwardEngine(settings)
    result = engine.run(
        ticker=settings.data.ticker,
        strategy_name=strategy,
        train_years=args.train_years,
        test_years=args.test_years,
        grid=grid,
    )
    paths = WalkForwardReportGenerator(settings).publish(result)
    print(format_walkforward_terminal(result))
    print(f"Results:     {paths['results']}")
    print(f"Parameters:  {paths['parameters']}")
    print(f"Equity:      {paths['equity_curve']}")
    print(f"Performance: {paths['performance_chart']}")
    print(f"Summary:     {paths['summary']}")
    logger.info("Open report: file://{}", paths["summary"].resolve())
    return 0


def cmd_optimize_universe(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)

    strategy = settings.strategy
    if strategy not in ("moving_average", "ma"):
        print(
            f"Universe optimization not supported for '{strategy}'. "
            "Use moving_average."
        )
        return 1

    tickers = _parse_csv_list(getattr(args, "tickers", None))
    grid = StrategyOptimizer.parse_grid_arg(getattr(args, "grid", None))
    optimizer = UniverseOptimizer(settings)
    result = optimizer.optimize_moving_average_universe(tickers=tickers, grid=grid)
    paths = UniverseOptimizationReportGenerator(settings).publish(result)
    print(format_universe_optimization_terminal(result))
    print(f"Full CSV:    {paths['full_results']}")
    print(f"Averages:    {paths['averages']}")
    print(f"Ranking:     {paths['ranking']}")
    print(f"Heatmaps:    {paths['heatmaps']}")
    print(f"HTML:        {paths['html']}")
    logger.info("Open report: file://{}", paths["html"].resolve())
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)

    strategy = settings.strategy
    if strategy not in ("moving_average", "ma"):
        print(f"Optimization not supported for strategy '{strategy}'. Use moving_average.")
        return 1

    grid = StrategyOptimizer.parse_grid_arg(getattr(args, "grid", None))
    optimizer = StrategyOptimizer(settings)
    result = optimizer.optimize_moving_average(
        ticker=settings.data.ticker,
        grid=grid,
    )
    paths = OptimizationReportGenerator(settings).publish(result)
    print(format_optimization_terminal(result))
    print(f"CSV:      {paths['csv']}")
    print(f"Heatmap:  {paths['heatmap']}")
    print(f"HTML:     {paths['html']}")
    logger.info("Open report: file://{}", paths["html"].resolve())
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)
    downloader = MarketDataDownloader(settings)
    df = downloader.get_data(force_refresh=args.refresh)
    print(f"Downloaded {len(df)} bars for {settings.data.ticker}")
    print(df.tail())
    return 0


def cmd_strategies(_: argparse.Namespace) -> int:
    for s in list_strategies():
        print(f"  {s['name']:20} — {s['description']}")
    return 0


def _parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_compare(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)

    strategies = _parse_csv_list(getattr(args, "strategies", None))
    comparator = StrategyComparator(settings)
    result = comparator.run(
        ticker=settings.data.ticker,
        strategies=strategies,
        save_individual_charts=not args.no_charts,
    )

    paths = ComparisonReportGenerator(settings).publish_strategy_comparison(result)
    print(format_strategy_terminal_table(result.comparison_df))
    print("Full metrics:")
    print(result.comparison_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nCSV:   {paths['csv']}")
    print(f"Chart: {paths['chart']}")
    print(f"HTML:  {paths['html']}")
    logger.info("Open report: file://{}", paths["html"].resolve())
    return 0


def cmd_compare_universe(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=args.env, overrides=_cli_overrides(args))
    configure_logging(settings)

    tickers = _parse_csv_list(getattr(args, "tickers", None))
    comparator = UniverseComparator(settings)
    result = comparator.run(
        strategy_name=settings.strategy,
        tickers=tickers,
        save_individual_charts=not args.no_charts,
    )

    paths = ComparisonReportGenerator(settings).publish_universe_comparison(result)
    print(format_universe_terminal_summary(result))
    print(f"Metrics:  {paths['metrics']}")
    print(f"Ranking:  {paths['ranking']}")
    print(f"Heatmaps: {paths['heatmaps']}")
    print(f"HTML:     {paths['html']}")
    logger.info("Open report: file://{}", paths["html"].resolve())
    return 0


def cmd_reports(args: argparse.Namespace) -> int:
    settings = load_settings(env_file=getattr(args, "env", None))
    reports = ReportGenerator(settings).list_reports()
    if not reports:
        print("No reports found.")
        return 0
    for r in reports:
        print(f"{r.get('run_id', '?')}: {r.get('strategy')} / {r.get('ticker')} → {r.get('path', '')}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "backtest": cmd_backtest,
        "download": cmd_download,
        "strategies": cmd_strategies,
        "reports": cmd_reports,
        "compare": cmd_compare,
        "compare-universe": cmd_compare_universe,
        "optimize": cmd_optimize,
        "optimize-universe": cmd_optimize_universe,
        "walkforward": cmd_walkforward,
        "walkforward-universe": cmd_walkforward_universe,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
