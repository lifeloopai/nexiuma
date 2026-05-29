"""Tests for strategy and universe comparison workflows."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from analytics.comparison_charts import ComparisonChartGenerator
from data.downloader import MarketDataDownloader
from analytics.performance import PerformanceAnalyzer, PerformanceReport
from config.settings import DataSettings, NexiumaSettings, RiskSettings
from core.engine import BacktestEngine, BacktestRunResult
from reports.comparison_report import (
    ComparisonReportGenerator,
    format_strategy_terminal_table,
    format_universe_terminal_summary,
)
from research.strategy_comparison import StrategyComparator
from research.universe_comparison import UniverseComparator


def _make_report(equity: pd.Series, trade_pnls: list[float]) -> PerformanceReport:
    analyzer = PerformanceAnalyzer()
    return analyzer.build_report(
        equity_curve=equity,
        trade_pnls=trade_pnls,
        initial_capital=100_000.0,
    )


def _make_run_result(
    strategy: str,
    ticker: str,
    equity: pd.Series,
    ohlcv: pd.DataFrame,
) -> BacktestRunResult:
    report = _make_report(equity, [100.0, -50.0])
    return BacktestRunResult(
        strategy_name=strategy,
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2020-12-31",
        ohlcv=ohlcv,
        equity_curve=equity,
        performance=report,
        run_id="test_run",
    )


@pytest.fixture
def comparison_settings(tmp_path: Path) -> NexiumaSettings:
    settings = NexiumaSettings(
        strategy="moving_average",
        data=DataSettings(
            ticker="TEST",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 10, 1),
        ),
        risk=RiskSettings(initial_capital=100_000.0),
        comparisons_dir=tmp_path / "comparisons",
    )
    settings.ensure_directories()
    return settings


def test_comparison_metrics_frame(sample_equity: pd.Series) -> None:
    eq1 = sample_equity
    eq2 = sample_equity * 0.98
    reports = {
        "alpha": _make_report(eq1, [200.0, -100.0]),
        "beta": _make_report(eq2, [50.0, -25.0]),
    }
    df = PerformanceAnalyzer.comparison_metrics_frame(reports)
    assert list(df.columns) == [
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "volatility",
        "win_rate",
        "profit_factor",
        "num_trades",
    ]
    assert df.loc["alpha", "total_return"] > df.loc["beta", "total_return"]


def test_strategy_comparator_runs_all_strategies(
    comparison_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    engine = BacktestEngine(comparison_settings)
    call_count = 0

    def fake_run(
        strategy_name: str | None = None,
        ticker: str | None = None,
        ohlcv: pd.DataFrame | None = None,
        save_charts: bool = True,
    ) -> BacktestRunResult:
        nonlocal call_count
        call_count += 1
        name = strategy_name or "moving_average"
        equity = sample_equity * (1.0 + 0.01 * call_count)
        data = sample_ohlcv if ohlcv is None else ohlcv
        return _make_run_result(name, ticker or "TEST", equity, data)

    engine.run = fake_run  # type: ignore[method-assign]
    comparator = StrategyComparator(comparison_settings, engine=engine)

    with patch.object(
        MarketDataDownloader,
        "get_data",
        return_value=sample_ohlcv,
    ):
        result = comparator.run(
            ticker="TEST",
            strategies=["moving_average", "rsi"],
            save_individual_charts=False,
        )

    assert set(result.results.keys()) == {"moving_average", "rsi"}
    assert len(result.comparison_df) == 2
    assert result.output_dir.exists()


def test_universe_comparator(
    comparison_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    engine = BacktestEngine(comparison_settings)

    def fake_run(
        strategy_name: str | None = None,
        ticker: str | None = None,
        ohlcv: pd.DataFrame | None = None,
        save_charts: bool = True,
    ) -> BacktestRunResult:
        mult = {"AAA": 1.0, "BBB": 1.1}.get(ticker or "AAA", 1.0)
        equity = sample_equity * mult
        return _make_run_result(
            strategy_name or "moving_average",
            ticker or "AAA",
            equity,
            sample_ohlcv,
        )

    engine.run = fake_run  # type: ignore[method-assign]
    comparator = UniverseComparator(comparison_settings, engine=engine)
    result = comparator.run(
        strategy_name="moving_average",
        tickers=["AAA", "BBB"],
        save_individual_charts=False,
    )

    assert result.averages["avg_return"] != 0
    assert len(result.ranking_df) == 2
    assert result.ranking_df.iloc[0]["rank"] == 1


def test_comparison_report_generator(
    comparison_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    from research.strategy_comparison import StrategyComparisonResult
    from research.universe_comparison import UniverseComparisonResult

    run_a = _make_run_result("moving_average", "TEST", sample_equity, sample_ohlcv)
    run_b = _make_run_result(
        "rsi",
        "TEST",
        sample_equity * 0.95,
        sample_ohlcv,
    )
    comp_df = PerformanceAnalyzer.comparison_metrics_frame(
        {"moving_average": run_a.performance, "rsi": run_b.performance}
    )
    out_dir = comparison_settings.comparisons_dir / "test_strategies"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy_result = StrategyComparisonResult(
        ticker="TEST",
        start_date="2020-01-01",
        end_date="2020-12-31",
        results={"moving_average": run_a, "rsi": run_b},
        comparison_df=comp_df,
        output_dir=out_dir,
        run_id="test",
    )

    gen = ComparisonReportGenerator(comparison_settings)
    paths = gen.publish_strategy_comparison(strategy_result)
    assert paths["csv"].exists()
    assert paths["chart"].exists()
    assert paths["html"].exists()
    assert "moving_average" in format_strategy_terminal_table(comp_df)

    uni_df = comp_df.copy()
    uni_df.index = ["AAA", "BBB"]
    uni_out = comparison_settings.comparisons_dir / "test_universe"
    uni_out.mkdir(parents=True, exist_ok=True)
    ranking = uni_df.sort_values("total_return", ascending=False).copy()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    universe_result = UniverseComparisonResult(
        strategy_name="moving_average",
        tickers=("AAA", "BBB"),
        results={"AAA": run_a, "BBB": run_b},
        metrics_df=uni_df,
        averages={
            "avg_return": float(uni_df["total_return"].mean()),
            "avg_sharpe": float(uni_df["sharpe_ratio"].mean()),
            "avg_drawdown": float(uni_df["max_drawdown"].mean()),
            "avg_cagr": float(uni_df["cagr"].mean()),
            "avg_sortino": float(uni_df["sortino_ratio"].mean()),
            "avg_volatility": float(uni_df["volatility"].mean()),
        },
        ranking_df=ranking,
        output_dir=uni_out,
        run_id="test",
    )

    uni_paths = gen.publish_universe_comparison(universe_result)
    assert uni_paths["metrics"].exists()
    assert uni_paths["heatmaps"].exists()
    summary = format_universe_terminal_summary(universe_result)
    assert "Average Return" in summary


def test_comparison_charts_write_html(
    comparison_settings: NexiumaSettings,
    sample_ohlcv: pd.DataFrame,
    sample_equity: pd.Series,
) -> None:
    from research.strategy_comparison import StrategyComparisonResult
    from research.universe_comparison import UniverseComparisonResult

    run = _make_run_result("moving_average", "TEST", sample_equity, sample_ohlcv)
    comp_df = PerformanceAnalyzer.comparison_metrics_frame(
        {"moving_average": run.performance}
    )
    out_dir = comparison_settings.comparisons_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_result = StrategyComparisonResult(
        ticker="TEST",
        start_date="2020-01-01",
        end_date="2020-12-31",
        results={"moving_average": run},
        comparison_df=comp_df,
        output_dir=out_dir,
    )
    charts = ComparisonChartGenerator()
    chart_path = charts.save_strategy_comparison(
        strategy_result, out_dir / "strategy.html"
    )
    assert chart_path.stat().st_size > 0

    universe_result = UniverseComparisonResult(
        strategy_name="moving_average",
        tickers=("TEST",),
        results={"TEST": run},
        metrics_df=comp_df,
        averages={"avg_return": 0.1, "avg_sharpe": 0.5, "avg_drawdown": -0.1,
                  "avg_cagr": 0.1, "avg_sortino": 0.5, "avg_volatility": 0.2},
        ranking_df=comp_df.reset_index(),
        output_dir=out_dir,
    )
    heatmap_path = charts.save_universe_heatmaps(
        universe_result, out_dir / "universe.html"
    )
    assert heatmap_path.exists()
