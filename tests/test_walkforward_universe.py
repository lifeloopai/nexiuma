"""Tests for walk-forward universe analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from config.settings import DataSettings, NexiumaSettings, RiskSettings
from research.walkforward_result import RobustnessMetrics, WalkForwardResult, WindowResult
from research.walkforward_universe import WalkForwardUniverseEngine
from research.walkforward_universe_report import (
    WalkForwardUniverseReportGenerator,
    format_walkforward_universe_terminal,
)
from research.walkforward_universe_result import (
    UniverseRobustnessScore,
    UniverseSummaryMetrics,
    WalkForwardUniverseResult,
)


def _make_wf_result(ticker: str, test_sharpe: float, train_sharpe: float) -> WalkForwardResult:
    windows = [
        WindowResult(
            window_id=1,
            train_start="2020-01-01",
            train_end="2022-12-31",
            test_start="2023-01-01",
            test_end="2023-12-31",
            best_params="10/50",
            fast_period=10,
            slow_period=50,
            train_sharpe=train_sharpe,
            train_return=0.5,
            test_sharpe=test_sharpe,
            test_return=test_sharpe * 0.3,
            test_cagr=test_sharpe * 0.2,
            test_sortino=test_sharpe,
            test_max_drawdown=-0.15,
            test_volatility=0.2,
            test_num_trades=10,
        ),
        WindowResult(
            window_id=2,
            train_start="2021-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
            best_params="20/50",
            fast_period=20,
            slow_period=50,
            train_sharpe=train_sharpe + 0.1,
            train_return=0.4,
            test_sharpe=test_sharpe - 0.1,
            test_return=test_sharpe * 0.2,
            test_cagr=test_sharpe * 0.15,
            test_sortino=test_sharpe - 0.1,
            test_max_drawdown=-0.2,
            test_volatility=0.22,
            test_num_trades=8,
        ),
    ]
    results_df = pd.DataFrame([w.to_dict() for w in windows])
    return WalkForwardResult(
        ticker=ticker,
        strategy_name="moving_average",
        train_years=3,
        test_years=1,
        start_date="2020-01-01",
        end_date="2024-12-31",
        windows=windows,
        results_df=results_df,
        parameter_history_df=results_df[
            ["window_id", "best_params", "train_sharpe", "test_sharpe"]
        ],
        combined_equity=pd.Series([100_000, 110_000], index=pd.date_range("2023", periods=2)),
        robustness=RobustnessMetrics(
            avg_test_sharpe=test_sharpe,
            avg_test_return=test_sharpe * 0.25,
            worst_test_drawdown=-0.2,
            parameter_stability=0.5,
            profitable_windows_pct=0.5,
            num_windows=2,
            avg_train_sharpe=train_sharpe,
        ),
        output_dir=Path("/tmp"),
    )


@pytest.fixture
def wf_uni_settings(tmp_path: Path) -> NexiumaSettings:
    settings = NexiumaSettings(
        strategy="moving_average",
        data=DataSettings(
            ticker="TEST",
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        ),
        risk=RiskSettings(initial_capital=100_000.0),
        walkforward_universe_dir=tmp_path / "walkforward_universe",
    )
    settings.ensure_directories()
    return settings


def test_parameter_frequency() -> None:
    window_df = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "window_id": [1, 2, 1, 2],
            "best_params": ["10/50", "10/50", "20/50", "10/50"],
            "train_sharpe": [0.8, 0.9, 0.7, 0.6],
            "test_sharpe": [0.4, 0.3, 0.2, 0.5],
            "test_return": [0.1, 0.2, 0.05, 0.15],
            "test_cagr": [0.08, 0.1, 0.04, 0.12],
            "test_max_drawdown": [-0.1, -0.15, -0.2, -0.12],
            "sharpe_degradation": [0.4, 0.6, 0.5, 0.1],
        }
    )
    freq = WalkForwardUniverseEngine._compute_parameter_frequency(window_df)
    assert freq.iloc[0]["params"] == "10/50"
    assert int(freq.iloc[0]["count"]) == 3
    assert freq.iloc[0]["frequency_pct"] == pytest.approx(0.75)


def test_robustness_score() -> None:
    summary = UniverseSummaryMetrics(
        avg_train_sharpe=0.9,
        avg_test_sharpe=0.6,
        sharpe_degradation=0.3,
        avg_return=0.15,
        avg_cagr=0.12,
        avg_drawdown=-0.15,
        win_rate=0.7,
        parameter_stability=0.8,
        most_frequent_params="10/50",
        best_asset="AAPL",
        worst_asset="META",
        pct_positive_test_sharpe=0.75,
        pct_positive_return=0.7,
        num_assets=6,
        num_windows_total=12,
    )
    window_df = pd.DataFrame({"train_sharpe": [0.9] * 12, "test_sharpe": [0.6] * 12})
    score = WalkForwardUniverseEngine._compute_robustness_score(summary, window_df)
    assert 0 <= score.score <= 100
    assert score.grade in ("Exceptional", "Strong", "Moderate", "Weak", "Poor")
    assert UniverseRobustnessScore.grade_from_score(95) == "Exceptional"
    assert UniverseRobustnessScore.grade_from_score(55) == "Moderate"


def test_universe_engine_aggregation(wf_uni_settings: NexiumaSettings) -> None:
    wf_results = {
        "AAA": _make_wf_result("AAA", test_sharpe=0.5, train_sharpe=0.9),
        "BBB": _make_wf_result("BBB", test_sharpe=0.2, train_sharpe=0.8),
    }

    def fake_run(**kwargs):
        ticker = kwargs.get("ticker") or "AAA"
        return wf_results[ticker]

    engine = WalkForwardUniverseEngine(wf_uni_settings)
    with patch.object(engine._wf_engine, "run", side_effect=fake_run):
        result = engine.run(
            tickers=["AAA", "BBB"],
            train_years=3,
            test_years=1,
        )

    assert len(result.asset_results_df) == 2
    assert result.summary_metrics.best_asset == "AAA"
    assert result.summary_metrics.worst_asset == "BBB"
    assert result.robustness_score.score > 0
    assert "AAA" in result.executive_summary


def test_universe_report_generation(wf_uni_settings: NexiumaSettings) -> None:
    window_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "window_id": [1, 1],
            "best_params": ["10/50", "20/50"],
            "train_sharpe": [0.9, 0.8],
            "test_sharpe": [0.4, 0.2],
            "test_return": [0.15, 0.05],
            "test_cagr": [0.12, 0.04],
            "test_max_drawdown": [-0.1, -0.2],
            "test_num_trades": [10, 8],
            "sharpe_degradation": [0.5, 0.6],
        }
    )
    asset_df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "num_windows": 1,
                "avg_train_sharpe": 0.9,
                "avg_test_sharpe": 0.4,
                "sharpe_degradation": 0.5,
                "avg_return": 0.15,
                "avg_cagr": 0.12,
                "avg_drawdown": -0.1,
                "win_rate": 1.0,
                "parameter_stability": 1.0,
                "pct_positive_test_sharpe": 1.0,
                "pct_positive_return": 1.0,
            },
            {
                "ticker": "BBB",
                "num_windows": 1,
                "avg_train_sharpe": 0.8,
                "avg_test_sharpe": 0.2,
                "sharpe_degradation": 0.6,
                "avg_return": 0.05,
                "avg_cagr": 0.04,
                "avg_drawdown": -0.2,
                "win_rate": 1.0,
                "parameter_stability": 1.0,
                "pct_positive_test_sharpe": 1.0,
                "pct_positive_return": 1.0,
            },
        ]
    ).set_index("ticker")

    summary = WalkForwardUniverseEngine._compute_summary_metrics(
        asset_df, window_df, ("AAA", "BBB")
    )
    param_freq = WalkForwardUniverseEngine._compute_parameter_frequency(window_df)
    score = WalkForwardUniverseEngine._compute_robustness_score(summary, window_df)
    out_dir = wf_uni_settings.walkforward_universe_dir / "test_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = WalkForwardUniverseResult(
        strategy_name="moving_average",
        tickers=("AAA", "BBB"),
        train_years=3,
        test_years=1,
        asset_results_df=asset_df,
        window_results_df=window_df,
        summary_df=pd.DataFrame([summary.to_dict()]).T,
        parameter_frequency_df=param_freq,
        summary_metrics=summary,
        robustness_score=score,
        per_asset_results={},
        output_dir=out_dir,
        executive_summary="Test summary.",
    )

    paths = WalkForwardUniverseReportGenerator(wf_uni_settings).publish(result)
    assert paths["asset_results"].exists()
    assert paths["summary"].exists()
    assert paths["parameter_frequency"].exists()
    assert paths["sharpe_heatmap"].exists()
    assert paths["index"].exists()

    terminal = format_walkforward_universe_terminal(result)
    assert "Robustness Score" in terminal
    assert "AAA" in terminal
