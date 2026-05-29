"""Tests for analytics metrics."""

from __future__ import annotations

import pandas as pd

from analytics.metrics import compute_metrics


def test_compute_metrics_basic(sample_equity: pd.Series) -> None:
    trade_pnls = [500.0, -200.0, 300.0, 100.0]
    metrics = compute_metrics(
        sample_equity,
        trade_pnls,
        initial_capital=100_000.0,
    )
    assert metrics.total_return > 0
    assert metrics.cagr == metrics.annualized_return
    assert metrics.num_trades == 4
    assert 0 <= metrics.win_rate <= 1
    assert metrics.sharpe_ratio != 0 or metrics.volatility == 0


def test_compute_metrics_empty_trades(sample_equity: pd.Series) -> None:
    metrics = compute_metrics(sample_equity, [], initial_capital=100_000.0)
    assert metrics.num_trades == 0
    assert metrics.win_rate == 0.0
