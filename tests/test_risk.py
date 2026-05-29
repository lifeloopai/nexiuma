"""Tests for risk analytics."""

from __future__ import annotations

import pandas as pd

from analytics.risk import RiskAnalyzer


def test_risk_analyzer(sample_equity: pd.Series) -> None:
    analyzer = RiskAnalyzer()
    metrics = analyzer.analyze(sample_equity, annualized_return=0.12)
    assert isinstance(metrics.var_95, float)
    assert metrics.ulcer_index >= 0
