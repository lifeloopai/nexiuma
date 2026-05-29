"""Pytest fixtures for Nexiuma."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV for unit tests (no network)."""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
    high = close + rng.uniform(0.5, 2, len(dates))
    low = close - rng.uniform(0.5, 2, len(dates))
    open_ = np.clip(close + rng.normal(0, 0.3, len(dates)), low, high)
    volume = rng.integers(1_000_000, 5_000_000, len(dates))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_equity() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    values = 100_000 * (1 + np.linspace(0, 0.15, len(dates)))
    return pd.Series(values, index=dates, name="equity")
