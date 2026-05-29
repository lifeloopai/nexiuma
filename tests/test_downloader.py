"""Tests for market data downloader and validator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config.settings import DataSettings, NexiumaSettings, RiskSettings
from data.downloader import MarketDataDownloader
from data.providers.yfinance_provider import OHLCV_COLUMNS
from data.validator import DataValidationError, OHLCVValidator


@pytest.fixture
def downloader(tmp_path: Path) -> MarketDataDownloader:
    settings = NexiumaSettings(
        data=DataSettings(cache_dir=tmp_path / "cache"),
        risk=RiskSettings(),
    )
    settings.ensure_directories()
    mock_provider = MagicMock()
    return MarketDataDownloader(settings=settings, provider=mock_provider)


def test_validator_rejects_empty(sample_ohlcv: pd.DataFrame) -> None:
    validator = OHLCVValidator()
    with pytest.raises(DataValidationError):
        validator.validate(sample_ohlcv.iloc[:0], "TEST")


def test_validator_rejects_bad_ohlc(sample_ohlcv: pd.DataFrame) -> None:
    bad = sample_ohlcv.copy()
    bad.loc[bad.index[0], "high"] = 1.0
    bad.loc[bad.index[0], "low"] = 100.0
    validator = OHLCVValidator()
    with pytest.raises(DataValidationError):
        validator.validate(bad, "TEST")


def test_get_data_uses_cache(
    sample_ohlcv: pd.DataFrame, downloader: MarketDataDownloader
) -> None:
    downloader._provider.fetch_ohlcv.return_value = sample_ohlcv
    df1 = downloader.get_data(
        ticker="TEST",
        start=sample_ohlcv.index[0].date(),
        end=sample_ohlcv.index[-1].date(),
    )
    df2 = downloader.get_data(
        ticker="TEST",
        start=sample_ohlcv.index[0].date(),
        end=sample_ohlcv.index[-1].date(),
    )
    assert len(df1) == len(df2)
    downloader._provider.fetch_ohlcv.assert_called_once()
    assert list(df1.columns) == list(OHLCV_COLUMNS)
