"""Market data downloader with caching, validation, and retry logic."""

from __future__ import annotations

import hashlib
import time
from datetime import date
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd
from loguru import logger

from config.settings import DataSettings, NexiumaSettings, get_settings
from data.providers.yfinance_provider import OHLCV_COLUMNS, YFinanceProvider
from data.validator import OHLCVValidator

__all__ = ["MarketDataDownloader", "OHLCV_COLUMNS"]

T = TypeVar("T")

MAX_RETRIES = 3
RETRY_DELAY_SEC = 1.5


def _retry(fn: Callable[[], T], retries: int = MAX_RETRIES) -> T:
    """Execute callable with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = RETRY_DELAY_SEC * (2 ** attempt)
            logger.warning("Attempt {} failed: {}. Retrying in {:.1f}s", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} attempts") from last_exc


class MarketDataDownloader:
    """Download, cache, validate, and clean OHLCV market data."""

    def __init__(
        self,
        settings: NexiumaSettings | None = None,
        provider: YFinanceProvider | None = None,
        validator: OHLCVValidator | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._data_settings: DataSettings = self._settings.data
        self._provider = provider or YFinanceProvider()
        self._validator = validator or OHLCVValidator()
        self._cache_dir = self._data_settings.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_data(
        self,
        ticker: str | None = None,
        start: date | None = None,
        end: date | None = None,
        force_refresh: bool | None = None,
    ) -> pd.DataFrame:
        """Return validated OHLCV data, using parquet cache when available."""
        ticker = (ticker or self._data_settings.ticker).upper()
        start = start or self._data_settings.start_date
        end = end or self._data_settings.end_date
        refresh = force_refresh if force_refresh is not None else self._data_settings.auto_refresh

        cache_path = self._cache_path(ticker, start, end)
        if not refresh and cache_path.exists():
            logger.info("Loading {} from cache: {}", ticker, cache_path)
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            if self._validator.validate(df, ticker, raise_on_error=False):
                return df
            logger.warning("Cached data invalid; re-downloading")

        df = _retry(lambda: self._provider.fetch_ohlcv(ticker, start, end))
        df = self._validator.clean(df)
        self._validator.validate(df, ticker, raise_on_error=True)
        self._write_cache(df, cache_path)
        logger.info("Cached {} rows for {}", len(df), ticker)
        return df

    def _cache_path(self, ticker: str, start: date, end: date) -> Path:
        key = f"{ticker}_{start.isoformat()}_{end.isoformat()}"
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return self._cache_dir / f"{ticker}_{digest}.parquet"

    def _write_cache(self, df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(path)

    def clear_cache(self, ticker: str | None = None) -> int:
        pattern = f"{ticker.upper()}_*.parquet" if ticker else "*.parquet"
        removed = 0
        for path in self._cache_dir.glob(pattern):
            path.unlink()
            removed += 1
        logger.info("Removed {} cache file(s)", removed)
        return removed
