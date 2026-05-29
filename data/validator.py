"""OHLCV data validation and cleaning."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from data.providers.yfinance_provider import OHLCV_COLUMNS


class DataValidationError(ValueError):
    """Raised when market data fails validation."""


class OHLCVValidator:
    """Validate and clean OHLCV market data."""

    REQUIRED_COLUMNS = OHLCV_COLUMNS

    def validate(self, df: pd.DataFrame, ticker: str, *, raise_on_error: bool = True) -> bool:
        """Run full validation suite on OHLCV data."""
        try:
            self._check_non_empty(df, ticker)
            self._check_columns(df, ticker)
            self._check_price_integrity(df, ticker)
            self._check_volume(df, ticker)
            self._check_index(df, ticker)
            return True
        except DataValidationError as exc:
            logger.error("Validation failed for {}: {}", ticker, exc)
            if raise_on_error:
                raise
            return False

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill minor gaps and drop invalid rows."""
        out = df.copy()
        out = out[~out.index.duplicated(keep="last")]
        out = out.sort_index()
        out = out.ffill(limit=3)
        out = out.dropna(how="any")
        return out

    def _check_non_empty(self, df: pd.DataFrame, ticker: str) -> None:
        if df.empty:
            raise DataValidationError(f"{ticker}: empty dataset")

    def _check_columns(self, df: pd.DataFrame, ticker: str) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataValidationError(f"{ticker}: missing columns {missing}")

    def _check_price_integrity(self, df: pd.DataFrame, ticker: str) -> None:
        if (df["high"] < df["low"]).any():
            raise DataValidationError(f"{ticker}: high < low")
        if (df["close"] > df["high"]).any() or (df["close"] < df["low"]).any():
            raise DataValidationError(f"{ticker}: close outside high/low")
        if (df["open"] > df["high"]).any() or (df["open"] < df["low"]).any():
            raise DataValidationError(f"{ticker}: open outside high/low")

    def _check_volume(self, df: pd.DataFrame, ticker: str) -> None:
        if (df["volume"] < 0).any():
            raise DataValidationError(f"{ticker}: negative volume")

    def _check_index(self, df: pd.DataFrame, ticker: str) -> None:
        if not df.index.is_monotonic_increasing:
            raise DataValidationError(f"{ticker}: non-monotonic index")
