"""Yahoo Finance data provider implementation."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf
from loguru import logger

from data.providers.base import BaseDataProvider

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class YFinanceProvider(BaseDataProvider):
    """Download OHLCV data via yfinance."""

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch and normalize OHLCV from Yahoo Finance."""
        logger.info("Fetching {} from {} to {} ({})", ticker, start, end, interval)
        try:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed for {}: {}", ticker, exc)
            raise RuntimeError(f"Failed to download data for {ticker}") from exc

        if raw is None or raw.empty:
            raise ValueError(f"No data returned for ticker {ticker}")

        df = self._normalize(raw)
        if df.empty:
            raise ValueError(f"Empty dataset after normalization for {ticker}")
        return df

    def validate_ticker(self, ticker: str) -> bool:
        """Check ticker exists via yfinance info endpoint."""
        try:
            info = yf.Ticker(ticker).history(period="5d")
            return info is not None and not info.empty
        except Exception:
            return False

    @staticmethod
    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output to standard OHLCV schema."""
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        rename_map = {
            "adj_close": "close",
        }
        df = df.rename(columns=rename_map)

        required = list(OHLCV_COLUMNS)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns after normalization: {missing}")

        df = df[list(required)].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.dropna(how="any")
