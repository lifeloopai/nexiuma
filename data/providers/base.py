"""Abstract data provider — implements core DataProvider protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

# Nexiuma core protocol (structural typing)
# from core.interfaces import DataProvider  # noqa: ERA001


class BaseDataProvider(ABC):
    """Contract for historical market data providers.

    Compatible with ``core.interfaces.DataProvider`` for future Alpaca/IBKR adapters.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a symbol."""

    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """Return True if the ticker is valid for this provider."""
