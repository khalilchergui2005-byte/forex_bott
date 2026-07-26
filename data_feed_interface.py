"""data_feed_interface.py"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd


class DataFeedInterface(ABC):
    """Contract every data source must fulfill."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data and return a DataFrame."""

    @abstractmethod
    def get_symbol(self) -> str:
        """Return instrument symbol."""

    @abstractmethod
    def get_timeframe(self) -> str:
        """Return timeframe string."""

    @abstractmethod
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return (first, last)."""

    @abstractmethod
    def validate(self) -> bool:
        """Run quality checks."""


class CsvDataFeed(DataFeedInterface):
    """Concrete feed backed by CSV data_loader."""

    def __init__(self, symbol: str = "EURUSD", timeframe: str = "H1"):
        self._symbol = symbol.upper()
        self._timeframe = timeframe.upper()

    def load(self) -> pd.DataFrame:
        from data_loader import load_hourly
        self._df = load_hourly()
        return self._df

    def get_symbol(self) -> str:
        return self._symbol

    def get_timeframe(self) -> str:
        return self._timeframe

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if not hasattr(self, "_df") or self._df is None:
            self.load()
        first = pd.Timestamp(self._df.index[0])
        last = pd.Timestamp(self._df.index[-1])
        return (first, last)

    def validate(self) -> bool:
        if not hasattr(self, "_df") or self._df is None:
            self.load()
        df = self._df
        cols = {"open", "high", "low", "close"}
        if not cols.issubset(set(df.columns)):
            return False
        if df.index.name != "Datetime":
            return False
        if len(df) == 0:
            return False
        if (df["high"] < df["low"]).any():
            return False
        return True
