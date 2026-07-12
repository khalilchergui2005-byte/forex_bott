"""strategy_base.py — Unified strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class Signal:
    """Standardized trading signal."""
    direction: int
    strength: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


class StrategyBase(ABC):
    """Every strategy must inherit from this."""

    SIGNAL_COL = "signal"

    @abstractmethod
    def generate(self, df, params=None):
        """Compute indicators, write signals into df[SIGNAL_COL]."""
        ...

    @classmethod
    def info(cls):
        return {"name": cls.__name__, "type": "unknown", "description": cls.__doc__ or ""}
