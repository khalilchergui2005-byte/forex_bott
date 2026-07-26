"""Spread Risk Monitor — tracks bid-ask spread health for trade decisions."""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SpreadRiskReport:
    current_spread_pips: float
    spread_percentile: float
    spread_zscore: float
    is_widening: bool
    should_avoid: bool
    cost_ratio: Optional[float]
    reason: str
    history_mean: float
    history_std: float


class SpreadRiskMonitor:

    def __init__(self, spread_threshold_pips=3.0,
                 percentile_threshold=90.0,
                 lookback=1000, widening_bars=10,
                 widening_slope_pct=0.1):
        self.spread_threshold = spread_threshold_pips
        self.pct_threshold = percentile_threshold
        self.lookback = lookback
        self.widening_bars = widening_bars
        self.widening_slope_pct = widening_slope_pct
        self._spreads: list[float] = []
        self._pip_value = 0.0001

    def update(self, bid: float, ask: float,
               pip_value: Optional[float] = None) -> None:
        if pip_value is not None:
            self._pip_value = pip_value
        spread = abs(ask - bid) / self._pip_value
        self._spreads.append(spread)
        if len(self._spreads) > self.lookback:
            self._spreads = self._spreads[-self.lookback:]

    def clear(self) -> None:
        self._spreads.clear()

    def check(self, expected_profit_pips=None):
        if len(self._spreads) < 5:
            return SpreadRiskReport(
                current_spread_pips=self._spreads[-1] if self._spreads else 0.0,
                spread_percentile=50.0, spread_zscore=0.0,
                is_widening=False, should_avoid=True,
                cost_ratio=None, reason="Insufficient data",
                history_mean=0.0, history_std=0.0)
        arr = np.array(self._spreads)
        current = float(arr[-1])
        hist = arr[:-1]
        mu = float(np.mean(hist))
        sd = float(np.std(hist))
        less = float(np.sum(hist < current))
        equal = float(np.sum(hist == current))
        pct = (less + 0.5 * equal) / len(hist) * 100.0
        z = float((current - mu) / sd) if sd > 1e-10 else 0.0
        is_w = False
        if len(arr) >= self.widening_bars:
            seg = arr[-self.widening_bars:]
            x = np.arange(len(seg), dtype=float)
            sx = np.std(x)
            if sx > 1e-10:
                slope = float(np.polyfit(x, seg, 1)[0])
                if slope > 0 and mu > 1e-10:
                    is_w = (slope / mu) > self.widening_slope_pct
        reasons = []
        if current > self.spread_threshold:
            reasons.append(f"spread {current:.2f} > {self.spread_threshold}")
        if pct > self.pct_threshold:
            reasons.append(f"pct {pct:.1f} > {self.pct_threshold}")
        if is_w:
            reasons.append("widening")
        avoid = len(reasons) > 0
        cr = None
        if expected_profit_pips is not None and expected_profit_pips > 0:
            cr = current / expected_profit_pips
        return SpreadRiskReport(
            current_spread_pips=current,
            spread_percentile=pct,
            spread_zscore=z,
            is_widening=is_w,
            should_avoid=avoid,
            cost_ratio=cr,
            reason="; ".join(reasons) if reasons else "Normal",
            history_mean=mu,
            history_std=sd)
