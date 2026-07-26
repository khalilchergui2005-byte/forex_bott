"""News Risk Detector - avoids trading around high-impact economic events."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class NewsEvent:
    timestamp: datetime
    currency: str
    impact: str
    name: str = ""

    def __post_init__(self):
        self.impact = self.impact.lower().strip()
        if self.impact not in ("high", "medium", "low"):
            raise ValueError(f"Invalid impact: {self.impact!r}")


@dataclass
class NewsRiskReport:
    risk_level: RiskLevel
    events_nearby: list
    should_avoid: bool
    recommended_reduction: float
    minutes_to_next_event: float
    reason: str = ""


_IMPACT_SCORE = {"high": 3.0, "medium": 2.0, "low": 1.0}
_REDUCTION = {RiskLevel.LOW: 0.0, RiskLevel.MEDIUM: 0.5,
              RiskLevel.HIGH: 0.75, RiskLevel.EXTREME: 1.0}


class NewsRiskDetector:

    def __init__(self, calendar_provider=None,
                 avoid_minutes_before=30, avoid_minutes_after=15):
        self.calendar_provider = calendar_provider
        self.avoid_before = timedelta(minutes=avoid_minutes_before)
        self.avoid_after = timedelta(minutes=avoid_minutes_after)
        self._events = []

    def add_event(self, event):
        if isinstance(event, dict):
            event = NewsEvent(**event)
        self._events.append(event)

    def add_events(self, events):
        for e in events:
            self.add_event(e)

    def clear(self):
        self._events.clear()

    @property
    def events(self):
        return list(self._events)

    def check(self, timestamp, currency=None):
        if self.calendar_provider is not None:
            fetched = self.calendar_provider(currency=currency, timestamp=timestamp)
            if fetched:
                self.add_events(fetched)
        nearby = []
        for ev in self._events:
            if currency and ev.currency.upper() != currency.upper():
                continue
            ws = ev.timestamp - self.avoid_before
            we = ev.timestamp + self.avoid_after
            if ws <= timestamp <= we:
                nearby.append(ev)
        if not nearby:
            nxt = self._next_min(timestamp, currency)
            return NewsRiskReport(
                risk_level=RiskLevel.LOW, events_nearby=[],
                should_avoid=False, recommended_reduction=0.0,
                minutes_to_next_event=nxt, reason="No events in window.")
        max_score = 0.0
        hw = (self.avoid_before + self.avoid_after).total_seconds() / 2.0
        for ev in nearby:
            dist = abs((timestamp - ev.timestamp).total_seconds())
            prox = max(0.0, 1.0 - dist / hw)
            score = _IMPACT_SCORE[ev.impact] * (0.85 + 0.15 * prox)
            if score > max_score:
                max_score = score
        if max_score >= 3.0:
            level = RiskLevel.EXTREME
        elif max_score >= 2.5:
            level = RiskLevel.HIGH
        elif max_score >= 1.5:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        nxt = self._next_min(timestamp, currency)
        names = ", ".join(e.name for e in nearby[:5])
        return NewsRiskReport(
            risk_level=level, events_nearby=nearby,
            should_avoid=level in (RiskLevel.HIGH, RiskLevel.EXTREME),
            recommended_reduction=_REDUCTION[level],
            minutes_to_next_event=nxt,
            reason=f"{level.value}: {names}")

    def _next_min(self, timestamp, currency):
        best = float("inf")
        for ev in self._events:
            if currency and ev.currency.upper() != currency.upper():
                continue
            gap = (ev.timestamp - self.avoid_before - timestamp)
            gap = gap.total_seconds() / 60.0
            if 0 < gap < best:
                best = gap
        return best
