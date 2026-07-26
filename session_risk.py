"""Session Risk Monitor - forex session quality with currency-specific scoring."""
from dataclasses import dataclass

SESSIONS = {
    "SYDNEY": (21, 6), "TOKYO": (0, 9),
    "LONDON": (7, 16), "NEW_YORK": (12, 21),
}
SQUALITY = {
    "trend_following": {"SYDNEY":0.3,"TOKYO":0.4,"LONDON":0.9,
        "NEW_YORK":0.8,"LONDON_NY":1.0,"SYDNEY_TOKYO":0.5},
    "mean_reversion": {"SYDNEY":0.7,"TOKYO":0.8,"LONDON":0.5,
        "NEW_YORK":0.4,"LONDON_NY":0.3,"SYDNEY_TOKYO":0.7},
    "breakout": {"SYDNEY":0.3,"TOKYO":0.5,"LONDON":0.8,
        "NEW_YORK":0.7,"LONDON_NY":1.0,"SYDNEY_TOKYO":0.4},
}
CPREF = {
    "EUR":{"SYDNEY":0.2,"TOKYO":0.3,"LONDON":1.0,"NEW_YORK":0.8},
    "GBP":{"SYDNEY":0.2,"TOKYO":0.2,"LONDON":1.0,"NEW_YORK":0.7},
    "USD":{"SYDNEY":0.3,"TOKYO":0.5,"LONDON":0.9,"NEW_YORK":1.0},
    "JPY":{"SYDNEY":0.4,"TOKYO":1.0,"LONDON":0.5,"NEW_YORK":0.8},
    "AUD":{"SYDNEY":1.0,"TOKYO":0.8,"LONDON":0.5,"NEW_YORK":0.4},
    "NZD":{"SYDNEY":1.0,"TOKYO":0.7,"LONDON":0.4,"NEW_YORK":0.3},
    "CHF":{"SYDNEY":0.2,"TOKYO":0.3,"LONDON":0.9,"NEW_YORK":0.8},
    "CAD":{"SYDNEY":0.2,"TOKYO":0.3,"LONDON":0.7,"NEW_YORK":1.0},
}
DWEIGHTS = {0: 0.8, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.7}

@dataclass
class SessionRiskReport:
    active_sessions: list
    overlaps: list
    quality_score: float
    should_avoid: bool
    recommended_reduction: float
    reason: str
    session_scores: dict

class SessionRiskMonitor:

    def __init__(self, strategy_type="trend_following",
                 quality_threshold=0.4,
                 edge_minutes=30, edge_factor=0.6):
        if strategy_type not in SQUALITY:
            raise ValueError(f"Unknown: {strategy_type}")
        self.stype = strategy_type
        self.qthr = quality_threshold
        self.edge_min = edge_minutes
        self.edge_f = edge_factor

    @staticmethod
    def _in(sess, h):
        s, e = SESSIONS[sess]
        return (s <= h < e) if s < e else (h >= s or h < e)

    def _active(self, h):
        return [s for s in SESSIONS if self._in(s, h)]

    @staticmethod
    def _overlaps(active):
        o = []
        if "LONDON" in active and "NEW_YORK" in active:
            o.append("LONDON_NY")
        if "SYDNEY" in active and "TOKYO" in active:
            o.append("SYDNEY_TOKYO")
        return o

    def _edge(self, ts, name):
        s, e = SESSIONS[name]
        edge = self.edge_min / 60.0
        h = ts.hour + ts.minute / 60.0
        if s < e:
            ds, de = h - s, e - h
        else:
            ds = (h - s) if h >= s else (24 - s + h)
            de = (e - h) if h < e else (24 - h + e)
        return self.edge_f if (ds < edge or de < edge) else 1.0

    def check(self, timestamp, currency=None):
        if timestamp.weekday() >= 5:
            return SessionRiskReport(
                [], [], 0.0, True, 1.0, "Weekend", {})
        if timestamp.weekday() == 4 and timestamp.hour >= 20:
            return SessionRiskReport(
                [], [], 0.0, True, 1.0, "Friday close", {})
        h = timestamp.hour + timestamp.minute / 60.0
        active = self._active(h)
        overlaps = self._overlaps(active)
        if not active:
            return SessionRiskReport(
                [], [], 0.0, True, 1.0, "No session", {})
        sq = SQUALITY[self.stype]
        dw = DWEIGHTS.get(timestamp.weekday(), 1.0)
        if timestamp.weekday() == 4 and timestamp.hour >= 16:
            dw *= 0.3
        scores = {}
        for sess in active:
            sc = sq.get(sess, 0.5) * dw * self._edge(timestamp, sess)
            if currency:
                sc *= CPREF.get(currency, {}).get(sess, 0.5)
            scores[sess] = float(sc)
        for ov in overlaps:
            sc = sq.get(ov, 0.5) * dw
            if currency:
                base = ov.split("_")[0]
                sc *= CPREF.get(currency, {}).get(base, 0.5)
            scores[ov] = float(sc)
        best = max(scores.values())
        red = max(0.0, 1.0 - best)
        avoid = best < self.qthr
        reason = "Normal"
        if avoid:
            reason = f"Low quality ({best:.2f}<{self.qthr})"
        return SessionRiskReport(
            active_sessions=active, overlaps=overlaps,
            quality_score=best, should_avoid=avoid,
            recommended_reduction=red, reason=reason,
            session_scores=scores)
