"""Bias x Trading Type Compatibility."""
from dataclasses import dataclass
from bias_score import BiasReport


@dataclass
class CompatReport:
    score: float
    compatible: bool
    side: str
    reason: str


class BiasTradingCompat:
    TYPES = ("trend_following", "mean_reversion", "breakout")

    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def _score(self, bias, ttype, side):
        d, s = bias.direction, bias.strength
        if ttype == "trend_following":
            return d if side == "long" else -d
        if ttype == "mean_reversion":
            return (-d * s) if side == "long" else (d * s)
        if ttype == "breakout":
            return max(0.0, 1.0 - abs(s - 0.3) / 0.3)
        return 0.0

    def _reason(self, bias, ttype, side, score):
        d, s = bias.direction, bias.strength
        if ttype == "trend_following":
            ok = (d > 0 and side == "long") or (d < 0 and side == "short")
            return f"TF {'with' if ok else 'against'} trend d={d:.2f}"
        if ttype == "mean_reversion":
            return f"MR fade {bias.label} str={s:.2f}"
        return f"BO coiling str={s:.2f}"

    def evaluate(self, bias, trading_type, side="any"):
        if trading_type not in self.TYPES:
            raise ValueError(f"Unknown: {trading_type}")
        if side == "any":
            return self._best_side(bias, trading_type)
        sc = self._score(bias, trading_type, side)
        r = self._reason(bias, trading_type, side, sc)
        ok = sc >= self.threshold
        return CompatReport(score=sc, compatible=ok,
                            side=side if ok else "none", reason=r)

    def _best_side(self, bias, ttype):
        if ttype == "breakout":
            sc = self._score(bias, ttype, "long")
            r = self._reason(bias, ttype, "long", sc)
            ok = sc >= self.threshold
            return CompatReport(score=sc, compatible=ok,
                                side="any" if ok else "none", reason=r)
        sl = self._score(bias, ttype, "long")
        ss = self._score(bias, ttype, "short")
        if sl >= ss:
            bs, bsc = "long", sl
        else:
            bs, bsc = "short", ss
        r = self._reason(bias, ttype, bs, bsc)
        ok = bsc >= self.threshold
        return CompatReport(score=bsc, compatible=ok,
                            side=bs if ok else "none", reason=r)
