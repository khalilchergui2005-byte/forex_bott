"""Bias Score - quantitative directional bias from normalized components."""
from dataclasses import dataclass
import numpy as np

@dataclass
class BiasReport:
    direction: float
    strength: float
    label: str
    components: dict

class BiasScorer:

    def __init__(self, ma_fast=20, ma_slow=50, momentum_period=20,
                 structure_lookback=20, atr_period=14, weights=None):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.mom_p = momentum_period
        self.str_lb = structure_lookback
        self.atr_p = atr_period
        self.w = weights or {"ma_position": 0.3, "ma_cross": 0.2,
                             "momentum": 0.3, "structure": 0.2}
        assert abs(sum(self.w.values()) - 1.0) < 1e-9

    @staticmethod
    def _sma(arr, p):
        out = np.full(len(arr), np.nan)
        if len(arr) < p:
            return out
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        out[p-1:] = (cs[p:] - cs[:-p]) / p
        return out

    @staticmethod
    def _atr(high, low, close, p):
        n = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]),
                        abs(low[i]-close[i-1]))
        a = np.zeros(n)
        if n <= p:
            return a
        a[p] = np.mean(tr[1:p+1])
        for i in range(p+1, n):
            a[i] = (a[i-1]*(p-1) + tr[i]) / p
        return a

    def _comp_ma_pos(self, c, ma_s, atr):
        if len(c) < self.ma_slow or atr[-1] < 1e-10:
            return 0.0
        return float(np.tanh((c[-1]-ma_s[-1]) / atr[-1] / 3.0))

    def _comp_ma_cross(self, ma_f, ma_s, atr):
        if len(ma_f) < 2 or atr[-1] < 1e-10:
            return 0.0
        return float(np.tanh((ma_f[-1]-ma_s[-1]) / atr[-1] / 3.0))

    def _comp_momentum(self, c, atr):
        if len(c) < self.mom_p + 1 or atr[-1] < 1e-10:
            return 0.0
        return float(np.tanh((c[-1]-c[-(self.mom_p+1)]) / atr[-1] / 5.0))

    def _comp_structure(self, h, l):
        lb = self.str_lb
        if len(h) < lb + 1:
            return 0.0
        up = float(np.sum(h[-lb:] > h[-(lb+1):-1]))
        dn = float(np.sum(l[-lb:] < l[-(lb+1):-1]))
        t = up + dn
        return (up - dn) / t if t > 1e-10 else 0.0

    def score(self, data):
        h = data["high"].values.astype(float)
        l = data["low"].values.astype(float)
        c = data["close"].values.astype(float)
        need = max(self.ma_slow, self.atr_p,
                   self.mom_p + 1, self.str_lb + 1)
        if len(c) < need:
            z = {k: 0.0 for k in self.w}
            return BiasReport(0.0, 0.0, "NEUTRAL", z)
        atr = self._atr(h, l, c, self.atr_p)
        mf = self._sma(c, self.ma_fast)
        ms = self._sma(c, self.ma_slow)
        comps = {
            "ma_position": self._comp_ma_pos(c, ms, atr),
            "ma_cross": self._comp_ma_cross(mf, ms, atr),
            "momentum": self._comp_momentum(c, atr),
            "structure": self._comp_structure(h, l),
        }
        d = sum(comps[k] * self.w[k] for k in self.w)
        s = abs(d)
        lbl = "NEUTRAL" if s < 0.15 else ("BULLISH" if d > 0 else "BEARISH")
        return BiasReport(float(d), float(s), lbl, comps)
