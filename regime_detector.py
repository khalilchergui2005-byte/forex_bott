"""Regime Detector."""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import pandas as pd
class Regime(IntEnum):
    TREND_UP = 1
    TREND_DOWN = 2
    RANGE = 3
    VOLATILE = 4
    QUIET = 5
RN = {1:"TREND_UP", 2:"TREND_DOWN", 3:"RANGE", 4:"VOLATILE", 5:"QUIET"}
@dataclass
class RegimeState:
    regime: Regime
    confidence: float
    trend_strength: float
    trend_direction: float
    volatility_score: float
    adx: float
    atr: float
    atr_percentile: float
    di_plus: float
    di_minus: float
class RegimeDetector:
    def __init__(self, adx_period=14, atr_period=14, ma_fast=20, ma_slow=50, vol_lookback=100):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.vol_lookback = vol_lookback
    @staticmethod
    def _compute_adx(high, low, close, period):
        n = len(close)
        tr = np.zeros(n)
        p_dm = np.zeros(n)
        m_dm = np.zeros(n)
        for i in range(1, n):
            hd = high[i] - high[i-1]
            ld = low[i-1] - low[i]
            tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            p_dm[i] = hd if (hd > ld and hd > 0) else 0.0
            m_dm[i] = ld if (ld > hd and ld > 0) else 0.0
        atr = np.zeros(n)
        sp = np.zeros(n)
        sm = np.zeros(n)
        if n <= period:
            return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        sp[period] = np.mean(p_dm[1:period+1])
        sm[period] = np.mean(m_dm[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1]*(period-1)+tr[i])/period
            sp[i] = (sp[i-1]*(period-1)+p_dm[i])/period
            sm[i] = (sm[i-1]*(period-1)+m_dm[i])/period
        dp = np.divide(100.0*sp, atr, out=np.zeros_like(atr), where=atr>0)
        dm = np.divide(100.0*sm, atr, out=np.zeros_like(atr), where=atr>0)
        dx = np.divide(100.0*np.abs(dp-dm), dp+dm, out=np.zeros_like(dp), where=(dp+dm)>0)
        adx = np.zeros(n)
        s = 2*period - 1
        if s < n:
            adx[s] = np.mean(dx[period:s+1])
            for i in range(s+1, n):
                adx[i] = (adx[i-1]*(period-1)+dx[i])/period
        return adx, dp, dm, atr
    def _classify_bar(self, adx_v, dp, dm, atr_v, atr_pct):
        ts = min(adx_v / 50.0, 1.0)
        td = np.clip((dp - dm) / 100.0, -1.0, 1.0)
        v = float(atr_pct) if not np.isnan(atr_pct) else 0.5
        if v > 0.85:
            return Regime.VOLATILE, 0.6 + 0.4*(v-0.85)/0.15
        if v < 0.15:
            return Regime.QUIET, 0.6 + 0.4*(0.15-v)/0.15
        if ts > 0.5:
            if td > 0.1:
                return Regime.TREND_UP, ts
            if td < -0.1:
                return Regime.TREND_DOWN, ts
        return Regime.RANGE, 0.5 + 0.3*(1.0-ts)
    def detect(self, data):
        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        adx, dp, dm, atr = self._compute_adx(h, l, c, self.adx_period)
        atr_pct = pd.Series(atr).rolling(self.vol_lookback, min_periods=20).rank(pct=True).iloc[-1]
        r, cf = self._classify_bar(adx[-1], dp[-1], dm[-1], atr[-1], atr_pct)
        return RegimeState(regime=r, confidence=cf,
            trend_strength=min(adx[-1]/50.0, 1.0),
            trend_direction=float(np.clip((dp[-1]-dm[-1])/100.0, -1.0, 1.0)),
            volatility_score=float(atr_pct) if not np.isnan(atr_pct) else 0.5,
            adx=float(adx[-1]), atr=float(atr[-1]),
            atr_percentile=float(atr_pct),
            di_plus=float(dp[-1]), di_minus=float(dm[-1]))
    def detect_history(self, data):
        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        adx, dp, dm, atr = self._compute_adx(h, l, c, self.adx_period)
        atr_pct = pd.Series(atr).rolling(self.vol_lookback, min_periods=20).rank(pct=True).values
        regs = []
        confs = []
        for i in range(len(c)):
            r, cf = self._classify_bar(adx[i], dp[i], dm[i], atr[i], atr_pct[i])
            regs.append(r)
            confs.append(cf)
        return pd.DataFrame({"regime": regs,
            "regime_name": [RN[int(r)] for r in regs],
            "confidence": confs,
            "trend_strength": np.clip(adx/50.0, 0, 1),
            "trend_direction": np.clip((dp-dm)/100.0, -1, 1),
            "volatility_score": atr_pct,
            "adx": adx, "atr": atr,
            "di_plus": dp, "di_minus": dm})
