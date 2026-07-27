"""
regime_detector.py - Institutional Regime Detector with Immune System
Classifies market state and triggers Extreme Volatility Lockout.
"""
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
    EXTREME_VOLATILE = 6  # Immune System Trigger

RN = {1:"TREND_UP", 2:"TREND_DOWN", 3:"RANGE", 4:"VOLATILE", 5:"QUIET", 6:"EXTREME_VOLATILE"}

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
    def __init__(self, adx_period=14, atr_period=14, ma_fast=20, ma_slow=50, vol_lookback=100, adx_threshold=25.0):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.vol_lookback = vol_lookback
        self.adx_threshold = adx_threshold

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        df["TR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        up_move = df["high"].diff()
        down_move = -df["low"].diff()
        
        df["+DM"] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        df["-DM"] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        alpha = 1.0 / self.adx_period
        df["ATR"] = df["TR"].ewm(alpha=alpha, adjust=False).mean()
        df["+DI"] = 100 * (df["+DM"].ewm(alpha=alpha, adjust=False).mean() / df["ATR"])
        df["-DI"] = 100 * (df["-DM"].ewm(alpha=alpha, adjust=False).mean() / df["ATR"])
        
        dx = 100 * np.abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]).replace(0, np.nan)
        df["ADX"] = dx.ewm(alpha=alpha, adjust=False).mean()
        
        df["MA_Fast"] = df["close"].rolling(self.ma_fast).mean()
        df["MA_Slow"] = df["close"].rolling(self.ma_slow).mean()
        
        df["ATR_Pct"] = df["ATR"].rolling(self.vol_lookback, min_periods=20).rank(pct=True)
        
        # FIX-INST: Removed bfill() to prevent look-ahead bias. Drop NaNs instead.
        df.dropna(inplace=True)
        return df

    def _classify_bar(self, row: pd.Series) -> tuple:
        adx = row["ADX"]
        di_plus = row["+DI"]
        di_minus = row["-DI"]
        atr_pct = row["ATR_Pct"]
        ma_fast = row["MA_Fast"]
        ma_slow = row["MA_Slow"]
        close = row["close"]
        
        # 1. Immune System: Extreme Volatility Lockout (Top 5% of historical volatility)
        if atr_pct > 0.95:
            return Regime.EXTREME_VOLATILE, 1.0
            
        # 2. Volatility Regimes
        if atr_pct > 0.85:
            return Regime.VOLATILE, 0.8
        if atr_pct < 0.15:
            return Regime.QUIET, 0.7
            
        # 3. Trend Regimes
        if adx >= self.adx_threshold:
            if close > ma_fast and ma_fast > ma_slow and di_plus > di_minus:
                return Regime.TREND_UP, min(adx / 50.0, 1.0)
            if close < ma_fast and ma_fast < ma_slow and di_minus > di_plus:
                return Regime.TREND_DOWN, min(adx / 50.0, 1.0)
                
        # 4. Default to Range
        return Regime.RANGE, 0.5 + 0.3 * (1.0 - min(adx / 50.0, 1.0))

    def detect(self, data: pd.DataFrame) -> RegimeState:
        df = self._compute_indicators(data)
        last_row = df.iloc[-1]
        r, cf = self._classify_bar(last_row)
        return RegimeState(
            regime=r, confidence=cf,
            trend_strength=min(last_row["ADX"]/50.0, 1.0),
            trend_direction=float(np.clip((last_row["+DI"]-last_row["-DI"])/100.0, -1.0, 1.0)),
            volatility_score=float(last_row["ATR_Pct"]),
            adx=float(last_row["ADX"]), atr=float(last_row["ATR"]),
            atr_percentile=float(last_row["ATR_Pct"]),
            di_plus=float(last_row["+DI"]), di_minus=float(last_row["-DI"])
        )

    def detect_history(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_indicators(data)
        results = df.apply(self._classify_bar, axis=1, result_type="expand")
        df["regime"] = results[0].astype(int)
        df["regime_name"] = results[0].map(RN)
        df["confidence"] = results[1]
        return df
