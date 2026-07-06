"""
strategy_momentum.py - وكيل الزخم (مُحسَّن)
المشروع: الكيان الفيروسي - أقوى شركة تداول
يستخدم أفضل المعاملات المستخرجة من المحسن الآلي.
"""
import pandas as pd
import numpy as np

def generate_momentum_signals(df):
    df = df.copy()
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    # MACD مع المعاملات المُحسَّنة
    ema_fast = df['close'].ewm(span=15, adjust=False).mean()
    ema_slow = df['close'].ewm(span=25, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # ADX مع المعاملات المُحسَّنة
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    atr_adx = true_range.rolling(14).mean()
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.ewm(span=20).mean() / atr_adx)
    minus_di = 100 * (minus_dm.ewm(span=20).mean() / atr_adx)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(20).mean()
    
    df.dropna(inplace=True)
    
    df['MO_Signal'] = 0
    buy_cond = (df['MACD_hist'] > 0) & (df['ADX'] > 23) & (df['MACD_hist'] > df['MACD_hist'].shift(1))
    sell_cond = (df['MACD_hist'] < 0) & (df['ADX'] > 23) & (df['MACD_hist'] < df['MACD_hist'].shift(1))
    df.loc[buy_cond, 'MO_Signal'] = 1
    df.loc[sell_cond, 'MO_Signal'] = -1
    
    return df
