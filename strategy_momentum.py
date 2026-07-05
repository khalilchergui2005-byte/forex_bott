"""
strategy_momentum.py - وكيل الزخم (Momentum Agent)
المشروع: الكيان الفيروسي - أقوى شركة تداول
يصطاد الانفجارات السعرية باستخدام MACD + ADX.
"""
import pandas as pd
import numpy as np

def generate_momentum_signals(df):
    """
    توليد إشارات الزخم على إطار الساعة (H1).
    """
    df = df.copy()
    
    # 1. حساب ATR (للمحرك)
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    # 2. مؤشر MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 3. مؤشر ADX
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean()
    
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff(-1).abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_di = 100 * (plus_dm.ewm(span=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    
    df.dropna(inplace=True)
    
    # 4. إشارات الزخم
    df['MO_Signal'] = 0
    
    # شراء: MACD إيجابي + ADX > 25 + زخم متسارع
    buy_cond = (df['MACD_hist'] > 0) & (df['ADX'] > 25) & (df['MACD_hist'] > df['MACD_hist'].shift(1))
    df.loc[buy_cond, 'MO_Signal'] = 1
    
    # بيع: MACD سلبي + ADX > 25 + زخم متسارع
    sell_cond = (df['MACD_hist'] < 0) & (df['ADX'] > 25) & (df['MACD_hist'] < df['MACD_hist'].shift(1))
    df.loc[sell_cond, 'MO_Signal'] = -1
    
    return df
