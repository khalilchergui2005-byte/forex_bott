import pandas as pd
import numpy as np

def generate_signals(df):
    """
    استراتيجية المصفوفة التسعة – مصدر الحقيقة الوحيد.
    المدخل: DataFrame يحتوي على أعمدة: open, high, low, close.
    المخرج: نفس الـ DataFrame مع أعمدة إضافية: Regime9, Buy, Sell, Final.
    """
    df = df.copy()
    
    # 1. التقلب والاتجاه
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    df['ATR_ratio'] = df['ATR'] / df['close']
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['Trend'] = 0
    df.loc[df['close'] > df['SMA_20'] * 1.001, 'Trend'] = 1
    df.loc[df['close'] < df['SMA_20'] * 0.999, 'Trend'] = -1

    vol = df['ATR_ratio'].dropna()
    low_thresh = vol.quantile(0.33)
    high_thresh = vol.quantile(0.66)
    df['VolClass'] = 1
    df.loc[df['ATR_ratio'] <= low_thresh, 'VolClass'] = 0
    df.loc[df['ATR_ratio'] >= high_thresh, 'VolClass'] = 2

    df['Regime9'] = df['VolClass'] * 3 + (df['Trend'] + 1)

    # 2. قواعد النظام
    regime_rules = {
        0: -1, 1: 0, 2: 1,
        3: -1, 4: 0, 5: 1,
        6: -1, 7: 0, 8: 1,
    }
    df['Allowed'] = df['Regime9'].map(regime_rules)

    # 3. مؤشرات الدخول
    df.dropna(inplace=True)
    df['SMA_S'] = df['close'].rolling(10).mean()
    df['SMA_L'] = df['close'].rolling(30).mean()
    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - 1.5 * df['BB_std']
    df['BB_upper'] = df['BB_mid'] + 1.5 * df['BB_std']
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
    df.dropna(inplace=True)

    df['Buy_SMA']  = ((df['SMA_S'] > df['SMA_L']) & (df['SMA_S'].shift(1) <= df['SMA_L'].shift(1))).astype(int)
    df['Sell_SMA'] = ((df['SMA_S'] < df['SMA_L']) & (df['SMA_S'].shift(1) >= df['SMA_L'].shift(1))).astype(int)
    df['Buy_BB']   = (df['low'] <= df['BB_lower']).astype(int)
    df['Sell_BB']  = (df['high'] >= df['BB_upper']).astype(int)
    df['Buy_RSI']  = (df['RSI'] < 48).astype(int)
    df['Sell_RSI'] = (df['RSI'] > 52).astype(int)

    df['Buy']  = ((df['Buy_SMA'] | df['Buy_BB']) & df['Buy_RSI']).astype(int)
    df['Sell'] = ((df['Sell_SMA'] | df['Sell_BB']) & df['Sell_RSI']).astype(int)

    # 4. الإشارات النهائية
    df['Final'] = 0
    df.loc[(df['Buy'] == 1) & (df['Allowed'].isin([1, 0])), 'Final'] = 1
    df.loc[(df['Sell'] == 1) & (df['Allowed'].isin([-1, 0])), 'Final'] = -1

    return df
