import pandas as pd
import numpy as np

def generate_mm_signals(df):
    """
    استراتيجية صانع السوق الديناميكي (M15).
    المدخل: DataFrame بأعمدة open, high, low, close (إطار 15 دقيقة).
    المخرج: نفس الـ DataFrame مع عمود MM_Active (True/False) والمؤشرات.
    """
    df = df.copy()
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    df['ATR_ratio'] = df['ATR'] / df['close']
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['Trend'] = 0
    df.loc[df['close'] > df['SMA_20'] * 1.001, 'Trend'] = 1
    df.loc[df['close'] < df['SMA_20'] * 0.999, 'Trend'] = -1

    vol = df['ATR_ratio'].dropna()
    low_thresh = vol.quantile(0.33)
    df['VolClass'] = 1
    df.loc[df['ATR_ratio'] <= low_thresh, 'VolClass'] = 0

    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_distance'] = (df['close'] - df['BB_mid']).abs() / df['BB_std']

    df['MM_Active'] = (
        (df['Trend'] == 0) &
        (df['VolClass'] == 0) &
        (df['BB_distance'] < 0.5)
    )
    return df
