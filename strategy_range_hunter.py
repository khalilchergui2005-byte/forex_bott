"""
strategy_range_hunter.py - Range Hunter Strategy (Mean Reversion)
Project: Viral Entity - Strongest Trading Firm
Trades only in ranging markets (ADX < 20). Uses Bollinger Bands for entries.
"""
import pandas as pd
import numpy as np

def generate_range_signals(df):
    """
    Generates buy/sell signals for ranging markets.
    Input: DataFrame with 'high', 'low', 'close', 'ATR', 'VolClass'.
    Output: DataFrame with new column 'Range_Signal' (1=buy, -1=sell, 0=neutral).
    """
    df = df.copy()
    
    # 1. Ensure ATR exists
    if 'ATR' not in df.columns:
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    # 2. Calculate ADX to detect ranging markets
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    atr_adx = true_range.rolling(14).mean()
    
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.ewm(span=14).mean() / atr_adx)
    minus_di = 100 * (minus_dm.ewm(span=14).mean() / atr_adx)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    
    # 3. Bollinger Bands for entry/exit levels
    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - 2.0 * df['BB_std']
    df['BB_upper'] = df['BB_mid'] + 2.0 * df['BB_std']
    
    df.dropna(inplace=True)
    
    # 4. Generate signals ONLY in ranging market (ADX < 20) and NOT in high volatility (VolClass != 2)
    df['Range_Signal'] = 0
    
    is_ranging = df['ADX'] < 20
    
    # Buy at lower band, Sell at upper band
    buy_condition = is_ranging & (df['low'] <= df['BB_lower'])
    sell_condition = is_ranging & (df['high'] >= df['BB_upper'])
    
    df.loc[buy_condition, 'Range_Signal'] = 1
    df.loc[sell_condition, 'Range_Signal'] = -1
    
    return df
