"""
strategy_hour00.py - Pure Hour_00 Temporal Strategy
Generates buy signal at 00:00 UTC.
"""
import pandas as pd
import numpy as np

def generate_hour00_signals(df):
    df = df.copy()
    
    # Calculate ATR (required by backtest engine)
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    # Generate signal at midnight UTC
    df['Signal'] = 0
    df.loc[df.index.hour == 0, 'Signal'] = 1
    
    df.dropna(inplace=True)
    
    return df
