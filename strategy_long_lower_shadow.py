"""
strategy_long_lower_shadow.py - Pure Long Lower Shadow Strategy
Generates buy signal when lower shadow >= 1.0 ATR.
"""
import pandas as pd
import numpy as np

def generate_long_lower_shadow_signals(df):
    df = df.copy()
    
    # Calculate ATR
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    # Calculate lower shadow ratio
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['ATR']
    
    # Generate signal
    df['Signal'] = 0
    df.loc[df['lower_shadow'] >= 1.0, 'Signal'] = 1
    
    df.dropna(inplace=True)
    
    return df
