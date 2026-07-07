"""
strategy_high_atr.py - Pure High ATR Strategy
Generates buy signal when ATR_ratio >= 66th percentile.
"""
import pandas as pd
import numpy as np

def generate_high_atr_signals(df):
    df = df.copy()
    
    # Calculate ATR
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    df['ATR_ratio'] = df['ATR'] / df['close']
    
    # Generate signal (buy when high volatility)
    threshold = df['ATR_ratio'].quantile(0.66)
    df['Signal'] = 0
    df.loc[df['ATR_ratio'] >= threshold, 'Signal'] = 1
    
    df.dropna(inplace=True)
    
    return df
