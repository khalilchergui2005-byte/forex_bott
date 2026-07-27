"""
strategy_news_deviation.py - News Deviation Strategy (Liquidity Shock)
Logic: Enters trade on Liquidity Shock (Volume Spike) and exits after 4 hours.
Outputs: 'News_Signal' column (1, -1, 0)
"""
import pandas as pd
import numpy as np

def generate_signals(df, hold_hours=4):
    df = df.copy()
    
    if "vol_shock" not in df.columns or "shock_dir" not in df.columns:
        df["News_Signal"] = 0
        return df
        
    df["News_Signal"] = 0
    in_trade = False
    trade_type = 0
    bars_held = 0
    
    for i in range(len(df)):
        shock = df["vol_shock"].iloc[i]
        direction = df["shock_dir"].iloc[i]
        
        if not in_trade:
            if shock and direction != 0:
                df.iloc[i, df.columns.get_loc("News_Signal")] = direction
                in_trade = True
                trade_type = direction
                bars_held = 1
        else:
            if bars_held < hold_hours:
                df.iloc[i, df.columns.get_loc("News_Signal")] = trade_type
                bars_held += 1
            else:
                # Exit after hold period
                df.iloc[i, df.columns.get_loc("News_Signal")] = 0
                in_trade = False
                trade_type = 0
                bars_held = 0
                
    return df
