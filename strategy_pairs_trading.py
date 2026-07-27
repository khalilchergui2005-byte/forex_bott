"""
strategy_pairs_trading.py - Statistical Arbitrage (Pairs Trading)
Logic: Trades the spread based on Z-Score mean reversion, ONLY if cointegration is valid.
"""
import pandas as pd
import numpy as np

def generate_signals(df, entry_z=1.7, exit_z=0.0, stop_z=4.5):
    df = df.copy()
    
    if "z_score" not in df.columns:
        df["Pairs_Signal"] = 0
        return df
        
    df["Pairs_Signal"] = 0
    in_trade = False
    trade_type = 0
    
    for i in range(len(df)):
        z = df["z_score"].iloc[i]
        coint_valid = df["coint_valid"].iloc[i] if "coint_valid" in df.columns else True
        
        if not in_trade:
            # Entry Conditions: Z-Score threshold AND Cointegration must be valid (Green Light)
            if coint_valid:
                if z > entry_z:
                    df.iloc[i, df.columns.get_loc("Pairs_Signal")] = -1
                    in_trade = True
                    trade_type = -1
                elif z < -entry_z:
                    df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 1
                    in_trade = True
                    trade_type = 1
        else:
            # Exit Condition (Z-Score reverts to exit_z) OR Cointegration breaks (Red Light)
            if (trade_type == 1 and z >= exit_z) or (trade_type == -1 and z <= exit_z) or not coint_valid:
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 0
                in_trade = False
                trade_type = 0
            # Stop Loss Condition
            elif (trade_type == 1 and z <= -stop_z) or (trade_type == -1 and z >= stop_z):
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 0
                in_trade = False
                trade_type = 0
            else:
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = trade_type
                
    return df
