code = r'''"""
strategy_pairs_trading.py - Statistical Arbitrage (Pairs Trading)
Logic: Trades the spread between two cointegrated pairs based on Z-Score mean reversion.
Outputs: 'Pairs_Signal' column (1, -1, 0)
"""
import pandas as pd
import numpy as np

def generate_signals(df, entry_z=2.0, exit_z=0.0):
    df = df.copy()
    
    if "z_score" not in df.columns:
        df["Pairs_Signal"] = 0
        return df
        
    df["Pairs_Signal"] = 0
    in_trade = False
    trade_type = 0  # 1 for Long Spread, -1 for Short Spread
    
    # Vectorized logic using a loop for state management (entry/exit)
    for i in range(len(df)):
        z = df["z_score"].iloc[i]
        
        if not in_trade:
            # Entry Conditions
            if z > entry_z:
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = -1  # Short Spread (Sell EUR, Buy GBP)
                in_trade = True
                trade_type = -1
            elif z < -entry_z:
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 1   # Long Spread (Buy EUR, Sell GBP)
                in_trade = True
                trade_type = 1
        else:
            # Exit Condition (Z-Score reverts to 0)
            if (trade_type == 1 and z >= exit_z) or (trade_type == -1 and z <= exit_z):
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 0  # Exit
                in_trade = False
                trade_type = 0
            else:
                # Maintain position state
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = trade_type
                
    return df
'''
open("strategy_pairs_trading.py", "w", encoding="utf-8").write(code)
print("SUCCESS: strategy_pairs_trading.py created")

import ast
ast.parse(code)
print("AST OK")
