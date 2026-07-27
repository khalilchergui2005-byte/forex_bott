# --- 1. إضافة News_Signal في parameter_controller.py ---
f = 'parameter_controller.py'
t = open(f, encoding='utf-8').read()
if 'News_Signal' not in t:
    t = t.replace("'Pairs_Signal': {'tp_mult': 0, 'sl_mult': 0, 'second_target_mult': 0}  # Pairs Trading (Exit on Z=0)", "'Pairs_Signal': {'tp_mult': 0, 'sl_mult': 0, 'second_target_mult': 0},  # Pairs Trading\n        'News_Signal': {'tp_mult': 3.0, 'sl_mult': 2.0, 'second_target_mult': 6.0}  # News Deviation")
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Added News_Signal to parameter_controller.py")

# --- 2. تعديل backtest_engine.py للتعرف على News_Signal ---
f = 'backtest_engine.py'
t = open(f, encoding='utf-8').read()
old_line = "is_trend = ('Trend' in col or 'trend' in col)"
new_line = "is_trend = ('Trend' in col or 'trend' in col or 'News' in col)"
if old_line in t:
    t = t.replace(old_line, new_line)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Engine updated to recognize News_Signal as Trend/Volatile")

# --- 3. كتابة استراتيجية strategy_news_deviation.py ---
code = r'''"""
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
'''
open("strategy_news_deviation.py", "w", encoding="utf-8").write(code)
print("SUCCESS: strategy_news_deviation.py created")

import ast
ast.parse(open("parameter_controller.py", encoding="utf-8").read())
ast.parse(open("backtest_engine.py", encoding="utf-8").read())
ast.parse(open("strategy_news_deviation.py", encoding="utf-8").read())
print("AST OK for all files")
