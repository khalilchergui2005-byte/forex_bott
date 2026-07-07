"""
smart_portfolio.py - Smart Portfolio with Dynamic Weights (Atom 8.10d)
Allocates capital based on current volatility regime.
"""
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_long_lower_shadow import generate_long_lower_shadow_signals
from strategy_high_atr import generate_high_atr_signals
from strategy_hour00 import generate_hour00_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params

print("=" * 60)
print("   SMART PORTFOLIO - Dynamic Strategy Weights")
print("=" * 60)

# 1. Load data and generate signals
print("\n[1/3] Loading data and generating signals...")
df = load_hourly()

# Generate all strategies
df_shadow = generate_long_lower_shadow_signals(df.copy())
df_atr = generate_high_atr_signals(df.copy())
df_hour = generate_hour00_signals(df.copy())

# Calculate volatility for regime detection
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR_ratio'] = df['ATR'] / df['close']
low_vol_threshold = df['ATR_ratio'].quantile(0.33)
high_vol_threshold = df['ATR_ratio'].quantile(0.66)
print(f"   Low Vol Threshold: {low_vol_threshold:.6f}, High Vol Threshold: {high_vol_threshold:.6f}")

# 2. Run each strategy separately
print("[2/3] Running individual strategies...")
trades_shadow, eq_shadow = dynamic_backtest(data=df_shadow, strategy_cols=['Signal'], risk_percent=0.005)
trades_atr, eq_atr = dynamic_backtest(data=df_atr, strategy_cols=['Signal'], risk_percent=0.005)
trades_hour, eq_hour = dynamic_backtest(data=df_hour, strategy_cols=['Signal'], risk_percent=0.005)

# 3. Combine equity curves with dynamic weights
print("[3/3] Building Smart Portfolio...")
min_len = min(len(eq_shadow), len(eq_atr), len(eq_hour))
eq_shadow = eq_shadow[:min_len]
eq_atr = eq_atr[:min_len]
eq_hour = eq_hour[:min_len]

# Generate dynamic weights based on volatility
weights = []
for i in range(min_len):
    atr_ratio = df['ATR_ratio'].iloc[i] if i < len(df['ATR_ratio']) else df['ATR_ratio'].iloc[-1]
    
    if atr_ratio >= high_vol_threshold:
        # High volatility: favor High_ATR
        w_shadow, w_atr, w_hour = 0.3, 0.5, 0.2
    elif atr_ratio <= low_vol_threshold:
        # Low volatility: favor Long_Lower_Shadow
        w_shadow, w_atr, w_hour = 0.5, 0.3, 0.2
    else:
        # Normal: balanced
        w_shadow, w_atr, w_hour = 0.4, 0.4, 0.2
    
    weights.append((w_shadow, w_atr, w_hour))

# Build combined equity curve
initial_capital = params.initial_capital
combined_equity = [initial_capital]
shadow_weight_sum = 0
atr_weight_sum = 0
hour_weight_sum = 0

for i in range(min_len - 1):
    shadow_ret = (eq_shadow[i+1] / eq_shadow[i]) - 1
    atr_ret = (eq_atr[i+1] / eq_atr[i]) - 1
    hour_ret = (eq_hour[i+1] / eq_hour[i]) - 1
    
    w_s, w_a, w_h = weights[i]
    shadow_weight_sum += w_s
    atr_weight_sum += w_a
    hour_weight_sum += w_h
    
    portfolio_ret = (w_s * shadow_ret) + (w_a * atr_ret) + (w_h * hour_ret)
    combined_equity.append(combined_equity[-1] * (1 + portfolio_ret))

# Results
total_ret = (combined_equity[-1] / initial_capital - 1) * 100
max_dd = (pd.Series(combined_equity) / pd.Series(combined_equity).cummax() - 1).min() * 100

print(f"\n{'='*60}")
print("   SMART PORTFOLIO RESULTS")
print(f"{'='*60}")
print(f"   💰 Final Balance: ${combined_equity[-1]:,.2f}")
print(f"   📈 Total Return: {total_ret:.2f}%")
print(f"   📉 Max Drawdown: {max_dd:.2f}%")
print(f"   📊 Avg Weights - Shadow: {shadow_weight_sum/min_len:.2f}, ATR: {atr_weight_sum/min_len:.2f}, Hour: {hour_weight_sum/min_len:.2f}")
print(f"{'='*60}")
