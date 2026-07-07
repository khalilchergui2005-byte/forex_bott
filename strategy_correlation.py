"""
strategy_correlation.py - Signal Correlation Test (Atom 8.10c)
Checks overlap between strategies to measure true diversification.
"""
import pandas as pd
from data_loader import load_hourly
from strategy_long_lower_shadow import generate_long_lower_shadow_signals
from strategy_high_atr import generate_high_atr_signals
from strategy_hour00 import generate_hour00_signals

print("=" * 60)
print("   STRATEGY DIVERSIFICATION TEST")
print("=" * 60)

# 1. Load data and generate all signals
print("\n[1/2] Loading data and generating signals...")
df = load_hourly()

df_shadow = generate_long_lower_shadow_signals(df.copy())
df_atr = generate_high_atr_signals(df.copy())
df_hour = generate_hour00_signals(df.copy())

# Get signal timestamps
shadow_times = set(df_shadow[df_shadow['Signal']==1].index)
atr_times = set(df_atr[df_atr['Signal']==1].index)
hour_times = set(df_hour[df_hour['Signal']==1].index)

# 2. Calculate overlaps
print("\n[2/2] Calculating overlaps...")
total_shadow = len(shadow_times)
total_atr = len(atr_times)
total_hour = len(hour_times)

overlap_shadow_atr = len(shadow_times & atr_times)
overlap_shadow_hour = len(shadow_times & hour_times)
overlap_atr_hour = len(atr_times & hour_times)
overlap_all = len(shadow_times & atr_times & hour_times)

print(f"\n{'='*60}")
print("   DIVERSIFICATION REPORT")
print(f"{'='*60}")
print(f"   Long_Lower_Shadow signals: {total_shadow}")
print(f"   High_ATR signals:         {total_atr}")
print(f"   Hour_00 signals:          {total_hour}")
print(f"   {'-'*40}")
print(f"   Overlap (Shadow & ATR):   {overlap_shadow_atr} ({overlap_shadow_atr/max(total_shadow,1)*100:.1f}%)")
print(f"   Overlap (Shadow & Hour):  {overlap_shadow_hour} ({overlap_shadow_hour/max(total_shadow,1)*100:.1f}%)")
print(f"   Overlap (ATR & Hour):     {overlap_atr_hour} ({overlap_atr_hour/max(total_atr,1)*100:.1f}%)")
print(f"   Overlap (All Three):      {overlap_all}")
print(f"{'='*60}")

if overlap_shadow_atr < total_shadow * 0.3 and overlap_shadow_hour < total_shadow * 0.3:
    print("✅ Strategies are DIVERSE. Smart Portfolio will add value.")
else:
    print("⚠️ Strategies overlap significantly. Use only Long_Lower_Shadow as core.")
