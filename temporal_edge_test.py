"""
temporal_edge_test.py - Temporal/Session Edge Test (Council Atom 7.4)
Tests if specific days/hours have a structural edge.
"""
import pandas as pd
import numpy as np
from data_loader import load_hourly
from scipy import stats

print("=" * 60)
print("   TEMPORAL EDGE TEST - Day of Week / Hour of Day")
print("=" * 60)

# 1. Load data
print("\n[1/3] Loading H1 data...")
df = load_hourly()
print(f"   Loaded {len(df)} candles")

# 2. Add forward return
print("[2/3] Calculating forward returns...")
df['forward_return_pips'] = (df['close'].shift(-1) - df['close']) * 10000
df.dropna(subset=['forward_return_pips'], inplace=True)

# 3. Analysis
print("[3/3] Analyzing temporal edges...")

# Day of week analysis
print("\n--- Day of Week Analysis ---")
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
day_stats = df.groupby('weekday')['forward_return_pips'].agg(['mean', 'std', 'count'])
day_stats['t_stat'] = day_stats['mean'] / (day_stats['std'] / np.sqrt(day_stats['count']))
day_stats['p_value'] = 2 * (1 - np.abs(day_stats['t_stat']).apply(lambda x: stats.norm.cdf(x) if not np.isnan(x) else 1))
for idx, row in day_stats.iterrows():
    name = day_names.get(idx, str(idx))
    sig = "SIGNIFICANT" if row['p_value'] < 0.05 else "noise"
    print(f"   {name}: Mean={row['mean']:.2f} pips, Count={int(row['count'])}, p={row['p_value']:.3f} [{sig}]")

# Hour of day analysis
print("\n--- Hour of Day Analysis (Top/Bottom 5) ---")
hour_stats = df.groupby('hour')['forward_return_pips'].agg(['mean', 'std', 'count'])
hour_stats['t_stat'] = hour_stats['mean'] / (hour_stats['std'] / np.sqrt(hour_stats['count']))
hour_stats['p_value'] = 2 * (1 - np.abs(hour_stats['t_stat']).apply(lambda x: stats.norm.cdf(x) if not np.isnan(x) else 1))

# Show top 5 best hours
top_hours = hour_stats.nlargest(5, 'mean')
print("   Best hours (by mean return):")
for h, row in top_hours.iterrows():
    sig = "SIGNIFICANT" if row['p_value'] < 0.05 else "noise"
    print(f"   Hour {int(h)}:00 - Mean={row['mean']:.2f} pips, p={row['p_value']:.3f} [{sig}]")

# Show bottom 5 worst hours
bottom_hours = hour_stats.nsmallest(5, 'mean')
print("   Worst hours (by mean return):")
for h, row in bottom_hours.iterrows():
    sig = "SIGNIFICANT" if row['p_value'] < 0.05 else "noise"
    print(f"   Hour {int(h)}:00 - Mean={row['mean']:.2f} pips, p={row['p_value']:.3f} [{sig}]")

print("\n" + "=" * 60)
print("If any temporal window shows SIGNIFICANT, it may be a new edge source.")
print("=" * 60)
