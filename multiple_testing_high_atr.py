"""
multiple_testing_high_atr.py - Bonferroni Correction for High ATR (Candidate 2)
Calculates original p-value and applies strict correction.
"""
import pandas as pd
import numpy as np
from scipy import stats
from data_loader import load_hourly

print("=" * 60)
print("   MULTIPLE TESTING CORRECTION - High ATR")
print("=" * 60)

# 1. Load data and calculate feature
print("\n[1/3] Loading data and calculating High ATR...")
df = load_hourly()
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR_ratio'] = df['ATR'] / df['close']
df['forward_return'] = df['close'].pct_change().shift(-1)
df.dropna(inplace=True)

# 2. Calculate original p-value (Spearman correlation)
print("[2/3] Calculating original p-value...")
corr, p_value = stats.spearmanr(df['ATR_ratio'], df['forward_return'])
print(f"   Spearman correlation: {corr:.6f}")
print(f"   Original p-value: {p_value:.6f}")

# 3. Apply Bonferroni correction
print("\n[3/3] Applying Bonferroni correction...")
total_trials = 2139  # Same as previous atoms
corrected_p = min(p_value * total_trials, 1.0)

print(f"   Total historical trials: {total_trials}")
print(f"   Corrected p-value: {corrected_p:.4f}")
print(f"   Still significant at 0.05? {'YES' if corrected_p < 0.05 else 'NO'}")

# 4. Final verdict
print(f"\n{'='*60}")
if corrected_p < 0.05:
    print("   ✅ High ATR SURVIVES Bonferroni correction.")
else:
    print("   ❌ High ATR FAILS Bonferroni correction.")
print(f"{'='*60}")
