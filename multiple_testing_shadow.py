"""
multiple_testing_shadow.py - Bonferroni Correction for Long Lower Shadow (Atom 8.9a)
Calculates original p-value and applies strict correction.
"""
import pandas as pd
import numpy as np
from scipy import stats
from data_loader import load_hourly

print("=" * 60)
print("   MULTIPLE TESTING CORRECTION - Long Lower Shadow")
print("=" * 60)

# 1. Load data and calculate feature
print("\n[1/3] Loading data and calculating Long Lower Shadow...")
df = load_hourly()
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['ATR']
df['forward_return'] = df['close'].pct_change().shift(-1)
df.dropna(inplace=True)

# 2. Calculate original p-value (Spearman correlation)
print("[2/3] Calculating original p-value...")
corr, p_value = stats.spearmanr(df['lower_shadow'], df['forward_return'])
print(f"   Spearman correlation: {corr:.6f}")
print(f"   Original p-value: {p_value:.6f}")

# 3. Apply Bonferroni correction
print("\n[3/3] Applying Bonferroni correction...")
# Total trials estimate (same as in atom 8.3)
total_trials = 2139
corrected_p = min(p_value * total_trials, 1.0)

print(f"   Total historical trials: {total_trials}")
print(f"   Corrected p-value: {corrected_p:.4f}")
print(f"   Still significant at 0.05? {'YES' if corrected_p < 0.05 else 'NO'}")

# 4. Final verdict
print(f"\n{'='*60}")
if corrected_p < 0.05:
    print("   ✅ Long Lower Shadow SURVIVES Bonferroni correction.")
    print("   Proceed to Atom 8.9b (Robustness ±15/±30 min).")
else:
    print("   ❌ Long Lower Shadow FAILS Bonferroni correction.")
    print("   Consider moving to next candidate (High_ATR).")
print(f"{'='*60}")
