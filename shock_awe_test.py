"""
shock_awe_test.py - Statistical Runs Test for Shock & Awe (Atom 7.3)
Tests if winning trades cluster together (momentum).
"""
import pandas as pd
import numpy as np
from scipy import stats
from data_loader import load_hourly
from strategy_long_lower_shadow import generate_long_lower_shadow_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   SHOCK & AWE - Runs Test for Trade Momentum")
print("=" * 60)

# 1. Generate trade sequence from our best strategy
print("\n[1/3] Generating trade history from Long_Lower_Shadow...")
df = load_hourly()
df = generate_long_lower_shadow_signals(df)
trades, equity, trade_log = dynamic_backtest(
    data=df,
    strategy_cols=['Signal'],
    risk_percent=0.005,
    log_trades=True
)

if not trades or len(trades) < 30:
    print("   ERROR: Not enough trades for statistical test (need >30).")
    exit()

# 2. Prepare win/loss sequence and run Runs Test
print(f"[2/3] Running Runs Test on {len(trades)} trades...")
wl_sequence = [1 if t > 0 else 0 for t in trades]
wins = sum(wl_sequence)
losses = len(wl_sequence) - wins

# Count runs
runs = 1
for i in range(1, len(wl_sequence)):
    if wl_sequence[i] != wl_sequence[i-1]:
        runs += 1

# Expected runs under null hypothesis (no momentum)
n = len(wl_sequence)
expected_runs = ((2 * wins * losses) / n) + 1
std_runs = np.sqrt((2 * wins * losses * (2 * wins * losses - n)) / (n**2 * (n - 1)))

# Z-score and p-value
z_stat = (runs - expected_runs) / std_runs if std_runs > 0 else 0
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# 3. Verdict
print(f"\n{'='*60}")
print("   RUNS TEST RESULTS")
print(f"{'='*60}")
print(f"   Total Trades: {n}")
print(f"   Wins: {wins}, Losses: {losses}")
print(f"   Observed Runs: {runs}")
print(f"   Expected Runs (if random): {expected_runs:.2f}")
print(f"   Z-statistic: {z_stat:.3f}")
print(f"   P-value: {p_value:.4f}")
print(f"{'='*60}")
if p_value < 0.05:
    print("   ✅ Momentum is STATISTICALLY SIGNIFICANT.")
    print("   Shock & Awe rule is VALID and should remain active.")
else:
    print("   ❌ No evidence of momentum (p > 0.05).")
    print("   Shock & Awe assumption is FALSE. Rule should be FROZEN.")
print(f"{'='*60}")
