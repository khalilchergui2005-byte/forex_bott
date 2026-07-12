"""
cpv_high_atr.py - CPCV Test for High ATR Edge (Candidate 2)
Tests if high volatility regimes offer a structural edge.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from data_loader import load_hourly
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   CPCV TEST - High ATR Edge")
print("=" * 60)

# 1. Load data
print("\n[1/4] Loading H1 data...")
full_data = load_hourly()

# Calculate required features
full_data['ATR'] = (full_data['high'] - full_data['low']).rolling(14).mean()
full_data['ATR_ratio'] = full_data['ATR'] / full_data['close']
full_data.dropna(inplace=True)

# Generate signal: buy when ATR_ratio > 66th percentile (high volatility)
threshold = full_data['ATR_ratio'].quantile(0.66)
full_data['Signal'] = 0
full_data.loc[full_data['ATR_ratio'] >= threshold, 'Signal'] = 1

print(f"   Clean candles: {len(full_data)}")
print(f"   High ATR threshold: {threshold:.6f}")
print(f"   Signals (High Volatility): {full_data['Signal'].sum()}")

# 2. CPCV Setup
print("\n[2/4] Setting up CPCV...")
n_groups = 5
group_size = len(full_data) // n_groups
groups = [full_data.iloc[i*group_size:(i+1)*group_size].copy() for i in range(n_groups)]
test_combinations = list(combinations(range(n_groups), 2))
print(f"   Testing {len(test_combinations)} combinations.")

# 3. Run CPCV
print("\n[3/4] Running CPCV across all paths...")
pf_results = []
for idx, test_idx in enumerate(test_combinations):
    train_idx = [i for i in range(n_groups) if i not in test_idx]
    test_data = pd.concat([groups[i] for i in test_idx])
    
    if test_data['Signal'].sum() < 5:
        pf_results.append(0.0)
        continue
    
    trades, equity = dynamic_backtest(
        data=test_data,
        strategy_cols=['Signal'],
        risk_percent=0.005
    )
    if trades and len(trades) > 5:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
        pf_results.append(pf)
    else:
        pf_results.append(0.0)

# 4. Analysis
print("\n[4/4] Analyzing results...")
if pf_results:
    avg_pf = np.mean(pf_results)
    std_pf = np.std(pf_results)
    n_trials = len(test_combinations)
    expected_max = np.sqrt(2 * np.log(n_trials))
    dsr = (avg_pf - 1.0) / max(std_pf, 0.001) * (1 - expected_max / np.sqrt(n_trials))
    
    print(f"\n{'='*60}")
    print(f"   CPCV RESULTS for HIGH ATR ({len(test_combinations)} paths)")
    print(f"{'='*60}")
    print(f"   Average PF: {avg_pf:.3f} (±{std_pf:.3f})")
    print(f"   PF Range: [{min(pf_results):.3f} - {max(pf_results):.3f}]")
    print(f"   % of paths PF>1.0: {sum(1 for p in pf_results if p>1.0)/len(pf_results)*100:.1f}%")
    print(f"   Deflated Sharpe Ratio: {dsr:.3f}")
    print(f"   Verdict: {'EDGE EXISTS' if dsr > 0.5 else 'NO RELIABLE EDGE'}")
    print(f"{'='*60}")
else:
    print("No results to display.")
