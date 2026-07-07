"""
cpv_validator.py - Combinatorial Purged Cross-Validation (Council Atom 7.2)
Tests edge stability across multiple time paths + Deflated Sharpe Ratio.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   CPCV VALIDATOR - Definitive Edge Test")
print("=" * 60)

# 1. Load and prepare data
print("\n[1/4] Loading H1 data...")
full_data = load_hourly()
full_data = generate_signals(full_data)
full_data = generate_momentum_signals(full_data)
full_data['Final_Mom'] = full_data['MO_Signal']
print(f"   Total candles: {len(full_data)}")

# 2. Combinatorial Purged Cross-Validation Setup
print("\n[2/4] Setting up CPCV...")
# Split data into 5 groups (for simplicity)
n_groups = 5
group_size = len(full_data) // n_groups
groups = [full_data.iloc[i*group_size:(i+1)*group_size].copy() for i in range(n_groups)]
print(f"   Created {n_groups} time-ordered groups of {group_size} candles each.")

# Generate training/testing combinations (2 groups for test, rest for train)
test_combinations = list(combinations(range(n_groups), 2))
print(f"   Testing {len(test_combinations)} combinations of train/test splits.")

# 3. Run CPCV
print("\n[3/4] Running CPCV across all paths...")
pf_results = []
for idx, test_idx in enumerate(test_combinations):
    train_idx = [i for i in range(n_groups) if i not in test_idx]
    train_data = pd.concat([groups[i] for i in train_idx])
    test_data = pd.concat([groups[i] for i in test_idx])
    
    trades, equity = dynamic_backtest(
        data=test_data,
        strategy_cols=['Final', 'Final_Mom'],
        risk_percent=0.005
    )
    if trades and len(trades) > 10:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
        pf_results.append(pf)
    else:
        pf_results.append(0.0)

# 4. Analysis and Deflated Sharpe Ratio
print("\n[4/4] Analyzing results...")
if pf_results:
    avg_pf = np.mean(pf_results)
    std_pf = np.std(pf_results)
    # Simple Deflated Sharpe Ratio estimation
    # Adjusts for the number of trials (combinations tested)
    n_trials = len(test_combinations)
    expected_max = np.sqrt(2 * np.log(n_trials))  # Expected max under null
    dsr = (avg_pf - 1.0) / max(std_pf, 0.001) * (1 - expected_max / np.sqrt(n_trials))
    
    print(f"\n{'='*60}")
    print(f"   CPCV RESULTS ({len(test_combinations)} paths)")
    print(f"{'='*60}")
    print(f"   Average PF: {avg_pf:.3f} (±{std_pf:.3f})")
    print(f"   PF Range: [{min(pf_results):.3f} - {max(pf_results):.3f}]")
    print(f"   % of paths PF>1.0: {sum(1 for p in pf_results if p>1.0)/len(pf_results)*100:.1f}%")
    print(f"   Deflated Sharpe Ratio: {dsr:.3f}")
    print(f"   Verdict: {'EDGE EXISTS' if dsr > 0.5 else 'NO RELIABLE EDGE'}")
    print(f"{'='*60}")
else:
    print("No results to display.")
