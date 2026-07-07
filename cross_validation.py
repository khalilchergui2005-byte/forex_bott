"""
cross_validation.py - Time-series Cross-Validation (Overfitting Gate)
Tests current parameters on 3 sequential data folds.
"""
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params

print("=" * 60)
print("   CROSS-VALIDATION (Overfitting Gate)")
print("=" * 60)

# 1. Load data
print("\n[1/3] Loading H1 data...")
data = load_hourly()
data = generate_signals(data)
data = generate_momentum_signals(data)
data['Final_Mom'] = data['MO_Signal']
print(f"   Total candles: {len(data)}")

# 2. Split into 3 time-sequential folds (NO shuffling)
fold_size = len(data) // 3
folds = []
for i in range(3):
    start = i * fold_size
    end = start + fold_size if i < 2 else len(data)
    folds.append(data.iloc[start:end].copy())

print(f"\n[2/3] Created 3 folds: {[len(f) for f in folds]} candles each")

# 3. Test on each fold
print("\n[3/3] Testing on each fold...")
results = []
for i, fold in enumerate(folds):
    trades, equity = dynamic_backtest(
        data=fold,
        strategy_cols=['Final', 'Final_Mom'],
        risk_percent=0.005
    )
    if trades:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
        ret = (equity[-1]/params.initial_capital - 1)*100
        results.append({'fold': i+1, 'pf': pf, 'return': ret, 'trades': len(trades)})
        print(f"   Fold {i+1}: PF={pf:.2f}, Return={ret:.2f}%, Trades={len(trades)}")
    else:
        results.append({'fold': i+1, 'pf': 0, 'return': 0, 'trades': 0})
        print(f"   Fold {i+1}: No trades!")

# 4. Summary
if results:
    df_results = pd.DataFrame(results)
    avg_pf = df_results['pf'].mean()
    std_pf = df_results['pf'].std()
    print(f"\n{'='*60}")
    print(f"   Average PF: {avg_pf:.2f} (±{std_pf:.2f})")
    print(f"   All folds PF > 1.0: {'YES' if df_results['pf'].min() > 1.0 else 'NO'}")
    print(f"   Stability: {'STABLE' if avg_pf > 1.0 and std_pf < 0.3 else 'UNSTABLE'}")
    print(f"{'='*60}")
else:
    print("No results.")
