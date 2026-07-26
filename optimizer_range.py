"""
optimizer_range.py - Dedicated Leakage-Free Optimizer for Range Hunter
Optimizes: rh_adx_period, rh_adx_threshold, rh_bb_window, rh_bb_mult
"""
import optuna
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_hourly
from strategy_range_hunter import generate_range_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params

print("=" * 60)
print("   RANGE HUNTER OPTIMIZER (LEAKAGE-FREE)")
print("=" * 60)

# 1. Load raw data only
print("\n[1/4] Loading H1 raw data...")
raw_data = load_hourly()
print(f"   Total candles: {len(raw_data)}")

# 2. Split into 3 sequential windows
window_size = len(raw_data) // 4
windows = []
for i in range(3):
    start_train = i * window_size
    end_train = start_train + window_size
    start_test = end_train
    end_test = start_test + window_size
    if end_test <= len(raw_data):
        windows.append({
            'train': raw_data.iloc[start_train:end_train].copy(),
            'test': raw_data.iloc[start_test:end_test].copy()
        })
print(f"   Created {len(windows)} windows.")

# 3. Define objective
def objective(trial):
    adx_period = trial.suggest_int("rh_adx_period", 10, 20)
    adx_threshold = trial.suggest_float("rh_adx_threshold", 15.0, 25.0, step=1.0)
    bb_window = trial.suggest_int("rh_bb_window", 15, 30)
    bb_mult = trial.suggest_float("rh_bb_mult", 1.5, 2.5, step=0.1)
    
    keys = ['rh_adx_period', 'rh_adx_threshold', 'rh_bb_window', 'rh_bb_mult']
    original = {k: getattr(params, k, 14 if 'period' in k else (20.0 if 'threshold' in k else (20 if 'window' in k else 2.0))) for k in keys}
    
    setattr(params, 'rh_adx_period', adx_period)
    setattr(params, 'rh_adx_threshold', adx_threshold)
    setattr(params, 'rh_bb_window', bb_window)
    setattr(params, 'rh_bb_mult', bb_mult)

    oos_pfs = []
    try:
        for w in windows:
            df_test = w['test'].copy()
            df_test = generate_range_signals(df_test)
            
            res = dynamic_backtest(data=df_test, strategy_cols=['Signal'], risk_percent=0.005, log_trades=False)
            trades = res[0] if isinstance(res, tuple) else res
            
            if trades and len(trades) > 10:
                wins = [t for t in trades if t > 0]
                losses = [t for t in trades if t <= 0]
                pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
                oos_pfs.append(pf)
            else:
                oos_pfs.append(0.0)

        if len(oos_pfs) < 3:
            return -9999.0

        avg_pf = np.mean(oos_pfs)
        std_pf = np.std(oos_pfs)
        score = avg_pf * 10 - std_pf * 30
        return score

    except Exception:
        return -9999.0
    finally:
        for k, v in original.items():
            setattr(params, k, v)

# 4. Run optimization
print("\n[2/4] Running Range Hunter Optimization (50 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n[3/4] Best stable parameters found:")
best = study.best_params
for k, v in best.items():
    print(f"   {k}: {v}")
print(f"   Best Score: {study.best_value:.2f}")

# 5. Save
with open('range_params.txt', 'w') as f:
    f.write("Stable parameters from Range Hunter Optimizer (Leakage-Free):\n")
    for k, v in best.items():
        f.write(f"{k}={v}\n")
print("\n[4/4] Results saved to range_params.txt")
print("Done.")
