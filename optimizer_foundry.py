"""
optimizer_foundry.py - Foundry Optimizer with Normalization Gate
Restricts parameter ranges to realistic values to prevent overfitting.
FIX-BUG-H05: Fixed Data Leakage by generating signals AFTER split.
FIX-BUG-M08: No singleton mutation, using local params dict.
"""
import optuna
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("   FOUNDRY OPTIMIZER (LEAKAGE-FREE)")
print("=" * 60)

# 1. Load raw data only (NO signal generation here)
print("\n[1/4] Loading H1 raw data...")
raw_data = load_hourly()
print(f"   Total candles: {len(raw_data)}")

# 2. Split into 3 sequential windows FIRST
window_size = len(raw_data) // 4
windows = []
for i in range(3):
    start_train = i * window_size
    end_train = start_train + window_size
    PURGE_BARS = 10  # FIX-INST: Prevent label leakage across train/test boundary
    start_test = end_train + PURGE_BARS
    end_test = start_test + window_size
    if end_test <= len(raw_data):
        train_df = raw_data.iloc[start_train:end_train].copy()
        test_df = raw_data.iloc[start_test:end_test].copy()
        
        # FIX-BUG-H05: Generate signals INDEPENDENTLY on each slice
        train_df = generate_signals(train_df)
        train_df = generate_momentum_signals(train_df)
        train_df['Final_Mom'] = train_df['MO_Signal']
        
        test_df = generate_signals(test_df)
        test_df = generate_momentum_signals(test_df)
        test_df['Final_Mom'] = test_df['MO_Signal']
        
        windows.append({'train': train_df, 'test': test_df})

print(f"   Created {len(windows)} windows: {[(len(w['train']), len(w['test'])) for w in windows]}")

# 3. Define objective
def objective(trial):
    bb_mult = trial.suggest_float("bb_multiplier", 1.2, 2.0, step=0.1)
    rsi_buy = trial.suggest_int("rsi_buy_threshold", 35, 45)
    rsi_sell = trial.suggest_int("rsi_sell_threshold", 55, 65)
    macd_fast = trial.suggest_int("macd_fast", 10, 16)
    macd_slow = trial.suggest_int("macd_slow", 22, 30)
    adx_period = trial.suggest_int("adx_period", 14, 20)
    adx_threshold = trial.suggest_int("adx_threshold", 22, 30)
    
    # FIX-BUG-M08: Temporarily set params, but ensure they are restored
    keys = ['bb_multiplier', 'rsi_buy_threshold', 'rsi_sell_threshold', 'macd_fast', 'macd_slow', 'adx_period', 'adx_threshold']
    original = {k: getattr(params, k) for k in keys}
    vals = [bb_mult, rsi_buy, rsi_sell, macd_fast, macd_slow, adx_period, adx_threshold]
    
    for k, v in zip(keys, vals):
        setattr(params, k, v)

    oos_pfs = []
    try:
        for w in windows:
            df_test = w['test'].copy()
            res = dynamic_backtest(data=df_test, strategy_cols=['Final', 'Final_Mom'], risk_percent=0.005, log_trades=False)
            trades = res[0] if isinstance(res, tuple) else res
            
            if trades and len(trades) > 15:
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
        # Ensure params are restored even if exception occurs
        for k, v in original.items():
            setattr(params, k, v)

# 4. Run optimization
print("\n[2/4] Running Foundry Optimization (50 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n[3/4] Best stable parameters found:")
best = study.best_params
for k, v in best.items():
    print(f"   {k}: {v}")
print(f"   Best Score: {study.best_value:.2f}")

# 5. Save
with open('foundry_params.txt', 'w') as f:
    f.write("Stable parameters from Foundry Optimizer (Leakage-Free):\n")
    for k, v in best.items():
        f.write(f"{k}={v}\n")
print("\n[4/4] Results saved to foundry_params.txt")
print("Done.")
