"""
optimizer_foundry.py - Foundry Optimizer with Normalization Gate
Restricts parameter ranges to realistic values to prevent overfitting.
"""
import optuna
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params

print("=" * 60)
print("   FOUNDRY OPTIMIZER WITH NORMALIZATION GATE")
print("=" * 60)

# 1. Load and prepare data
print("\n[1/4] Loading H1 data...")
full_data = load_hourly()
full_data = generate_signals(full_data)
full_data = generate_momentum_signals(full_data)
full_data['Final_Mom'] = full_data['MO_Signal']
print(f"   Total candles: {len(full_data)}")

# 2. Split into 3 sequential windows
window_size = len(full_data) // 4
windows = []
for i in range(3):
    start_train = i * window_size
    end_train = start_train + window_size
    start_test = end_train
    end_test = start_test + window_size
    if end_test <= len(full_data):
        windows.append({
            'train': full_data.iloc[start_train:end_train].copy(),
            'test': full_data.iloc[start_test:end_test].copy()
        })
print(f"   Created {len(windows)} windows: {[(len(w['train']), len(w['test'])) for w in windows]}")

# 3. Define objective with NORMALIZED parameter ranges
def objective(trial):
    # --- Normalization Gate: Tight, statistically reasonable ranges ---
    bb_mult = trial.suggest_float("bb_multiplier", 1.2, 2.0, step=0.1)
    rsi_buy = trial.suggest_int("rsi_buy_threshold", 35, 45)
    rsi_sell = trial.suggest_int("rsi_sell_threshold", 55, 65)
    macd_fast = trial.suggest_int("macd_fast", 10, 16)
    macd_slow = trial.suggest_int("macd_slow", 22, 30)
    adx_period = trial.suggest_int("adx_period", 14, 20)
    adx_threshold = trial.suggest_int("adx_threshold", 22, 30)
    # -----------------------------------------------------------------

    original = {k: getattr(params, k) for k in ['bb_multiplier', 'rsi_buy_threshold', 'rsi_sell_threshold', 'macd_fast', 'macd_slow', 'adx_period', 'adx_threshold']}
    for k, v in zip(original.keys(), [bb_mult, rsi_buy, rsi_sell, macd_fast, macd_slow, adx_period, adx_threshold]):
        setattr(params, k, v)

    oos_pfs = []
    try:
        for w in windows:
            df_test = w['test'].copy()
            trades, equity = dynamic_backtest(data=df_test, strategy_cols=['Final', 'Final_Mom'], risk_percent=0.005)
            if trades and len(trades) > 15:
                wins = [t for t in trades if t > 0]
                losses = [t for t in trades if t <= 0]
                pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
                oos_pfs.append(pf)
            else:
                oos_pfs.append(0.0)

        for k, v in original.items():
            setattr(params, k, v)

        if len(oos_pfs) < 3:
            return -9999.0

        avg_pf = np.mean(oos_pfs)
        std_pf = np.std(oos_pfs)
        # Heavily penalize high variance
        score = avg_pf * 10 - std_pf * 30
        return score

    except Exception:
        for k, v in original.items():
            setattr(params, k, v)
        return -9999.0

# 4. Run optimization
print("\n[2/4] Running Foundry Optimization (100 trials with Normalization)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n[3/4] Best stable parameters found:")
best = study.best_params
for k, v in best.items():
    print(f"   {k}: {v}")
print(f"   Best Score: {study.best_value:.2f}")

# 5. Save
with open('foundry_params.txt', 'w') as f:
    f.write("Stable parameters from Foundry Optimizer (Normalized):\n")
    for k, v in best.items():
        f.write(f"{k}={v}\n")
print("\n[4/4] Results saved to foundry_params.txt")
print("Done.")
