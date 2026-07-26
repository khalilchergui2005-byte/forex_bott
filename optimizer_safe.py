"""
optimizer_safe.py - Safe Bayesian Optimization for Matrix + Momentum
Project: Viral Entity - Strongest Trading Firm
Does NOT modify any existing file. Only suggests best parameters.
"""
import optuna
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params
import copy

print("=" * 60)
print("   SAFE BAYESIAN OPTIMIZER")
print("=" * 60)

# 1. Load and prepare data
print("\n[1/4] Loading H1 data...")
full_data = load_hourly()
full_data = full_data.iloc[-4000:].copy()
split = int(len(full_data) * 0.7)
train_data = full_data.iloc[:split].copy()
test_data = full_data.iloc[split:].copy()
print(f"   Train: {len(train_data)} candles, Test: {len(test_data)} candles")

# 2. Define objective function
def objective(trial):
    # Suggest parameters
    bb_mult = trial.suggest_float("bb_multiplier", 1.0, 2.5, step=0.1)
    rsi_buy = trial.suggest_int("rsi_buy_threshold", 30, 50)
    rsi_sell = trial.suggest_int("rsi_sell_threshold", 50, 70)
    macd_fast = trial.suggest_int("macd_fast", 8, 20)
    macd_slow = trial.suggest_int("macd_slow", 20, 40)
    adx_period = trial.suggest_int("adx_period", 10, 25)
    adx_threshold = trial.suggest_int("adx_threshold", 20, 35)

    # Save original values
    original = {k: getattr(params, k) for k in ['bb_multiplier', 'rsi_buy_threshold', 'rsi_sell_threshold', 'macd_fast', 'macd_slow', 'adx_period', 'adx_threshold']}
    
    # Apply trial values
    params.bb_multiplier = bb_mult
    params.rsi_buy_threshold = rsi_buy
    params.rsi_sell_threshold = rsi_sell
    params.macd_fast = macd_fast
    params.macd_slow = macd_slow
    params.adx_period = adx_period
    params.adx_threshold = adx_threshold

    try:
        df = generate_signals(train_data.copy())
        df = generate_momentum_signals(df)
        df['Final_Mom'] = df['MO_Signal']
        trades, equity = dynamic_backtest(data=df, strategy_cols=['Final', 'Final_Mom'], risk_percent=0.005)

        # Restore original params
        for k, v in original.items():
            setattr(params, k, v)

        if not trades or len(trades) < 30:
            return -9999.0

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 10.0
        total_return = (equity[-1] / params.initial_capital) - 1
        max_dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min()

        # Penalize drawdowns over 2%
        penalty = max(0, abs(max_dd) - 0.02) * 500

        return pf * 10 + total_return * 100 - penalty

    except Exception as e:
        for k, v in original.items():
            setattr(params, k, v)
        return -9999.0

# 3. Run optimization
print("\n[2/4] Running Bayesian Optimization (100 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n[3/4] Optimization complete. Best parameters found:")
best = study.best_params
for k, v in best.items():
    print(f"   {k}: {v}")
print(f"   Best Score: {study.best_value:.2f}")

# 4. Test on Out-of-Sample with best parameters
print("\n[4/4] Testing best parameters on Out-of-Sample data...")
_saved_params = {k: getattr(params, k) for k in best.keys()}
for k, v in best.items():
    setattr(params, k, v)

df_test = generate_signals(test_data.copy())
df_test = generate_momentum_signals(df_test)
df_test['Final_Mom'] = df_test['MO_Signal']

trades_oos, eq_oos = dynamic_backtest(data=df_test, strategy_cols=['Final', 'Final_Mom'], risk_percent=0.005)

if trades_oos and len(trades_oos) > 0:
    wins = [t for t in trades_oos if t > 0]
    losses = [t for t in trades_oos if t <= 0]
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0
    ret = (eq_oos[-1]/10000 - 1)*100
    dd = (pd.Series(eq_oos)/pd.Series(eq_oos).cummax() - 1).min()*100
    print(f"   OOS Balance: {eq_oos[-1]:.2f} | Return: {ret:.2f}% | Trades: {len(trades_oos)}")
    print(f"   OOS PF: {pf:.2f} | MaxDD: {dd:.2f}% | WinRate: {len(wins)/len(trades_oos)*100:.1f}%")
else:
    print("   No trades generated on OOS data.")

# Save best parameters to a safe file
with open('optimizer_safe_results.txt', 'w') as f:
    f.write("Best parameters found by optimizer_safe.py:\n")
    for k, v in best.items():
        f.write(f"{k}={v}\n")
print("\nResults saved to optimizer_safe_results.txt. You can review and apply them manually.")
# BUG-M08 FIX: restore original params after OOS
for k, v in _saved_params.items():
    setattr(params, k, v)
print("Original params restored.")
print("Done.")
