"""
optimizer_bayesian.py - Bayesian Optimization for Matrix + Momentum
Uses Optuna to find the strongest parameter set for PF and stability.
"""
import optuna
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params

# Load full data once
print("Loading H1 data...")
full_data = load_hourly()
full_data = full_data.iloc[-4000:].copy()  # Use last 4000 candles for speed
print(f"Data loaded: {len(full_data)} candles")

# Split into train (70%) and test (30%)
split = int(len(full_data) * 0.7)
train_data = full_data.iloc[:split].copy()
test_data = full_data.iloc[split:].copy()
print(f"Train: {len(train_data)} candles, Test: {len(test_data)} candles")

def objective(trial):
    """Objective function for Optuna."""
    # --- Suggest parameters ---
    bb_mult = trial.suggest_float("bb_multiplier", 1.0, 2.5, step=0.1)
    rsi_buy = trial.suggest_int("rsi_buy_threshold", 30, 50)
    rsi_sell = trial.suggest_int("rsi_sell_threshold", 50, 70)
    macd_fast = trial.suggest_int("macd_fast", 8, 20)
    macd_slow = trial.suggest_int("macd_slow", 20, 40)
    adx_period = trial.suggest_int("adx_period", 10, 25)
    adx_threshold = trial.suggest_int("adx_threshold", 20, 35)
    
    # --- Temporarily set params ---
    original_values = {
        'bb_multiplier': params.bb_multiplier,
        'rsi_buy_threshold': params.rsi_buy_threshold,
        'rsi_sell_threshold': params.rsi_sell_threshold,
        'macd_fast': params.macd_fast,
        'macd_slow': params.macd_slow,
        'adx_period': params.adx_period,
        'adx_threshold': params.adx_threshold,
    }
    params.bb_multiplier = bb_mult
    params.rsi_buy_threshold = rsi_buy
    params.rsi_sell_threshold = rsi_sell
    params.macd_fast = macd_fast
    params.macd_slow = macd_slow
    params.adx_period = adx_period
    params.adx_threshold = adx_threshold

    try:
        # Generate signals on train data
        df = generate_signals(train_data.copy())
        df = generate_momentum_signals(df)
        df['Final_Mom'] = df['MO_Signal']

        # Backtest on train data
        trades, equity = dynamic_backtest(
            data=df,
            strategy_cols=['Final', 'Final_Mom'],
            risk_percent=0.005
        )

        # Restore original params
        for k, v in original_values.items():
            setattr(params, k, v)

        if not trades or len(trades) < 30:
            return -9999

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 10
        total_return = (equity[-1]/10000) - 1
        max_dd = (pd.Series(equity)/pd.Series(equity).cummax() - 1).min()

        # Penalize drawdowns over 2%
        penalty = max(0, abs(max_dd) - 0.02) * 500

        score = pf * 10 + total_return * 100 - penalty
        return score

    except Exception as e:
        # Restore original params in case of error
        for k, v in original_values.items():
            setattr(params, k, v)
        print(f"Trial failed: {e}")
        return -9999

# --- Run optimization ---
print("\nStarting Bayesian Optimization (100 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n" + "=" * 60)
print("   Best Parameters Found:")
print("=" * 60)
best = study.best_params
for k, v in best.items():
    print(f"   {k}: {v}")
print(f"   Best Score: {study.best_value:.2f}")

# --- Set best params and test on Out-of-Sample ---
params.bb_multiplier = best['bb_multiplier']
params.rsi_buy_threshold = best['rsi_buy_threshold']
params.rsi_sell_threshold = best['rsi_sell_threshold']
params.macd_fast = best['macd_fast']
params.macd_slow = best['macd_slow']
params.adx_period = best['adx_period']
params.adx_threshold = best['adx_threshold']

df_test = generate_signals(test_data.copy())
df_test = generate_momentum_signals(df_test)
df_test['Final_Mom'] = df_test['MO_Signal']

trades_oos, eq_oos = dynamic_backtest(
    data=df_test,
    strategy_cols=['Final', 'Final_Mom'],
    risk_percent=0.005
)

if trades_oos and len(trades_oos) > 0:
    wins = [t for t in trades_oos if t > 0]
    losses = [t for t in trades_oos if t <= 0]
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0
    ret = (eq_oos[-1]/10000 - 1)*100
    dd = (pd.Series(eq_oos)/pd.Series(eq_oos).cummax() - 1).min()*100
    print(f"\n--- Out-of-Sample Result with Best Params ---")
    print(f"Balance: {eq_oos[-1]:.2f} | Return: {ret:.2f}% | Trades: {len(trades_oos)}")
    print(f"PF: {pf:.2f} | MaxDD: {dd:.2f}% | WinRate: {len(wins)/len(trades_oos)*100:.1f}%")

# Save best params to file
with open('best_bayesian_params.txt', 'w') as f:
    for k, v in best.items():
        f.write(f"{k}={v}\n")
print("\nBest parameters saved to best_bayesian_params.txt")
print("Done.")
