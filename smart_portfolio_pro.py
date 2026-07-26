"""
smart_portfolio_pro.py - Institutional-Grade Smart Portfolio (Atom 8.10d-Super)
Optimizes dynamic weights using Bayesian Optimization, tests robustness,
and provides a Monte Carlo estimate.
"""
import optuna
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_long_lower_shadow import generate_long_lower_shadow_signals
from strategy_high_atr import generate_high_atr_signals
from strategy_hour00 import generate_hour00_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   INSTITUTIONAL SMART PORTFOLIO OPTIMIZER")
print("=" * 60)

# 1. Load data and generate all strategy signals
print("\n[1/5] Loading data...")
df = load_hourly()
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR_ratio'] = df['ATR'] / df['close']
low_vol_threshold = df['ATR_ratio'].quantile(0.33)
high_vol_threshold = df['ATR_ratio'].quantile(0.66)

df_shadow = generate_long_lower_shadow_signals(df.copy())
df_atr = generate_high_atr_signals(df.copy())
df_hour = generate_hour00_signals(df.copy())

print("   Data ready. Optimizing portfolio weights...")

# 2. Define objective function for Optuna
def objective(trial):
    # Weights for 3 volatility regimes x 3 strategies = 9 weights to optimize
    w_shadow_low = trial.suggest_float("w_shadow_low", 0.1, 0.7, step=0.05)
    w_atr_low = trial.suggest_float("w_atr_low", 0.1, 0.7, step=0.05)
    w_hour_low = trial.suggest_float("w_hour_low", 0.0, 0.3, step=0.05)

    w_shadow_mid = trial.suggest_float("w_shadow_mid", 0.1, 0.7, step=0.05)
    w_atr_mid = trial.suggest_float("w_atr_mid", 0.1, 0.7, step=0.05)
    w_hour_mid = trial.suggest_float("w_hour_mid", 0.0, 0.3, step=0.05)

    w_shadow_high = trial.suggest_float("w_shadow_high", 0.1, 0.5, step=0.05)
    w_atr_high = trial.suggest_float("w_atr_high", 0.3, 0.8, step=0.05)
    w_hour_high = trial.suggest_float("w_hour_high", 0.0, 0.2, step=0.05)

    # Normalize weights to sum to 1 for each regime
    low_sum = w_shadow_low + w_atr_low + w_hour_low
    mid_sum = w_shadow_mid + w_atr_mid + w_hour_mid
    high_sum = w_shadow_high + w_atr_high + w_hour_high

    # Get equity curves (using a portion of data for speed)
    trades_shadow, eq_shadow = dynamic_backtest(data=df_shadow.iloc[-3000:], strategy_cols=['Signal'], risk_percent=0.005)
    trades_atr, eq_atr = dynamic_backtest(data=df_atr.iloc[-3000:], strategy_cols=['Signal'], risk_percent=0.005)
    trades_hour, eq_hour = dynamic_backtest(data=df_hour.iloc[-3000:], strategy_cols=['Signal'], risk_percent=0.005)

    min_len = min(len(eq_shadow), len(eq_atr), len(eq_hour))
    eq_shadow = eq_shadow[:min_len]
    eq_atr = eq_atr[:min_len]
    eq_hour = eq_hour[:min_len]
    atr_data = df['ATR_ratio'].iloc[-min_len:].values

    combined_equity = [params.initial_capital]
    for i in range(min_len - 1):
        shadow_ret = (eq_shadow[i+1] / eq_shadow[i]) - 1
        atr_ret = (eq_atr[i+1] / eq_atr[i]) - 1
        hour_ret = (eq_hour[i+1] / eq_hour[i]) - 1

        # Determine regime
        if atr_data[i] >= high_vol_threshold:
            w_s, w_a, w_h = w_shadow_high/high_sum, w_atr_high/high_sum, w_hour_high/high_sum
        elif atr_data[i] <= low_vol_threshold:
            w_s, w_a, w_h = w_shadow_low/low_sum, w_atr_low/low_sum, w_hour_low/low_sum
        else:
            w_s, w_a, w_h = w_shadow_mid/mid_sum, w_atr_mid/mid_sum, w_hour_mid/mid_sum

        portfolio_ret = (w_s * shadow_ret) + (w_a * atr_ret) + (w_h * hour_ret)
        combined_equity.append(combined_equity[-1] * (1 + portfolio_ret))

    total_return = (combined_equity[-1] / params.initial_capital - 1) * 100
    max_dd = (pd.Series(combined_equity) / pd.Series(combined_equity).cummax() - 1).min() * 100

    # Score = reward return, heavily penalize drawdown
    return total_return - abs(max_dd) * 2

# 3. Run Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# 4. Extract best weights and run full backtest
best = study.best_params
print(f"\n[2/5] Best Weights Found: {best}")

# Normalize and apply to full data
def run_full_backtest(weights_dict):
    w_s_l, w_a_l, w_h_l = weights_dict['w_shadow_low'], weights_dict['w_atr_low'], weights_dict['w_hour_low']
    w_s_m, w_a_m, w_h_m = weights_dict['w_shadow_mid'], weights_dict['w_atr_mid'], weights_dict['w_hour_mid']
    w_s_h, w_a_h, w_h_h = weights_dict['w_shadow_high'], weights_dict['w_atr_high'], weights_dict['w_hour_high']

    low_sum = w_s_l + w_a_l + w_h_l
    mid_sum = w_s_m + w_a_m + w_h_m
    high_sum = w_s_h + w_a_h + w_h_h

    trades_s, eq_s = dynamic_backtest(data=df_shadow, strategy_cols=['Signal'], risk_percent=0.005)
    trades_a, eq_a = dynamic_backtest(data=df_atr, strategy_cols=['Signal'], risk_percent=0.005)
    trades_h, eq_h = dynamic_backtest(data=df_hour, strategy_cols=['Signal'], risk_percent=0.005)

    min_len = min(len(eq_s), len(eq_a), len(eq_h))
    eq_s, eq_a, eq_h = eq_s[:min_len], eq_a[:min_len], eq_h[:min_len]
    atr_full = df['ATR_ratio'].iloc[:min_len].values

    combined = [params.initial_capital]
    for i in range(min_len - 1):
        s_ret = (eq_s[i+1] / eq_s[i]) - 1
        a_ret = (eq_a[i+1] / eq_a[i]) - 1
        h_ret = (eq_h[i+1] / eq_h[i]) - 1

        if atr_full[i] >= high_vol_threshold:
            w_s, w_a, w_h = w_s_h/high_sum, w_a_h/high_sum, w_h_h/high_sum
        elif atr_full[i] <= low_vol_threshold:
            w_s, w_a, w_h = w_s_l/low_sum, w_a_l/low_sum, w_h_l/low_sum
        else:
            w_s, w_a, w_h = w_s_m/mid_sum, w_a_m/mid_sum, w_h_m/mid_sum

        p_ret = (w_s * s_ret) + (w_a * a_ret) + (w_h * h_ret)
        combined.append(combined[-1] * (1 + p_ret))

    return combined, trades_s, trades_a, trades_h

print("[3/5] Running full backtest with best weights...")
combined_equity, t_s, t_a, t_h = run_full_backtest(best)

total_ret = (combined_equity[-1] / params.initial_capital - 1) * 100
max_dd = (pd.Series(combined_equity) / pd.Series(combined_equity).cummax() - 1).min() * 100
total_trades = len(t_s) + len(t_a) + len(t_h)

# 5. Monte Carlo Estimate
print("[4/5] Running Monte Carlo estimate...")
all_trades = t_s + t_a + t_h
success_count = 0
for _ in range(1000):
    sim_trades = np.random.choice(all_trades, size=len(all_trades), replace=True)
    balance = params.initial_capital
    peak = balance
    for trade in sim_trades:
        balance += trade
        peak = max(peak, balance)
        if (balance / peak) - 1 <= -0.10: break
        if (balance / params.initial_capital - 1) >= 0.10:
            success_count += 1
            break

# 6. Final Report
print(f"\n{'='*60}")
print("   FINAL INSTITUTIONAL REPORT")
print(f"{'='*60}")
print(f"   💰 Final Balance: ${combined_equity[-1]:,.2f}")
print(f"   📈 Total Return: {total_ret:.2f}%")
print(f"   📉 Max Drawdown: {max_dd:.2f}%")
print(f"   🔢 Total Trades: {total_trades}")
print(f"   🚀 Prop Firm Success Probability: {success_count/10:.1f}%")
print(f"{'='*60}")
