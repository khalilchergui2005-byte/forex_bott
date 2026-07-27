import optuna
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from spread_calculator import calculate_spread
from strategy_pairs_trading import generate_signals as gen_pairs
from pairs_backtest_engine import dynamic_pairs_backtest

print("=" * 60)
print("   PAIRS TRADING OPTIMIZER (Min Trades Constraint)")
print("=" * 60)

print("Loading and calculating base spread (5 Years)...")
base_df = calculate_spread()

window_size = len(base_df) // 4
windows = []
for i in range(3):
    start_test = (i + 1) * window_size
    end_test = start_test + window_size
    if end_test <= len(base_df):
        windows.append(base_df.iloc[start_test:end_test].copy())

def objective(trial):
    entry_z = trial.suggest_float("entry_z", 1.5, 2.5, step=0.1) # Lowered max to 2.5 to ensure trades
    exit_z = trial.suggest_float("exit_z", 0.0, 0.5, step=0.1)
    stop_z = trial.suggest_float("stop_z", 3.0, 5.0, step=0.5)
    lookback = trial.suggest_int("lookback", 50, 200, step=10)
    
    oos_pfs = []
    oos_trades = []
    try:
        for w in windows:
            # Recalculate spread for the window with the trial lookback
            from spread_calculator import load_pair_data
            df1 = load_pair_data("EURUSD").add_prefix("EURUSD_")
            df2 = load_pair_data("GBPUSD").add_prefix("GBPUSD_")
            df = pd.merge(df1, df2, left_index=True, right_index=True, how="inner")
            
            s1 = df["EURUSD_close"]
            s2 = df["GBPUSD_close"]
            cov = s1.rolling(lookback).cov(s2)
            var = s2.rolling(lookback).var()
            df["hedge_ratio"] = cov / var
            df["spread"] = s1 - (df["hedge_ratio"] * s2)
            rolling_mean = df["spread"].rolling(lookback).mean()
            rolling_std = df["spread"].rolling(lookback).std()
            df["z_score"] = (df["spread"] - rolling_mean) / (rolling_std + 1e-10)
            df.dropna(inplace=True)
            
            start_idx = w.index[0]
            end_idx = w.index[-1]
            df_w = df.loc[start_idx:end_idx].copy()
            
            df_w = gen_pairs(df_w, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)
            
            res = dynamic_pairs_backtest(data=df_w, log_trades=False)
            trades = res[0] if isinstance(res, tuple) else res
            
            # CONSTRAINT: Minimum 20 trades per window
            if not trades or len(trades) < 20:
                return -9999.0
                
            oos_trades.append(len(trades))
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t <= 0]
            pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
            oos_pfs.append(pf)

        avg_pf = np.mean(oos_pfs)
        std_pf = np.std(oos_pfs)
        # Reward high PF, penalize high variance
        score = avg_pf * 10 - (std_pf * 20)
        return score

    except Exception as e:
        return -9999.0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\nBest Institutional Parameters found (with min trades):")
best = study.best_params
for k, v in best.items():
    print(f"   {k}: {v}")
print(f"   Best Score: {study.best_value:.2f}")

with open('pairs_opt_params.txt', 'w') as f:
    for k, v in best.items():
        f.write(f"{k}={v}\n")
print("Saved to pairs_opt_params.txt")
