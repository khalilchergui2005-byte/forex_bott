"""
walk_forward_optimizer.py - Institutional Walk-Forward Validation (0.2.5)
Project: Viral Entity - Strongest Trading Firm
Proves strategy stability across multiple independent time windows.
"""
import optuna
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from strategy_range_hunter import generate_range_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params
import copy, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   🔬 WALK-FORWARD VALIDATION (0.2.5)")
print("=" * 60)

# 1. Setup
WINDOW_TRAIN = 2000
WINDOW_TEST = 1000
WINDOW_STEP = 500
NUM_WINDOWS = 5

print(f"\n⚙️ Configuration: {NUM_WINDOWS} windows | Train={WINDOW_TRAIN} | Test={WINDOW_TEST} | Step={WINDOW_STEP}")

# 2. Load and prepare data
print("\n📥 Loading H1 data...")
full_data = load_hourly()
# Ensure all required signal generators work
full_data = generate_signals(full_data)
full_data = generate_momentum_signals(full_data)
full_data = generate_range_signals(full_data)
full_data['Final_Mom'] = full_data['MO_Signal']
full_data['Final_Range'] = full_data['Range_Signal']
print(f"   ✅ Data ready: {len(full_data)} candles")

# 3. Walk-Forward Loop
results = []
best_params_overall = None
best_score_overall = -9999

for w in range(NUM_WINDOWS):
    start_train = w * WINDOW_STEP
    end_train = start_train + WINDOW_TRAIN
    start_test = end_train
    end_test = start_test + WINDOW_TEST
    
    if end_test > len(full_data):
        break
        
    train_data = full_data.iloc[start_train:end_train].copy()
    test_data = full_data.iloc[start_test:end_test].copy()
    
    print(f"\n{'='*40}")
    print(f"📊 Window {w+1}: Train [{start_train}:{end_train}] → Test [{start_test}:{end_test}]")
    
    def objective(trial):
        bb_mult = trial.suggest_float("bb_multiplier", 1.0, 2.5, step=0.1)
        rsi_buy = trial.suggest_int("rsi_buy_threshold", 30, 50)
        rsi_sell = trial.suggest_int("rsi_sell_threshold", 50, 70)
        macd_fast = trial.suggest_int("macd_fast", 8, 20)
        macd_slow = trial.suggest_int("macd_slow", 20, 40)
        adx_period = trial.suggest_int("adx_period", 10, 25)
        adx_threshold = trial.suggest_int("adx_threshold", 20, 35)

        original = {k: getattr(params, k) for k in ['bb_multiplier', 'rsi_buy_threshold', 'rsi_sell_threshold', 'macd_fast', 'macd_slow', 'adx_period', 'adx_threshold']}
        for k, v in zip(original.keys(), [bb_mult, rsi_buy, rsi_sell, macd_fast, macd_slow, adx_period, adx_threshold]):
            setattr(params, k, v)

        try:
            df = train_data.copy()
            df = generate_signals(df)
            df = generate_momentum_signals(df)
            df = generate_range_signals(df)
            df['Final_Mom'] = df['MO_Signal']
            df['Final_Range'] = df['Range_Signal']
            
            trades, equity = dynamic_backtest(data=df, strategy_cols=['Final', 'Final_Mom', 'Final_Range'], risk_percent=0.005)
            
            for k, v in original.items():
                setattr(params, k, v)
                
            if not trades or len(trades) < 30:
                return -9999.0
                
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t <= 0]
            pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 10.0
            total_return = (equity[-1] / params.initial_capital) - 1
            max_dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min()
            penalty = max(0, abs(max_dd) - 0.02) * 500
            
            return pf * 20 + total_return * 200 - penalty
        except Exception:
            for k, v in original.items():
                setattr(params, k, v)
            return -9999.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best = study.best_params
    
    # Test on OOS
    for k, v in best.items():
        setattr(params, k, v)
    df_test = test_data.copy()
    df_test = generate_signals(df_test)
    df_test = generate_momentum_signals(df_test)
    df_test = generate_range_signals(df_test)
    df_test['Final_Mom'] = df_test['MO_Signal']
    df_test['Final_Range'] = df_test['Range_Signal']
    trades_oos, eq_oos = dynamic_backtest(data=df_test, strategy_cols=['Final', 'Final_Mom', 'Final_Range'], risk_percent=0.005)
    
    if trades_oos and len(trades_oos) > 0:
        wins = [t for t in trades_oos if t > 0]
        losses = [t for t in trades_oos if t <= 0]
        pf_oos = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0
        ret_oos = (eq_oos[-1]/params.initial_capital - 1)*100
        dd_oos = (pd.Series(eq_oos)/pd.Series(eq_oos).cummax() - 1).min()*100
        
        results.append({'window': w+1, 'pf_train': study.best_value, 'pf_oos': pf_oos, 'return_oos': ret_oos, 'dd_oos': dd_oos, 'trades': len(trades_oos)})
        print(f"   OOS PF: {pf_oos:.2f} | Return: {ret_oos:.2f}% | MaxDD: {dd_oos:.2f}% | Trades: {len(trades_oos)}")
    else:
        results.append({'window': w+1, 'pf_train': study.best_value, 'pf_oos': 0, 'return_oos': 0, 'dd_oos': 0, 'trades': 0})
        print("   ❌ No trades in this window!")

# 4. Summary Report
print(f"\n{'='*60}")
print("   📋 WALK-FORWARD FINAL REPORT")
print(f"{'='*60}")
if results:
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print(f"\n📊 Averages across {len(results)} windows:")
    print(f"   PF OOS: {df_results['pf_oos'].mean():.2f} (±{df_results['pf_oos'].std():.2f})")
    print(f"   Return OOS: {df_results['return_oos'].mean():.2f}%")
    print(f"   MaxDD OOS: {df_results['dd_oos'].mean():.2f}%")
    print(f"   Trades OOS: {df_results['trades'].mean():.0f}")
    
    stability = "✅ STABLE" if df_results['pf_oos'].mean() > 1.0 and df_results['pf_oos'].std() < 0.5 else "⚠️ UNSTABLE"
    print(f"\n🎯 Stability Verdict: {stability}")
    print("   (Stable = Avg PF > 1.0 & Std < 0.5)")
else:
    print("❌ No results to show.")

print(f"\n{'='*60}")
print("Walk-Forward validation complete.")


