"""
correlation_test.py - Correlation Matrix for Three Candidate Strategies (Atom 8.10c)
Measures daily return correlation between Long_Lower_Shadow, High_ATR, and Hour_00.
"""
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_long_lower_shadow import generate_long_lower_shadow_signals
from strategy_high_atr import generate_high_atr_signals
from strategy_hour00 import generate_hour00_signals
from backtest_engine import dynamic_backtest

if __name__ == '__main__':
    print("=" * 60)
    print("   STRATEGY CORRELATION TEST")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/3] Loading data...")
    df = load_hourly()
    
    # 2. Generate daily returns for each strategy
    print("[2/3] Calculating daily returns...")
    strategies = {
        'Long_Lower_Shadow': (generate_long_lower_shadow_signals, ['Signal']),
        'High_ATR': (generate_high_atr_signals, ['Signal']),
        'Hour_00': (generate_hour00_signals, ['Signal']),
    }
    
    daily_returns = {}
    for name, (func, cols) in strategies.items():
        df_signals = func(df.copy())
        trades, equity = dynamic_backtest(
            data=df_signals,
            strategy_cols=cols,
            risk_percent=0.005
        )
        # Convert equity to daily returns
        equity_series = pd.Series(equity[1:], index=df_signals.index)
        daily_equity = equity_series.resample('D').last().dropna()
        daily_returns[name] = daily_equity.pct_change().dropna()
        print(f"   {name}: {len(trades)} trades, Avg Return: {daily_returns[name].mean()*100:.4f}%")
    
    # 3. Correlation matrix
    print("\n[3/3] Correlation Matrix:")
    df_returns = pd.DataFrame(daily_returns)
    corr_matrix = df_returns.corr()
    
    print("\n" + "=" * 60)
    print(corr_matrix.to_string())
    print("=" * 60)
    
    # Interpretation
    print("\n📊 Interpretation:")
    if corr_matrix.max().max() < 0.3:
        print("   ✅ Strategies are UNCORRELATED. Combining them will add REAL diversification.")
        print("   Proceed to Atom 8.10d (Smart Portfolio Allocation).")
    else:
        print("   ⚠️ Some strategies are correlated (>0.3). Combining them may not add diversification.")
        print("   Consider using only the BEST performer (Long_Lower_Shadow) as core.")
