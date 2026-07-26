"""
6.3_market_maker.py - صانع السوق الديناميكي (M15)
يستخدم DataHub للحصول على البيانات.
"""
import pandas as pd
from strategy_market_maker import generate_mm_signals
from backtest_engine_mm import dynamic_backtest_mm
from data_loader import load_m15

df = load_m15()
df = generate_mm_signals(df)
df.dropna(inplace=True)

split = int(0.7 * len(df))
train_df, test_df = df.iloc[:split], df.iloc[split:]

trades_train, equity_train = dynamic_backtest_mm(train_df, 10000)
trades_test, equity_test = dynamic_backtest_mm(test_df, 10000)

def calc_metrics(trades, equity, initial=10000):
    if not trades: return 0,0,0,0,0,initial
    wins, losses = [t for t in trades if t>0], [t for t in trades if t<=0]
    n = len(trades)
    wr = len(wins)/n*100
    ret = (equity[-1]/initial - 1)*100
    dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    return ret, n, wr, dd, pf, equity[-1]

ret_train, n_train, wr_train, dd_train, pf_train, cap_train = calc_metrics(trades_train, equity_train)
ret_test, n_test, wr_test, dd_test, pf_test, cap_test = calc_metrics(trades_test, equity_test)

print(f"\n📊 تدريب: {cap_train:,.2f} \$ | عائد: {ret_train:.2f}% | صفقات: {n_train} | فوز: {wr_train:.1f}% | DD: {dd_train:.2f}% | PF: {pf_train:.2f}")
print(f"📊 اختبار: {cap_test:,.2f} \$ | عائد: {ret_test:.2f}% | صفقات: {n_test} | فوز: {wr_test:.1f}% | DD: {dd_test:.2f}% | PF: {pf_test:.2f}")
print("🎉 الذرة 6.3 (متصلة بالمركز الموحد) اكتملت.")
