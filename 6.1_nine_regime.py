"""
6.1_nine_regime.py - مصفوفة الأنظمة التسعة (ديناميكية)
تستخدم DataHub للحصول على البيانات.
"""
import pandas as pd
from pathlib import Path
from strategy_matrix import generate_signals
from backtest_engine import dynamic_backtest
from data_loader import load_hourly

print("📥 تحميل البيانات من المركز الموحد...")
df = load_hourly()
df = generate_signals(df)
print(f"🔍 إجمالي الإشارات النهائية: {(df['Final']!=0).sum()}")

split = int(0.7 * len(df))
train_df, test_df = df.iloc[:split], df.iloc[split:]

trades_train, equity_train = dynamic_backtest(train_df, 10000)
trades_test, equity_test = dynamic_backtest(test_df, 10000)

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

print(f"\n📊 تدريب (أول 70%):")
print(f"💰 رأس المال: {cap_train:,.2f} \$ | عائد: {ret_train:.2f}% | صفقات: {n_train} | فوز: {wr_train:.1f}% | DD: {dd_train:.2f}% | PF: {pf_train:.2f}")
print(f"\n📊 اختبار (آخر 30%):")
print(f"💰 رأس المال: {cap_test:,.2f} \$ | عائد: {ret_test:.2f}% | صفقات: {n_test} | فوز: {wr_test:.1f}% | DD: {dd_test:.2f}% | PF: {pf_test:.2f}")
print("\n🎉 الذرة 6.1 (متصلة بالمركز الموحد) اكتملت.")
