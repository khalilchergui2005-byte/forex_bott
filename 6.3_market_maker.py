import pandas as pd
import numpy as np
from pathlib import Path
from strategy_market_maker import generate_mm_signals
from backtest_engine_mm import dynamic_backtest_mm

print("📥 تحميل بيانات الساعة...")
train = pd.read_csv("data/train_hourly.csv", index_col=0)
test  = pd.read_csv("data/test_hourly.csv", index_col=0)
df_h1 = pd.concat([train, test]).sort_index()
df_h1.index = pd.to_datetime(df_h1.index, utc=True)
df = df_h1[['open','high','low','close']].copy()

df_m15 = df.resample('15min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()
print(f"✅ شموع 15 دقيقة: {len(df_m15)}")

df_m15 = generate_mm_signals(df_m15)
df_m15.dropna(inplace=True)

# تقسيم البيانات
split = int(0.7 * len(df_m15))
train_df = df_m15.iloc[:split]
test_df  = df_m15.iloc[split:]

print("🚀 بدء الاختبار الخلفي الديناميكي...")
trades_train, equity_train = dynamic_backtest_mm(train_df, 10000)
trades_test, equity_test = dynamic_backtest_mm(test_df, 10000)

def calc_metrics(trades, equity, initial=10000):
    if not trades:
        return 0, 0, 0, 0, 0, initial
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    total = sum(trades)
    ret = (equity[-1]/initial - 1)*100
    dd_pct = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    return ret, n, wr, dd_pct, pf, equity[-1]

ret_train, n_train, wr_train, dd_train, pf_train, cap_train = calc_metrics(trades_train, equity_train)
ret_test, n_test, wr_test, dd_test, pf_test, cap_test = calc_metrics(trades_test, equity_test)

print(f"\n📊 نتائج صانع السوق الديناميكي (M15):")
print(f"💰 تدريب: {cap_train:,.2f} $ | عائد: {ret_train:.2f}% | صفقات: {n_train} | فوز: {wr_train:.1f}% | DD: {dd_train:.2f}% | PF: {pf_train:.2f}")
print(f"💰 اختبار: {cap_test:,.2f} $ | عائد: {ret_test:.2f}% | صفقات: {n_test} | فوز: {wr_test:.1f}% | DD: {dd_test:.2f}% | PF: {pf_test:.2f}")
print("🎉 الذرة 6.3 (صانع السوق الديناميكي) اكتملت.")
