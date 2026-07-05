import pandas as pd
import numpy as np
from pathlib import Path
from strategy_matrix import generate_signals

print("📥 تحميل بيانات الساعة...")
train = pd.read_csv("data/train_hourly.csv", index_col=0)
test  = pd.read_csv("data/test_hourly.csv", index_col=0)
df_h1 = pd.concat([train, test]).sort_index()
df_h1.index = pd.to_datetime(df_h1.index, utc=True)
df = df_h1[['open','high','low','close']].copy()
print(f"✅ إجمالي شموع الساعة: {len(df)}")

# استدعاء عقل الاستراتيجية (مصدر الحقيقة الوحيد)
df = generate_signals(df)
print(f"🔍 إجمالي الإشارات النهائية: {(df['Final']!=0).sum()}")

# تقسيم
split = int(0.7 * len(df))
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

def backtest(data):
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    trades = []
    for i in range(len(data)):
        p = data["close"].iloc[i]
        a = data["ATR"].iloc[i]
        s = data["Final"].iloc[i]
        if pos and s and s != pos:
            pnl = (p - entry) * pos * units
            cost = (units / 100_000) * 7
            capital += (pnl - cost)
            trades.append(pnl - cost)
            pos = 0
            continue
        if pos == 0 and s != 0:
            risk = capital * 0.01
            stop_d = 2.0 * a
            if stop_d > 0.0001:
                units = min(risk / stop_d, 50000)
                pos = s
                entry = p
    if not trades:
        return 0,0,0,0,0,0
    wins = [t for t in trades if t>0]
    losses = [t for t in trades if t<=0]
    n = len(trades)
    wr = len(wins)/n*100
    total = sum(trades)
    ret = (capital/10_000-1)*100
    dd = (pd.Series([10_000]+trades).cumsum()/10_000-1).min()*100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    return ret, n, wr, dd, pf, capital

ret_train, n_train, wr_train, dd_train, pf_train, cap_train = backtest(train_df)
ret_test, n_test, wr_test, dd_test, pf_test, cap_test = backtest(test_df)

print(f"\n📊 تدريب (أول 70%):")
print(f"💰 رأس المال: {cap_train:,.2f} $ | عائد: {ret_train:.2f}% | صفقات: {n_train} | فوز: {wr_train:.1f}% | DD: {dd_train:.2f}% | PF: {pf_train:.2f}")
print(f"\n📊 اختبار (آخر 30%):")
print(f"💰 رأس المال: {cap_test:,.2f} $ | عائد: {ret_test:.2f}% | صفقات: {n_test} | فوز: {wr_test:.1f}% | DD: {dd_test:.2f}% | PF: {pf_test:.2f}")
print("\n🔁 مقارنة مع HMM القديم (D1): 5 صفقات | 80% فوز | PF=28")
print(f"   المصفوفة 9 المصححة (H1): {n_test} صفقة | {wr_test:.1f}% فوز| PF={pf_test:.2f}")
print("🎉 الذرة 6.1 (مُصلحة) اكتملت.")
