# ============================================================
# الذرة 5.5 (مصححة): Out-of-Sample حقيقي للجين الثالث
# التقسيم: تدريب 2015-2022 | اختبار 2023-2024
# ============================================================
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

print("📥 تحميل EUR/USD (2015-2024)...")
data = yf.download("EURUSD=X", start="2015-01-01", end="2024-12-31", progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
df = data[['Open','High','Low','Close']].copy()
df.columns = ['open','high','low','close']
df.dropna(inplace=True)

# تقسيم: تدريب 2015-2022 | اختبار 2023-2024
train_df = df.loc["2015-01-01":"2022-12-31"].copy()
test_df  = df.loc["2023-01-01":"2024-12-31"].copy()
print(f"✅ تدريب: {len(train_df)} شمعة (2015-2022) | اختبار: {len(test_df)} شمعة (2023-2024)")

# الجين الثالث
SMA_S, SMA_L = 21, 59
RSI_P, RSI_LOW, RSI_HIGH = 14, 33, 65
ATR_MULT = 0.5

def backtest(data, label=""):
    d = data.copy()
    d["SMA_S"] = d["close"].rolling(SMA_S).mean()
    d["SMA_L"] = d["close"].rolling(SMA_L).mean()
    d["ATR_dyn"] = (d["high"] - d["low"]).rolling(14).mean()
    delta = d["close"].diff()
    d["RSI_dyn"] = 100 - (100 / (1 + delta.clip(lower=0).rolling(RSI_P).mean() / (-delta.clip(upper=0)).rolling(RSI_P).mean()))
    d.dropna(inplace=True)
    
    d["Signal"] = 0
    d.loc[(d["SMA_S"] > d["SMA_L"]) & (d["SMA_S"].shift(1) <= d["SMA_L"].shift(1)), "Signal"] = 1
    d.loc[(d["SMA_S"] < d["SMA_L"]) & (d["SMA_S"].shift(1) >= d["SMA_L"].shift(1)), "Signal"] = -1
    d["RSI_Sig"] = 0
    d.loc[(d["RSI_dyn"] < RSI_LOW) & (d["close"] > d["SMA_L"]), "RSI_Sig"] = 1
    d.loc[(d["RSI_dyn"] > RSI_HIGH) & (d["close"] < d["SMA_L"]), "RSI_Sig"] = -1
    d["Final"] = 0
    d.loc[(d["Signal"] == 1) | (d["RSI_Sig"] == 1), "Final"] = 1
    d.loc[(d["Signal"] == -1) | (d["RSI_Sig"] == -1), "Final"] = -1
    
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    trades = []
    
    for i in range(len(d)):
        price = d["close"].iloc[i]
        atr_val = d["ATR_dyn"].iloc[i]
        sig = d["Final"].iloc[i]
        
        if pos and sig and sig != pos:
            pnl = (price - entry) * pos * units
            cost = (units / 100_000) * 7
            capital += (pnl - cost)
            trades.append(pnl - cost)
            pos = 0
            continue
        
        if pos == 0 and sig != 0:
            risk = capital * 0.01
            stop_dist = ATR_MULT * atr_val
            if stop_dist > 0.0001:
                units = min(risk / stop_dist, 50000)
                pos = sig
                entry = price
    
    if not trades:
        return 0, 0, 0, 0, 0, 0
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    total = sum(trades)
    ret = (capital/10_000 - 1)*100
    dd = (pd.Series([10_000] + trades).cumsum()/10_000 - 1).min()*100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    return ret, n, wr, dd, pf, capital

ret_train, n_train, wr_train, dd_train, pf_train, cap_train = backtest(train_df)
ret_test, n_test, wr_test, dd_test, pf_test, cap_test = backtest(test_df)

print(f"\n📊 تدريب (2015-2022):")
print(f"💰 رأس المال: {cap_train:,.2f} $ | عائد: {ret_train:.2f}% | صفقات: {n_train} | فوز: {wr_train:.1f}% | DD: {dd_train:.2f}% | PF: {pf_train:.2f}")

print(f"\n📊 اختبار Out-of-Sample (2023-2024):")
print(f"💰 رأس المال: {cap_test:,.2f} $ | عائد: {ret_test:.2f}% | صفقات: {n_test} | فوز: {wr_test:.1f}% | DD: {dd_test:.2f}% | PF: {pf_test:.2f}")

print(f"\n🔁 الحكم:")
if ret_test > 15 and pf_test > 1.4 and dd_test > -10:
    print("✅ الجين الثالث صامد على بيانات لم يرها. جاهز للاستخدام الحي.")
elif ret_test > 0 and pf_test > 1.0:
    print("⚠️ الجين يعمل لكن أداؤه أضعف. يمكن استخدامه بحجم مخاطرة 0.5%.")
else:
    print("❌ الجين فشل في Out-of-Sample. يحتاج إلى جولة رابعة.")

print("🎉 الذرة 5.5 اكتملت.")
