# ============================================================
# الذرة 5.1: Out-of-Sample Test للجين الفائز (2015-2019)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
import yfinance as yf

# 1. تحميل بيانات 2015-2019 (لم يرها الجين مطلقاً)
print("📥 تحميل EUR/USD (2015-2019) ...")
data = yf.download("EURUSD=X", start="2015-01-01", end="2019-12-31", progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
df = data[['Open','High','Low','Close']].copy()
df.columns = ['open','high','low','close']
df.dropna(inplace=True)
print(f"✅ تم تحميل {len(df)} شمعة يومية")

# 2. تحضير الميزات (نفس منطق الجين)
df["Return"] = df["close"].pct_change()
df["Range"] = (df["high"] - df["low"]) / df["close"]
df["Volatility"] = df["Return"].rolling(20).std()
df["Momentum"] = df["close"].pct_change(10)
df.dropna(inplace=True)

# HMM لتصنيف السوق (يُطبّق على هذه البيانات الجديدة)
gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
df["Regime"] = gmm.fit_predict(df[["Return","Range","Volatility","Momentum"]])
state_returns = df.groupby("Regime")["Return"].mean().sort_values()
df["Regime"] = df["Regime"].map({state_returns.index[0]: -1, state_returns.index[1]: 0, state_returns.index[2]: 1})

# 3. الجين الفائز (ثوابت)
SMA_S, SMA_L = 16, 65
RSI_P, RSI_LOW, RSI_HIGH = 20, 30, 68
ATR_MULT = 1.1
REGIME_FILTER = True

df["SMA_S"] = df["close"].rolling(SMA_S).mean()
df["SMA_L"] = df["close"].rolling(SMA_L).mean()
df["ATR_dyn"] = (df["high"] - df["low"]).rolling(14).mean()

delta = df["close"].diff()
df["RSI_dyn"] = 100 - (100 / (1 + delta.clip(lower=0).rolling(RSI_P).mean() / (-delta.clip(upper=0)).rolling(RSI_P).mean()))
df.dropna(inplace=True)

# الإشارات
df["Signal"] = 0
df.loc[(df["SMA_S"] > df["SMA_L"]) & (df["SMA_S"].shift(1) <= df["SMA_L"].shift(1)), "Signal"] = 1
df.loc[(df["SMA_S"] < df["SMA_L"]) & (df["SMA_S"].shift(1) >= df["SMA_L"].shift(1)), "Signal"] = -1

df["RSI_Sig"] = 0
df.loc[(df["RSI_dyn"] < RSI_LOW) & (df["close"] > df["SMA_L"]), "RSI_Sig"] = 1
df.loc[(df["RSI_dyn"] > RSI_HIGH) & (df["close"] < df["SMA_L"]), "RSI_Sig"] = -1

df["Final"] = 0
df.loc[(df["Signal"] == 1) | (df["RSI_Sig"] == 1), "Final"] = 1
df.loc[(df["Signal"] == -1) | (df["RSI_Sig"] == -1), "Final"] = -1

if REGIME_FILTER:
    df.loc[(df["Final"] == 1) & (df["Regime"] < 0), "Final"] = 0
    df.loc[(df["Final"] == -1) & (df["Regime"] > 0), "Final"] = 0

# 4. المحاكاة مع تكاليف IBKR الحقيقية (نقطة أساسية)
capital = 10_000.0
pos = 0
entry = 0.0
units = 0.0
trades = []

for i in range(len(df)):
    price = df["close"].iloc[i]
    atr_val = df["ATR_dyn"].iloc[i]
    sig = df["Final"].iloc[i]

    if pos and sig and sig != pos:
        pnl = (price - entry) * pos * units
        # تكلفة IBKR: حوالي 0.7 نقطة سبريد + عمولة 5$ لكل لوت
        cost = (units / 100_000) * 5 * 2 + 0.00007 * units
        net = pnl - cost
        capital += net
        trades.append(net)
        pos = 0
        continue

    if pos == 0 and sig != 0:
        risk = capital * 0.01
        stop_dist = ATR_MULT * atr_val
        if stop_dist > 0.0001:
            units = min(risk / stop_dist, 50000)
            pos = sig
            entry = price

# 5. النتائج
if trades:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    total_pnl = sum(trades)
    ret = (capital/10_000 - 1)*100
    dd = (pd.Series([10_000] + trades).cumsum()/10_000 - 1).min()*100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)

    print(f"\n📊 Out-of-Sample (2015-2019) - الجين الفائز:")
    print(f"💰 رأس المال النهائي: {capital:,.2f} $")
    print(f"📈 العائد: {ret:.2f}%")
    print(f"📉 أقصى انخفاض: {dd:.2f}%")
    print(f"🔢 عدد الصفقات: {n}")
    print(f"✅ نسبة الفوز: {wr:.1f}%")
    print(f"⚖️ Profit Factor: {pf:.2f}")

    if ret > 25 and pf > 1.4 and dd > -15:
        print("\n✅ الجين صلب! أداؤه حقيقي حتى على بيانات لم يرها.")
    elif ret > 0 and pf > 1.0:
        print("\n⚠️ الجين يعمل لكن أداؤه أضعف. قد يكون هناك overfitting خفيف.")
    else:
        print("\n❌ الجين انهار على بيانات 2015-2019. Overfitting مؤكد. يجب إعادة تطويره.")
else:
    print("❌ لم تُنفذ أي صفقة. الجين فشل في هذا الإطار الزمني.")

print("🎉 الذرة 5.1 اكتملت.")
