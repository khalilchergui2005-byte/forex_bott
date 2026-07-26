import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("📥 تحميل بيانات الساعة...")
train = pd.read_csv("data/train_hourly.csv", index_col=0)
test  = pd.read_csv("data/test_hourly.csv", index_col=0)
df_h1 = pd.concat([train, test]).sort_index()
df_h1.index = pd.to_datetime(df_h1.index, utc=True)
print(f"✅ إجمالي شموع الساعة: {len(df_h1)}")

# إعادة تجميع إلى 4 ساعات
df = df_h1.resample('4h').agg({
    'open':  'first',
    'high':  'max',
    'low':   'min',
    'close': 'last'
}).dropna()
print(f"✅ شموع 4 ساعات: {len(df)}")

# الميزات
df["Return"] = df["close"].pct_change()
df["Range"] = (df["high"] - df["low"]) / df["close"]
df["Volatility"] = df["Return"].rolling(20).std()
df["Momentum"] = df["close"].pct_change(10)
df.dropna(inplace=True)

# HMM
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=5)
df["Regime"] = gmm.fit_predict(df[["Return", "Range", "Volatility", "Momentum"]])
sr = df.groupby("Regime")["Return"].mean().sort_values()
df["Regime"] = df["Regime"].map({sr.index[0]: -1, sr.index[1]: 0, sr.index[2]: 1})

# مؤشرات
df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
df["RSI"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
df.dropna(inplace=True)

# الإشارات
df["Signal"] = 0
df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["SMA_20"].shift(1) <= df["SMA_50"].shift(1)), "Signal"] = 1
df.loc[(df["SMA_20"] < df["SMA_50"]) & (df["SMA_20"].shift(1) >= df["SMA_50"].shift(1)), "Signal"] = -1

df["Final"] = 0
df.loc[(df["Signal"] == 1) & (df["Regime"].isin([0, 1])), "Final"] = 1
df.loc[(df["Signal"] == -1) & (df["Regime"].isin([-1, 0])), "Final"] = -1

print(f"🔍 الإشارات الخام: {(df['Signal']!=0).sum()} | بعد HMM: {(df['Final']!=0).sum()}")

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
        price = data["close"].iloc[i]
        atr   = data["ATR"].iloc[i]
        sig   = data["Final"].iloc[i]
        if pos and sig and sig != pos:
            pnl = (price - entry) * pos * units
            cost = (units / 100_000) * 7
            net = pnl - cost
            capital += net
            trades.append(net)
            pos = 0
            continue
        if pos == 0 and sig != 0:
            risk = capital * 0.01
            stop_d = 2.0 * atr
            if stop_d > 0.0001:
                units = min(risk / stop_d, 50000)
                pos = sig
                entry = price
    if not trades:
        return 0, 0, 0, 0, 0, 0
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

print(f"\n📊 نتائج H4 على التدريب:")
print(f"💰 رأس المال: {cap_train:,.2f} $ | عائد: {ret_train:.2f}% | صفقات: {n_train} | فوز: {wr_train:.1f}% | DD: {dd_train:.2f}% | PF: {pf_train:.2f}")

print(f"\n📊 نتائج H4 على الاختبار:")
print(f"💰 رأس المال: {cap_test:,.2f} $ | عائد: {ret_test:.2f}% | صفقات: {n_test} | فوز: {wr_test:.1f}% | DD: {dd_test:.2f}% | PF: {pf_test:.2f}")

print("\n🔁 مقارنة مع اليومي (HMM D1):")
print("   اليومي: ربح 1.15% | 5 صفقات | فوز 80% | PF=28.10 | DD=-0.04%")
print(f"   H4:     ربح {ret_test:.2f}% | {n_test} صفقة | فوز {wr_test:.1f}% | PF={pf_test:.2f} | DD={dd_test:.2f}%")
print("🎉 الذرة 5.4 اكتملت.")
