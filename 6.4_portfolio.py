# ============================================================
# الذرة 6.4: دمج HMM اليومي + المصفوفة ذات 9 أنظمة على H1 (مصححة)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path

print("📥 تحميل بيانات اليومي + الساعة...")
# اليومي
df_d1 = pd.read_csv("data/eurusd_daily.csv", index_col=0, parse_dates=True)
df_d1.index = pd.to_datetime(df_d1.index, utc=True)

# الساعة
train_h1 = pd.read_csv("data/train_hourly.csv", index_col=0)
test_h1  = pd.read_csv("data/test_hourly.csv", index_col=0)
df_h1 = pd.concat([train_h1, test_h1]).sort_index()
df_h1.index = pd.to_datetime(df_h1.index, utc=True)
print(f"✅ يومي: {len(df_d1)} | ساعة: {len(df_h1)}")

# ============================================================
# 1. HMM اليومي (القناص)
# ============================================================
df_d1["Return"] = df_d1["close"].pct_change()
df_d1["Range"] = (df_d1["high"] - df_d1["low"]) / df_d1["close"]
df_d1["Volatility"] = df_d1["Return"].rolling(20).std()
df_d1["Momentum"] = df_d1["close"].pct_change(10)
df_d1.dropna(inplace=True)

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=5)
df_d1["Regime"] = gmm.fit_predict(df_d1[["Return", "Range", "Volatility", "Momentum"]])
state_returns = df_d1.groupby("Regime")["Return"].mean().sort_values()
df_d1["Regime"] = df_d1["Regime"].map({state_returns.index[0]: -1, state_returns.index[1]: 0, state_returns.index[2]: 1})

df_d1["SMA_20"] = df_d1["close"].rolling(20).mean()
df_d1["SMA_50"] = df_d1["close"].rolling(50).mean()
df_d1["Signal"] = 0
df_d1.loc[(df_d1["SMA_20"] > df_d1["SMA_50"]) & (df_d1["SMA_20"].shift(1) <= df_d1["SMA_50"].shift(1)), "Signal"] = 1
df_d1.loc[(df_d1["SMA_20"] < df_d1["SMA_50"]) & (df_d1["SMA_20"].shift(1) >= df_d1["SMA_50"].shift(1)), "Signal"] = -1
df_d1["HMM_Final"] = 0
df_d1.loc[(df_d1["Signal"] == 1) & (df_d1["Regime"].isin([0, 1])), "HMM_Final"] = 1
df_d1.loc[(df_d1["Signal"] == -1) & (df_d1["Regime"].isin([-1, 0])), "HMM_Final"] = -1
df_d1.dropna(inplace=True)
print(f"   HMM اليومي: {(df_d1['HMM_Final']!=0).sum()} إشارة")

# ============================================================
# 2. المصفوفة ذات 9 أنظمة على H1 (المهاجم)
# ============================================================
df = df_h1[['open','high','low','close']].copy()
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR_ratio'] = df['ATR'] / df['close']
df['SMA_20'] = df['close'].rolling(20).mean()
df['Trend'] = 0
df.loc[df['close'] > df['SMA_20'] * 1.001, 'Trend'] = 1
df.loc[df['close'] < df['SMA_20'] * 0.999, 'Trend'] = -1
vol = df['ATR_ratio'].dropna()
low_thresh  = vol.quantile(0.33)
high_thresh = vol.quantile(0.66)
df['VolClass'] = 1
df.loc[df['ATR_ratio'] <= low_thresh, 'VolClass'] = 0
df.loc[df['ATR_ratio'] >= high_thresh, 'VolClass'] = 2
df['Regime9'] = df['VolClass'] * 3 + (df['Trend'] + 1)

regime_rules = {0: -1, 1: 0, 2: 1, 3: -1, 4: 0, 5: 1, 6: -1, 7: 0, 8: 1}
df['Allowed'] = df['Regime9'].map(regime_rules)

df.dropna(inplace=True)
df['SMA_S'] = df['close'].rolling(10).mean()
df['SMA_L'] = df['close'].rolling(30).mean()
df['BB_mid'] = df['close'].rolling(20).mean()
df['BB_std'] = df['close'].rolling(20).std()
df['BB_lower'] = df['BB_mid'] - 1.5 * df['BB_std']
df['BB_upper'] = df['BB_mid'] + 1.5 * df['BB_std']
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
df['RSI'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
df.dropna(inplace=True)

df['Buy_SMA']  = ((df['SMA_S'] > df['SMA_L']) & (df['SMA_S'].shift(1) <= df['SMA_L'].shift(1))).astype(int)
df['Sell_SMA'] = ((df['SMA_S'] < df['SMA_L']) & (df['SMA_S'].shift(1) >= df['SMA_L'].shift(1))).astype(int)
df['Buy_BB']   = (df['low'] <= df['BB_lower']).astype(int)
df['Sell_BB']  = (df['high'] >= df['BB_upper']).astype(int)
df['Buy_RSI']  = (df['RSI'] < 48).astype(int)
df['Sell_RSI'] = (df['RSI'] > 52).astype(int)
df['Buy']  = ((df['Buy_SMA'] | df['Buy_BB']) & df['Buy_RSI']).astype(int)
df['Sell'] = ((df['Sell_SMA'] | df['Sell_BB']) & df['Sell_RSI']).astype(int)
df['Final'] = 0
df.loc[(df['Buy'] == 1) & (df['Allowed'].isin([1, 0])), 'Final'] = 1
df.loc[(df['Sell'] == 1) & (df['Allowed'].isin([-1, 0])), 'Final'] = -1
print(f"   المصفوفة H1: {(df['Final']!=0).sum()} إشارة")

# ============================================================
# 3. دمج الإشارات في محفظة واحدة (مصحح)
# ============================================================
df['DateOnly'] = df.index.date
df_d1['DateOnly'] = df_d1.index.date
df = df.merge(df_d1[['DateOnly', 'HMM_Final', 'Regime']], on='DateOnly', how='left', suffixes=('', '_D1'))
df['HMM_Final'] = df['HMM_Final'].fillna(0)
df['Regime_D1'] = df['Regime'].fillna(0)

# القاعدة: نستخدم إشارة المصفوفة فقط إذا كانت متوافقة مع HMM اليومي
df['Portfolio'] = 0
df.loc[(df['Final'] == 1) & (df['Regime_D1'].isin([0, 1])), 'Portfolio'] = 1
df.loc[(df['Final'] == -1) & (df['Regime_D1'].isin([-1, 0])), 'Portfolio'] = -1
print(f"   المحفظة المدمجة: {(df['Portfolio']!=0).sum()} إشارة")

# ============================================================
# 4. Backtest موحد
# ============================================================
capital = 10_000
pos = 0
entry = 0.0
units = 0.0
trades = []
equity = [capital]

for i in range(len(df)):
    price = df['close'].iloc[i]
    atr_val = df['ATR'].iloc[i]
    sig = df['Portfolio'].iloc[i]
    
    if pos and sig and sig != pos:
        pnl = (price - entry) * pos * units
        cost = (units / 100_000) * 7
        capital += (pnl - cost)
        trades.append(pnl - cost)
        pos = 0
        continue
    
    if pos == 0 and sig != 0:
        risk = capital * 0.01
        stop_d = 2.0 * atr_val
        if stop_d > 0.0001:
            units = min(risk / stop_d, 50000)
            pos = sig
            entry = price
    
    equity.append(capital)

if trades:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    ret = (capital/10_000 - 1)*100
    dd_pct = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    
    print(f"\n📊 نتائج المحفظة المتكاملة (HMM + المصفوفة):")
    print(f"💰 رأس المال: 10,000 -> {capital:,.2f} $")
    print(f"📈 العائد: {ret:.2f}%")
    print(f"📉 أقصى انخفاض: {dd_pct:.2f}%")
    print(f"🔢 الصفقات: {n}")
    print(f"✅ فوز: {wr:.1f}%")
    print(f"⚖️ Profit Factor: {pf:.2f}")
    
    print(f"\n🔁 مقارنة:")
    print(f"   HMM اليومي وحده: 5 صفقات | 80% فوز | PF=28")
    print(f"   المصفوفة H1 وحدها: 75 صفقة | 76% فوز | PF=2.02")
    print(f"   المحفظة المدمجة:   {n} صفقة | {wr:.1f}% فوز | PF={pf:.2f}")
else:
    print("⚠️ لم تُنفذ أي صفقة.")

print("🎉 الذرة 6.4 اكتملت.")
