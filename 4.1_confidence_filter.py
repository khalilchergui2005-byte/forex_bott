import pandas as pd
import numpy as np
from pathlib import Path
import re

print("=" * 60)
print("   4.1 – تفعيل طبقة التنبؤ (مرشح الثقة)")
print("=" * 60)

print("تحميل بيانات EUR/USD (MT5 2025)...")
df = pd.read_csv("data/eurusd_mt5_2025.csv", sep=r'\s+', engine='python')
df.columns = [re.sub(r'[<>]', '', c) for c in df.columns]
df['Datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str), utc=True)
df = df.set_index('Datetime')
df = df.rename(columns={'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close'})
df = df[['open', 'high', 'low', 'close']].copy()
df.dropna(inplace=True)
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['date'] = df.index.date
print(f"   {len(df)} شمعة EUR/USD")

print("تحميل درجات الثقة...")
conf = pd.read_csv("data/eurusd_with_confidence.csv")
print(f"   أعمدة ملف الثقة: {list(conf.columns)}")

# التعامل مع أي تنسيق للتاريخ
date_col = None
for c in conf.columns:
    low = c.lower()
    if 'date' in low or 'time' in low or 'timestamp' in low:
        date_col = c
        break
if date_col is None:
    date_col = conf.columns[0]
    print(f"   لم أجد عمود تاريخ محدد، استخدمت: {date_col}")

conf['date_parsed'] = pd.to_datetime(conf[date_col], utc=True).dt.date
daily_conf = conf.groupby('date_parsed')['confidence'].mean()
print(f"   {len(daily_conf)} يوم مع درجة ثقة")

# المؤشرات
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR_ratio'] = df['ATR'] / df['close']
df['SMA_20'] = df['close'].rolling(20).mean()
df['Trend'] = 0
df.loc[df['close'] > df['SMA_20'] * 1.001, 'Trend'] = 1
df.loc[df['close'] < df['SMA_20'] * 0.999, 'Trend'] = -1
vol = df['ATR_ratio'].dropna()
df['VolClass'] = 1
df.loc[df['ATR_ratio'] <= vol.quantile(0.33), 'VolClass'] = 0
df.loc[df['ATR_ratio'] >= vol.quantile(0.66), 'VolClass'] = 2
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
buy_sma  = (df['SMA_S'] > df['SMA_L']) & (df['SMA_S'].shift(1) <= df['SMA_L'].shift(1))
sell_sma = (df['SMA_S'] < df['SMA_L']) & (df['SMA_S'].shift(1) >= df['SMA_L'].shift(1))
buy_bb   = df['low'] <= df['BB_lower']
sell_bb  = df['high'] >= df['BB_upper']
buy_rsi  = df['RSI'] < 48
sell_rsi = df['RSI'] > 52
buy  = (buy_sma | buy_bb) & buy_rsi
sell = (sell_sma | sell_bb) & sell_rsi
df['Final'] = 0
df.loc[buy & df['Allowed'].isin([1, 0]), 'Final'] = 1
df.loc[sell & df['Allowed'].isin([-1, 0]), 'Final'] = -1

london_ny_hours = [7, 8, 12]
time_filter = df['hour'].isin(london_ny_hours)
good_days = [1, 2, 4]
day_filter = df['weekday'].isin(good_days)
high_vol = df['VolClass'] == 2
df['Signal_Base'] = 0
df.loc[(df['Final'] != 0) & time_filter & day_filter, 'Signal_Base'] = df['Final']
df.loc[(df['Final'] != 0) & high_vol & day_filter, 'Signal_Base'] = df['Final']

df['Confidence'] = df['date'].map(daily_conf).fillna(0.5)
CONF_THRESH = 0.6
df['Signal'] = 0
df.loc[(df['Signal_Base'] != 0) & (df['Confidence'] >= CONF_THRESH), 'Signal'] = df['Signal_Base']

print(f"   إشارات بدون مرشح: {(df['Signal_Base']!=0).sum()}")
print(f"   إشارات مع مرشح: {(df['Signal']!=0).sum()}")

def backtest(sig_col):
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    trades = []
    for i in range(len(df)):
        p = df['close'].iloc[i]
        a = df['ATR'].iloc[i]
        s = df[sig_col].iloc[i]
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
        return 0, 0, 0, 0, 0
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    ret = (capital/10_000 - 1)*100
    dd = (pd.Series([10_000] + trades).cumsum()/10_000 - 1).min()*100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    return ret, n, wr, dd, pf

r0, n0, w0, d0, p0 = backtest('Signal_Base')
r1, n1, w1, d1, p1 = backtest('Signal')

print(f"\n📊 مقارنة:")
print(f"   {'':20} {'بدون مرشح':>12} {'مع مرشح الثقة':>14}")
print(f"   {'العائد':20} {r0:>11.2f}% {r1:>13.2f}%")
print(f"   {'الصفقات':20} {n0:>12} {n1:>14}")
print(f"   {'نسبة الفوز':20} {w0:>11.1f}% {w1:>13.1f}%")
print(f"   {'أقصى انخفاض':20} {d0:>11.2f}% {d1:>13.2f}%")
print(f"   {'Profit Factor':20} {p0:>11.2f} {p1:>13.2f}")

if p1 > p0 * 1.05:
    print(f"\n🎉 مرشح الثقة يُحسّن PF من {p0:.2f} إلى {p1:.2f}!")
elif p1 >= p0 * 0.95:
    print(f"\n✅ مرشح الثقة مقبول.")
else:
    print(f"\n⚠️ مرشح الثقة يضر النظام.")

print("🎉 الذرة 4.1 اكتملت.")
