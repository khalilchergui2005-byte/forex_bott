# ============================================================
# الذرة 1.3: دمج الفلاتر الأكاديمية في نظام واحد
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("   1.3 – دمج Time-of-Day + Weekly Bias")
print("=" * 60)

df = pd.read_csv("data/eurusd_hourly_dukascopy.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday

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

london_ny_hours = [7, 8, 12, 13]
time_filter = df['hour'].isin(london_ny_hours)
good_days = [1, 2, 4]
day_filter = df['weekday'].isin(good_days)

df['NoFilter'] = df['Final']
df['TimeOnly'] = 0
df.loc[(df['Final'] != 0) & time_filter, 'TimeOnly'] = df['Final']
df['DayOnly'] = 0
df.loc[(df['Final'] != 0) & day_filter, 'DayOnly'] = df['Final']
df['BothFilters'] = 0
df.loc[(df['Final'] != 0) & time_filter & day_filter, 'BothFilters'] = df['Final']

print(f"   إشارات بدون فلتر: {(df['NoFilter']!=0).sum()}")
print(f"   إشارات Time-of-Day فقط: {(df['TimeOnly']!=0).sum()}")
print(f"   إشارات Weekly Bias فقط: {(df['DayOnly']!=0).sum()}")
print(f"   إشارات الفلترين معاً: {(df['BothFilters']!=0).sum()}")

def backtest(signal_col):
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    trades = []
    for i in range(len(df)):
        p = df['close'].iloc[i]
        a = df['ATR'].iloc[i]
        s = df[signal_col].iloc[i]
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

r0, n0, w0, d0, p0 = backtest('NoFilter')
r1, n1, w1, d1, p1 = backtest('TimeOnly')
r2, n2, w2, d2, p2 = backtest('DayOnly')
r3, n3, w3, d3, p3 = backtest('BothFilters')

print(f"\n📊 النتائج النهائية:")
print(f"   {'':22} {'بدون فلتر':>10} {'Time-of-Day':>12} {'Weekly Bias':>12} {'الفلتران معاً':>14}")
print(f"   {'العائد':22} {r0:>9.2f}% {r1:>11.2f}% {r2:>11.2f}% {r3:>13.2f}%")
print(f"   {'الصفقات':22} {n0:>10} {n1:>12} {n2:>12} {n3:>14}")
print(f"   {'نسبة الفوز':22} {w0:>9.1f}% {w1:>11.1f}% {w2:>11.1f}% {w3:>13.1f}%")
print(f"   {'أقصى انخفاض':22} {d0:>9.2f}% {d1:>11.2f}% {d2:>11.2f}% {d3:>13.2f}%")
print(f"   {'Profit Factor':22} {p0:>9.2f} {p1:>11.2f} {p2:>11.2f} {p3:>13.2f}")

if p3 > p0 * 1.1 and d3 > d0:
    print(f"\n🎉 نجاح! الفلتران معاً أفضل بكثير من بدون فلتر.")
elif p3 >= p0 * 0.9:
    print(f"\n✅ الفلتران معاً مقبولان. يُحافظان على الأداء مع صفقات أنظف.")
else:
    print(f"\n⚠️ الفلتران معاً قللا الصفقات أكثر من اللازم.")

print("🎉 الذرة 1.3 اكتملت – النظام متعدد الطبقات جاهز.")
