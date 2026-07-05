# ============================================================
# الذرة 1.0 (مصححة): Time-of-Day كفلتر – وليس كاستراتيجية
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("   1.0 – Time-of-Day كفلتر للمصفوفة H1")
print("=" * 60)

# 1. تحميل البيانات
df = pd.read_csv("data/eurusd_hourly_dukascopy.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
df['hour'] = df.index.hour
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
buy_sma  = (df['SMA_S'] > df['SMA_L']) & (df['SMA_S'].shift(1) <= df['SMA_L'].shift(1))
sell_sma = (df['SMA_S'] < df['SMA_L']) & (df['SMA_S'].shift(1) >= df['SMA_L'].shift(1))
buy_bb   = df['low'] <= df['BB_lower']
sell_bb  = df['high'] >= df['BB_upper']
buy_rsi  = df['RSI'] < 48
sell_rsi = df['RSI'] > 52
buy  = (buy_sma | buy_bb) & buy_rsi
sell = (sell_sma | sell_bb) & sell_rsi

# إشارات بدون فلتر
df['Final_NoFilter'] = 0
df.loc[buy & df['Allowed'].isin([1, 0]), 'Final_NoFilter'] = 1
df.loc[sell & df['Allowed'].isin([-1, 0]), 'Final_NoFilter'] = -1

# فلتر Time-of-Day: London (7-8 UTC) + New York (12-13 UTC)
london_ny_hours = [7, 8, 12, 13]
time_filter = df['hour'].isin(london_ny_hours)

# إشارات مع الفلتر
df['Final_Filtered'] = 0
df.loc[(df['Final_NoFilter'] == 1) & time_filter, 'Final_Filtered'] = 1
df.loc[(df['Final_NoFilter'] == -1) & time_filter, 'Final_Filtered'] = -1

print(f"   إشارات بدون فلتر: {(df['Final_NoFilter']!=0).sum()}")
print(f"   إشارات مع فلتر Time-of-Day: {(df['Final_Filtered']!=0).sum()}")

# 2. اختبار مقارن
def backtest(signal_col, label):
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

ret_no, n_no, wr_no, dd_no, pf_no, cap_no = backtest('Final_NoFilter', 'بدون فلتر')
ret_f, n_f, wr_f, dd_f, pf_f, cap_f = backtest('Final_Filtered', 'مع فلتر Time-of-Day')

print(f"\n📊 مقارنة:")
print(f"   {'':20} {'بدون فلتر':>12} {'مع الفلتر':>12}")
print(f"   {'العائد':20} {ret_no:>11.2f}% {ret_f:>11.2f}%")
print(f"   {'الصفقات':20} {n_no:>12} {n_f:>12}")
print(f"   {'نسبة الفوز':20} {wr_no:>11.1f}% {wr_f:>11.1f}%")
print(f"   {'أقصى انخفاض':20} {dd_no:>11.2f}% {dd_f:>11.2f}%")
print(f"   {'Profit Factor':20} {pf_no:>11.2f} {pf_f:>11.2f}")

if pf_f > pf_no and dd_f > dd_no:
    print(f"\n✅ فلتر Time-of-Day يُحسّن المصفوفة! PF ارتفع من {pf_no:.2f} إلى {pf_f:.2f}.")
elif pf_f > 0 and pf_f > pf_no * 0.8:
    print(f"\n⚠️ فلتر Time-of-Day مقبول. PF انخفض قليلاً لكن DD تحسن.")
else:
    print(f"\n❌ فلتر Time-of-Day يضر المصفوفة. PF انخفض.")

print("🎉 الذرة 1.0 (مصححة) اكتملت.")
