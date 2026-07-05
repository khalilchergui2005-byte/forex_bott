# ============================================================
# الذرة 1.2: Weekly Bias – أيام الأسبوع المواتية للمصفوفة
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("   1.2 – Weekly Bias (أداء المصفوفة حسب يوم الأسبوع)")
print("=" * 60)

df = pd.read_csv("data/eurusd_hourly_dukascopy.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday  # 0=Monday, 6=Sunday
df['date'] = df.index.date

# مؤشرات المصفوفة
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

# فلتر Time-of-Day
london_ny_hours = [7, 8, 12, 13]
time_filter = df['hour'].isin(london_ny_hours)
df['Filtered'] = 0
df.loc[(df['Final'] == 1) & time_filter, 'Filtered'] = 1
df.loc[(df['Final'] == -1) & time_filter, 'Filtered'] = -1

# 1. تحليل الأداء حسب يوم الأسبوع
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
print(f"\n📊 أداء المصفوفة حسب يوم الأسبوع:")
print(f"   {'اليوم':<12} {'صفقات':>8} {'ربح ($)':>12} {'فوز':>8}")

day_stats = {}
for day in range(5):
    day_data = df[df['weekday'] == day]
    if len(day_data) == 0:
        continue
    # محاكاة سريعة لهذا اليوم فقط
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    day_trades = []
    day_signals = day_data[day_data['Filtered'] != 0]
    
    for i in range(len(day_data)):
        p = day_data['close'].iloc[i]
        a = day_data['ATR'].iloc[i]
        s = day_data['Filtered'].iloc[i]
        if pos and s and s != pos:
            pnl = (p - entry) * pos * units
            cost = (units / 100_000) * 7
            capital += (pnl - cost)
            day_trades.append(pnl - cost)
            pos = 0
            continue
        if pos == 0 and s != 0:
            risk = capital * 0.01
            stop_d = 2.0 * a
            if stop_d > 0.0001:
                units = min(risk / stop_d, 50000)
                pos = s
                entry = p
    
    if day_trades:
        wins = [t for t in day_trades if t > 0]
        wr = len(wins)/len(day_trades)*100
        total = sum(day_trades)
        day_stats[day] = {'trades': len(day_trades), 'pnl': total, 'wr': wr, 'signals': len(day_signals)}
        print(f"   {day_names[day]:<12} {len(day_trades):>8} {total:>11.0f}$ {wr:>7.1f}%")
    else:
        day_stats[day] = {'trades': 0, 'pnl': 0, 'wr': 0, 'signals': len(day_signals)}
        print(f"   {day_names[day]:<12} {0:>8} {0:>11}$ {0:>7.1f}%")

# 2. تحديد الأيام الجيدة (ربح > 0 أو عدد صفقات > المتوسط)
avg_pnl = np.mean([s['pnl'] for s in day_stats.values() if s['trades'] > 0])
good_days = [day for day, s in day_stats.items() if s['pnl'] > 0 or (s['trades'] > 0 and s['pnl'] > avg_pnl * 0.5)]
bad_days = [day for day in range(5) if day not in good_days]

print(f"\n   ✅ أيام جيدة: {[day_names[d] for d in good_days]}")
print(f"   ❌ أيام تجنب: {[day_names[d] for d in bad_days]}")

# 3. اختبار الفلتر
# بدون فلتر الأسبوع
capital = 10_000
pos = 0
entry = 0.0
units = 0.0
trades_no = []
for i in range(len(df)):
    p = df['close'].iloc[i]
    a = df['ATR'].iloc[i]
    s = df['Filtered'].iloc[i]
    if pos and s and s != pos:
        pnl = (p - entry) * pos * units
        cost = (units / 100_000) * 7
        capital += (pnl - cost)
        trades_no.append(pnl - cost)
        pos = 0
        continue
    if pos == 0 and s != 0:
        risk = capital * 0.01
        stop_d = 2.0 * a
        if stop_d > 0.0001:
            units = min(risk / stop_d, 50000)
            pos = s
            entry = p

# مع فلتر الأسبوع
capital2 = 10_000
pos = 0
entry = 0.0
units = 0.0
trades_yes = []
for i in range(len(df)):
    p = df['close'].iloc[i]
    a = df['ATR'].iloc[i]
    s = df['Filtered'].iloc[i]
    wd = df['weekday'].iloc[i]
    # لا تتداول في الأيام السيئة
    if wd in bad_days:
        s = 0
    if pos and s and s != pos:
        pnl = (p - entry) * pos * units
        cost = (units / 100_000) * 7
        capital2 += (pnl - cost)
        trades_yes.append(pnl - cost)
        pos = 0
        continue
    if pos == 0 and s != 0:
        risk = capital2 * 0.01
        stop_d = 2.0 * a
        if stop_d > 0.0001:
            units = min(risk / stop_d, 50000)
            pos = s
            entry = p

def calc_metrics(trades, cap):
    if not trades:
        return 0, 0, 0, 0, 0
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    ret = (cap/10_000 - 1)*100
    dd = (pd.Series([10_000] + trades).cumsum()/10_000 - 1).min()*100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    return ret, n, wr, dd, pf

ret_no, n_no, wr_no, dd_no, pf_no = calc_metrics(trades_no, capital)
ret_yes, n_yes, wr_yes, dd_yes, pf_yes = calc_metrics(trades_yes, capital2)

print(f"\n📊 مقارنة:")
print(f"   {'':20} {'بدون فلتر الأسبوع':>18} {'مع فلتر الأسبوع':>18}")
print(f"   {'العائد':20} {ret_no:>17.2f}% {ret_yes:>17.2f}%")
print(f"   {'الصفقات':20} {n_no:>18} {n_yes:>18}")
print(f"   {'نسبة الفوز':20} {wr_no:>17.1f}% {wr_yes:>17.1f}%")
print(f"   {'أقصى انخفاض':20} {dd_no:>17.2f}% {dd_yes:>17.2f}%")
print(f"   {'Profit Factor':20} {pf_no:>17.2f} {pf_yes:>17.2f}")

if pf_yes > pf_no * 1.05:
    print(f"\n✅ فلتر Weekly Bias يُحسّن المصفوفة!")
elif pf_yes >= pf_no * 0.95:
    print(f"\n⚠️ فلتر Weekly Bias محايد. لا يضر ولا ينفع.")
else:
    print(f"\n❌ فلتر Weekly Bias يضر المصفوفة.")

print("🎉 الذرة 1.2 اكتملت.")
