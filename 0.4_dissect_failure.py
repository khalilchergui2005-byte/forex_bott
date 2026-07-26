# ============================================================
# الذرة 0.4: تشريح فشل Out-of-Sample 2025
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
import re

print("=" * 70)
print("   🩺 تشريح فشل Out-of-Sample 2025")
print("=" * 70)

# تحميل بيانات MT5
df = pd.read_csv("data/eurusd_mt5_2025.csv", sep=r'\s+', engine='python')
df.columns = [re.sub(r'[<>]', '', c) for c in df.columns]
df['Datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
df = df.set_index('Datetime')
df = df.rename(columns={'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close'})
df = df[['open', 'high', 'low', 'close']].copy()
df.dropna(inplace=True)
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['month'] = df.index.month

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

# الفلاتر (نفس النظام)
london_ny_hours = [7, 8, 12, 13]
time_filter = df['hour'].isin(london_ny_hours)
good_days = [1, 2, 4]
day_filter = df['weekday'].isin(good_days)
high_vol = df['VolClass'] == 2
df['Signal'] = 0
df.loc[(df['Final'] != 0) & time_filter & day_filter, 'Signal'] = df['Final']
df.loc[(df['Final'] != 0) & high_vol & day_filter, 'Signal'] = df['Final']

# محاكاة مع تسجيل كل صفقة
capital = 10_000
pos = 0
entry = 0.0
entry_regime = None
entry_hour = None
entry_day = None
entry_month = None
units = 0.0
trades = []

for i in range(len(df)):
    p = df['close'].iloc[i]
    a = df['ATR'].iloc[i]
    s = df['Signal'].iloc[i]
    if pos and s and s != pos:
        pnl = (p - entry) * pos * units
        cost = (units / 100_000) * 7
        net = pnl - cost
        capital += net
        trades.append({
            'pnl': net,
            'regime': entry_regime,
            'hour': entry_hour,
            'day': entry_day,
            'month': entry_month,
            'direction': 'LONG' if pos == 1 else 'SHORT'
        })
        pos = 0
        continue
    if pos == 0 and s != 0:
        risk = capital * 0.01
        stop_d = 2.0 * a
        if stop_d > 0.0001:
            units = min(risk / stop_d, 50000)
            pos = s
            entry = p
            entry_regime = df['Regime9'].iloc[i]
            entry_hour = df['hour'].iloc[i]
            entry_day = df['weekday'].iloc[i]
            entry_month = df['month'].iloc[i]

if not trades:
    print("لا توجد صفقات للتحليل.")
    exit()

trades_df = pd.DataFrame(trades)

# 1. تحليل حسب خلية المصفوفة
print("\n" + "=" * 70)
print("   1️⃣  الأداء حسب خلية المصفوفة (Regime9)")
print("=" * 70)
regime_names = {
    0: 'منخفض/هابط', 1: 'منخفض/محايد', 2: 'منخفض/صاعد',
    3: 'متوسط/هابط', 4: 'متوسط/محايد', 5: 'متوسط/صاعد',
    6: 'مرتفع/هابط', 7: 'مرتفع/محايد', 8: 'مرتفع/صاعد'
}
for regime in range(9):
    reg_trades = trades_df[trades_df['regime'] == regime]
    if len(reg_trades) > 0:
        total = reg_trades['pnl'].sum()
        wins = (reg_trades['pnl'] > 0).sum()
        wr = wins / len(reg_trades) * 100
        print(f"   {regime_names[regime]:<18} {len(reg_trades):>4} صفقة | ربح={total:>8.0f}$ | فوز={wr:>5.1f}%")
    else:
        print(f"   {regime_names[regime]:<18}    0 صفقة")

# 2. تحليل حسب يوم الأسبوع
print("\n" + "=" * 70)
print("   2️⃣  الأداء حسب يوم الأسبوع")
print("=" * 70)
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
for day in range(7):
    day_trades = trades_df[trades_df['day'] == day]
    if len(day_trades) > 0:
        total = day_trades['pnl'].sum()
        wins = (day_trades['pnl'] > 0).sum()
        wr = wins / len(day_trades) * 100
        print(f"   {day_names[day]:<12} {len(day_trades):>4} صفقة | ربح={total:>8.0f}$ | فوز={wr:>5.1f}%")
    else:
        print(f"   {day_names[day]:<12}    0 صفقة")

# 3. تحليل حسب الساعة
print("\n" + "=" * 70)
print("   3️⃣  الأداء حسب ساعة التداول")
print("=" * 70)
for hour in range(24):
    hour_trades = trades_df[trades_df['hour'] == hour]
    if len(hour_trades) > 0:
        total = hour_trades['pnl'].sum()
        wins = (hour_trades['pnl'] > 0).sum()
        wr = wins / len(hour_trades) * 100
        print(f"   {hour:02d}:00 UTC        {len(hour_trades):>4} صفقة | ربح={total:>8.0f}$ | فوز={wr:>5.1f}%")

# 4. تحليل حسب الشهر
print("\n" + "=" * 70)
print("   4️⃣  الأداء حسب الشهر")
print("=" * 70)
for month in range(1, 13):
    month_trades = trades_df[trades_df['month'] == month]
    if len(month_trades) > 0:
        total = month_trades['pnl'].sum()
        wins = (month_trades['pnl'] > 0).sum()
        wr = wins / len(month_trades) * 100
        print(f"   شهر {month:02d}          {len(month_trades):>4} صفقة | ربح={total:>8.0f}$ | فوز={wr:>5.1f}%")

# 5. الخلاصة
print("\n" + "=" * 70)
print("   🎯 الخلاصة – أين تكمن المشكلة؟")
print("=" * 70)
worst_regime = trades_df.groupby('regime')['pnl'].sum().idxmin()
worst_day = trades_df.groupby('day')['pnl'].sum().idxmin()
print(f"   أسوأ خلية مصفوفة: {regime_names[worst_regime]}")
print(f"   أسوأ يوم تداول: {day_names[worst_day]}")
print(f"   إجمالي الصفقات: {len(trades_df)}")
print(f"   إجمالي الربح/الخسارة: {trades_df['pnl'].sum():,.0f}$")
print(f"   متوسط الربح للرابحة: {trades_df[trades_df['pnl']>0]['pnl'].mean():,.0f}$")
print(f"   متوسط الخسارة للخاسرة: {trades_df[trades_df['pnl']<=0]['pnl'].mean():,.0f}$")

print("\n🎉 انتهى التشريح. هذه هي الحقائق.")
