import pandas as pd
import numpy as np
from pathlib import Path

print("📥 تحميل بيانات الساعة...")
train = pd.read_csv("data/train_hourly.csv", index_col=0)
test  = pd.read_csv("data/test_hourly.csv", index_col=0)
df_h1 = pd.concat([train, test]).sort_index()
df_h1.index = pd.to_datetime(df_h1.index, utc=True)
df = df_h1[['open','high','low','close']].copy()

df_m15 = df.resample('15min').agg({
    'open':  'first', 'high':  'max', 'low':   'min', 'close': 'last'
}).dropna()
print(f"✅ شموع 15 دقيقة: {len(df_m15)}")

df_m15['ATR'] = (df_m15['high'] - df_m15['low']).rolling(14).mean()
df_m15['ATR_ratio'] = df_m15['ATR'] / df_m15['close']
df_m15['SMA_20'] = df_m15['close'].rolling(20).mean()
df_m15['Trend'] = 0
df_m15.loc[df_m15['close'] > df_m15['SMA_20'] * 1.001, 'Trend'] = 1
df_m15.loc[df_m15['close'] < df_m15['SMA_20'] * 0.999, 'Trend'] = -1

vol = df_m15['ATR_ratio'].dropna()
low_thresh = vol.quantile(0.33)
df_m15['VolClass'] = 1
df_m15.loc[df_m15['ATR_ratio'] <= low_thresh, 'VolClass'] = 0

# Bollinger Bands للمسافة من المنتصف
df_m15['BB_mid'] = df_m15['close'].rolling(20).mean()
df_m15['BB_std'] = df_m15['close'].rolling(20).std()
df_m15['BB_upper'] = df_m15['BB_mid'] + 2 * df_m15['BB_std']
df_m15['BB_lower'] = df_m15['BB_mid'] - 2 * df_m15['BB_std']
df_m15['BB_distance'] = (df_m15['close'] - df_m15['BB_mid']).abs() / df_m15['BB_std']

# شروط التنشيط مشددة
df_m15['MM_Active'] = (
    (df_m15['Trend'] == 0) &
    (df_m15['VolClass'] == 0) &  # فقط تقلب منخفض
    (df_m15['BB_distance'] < 0.5)  # السعر قريب من المنتصف
)

SPREAD_TARGET = 0.0005  # 5 نقاط
STOP_MULT = 1.5
POS_SIZE = 2000

df_m15.dropna(inplace=True)

capital = 10_000
pos = 0
entry = 0.0
direction = 0
trades = []
equity = [capital]

for i in range(1, len(df_m15)):
    price = df_m15['close'].iloc[i]
    active = df_m15['MM_Active'].iloc[i]

    if pos != 0:
        target = entry + direction * SPREAD_TARGET
        stop   = entry - direction * (SPREAD_TARGET * STOP_MULT)
        close_trade = False
        if direction == 1:
            if price >= target or price <= stop: close_trade = True
        else:
            if price <= target or price >= stop: close_trade = True
        if not active: close_trade = True

        if close_trade:
            pnl = (price - entry) * direction * POS_SIZE
            cost = (POS_SIZE / 100_000) * 7
            capital += (pnl - cost)
            trades.append(pnl - cost)
            pos = 0

    if pos == 0 and active:
        direction = 1 if i % 2 == 0 else -1
        entry = price
        pos = direction

    equity.append(capital)

if trades:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    total = sum(trades)
    ret = (capital/10_000 - 1)*100
    dd_pct = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    
    print(f"\n📊 نتائج صانع السوق الذكي (M15):")
    print(f"💰 رأس المال: 10,000 -> {capital:,.2f} $")
    print(f"📈 العائد: {ret:.2f}%")
    print(f"📉 أقصى انخفاض: {dd_pct:.2f}%")
    print(f"🔢 الصفقات: {n}")
    print(f"✅ فوز: {wr:.1f}%")
    print(f"⚖️ Profit Factor: {pf:.2f}")
else:
    print("⚠️ لم تُنفذ أي صفقة.")

print("🎉 الذرة 6.3 (مُصلحة) اكتملت.")
