# ============================================================
# الذرة 2.1: نظام تسجيل كامل (Trade Log)
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("   2.1 – نظام تسجيل كامل (Trade Log)")
print("=" * 60)

# تحميل البيانات وبناء المصفوفة مع الفلاتر
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
df['Signal'] = 0
df.loc[(df['Final'] != 0) & time_filter & day_filter, 'Signal'] = df['Final']

# محاكاة مع تسجيل كامل
capital = 10_000
pos = 0
entry = 0.0
entry_time = None
entry_regime = None
entry_vol = None
entry_hour = None
entry_day = None
units = 0.0
trades_log = []

for i in range(len(df)):
    p = df['close'].iloc[i]
    a = df['ATR'].iloc[i]
    s = df['Signal'].iloc[i]
    t = df.index[i]
    
    if pos and s and s != pos:
        exit_price = p
        pnl = (exit_price - entry) * pos * units
        cost = (units / 100_000) * 7
        net = pnl - cost
        capital += net
        
        # تسجيل كامل
        trades_log.append({
            'entry_time': entry_time,
            'exit_time': t,
            'direction': 'LONG' if pos == 1 else 'SHORT',
            'entry_price': entry,
            'exit_price': exit_price,
            'units': units,
            'net_pnl': net,
            'gross_pnl': pnl,
            'cost': cost,
            'rr_ratio': abs(net / (2.0 * a * units)) if a > 0 else 0,
            'entry_regime': entry_regime,
            'exit_regime': df['Regime9'].iloc[i],
            'vol_regime': entry_vol,
            'hour': entry_hour,
            'day': entry_day,
            'signal_source': 'Matrix',
            'filter_applied': 'Time-of-Day + Weekly Bias',
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
            entry_time = t
            entry_regime = df['Regime9'].iloc[i]
            entry_vol = df['VolClass'].iloc[i]
            entry_hour = df['hour'].iloc[i]
            entry_day = df['weekday'].iloc[i]

if trades_log:
    log_df = pd.DataFrame(trades_log)
    Path("logs").mkdir(exist_ok=True)
    log_df.to_csv("logs/trade_log_detailed.csv", index=False)
    
    print(f"✅ تم تسجيل {len(log_df)} صفقة في logs/trade_log_detailed.csv")
    print(f"\n📊 أعمدة السجل:")
    for col in log_df.columns:
        print(f"   - {col}")
    print(f"\n📋 أول صفقة مسجلة:")
    first = log_df.iloc[0]
    print(f"   دخول: {first['entry_time']} | خروج: {first['exit_time']}")
    print(f"   اتجاه: {first['direction']} | ربح صافي: {first['net_pnl']:.2f}$")
    print(f"   نظام الدخول: {int(first['entry_regime'])} | الساعة: {int(first['hour'])} | اليوم: {int(first['day'])}")
else:
    print("⚠️ لا توجد صفقات للتسجيل.")

print("🎉 الذرة 2.1 اكتملت – نظام التسجيل الكامل جاهز.")
