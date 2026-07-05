# ============================================================
# الذرة 1.1 (مصححة): News Drift كمُضخم حجم للمصفوفة
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("   1.1 – News Drift كمُضخم حجم للمصفوفة H1")
print("=" * 60)

df = pd.read_csv("data/eurusd_hourly_dukascopy.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
df['hour'] = df.index.hour
df['date'] = df.index.date

news_dates = [
    "2024-01-05","2024-02-02","2024-03-01","2024-04-05","2024-05-03","2024-06-07",
    "2024-07-05","2024-08-02","2024-09-06","2024-10-04","2024-11-01","2024-12-06",
    "2025-01-03","2025-02-07","2025-03-07","2025-04-04","2025-05-02","2025-06-06",
    "2025-07-04","2025-08-01","2025-09-05","2025-10-03","2025-11-07","2025-12-05",
    "2026-01-02","2026-02-06","2026-03-06","2026-04-03","2026-05-01","2026-06-05",
    "2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15","2024-06-12",
    "2024-07-11","2024-08-14","2024-09-11","2024-10-10","2024-11-13","2024-12-11",
    "2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-14","2025-06-11",
    "2025-07-16","2025-08-13","2025-09-11","2025-10-15","2025-11-13","2025-12-11",
    "2026-01-14","2026-02-12","2026-03-12","2026-04-15","2026-05-14","2026-06-11",
]
news_dates = set(pd.to_datetime(news_dates).date)

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

# اختبار: حجم عادي vs حجم مضاعف في أيام الأخبار
def backtest(news_multiplier=1.0):
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    trades = []
    for i in range(len(df)):
        p = df['close'].iloc[i]
        a = df['ATR'].iloc[i]
        s = df['Filtered'].iloc[i]
        is_news_day = df.index[i].date() in news_dates
        mult = news_multiplier if is_news_day else 1.0
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
                units = min(risk / stop_d * mult, 50000)
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
    return ret, n, wr, dd, pf, total

ret1, n1, wr1, dd1, pf1, tot1 = backtest(1.0)
ret2, n2, wr2, dd2, pf2, tot2 = backtest(1.5)
ret3, n3, wr3, dd3, pf3, tot3 = backtest(2.0)

print(f"\n📊 مقارنة مضاعف حجم الأخبار:")
print(f"   {'المضاعف':<10} {'العائد':>10} {'الصفقات':>8} {'الفوز':>8} {'DD':>10} {'PF':>8} {'الربح':>10}")
print(f"   {'1.0x':<10} {ret1:>9.2f}% {n1:>8} {wr1:>7.1f}% {dd1:>9.2f}% {pf1:>7.2f} {tot1:>9.0f}$")
print(f"   {'1.5x':<10} {ret2:>9.2f}% {n2:>8} {wr2:>7.1f}% {dd2:>9.2f}% {pf2:>7.2f} {tot2:>9.0f}$")
print(f"   {'2.0x':<10} {ret3:>9.2f}% {n3:>8} {wr3:>7.1f}% {dd3:>9.2f}% {pf3:>7.2f} {tot3:>9.0f}$")

if tot2 > tot1 or pf2 > pf1:
    print(f"\n✅ News Drift كمُضخم حجم يُحسّن النتائج!")
else:
    print(f"\n⚠️ News Drift كمُضخم حجم لا يُضيف قيمة واضحة.")

print("🎉 الذرة 1.1 (مصححة) اكتملت.")
