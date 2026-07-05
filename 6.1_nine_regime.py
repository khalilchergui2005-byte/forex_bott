# ============================================================
# الذرة 6.1 (مُصلحة): مصفوفة 9 أنظمة + Bollinger Bands
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

print("📥 تحميل بيانات الساعة...")
train = pd.read_csv("data/train_hourly.csv", index_col=0)
test  = pd.read_csv("data/test_hourly.csv", index_col=0)
df_h1 = pd.concat([train, test]).sort_index()
df_h1.index = pd.to_datetime(df_h1.index, utc=True)
df = df_h1[['open','high','low','close']].copy()
print(f"✅ إجمالي شموع الساعة: {len(df)}")

# 1. التقلب والاتجاه (نفس المنطق)
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

# 2. قواعد النظام (معدلة: نسمح بالتداول في أنظمة أكثر)
# 0=منخفض/هابط, 1=منخفض/محايد, 2=منخفض/صاعد, 3=متوسط/هابط, ...
regime_rules = {
    0: -1,   # منخفض+هابط → بيع
    1: 0,    # منخفض+محايد → لا شيء
    2: 1,    # منخفض+صاعد → شراء
    3: -1,   # متوسط+هابط → بيع
    4: 0,    # متوسط+محايد → لا شيء
    5: 1,    # متوسط+صاعد → شراء
    6: -1,   # مرتفع+هابط → بيع
    7: 0,    # مرتفع+محايد → لا شيء
    8: 1,    # مرتفع+صاعد → شراء
}
df['Allowed'] = df['Regime9'].map(regime_rules)

# 3. مؤشرات دخول مرنة
df.dropna(inplace=True)
df['SMA_S'] = df['close'].rolling(10).mean()
df['SMA_L'] = df['close'].rolling(30).mean()

# Bollinger Bands
df['BB_mid'] = df['close'].rolling(20).mean()
df['BB_std'] = df['close'].rolling(20).std()
df['BB_lower'] = df['BB_mid'] - 1.5 * df['BB_std']
df['BB_upper'] = df['BB_mid'] + 1.5 * df['BB_std']

# RSI مخفف
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
df['RSI'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
df.dropna(inplace=True)

# إشارات متعددة (نستخدم أي إشارة متاحة)
df['Buy_SMA']  = ((df['SMA_S'] > df['SMA_L']) & (df['SMA_S'].shift(1) <= df['SMA_L'].shift(1))).astype(int)
df['Sell_SMA'] = ((df['SMA_S'] < df['SMA_L']) & (df['SMA_S'].shift(1) >= df['SMA_L'].shift(1))).astype(int)
df['Buy_BB']   = (df['low'] <= df['BB_lower']).astype(int)
df['Sell_BB']  = (df['high'] >= df['BB_upper']).astype(int)
df['Buy_RSI']  = (df['RSI'] < 48).astype(int)
df['Sell_RSI'] = (df['RSI'] > 52).astype(int)

# دمج: أي إشارة شراء + RSI يؤكد
df['Buy']  = ((df['Buy_SMA'] | df['Buy_BB']) & df['Buy_RSI']).astype(int)
df['Sell'] = ((df['Sell_SMA'] | df['Sell_BB']) & df['Sell_RSI']).astype(int)

# 4. الإشارات النهائية (بفلتر النظام)
df['Final'] = 0
df.loc[(df['Buy'] == 1) & (df['Allowed'].isin([1, 0])), 'Final'] = 1
df.loc[(df['Sell'] == 1) & (df['Allowed'].isin([-1, 0])), 'Final'] = -1

print(f"🔍 إجمالي الإشارات النهائية: {(df['Final']!=0).sum()}")

# 5. تقسيم
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
        p = data["close"].iloc[i]
        a = data["ATR"].iloc[i]
        s = data["Final"].iloc[i]
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
        return 0,0,0,0,0,0
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

print(f"\n📊 تدريب (أول 70%):")
print(f"💰 رأس المال: {cap_train:,.2f} $ | عائد: {ret_train:.2f}% | صفقات: {n_train} | فوز: {wr_train:.1f}% | DD: {dd_train:.2f}% | PF: {pf_train:.2f}")
print(f"\n📊 اختبار (آخر 30%):")
print(f"💰 رأس المال: {cap_test:,.2f} $ | عائد: {ret_test:.2f}% | صفقات: {n_test} | فوز: {wr_test:.1f}% | DD: {dd_test:.2f}% | PF: {pf_test:.2f}")

print("\n🔁 مقارنة مع HMM القديم (D1): 5 صفقات | 80% فوز | PF=28")
print(f"   المصفوفة 9 المصححة (H1): {n_test} صفقة | {wr_test:.1f}% فوز | PF={pf_test:.2f}")
print("🎉 الذرة 6.1 (مُصلحة) اكتملت.")
