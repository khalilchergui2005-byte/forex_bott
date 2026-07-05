# ============================================================
# الذرة 0.1: الأرقام الأربعة المفقودة (نسخة مصححة)
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   🩺 تشريح مصفوفة H1 – الأرقام الأربعة المفقودة")
print("=" * 60)

# 1. تحميل البيانات وبناء المصفوفة
train = pd.read_csv("data/train_hourly.csv", index_col=0)
test  = pd.read_csv("data/test_hourly.csv", index_col=0)
df = pd.concat([train, test]).sort_index()
df.index = pd.to_datetime(df.index, utc=True)
df = df[['open','high','low','close']].copy()

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

df['Final'] = 0
df.loc[buy & df['Allowed'].isin([1, 0]), 'Final'] = 1
df.loc[sell & df['Allowed'].isin([-1, 0]), 'Final'] = -1

# 2. محاكاة مع تسجيل تفاصيل كل صفقة
capital = 10_000
pos = 0
entry = 0.0
entry_time = None
units = 0.0
trades = []

for i in range(len(df)):
    p = df['close'].iloc[i]
    a = df['ATR'].iloc[i]
    s = df['Final'].iloc[i]
    t = df.index[i]
    
    if pos and s and s != pos:
        exit_price = p
        pnl = (exit_price - entry) * pos * units
        cost = (units / 100_000) * 7
        net = pnl - cost
        
        stop_distance = 2.0 * entry_atr
        risk_amount = stop_distance * units
        reward_amount = net if net > 0 else 0
        risk_actual = abs(net) if net < 0 else risk_amount
        rr_ratio = reward_amount / risk_actual if risk_actual > 0 else 0
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': t,
            'direction': 'LONG' if pos == 1 else 'SHORT',
            'entry_price': entry,
            'exit_price': exit_price,
            'units': units,
            'net_pnl': net,
            'rr_ratio': rr_ratio,
            'regime': entry_regime,
            'year': entry_time.year,
            'month': entry_time.month
        })
        capital += net
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
            entry_atr = a
            entry_regime = df['Regime9'].iloc[i]

if not trades:
    print("❌ لا توجد صفقات في المصفوفة.")
    exit()

trades_df = pd.DataFrame(trades)

# ============================================================
# الرقم 1: متوسط R:R
# ============================================================
print("\n" + "=" * 60)
print("   1️⃣  متوسط R:R (نسبة المخاطرة إلى العائد)")
print("=" * 60)
wins = trades_df[trades_df['net_pnl'] > 0]
losses = trades_df[trades_df['net_pnl'] <= 0]
avg_rr = trades_df['rr_ratio'].mean()
avg_win_rr = wins['rr_ratio'].mean() if len(wins) > 0 else 0
avg_loss_rr = losses['rr_ratio'].mean() if len(losses) > 0 else 0

print(f"   عدد الصفقات الكلي: {len(trades_df)}")
print(f"   الصفقات الرابحة: {len(wins)} ({len(wins)/len(trades_df)*100:.1f}%)")
print(f"   الصفقات الخاسرة: {len(losses)} ({len(losses)/len(trades_df)*100:.1f}%)")
print(f"   متوسط R:R: {avg_rr:.2f}")
print(f"   متوسط R:R للرابحة: {avg_win_rr:.2f}")
print(f"   متوسط R:R للخاسرة: {avg_loss_rr:.2f}")
print(f"   متوسط الربح: {wins['net_pnl'].mean():.2f} $" if len(wins) > 0 else "   لا صفقات رابحة")
print(f"   متوسط الخسارة: {losses['net_pnl'].mean():.2f} $" if len(losses) > 0 else "   لا صفقات خاسرة")

if avg_win_rr > avg_loss_rr and avg_rr > 0.5:
    print("   ✅ الحكم: R:R صحي – الصفقات الرابحة أكبر من الخاسرة")
else:
    print("   ❌ الحكم: R:R معطوب – Win Rate بدون R:R جيد = وهم")

# ============================================================
# الرقم 2: التوزيع الزمني للصفقات
# ============================================================
print("\n" + "=" * 60)
print("   2️⃣  التوزيع الزمني للصفقات")
print("=" * 60)
yearly = trades_df.groupby('year').agg(
    trades=('net_pnl', 'count'),
    total_pnl=('net_pnl', 'sum'),
    win_rate=('net_pnl', lambda x: (x > 0).mean() * 100)
)
print(yearly.to_string())

covid_trades = trades_df[trades_df['year'].isin([2020, 2021])]
covid_pct = len(covid_trades) / len(trades_df) * 100
print(f"\n   نسبة الصفقات في 2020-2021 (COVID): {covid_pct:.1f}%")
if covid_pct > 50:
    print("   ❌ تحذير: أكثر من نصف الصفقات في فترة COVID.")
else:
    print("   ✅ التوزيع الزمني مقبول.")

# ============================================================
# الرقم 3: ارتباط الأنظمة الثلاثة
# ============================================================
print("\n" + "=" * 60)
print("   3️⃣  ارتباط الأنظمة الثلاثة")
print("=" * 60)
# استخدام نفس الفهرس
df['MM_Active'] = ((df['Trend'] == 0) & (df['VolClass'] == 0)).astype(int)
daily = df.resample('D').agg({
    'Final': 'sum',
    'MM_Active': 'max'
}).fillna(0)
daily.columns = ['Matrix', 'MM']

# HMM تقديري: إشارات اليومي
hmm_daily = df['Final'].resample('D').apply(lambda x: x.mean()).fillna(0)
daily['HMM'] = hmm_daily

corr_matrix = daily.corr()
print(corr_matrix)
if corr_matrix.abs().max().max() > 0.7:
    print("   ❌ تحذير: ارتباط عالي بين الأنظمة.")
else:
    print("   ✅ ارتباط مقبول بين الأنظمة.")

# ============================================================
# الرقم 4: Capital Adequacy
# ============================================================
print("\n" + "=" * 60)
print("   4️⃣  Capital Adequacy (كفاية رأس المال)")
print("=" * 60)
ACCOUNT_SIZE = 10_000
TARGET_PROFIT = 0.10
num_years = trades_df['year'].max() - trades_df['year'].min() + 1
avg_trades_per_month = len(trades_df) / (num_years * 12)
avg_commission_per_trade = 7
monthly_commission = avg_trades_per_month * avg_commission_per_trade
monthly_commission_pct = monthly_commission / ACCOUNT_SIZE * 100
required_gross_profit = ACCOUNT_SIZE * TARGET_PROFIT
required_net_profit = required_gross_profit + monthly_commission
required_net_pct = required_net_profit / ACCOUNT_SIZE * 100

print(f"   متوسط الصفقات شهرياً: {avg_trades_per_month:.1f}")
print(f"   إجمالي التكاليف الشهرية: {monthly_commission:.2f} $ ({monthly_commission_pct:.1f}% من رأس المال)")
print(f"   الربح الإجمالي المطلوب لـ 10% صافي: {required_net_profit:.2f} $ ({required_net_pct:.1f}%)")
if required_net_pct > 15:
    print("   ❌ تحذير: تحتاج ربح إجمالي > 15% لتحقيق 10% صافي.")
else:
    print("   ✅ Capital Adequacy مقبول.")

print("\n🎉 الذرة 0.1 اكتملت – الأرقام الأربعة بين يديك الآن.")
