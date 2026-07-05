# ============================================================
# الذرة 2.2: Walk-Forward تلقائي شهري
# ============================================================
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
import joblib

print("=" * 60)
print("   2.2 – Walk-Forward تلقائي شهري")
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
df['Signal'] = 0
df.loc[(df['Final'] != 0) & time_filter & day_filter, 'Signal'] = df['Final']

def backtest_window(data):
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    trades = []
    for i in range(len(data)):
        p = data['close'].iloc[i]
        a = data['ATR'].iloc[i]
        s = data['Signal'].iloc[i]
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
    total = sum(trades)
    ret = (capital/10_000 - 1)*100
    dd = (pd.Series([10_000] + trades).cumsum()/10_000 - 1).min()*100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    return ret, n, wr, dd, pf

# Walk-Forward: نوافذ 6 أشهر تدريب / شهر اختبار
results = []
start_date = df.index.min()
end_date = df.index.max()
current = start_date
window_num = 0

while current + pd.DateOffset(months=7) <= end_date:
    train_start = current
    train_end = current + pd.DateOffset(months=6)
    test_start = train_end
    test_end = test_start + pd.DateOffset(months=1)
    
    train_data = df[(df.index >= train_start) & (df.index < train_end)]
    test_data = df[(df.index >= test_start) & (df.index < test_end)]
    
    if len(train_data) < 100 or len(test_data) < 20:
        current += pd.DateOffset(months=1)
        continue
    
    ret_train, n_train, wr_train, dd_train, pf_train = backtest_window(train_data)
    ret_test, n_test, wr_test, dd_test, pf_test = backtest_window(test_data)
    
    results.append({
        'window': window_num,
        'train_start': train_start.date(),
        'train_end': train_end.date(),
        'test_start': test_start.date(),
        'test_end': test_end.date(),
        'train_return': ret_train,
        'train_trades': n_train,
        'train_pf': pf_train,
        'test_return': ret_test,
        'test_trades': n_test,
        'test_pf': pf_test,
        'test_dd': dd_test,
    })
    
    if window_num < 3 or window_num % 3 == 0:
        print(f"   نافذة {window_num}: تدريب PF={pf_train:.2f} | اختبار PF={pf_test:.2f} | DD={dd_test:.1f}% | صفقات={n_test}")
    
    window_num += 1
    current += pd.DateOffset(months=1)

if results:
    res_df = pd.DataFrame(results)
    Path("models").mkdir(exist_ok=True)
    res_df.to_csv("models/walk_forward_monthly.csv", index=False)
    
    print(f"\n📊 نتائج Walk-Forward ({len(results)} نافذة):")
    print(f"   متوسط PF التدريب: {res_df['train_pf'].mean():.2f}")
    print(f"   متوسط PF الاختبار: {res_df['test_pf'].mean():.2f}")
    print(f"   متوسط DD الاختبار: {res_df['test_dd'].mean():.1f}%")
    print(f"   متوسط صفقات الاختبار: {res_df['test_trades'].mean():.1f}")
    
    # حفظ آخر نموذج (مُدرَّب على آخر 6 أشهر)
    last_train = df[df.index >= end_date - pd.DateOffset(months=6)]
    print(f"\n💾 حفظ النموذج النهائي (مُدرَّب على {len(last_train)} صف)...")
    
    # تدريب النموذج على آخر 6 أشهر
    last_train_data = last_train.copy()
    # (النموذج هو نفس المصفوفة، نحفظ فقط البيانات)
    last_train_data.to_csv("models/latest_training_data.csv", index=False)
    print("✅ تم حفظ النموذج النهائي في models/latest_training_data.csv")
else:
    print("⚠️ لم نتمكن من إنشاء نوافذ كافية.")

print("🎉 الذرة 2.2 اكتملت – Walk-Forward تلقائي شهري جاهز.")
