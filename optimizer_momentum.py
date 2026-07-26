"""
optimizer_momentum.py - المحسن الآلي لوكيل الزخم
المشروع: الكيان الفيروسي - أقوى شركة تداول
يبحث عن أفضل معاملات لاستراتيجية الزخم (Momentum).
"""
import pandas as pd
import numpy as np
import random
from backtest_engine import dynamic_backtest
from data_loader import load_hourly
from parameter_controller import params as opt_guard

def generate_momentum_signals_optimized(df, adx_period, adx_threshold, ema_fast, ema_slow):
    """توليد إشارات الزخم بمعاملات محددة."""
    df = df.copy()
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    # MACD
    ema_f = df['close'].ewm(span=int(ema_fast), adjust=False).mean()
    ema_s = df['close'].ewm(span=int(ema_slow), adjust=False).mean()
    df['MACD'] = ema_f - ema_s
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # ADX
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    atr = true_range.rolling(14).mean()
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.ewm(span=int(adx_period)).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=int(adx_period)).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(int(adx_period)).mean()
    
    df.dropna(inplace=True)
    
    df['MO_Signal'] = 0
    buy_cond = (df['MACD_hist'] > 0) & (df['ADX'] > adx_threshold) & (df['MACD_hist'] > df['MACD_hist'].shift(1))
    sell_cond = (df['MACD_hist'] < 0) & (df['ADX'] > adx_threshold) & (df['MACD_hist'] < df['MACD_hist'].shift(1))
    df.loc[buy_cond, 'MO_Signal'] = 1
    df.loc[sell_cond, 'MO_Signal'] = -1
    return df

def evaluate_params(params, data):
    adx_period, adx_threshold, ema_fast, ema_slow = params
    df = generate_momentum_signals_optimized(data.copy(), adx_period, adx_threshold, ema_fast, ema_slow)
    trades, equity = dynamic_backtest(df, 10000, risk_percent=0.005, strategy_cols=['MO_Signal'])
    if len(trades) < opt_guard.min_trades_optimizer:
        return -9999
    if not trades:
        return -999
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    pf = 10 if not losses or sum(losses)==0 else abs(sum(wins)/sum(losses))
    total_return = (equity[-1]/10000)-1
    return pf*10 + total_return

print("=" * 60)
print("   🧬 مُحسِّن وكيل الزخم الآلي")
print("=" * 60)
print("\n📥 تحميل بيانات H1...")
df = load_hourly()
df = df.iloc[-3000:]
print(f"   ✅ تم تحميل {len(df)} صف (عينة سريعة)")

NUM_TRIALS = 200
print(f"\n🔬 عدد التركيبات العشوائية: {NUM_TRIALS}")

best_score = -9999
best_params = None

print("\n⚙️ بدء عملية التحسين...")
for i in range(NUM_TRIALS):
    if i % 40 == 0:
        print(f"   تقدم: {i}/{NUM_TRIALS}")
    params = (
        random.randint(10, 20),      # ADX period
        random.randint(20, 35),      # ADX threshold
        random.randint(8, 16),       # EMA fast
        random.randint(20, 30),      # EMA slow
    )
    score = evaluate_params(params, df)
    if score > best_score:
        best_score = score
        best_params = params

print(f"\n🏆 أفضل معاملات تم العثور عليها:")
print(f"   📏 ADX Period: {best_params[0]}")
print(f"   📈 ADX Threshold: {best_params[1]}")
print(f"   ⚡ EMA Fast: {best_params[2]}")
print(f"   🐢 EMA Slow: {best_params[3]}")
print(f"   ⭐ Best Score: {best_score:.2f}")

with open('best_momentum_params.txt', 'w') as f:
    f.write(f"ADX_PERIOD={best_params[0]}\n")
    f.write(f"ADX_THRESHOLD={best_params[1]}\n")
    f.write(f"EMA_FAST={best_params[2]}\n")
    f.write(f"EMA_SLOW={best_params[3]}\n")

print(f"\n💾 تم حفظ أفضل المعاملات في best_momentum_params.txt")
print("🎉 التحسين الآلي اكتمل.")


