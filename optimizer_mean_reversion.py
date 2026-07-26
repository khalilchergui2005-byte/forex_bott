"""
optimizer_mean_reversion.py - المحسن الآلي لوكيل الظل (مُصحح)
المشروع: الكيان الفيروسي - أقوى شركة تداول
يبحث عن أفضل معاملات لاستراتيجية الارتداد نحو المتوسط.
"""
import pandas as pd
import numpy as np
import itertools
from backtest_engine import dynamic_backtest
from data_loader import load_m15

def generate_mr_signals_optimized(df, bb_mult, rsi_low, rsi_high, sma_period):
    """توليد إشارات وكيل الظل بمعاملات محددة. يتضمن حساب ATR."""
    df = df.copy()
    
    # 0. حساب ATR المطلوب من قبل المحرك
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    # 1. المؤشرات
    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - bb_mult * df['BB_std']
    df['BB_upper'] = df['BB_mid'] + bb_mult * df['BB_std']
    
    df['SMA'] = df['close'].rolling(int(sma_period)).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    
    # 2. الإشارات
    df['MR_Signal'] = 0
    buy_cond = (df['low'] <= df['BB_lower']) & (df['RSI'] < rsi_low) & (df['close'] < df['SMA'])
    sell_cond = (df['high'] >= df['BB_upper']) & (df['RSI'] > rsi_high) & (df['close'] > df['SMA'])
    df.loc[buy_cond, 'MR_Signal'] = 1
    df.loc[sell_cond, 'MR_Signal'] = -1
    
    return df

def evaluate_params(params, data):
    """تقييم أداء تركيبة معاملات معينة."""
    bb_mult, rsi_low, rsi_high, sma_period = params
    
    # توليد الإشارات
    df = generate_mr_signals_optimized(data.copy(), bb_mult, rsi_low, rsi_high, sma_period)
    
    # تشغيل المحاكاة
    trades, equity = dynamic_backtest(df, 10000, risk_percent=0.005, strategy_cols=['MR_Signal'])
    
    # حساب النتيجة
    if not trades:
        return -999  # عقوبة لعدم وجود صفقات
    
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    if not losses or sum(losses) == 0:
        pf = 10
    else:
        pf = abs(sum(wins) / sum(losses))
    
    total_return = (equity[-1] / 10000) - 1
    
    # النتيجة النهائية: نعطي وزناً أعلى للـ Profit Factor
    score = pf * 10 + total_return
    return score

print("=" * 60)
print("   🧬 مُحسِّن وكيل الظل الآلي (مُصحح)")
print("=" * 60)

# تحميل البيانات
print("\n📥 تحميل بيانات M15...")
df = load_m15()
print(f"   ✅ تم تحميل {len(df)} صف")

# تعريف نطاقات البحث
bb_mult_range = [1.5, 1.8, 2.0, 2.2, 2.5]
rsi_low_range = [20, 25, 28, 30, 32, 35, 40]
rsi_high_range = [60, 65, 68, 70, 72, 75, 80]
sma_period_range = [30, 40, 50, 60, 70]

# توليد جميع التركيبات الممكنة
all_combinations = list(itertools.product(bb_mult_range, rsi_low_range, rsi_high_range, sma_period_range))
total_combos = len(all_combinations)
print(f"\n🔬 عدد التركيبات المطلوب اختبارها: {total_combos}")

# تقييم كل تركيبة
best_score = -9999
best_params = None

print("\n⚙️ بدء عملية التحسين...")
for i, params in enumerate(all_combinations):
    if i % 100 == 0:
        print(f"   تقدم: {i}/{total_combos} ({(i/total_combos)*100:.1f}%)")
    
    score = evaluate_params(params, df)
    
    if score > best_score:
        best_score = score
        best_params = params

# عرض النتائج
print(f"\n🏆 أفضل معاملات تم العثور عليها:")
print(f"   📊 Bollinger Multiplier: {best_params[0]}")
print(f"   📉 RSI Low Threshold: {best_params[1]}")
print(f"   📈 RSI High Threshold: {best_params[2]}")
print(f"   📏 SMA Period: {best_params[3]}")
print(f"   ⭐ Best Score: {best_score:.2f}")

# حفظ المعاملات الأفضل
with open('best_mr_params.txt', 'w') as f:
    f.write(f"BB_MULT={best_params[0]}\n")
    f.write(f"RSI_LOW={best_params[1]}\n")
    f.write(f"RSI_HIGH={best_params[2]}\n")
    f.write(f"SMA_PERIOD={int(best_params[3])}\n")

print(f"\n💾 تم حفظ أفضل المعاملات في best_mr_params.txt")
print("🎉 التحسين الآلي اكتمل.")
