"""
dry_run_simulator.py - بيئة المحاكاة الجافة
المشروع: الكيان الفيروسي - أقوى شركة تداول
تختبر الاستراتيجية الكاملة بدون لمس السوق.
"""
import pandas as pd
from pathlib import Path
from data_loader import load_hourly
from strategy_matrix import generate_signals
from backtest_engine import dynamic_backtest
from unified_filters import apply_time_filter, apply_weekly_bias_filter

print("=" * 60)
print("   🧪 بيئة المحاكاة الجافة – Dry Run Simulator")
print("=" * 60)

# 1. تحميل البيانات من المركز الموحد
print("\n📥 تحميل البيانات من DataHub...")
df = load_hourly()
print(f"   ✅ تم تحميل {len(df)} صف | {df.index[0]} → {df.index[-1]}")

# 2. توليد الإشارات من الاستراتيجية
print("\n🧠 توليد الإشارات من المصفوفة...")
df = generate_signals(df)
print(f"   🔍 إجمالي الإشارات الخام: {(df['Final']!=0).sum()}")

# 3. تطبيق الفلاتر الموحدة
print("\n🔍 تطبيق الفلاتر...")
time_ok = apply_time_filter(df)
day_ok = apply_weekly_bias_filter(df)
df['Trade_OK'] = (df['Final'] != 0) & time_ok & day_ok
print(f"   ✅ إشارات صالحة بعد الفلاتر: {df['Trade_OK'].sum()}")

# 4. تشغيل المحاكاة
print("\n⚙️ تشغيل المحاكاة مع RiskManager...")
df['Final_Filtered'] = 0
df.loc[df['Trade_OK'], 'Final_Filtered'] = df['Final']
trades, equity, trade_log = dynamic_backtest(df, 10000, log_trades=True)

# 5. حساب النتائج
print("\n📊 نتائج المحاكاة الجافة:")
if trades:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    total = sum(trades)
    ret = (equity[-1]/10000 - 1)*100
    dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    print(f"   💰 رأس المال النهائي: {equity[-1]:,.2f} \$")
    print(f"   📈 العائد الكلي: {ret:.2f}%")
    print(f"   📉 أقصى انخفاض: {dd:.2f}%")
    print(f"   🔢 عدد الصفقات: {n}")
    print(f"   ✅ نسبة الفوز: {wr:.1f}%")
    print(f"   ⚖️ Profit Factor: {pf:.2f}")
else:
    print("   ⚠️ لا توجد صفقات في هذه الفترة.")

# 6. حفظ السجل
if trade_log:
    log_df = pd.DataFrame(trade_log)
    Path("logs").mkdir(exist_ok=True)
    log_df.to_csv("logs/dry_run_log.csv", index=False)
    print(f"\n📋 تم تسجيل {len(log_df)} صفقة في logs/dry_run_log.csv")

print("\n🎉 المحاكاة الجافة اكتملت. النظام جاهز للاختبار الحي.")
print("=" * 60)
