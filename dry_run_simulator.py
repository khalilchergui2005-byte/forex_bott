"""
dry_run_simulator.py - برج المراقبة والتنفيذ الذهبي (مُحسَّن)
المشروع: الكيان الفيروسي - أقوى شركة تداول
يمنح كل إشارة درجة تأكيد، ويضاعف الرافعة على الذهبية.
"""
import pandas as pd
from pathlib import Path
from data_loader import load_hourly
from strategy_matrix import generate_signals
from backtest_engine import dynamic_backtest
from unified_filters import apply_time_filter, apply_weekly_bias_filter

print("=" * 60)
print("   🦅 برج المراقبة – Dry Run Simulator (محسن)")
print("=" * 60)

# 1. تحميل البيانات
print("\n📥 تحميل البيانات من DataHub...")
df = load_hourly()
print(f"   ✅ تم تحميل {len(df)} صف | {df.index[0]} → {df.index[-1]}")

# 2. توليد الإشارات
print("\n🧠 توليد الإشارات من المصفوفة...")
df = generate_signals(df)
print(f"   🔍 إجمالي الإشارات الخام: {(df['Final']!=0).sum()}")

# 3. حساب مصفوفة التأكيد (محسنة)
print("\n🛡️ حساب درجة التأكيد (محسنة)...")
time_ok = apply_time_filter(df)
day_ok = apply_weekly_bias_filter(df)
vol_ok = (df['VolClass'] == 2)  # مكافأة إضافية فقط

# درجة التأكيد تبدأ من 1.0 (أساسي لكل إشارة صالحة)
df['Confirmation'] = 1.0
df.loc[time_ok, 'Confirmation'] += 0.5
df.loc[day_ok, 'Confirmation'] += 0.5
df.loc[vol_ok, 'Confirmation'] += 0.5
# الحد الأقصى 2.0 (لتبقى المخاطرة القصوى 1.0%)
df['Confirmation'] = df['Confirmation'].clip(upper=2.0)

# الإشارات الصالحة هي التي تحمل إشارة تداول
df['Final_Filtered'] = df['Final']
print(f"   ✅ إشارات صالحة للتداول: {df['Final_Filtered'].sum()}")

# 4. تشغيل المحاكاة
print("\n⚙️ تشغيل المحاكاة مع مضاعف الرافعة الذكي...")
trades, equity, trade_log = dynamic_backtest(df, 10000, risk_percent=0.005, log_trades=True)

# 5. النتائج
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

if trade_log:
    log_df = pd.DataFrame(trade_log)
    Path("logs").mkdir(exist_ok=True)
    log_df.to_csv("logs/dry_run_log.csv", index=False)
    print(f"\n📋 تم تسجيل {len(log_df)} صفقة في logs/dry_run_log.csv")

print("\n🎉 المحاكاة الجافة اكتملت. النظام جاهز للاختبار الحي.")
print("=" * 60)
