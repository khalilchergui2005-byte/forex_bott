"""
dry_run_simulator.py - برج المراقبة + الفلاتر الديناميكية + المحلل الأسبوعي
المشروع: الكيان الفيروسي - أقوى شركة تداول
الفلاتر تتكيف تلقائياً كل 30 يوماً بناءً على الأداء الفعلي.
"""
import pandas as pd
from pathlib import Path
from data_loader import load_hourly
from strategy_matrix import generate_signals
from backtest_engine import dynamic_backtest
from weekly_analyzer import analyze_week, apply_weekly_rules, get_effective_risk, recalculate_filters

print("=" * 60)
print("   🦅 برج المراقبة – الفلاتر الديناميكية + الدماغ الأسبوعي")
print("=" * 60)

# 1. تحميل البيانات
print("\n📥 تحميل البيانات من DataHub...")
df = load_hourly()
print(f"   ✅ تم تحميل {len(df)} صف | {df.index[0]} → {df.index[-1]}")

# 2. توليد الإشارات
print("\n🧠 توليد الإشارات من المصفوفة...")
df = generate_signals(df)
print(f"   🔍 إجمالي الإشارات الخام: {(df['Final']!=0).sum()}")

# 3. محاولة تحميل سجل سابق لتوليد فلاتر ديناميكية
print("\n🔄 حساب الفلاتر الديناميكية من السجل السابق...")
try:
    past_log = pd.read_csv('logs/dry_run_log.csv')
    dynamic_filters = recalculate_filters(past_log)
    good_days = dynamic_filters['good_days']
    good_hours = dynamic_filters['good_hours']
    good_vol = dynamic_filters['good_vol_classes']
    print(f"   ✅ أيام جيدة (ديناميكي): {good_days}")
    print(f"   ✅ ساعات جيدة (ديناميكي): {good_hours}")
    print(f"   ✅ فئات تقلب جيدة (ديناميكي): {good_vol}")
except FileNotFoundError:
    # استخدام القيم الافتراضية كأساس أولي
    good_days = [1, 2, 4]
    good_hours = [7, 8, 12, 13]
    good_vol = [1, 2]
    print("   ⚠️ لا يوجد سجل سابق. استخدام القيم الافتراضية.")

# 4. بناء الفلاتر الديناميكية
print("\n🛡️ تطبيق الفلاتر الديناميكية...")
time_ok = df['hour'].isin(good_hours)
day_ok = df['weekday'].isin(good_days)
vol_ok = df['VolClass'].isin(good_vol)

# 5. حساب درجة التأكيد
df['Confirmation'] = 1.0
df.loc[time_ok, 'Confirmation'] += 0.5
df.loc[day_ok, 'Confirmation'] += 0.5
df.loc[vol_ok, 'Confirmation'] += 0.5
df['Confirmation'] = df['Confirmation'].clip(upper=2.0)
print(f"   ✅ إشارات صالحة للتداول: {(df['Final']!=0).sum()}")

# 6. محاكاة المحلل الأسبوعي
print("\n📊 تحليل الأداء الأسبوعي السابق...")
try:
    past_log = pd.read_csv('logs/dry_run_log.csv')
    analysis = analyze_week(past_log)
    print(f"   {analysis['message']}")
    weekly_rules = analysis['rules']
except FileNotFoundError:
    weekly_rules = {}
    print("   ⚠️ لا يوجد سجل سابق. بداية جديدة.")

# 7. تطبيق القواعد الأسبوعية
if weekly_rules:
    df = apply_weekly_rules(df, weekly_rules)
    print(f"   🔧 تم تطبيق {len(weekly_rules)} قاعدة أسبوعية.")
else:
    print("   ✅ لا توجد قواعد أسبوعية للتطبيق.")

# 8. تشغيل المحاكاة
base_risk = 0.005
effective_risk = get_effective_risk(base_risk, weekly_rules)
print(f"\n⚙️ تشغيل المحاكاة (المخاطرة الأساسية: {effective_risk*100:.2f}%)...")
trades, equity, trade_log = dynamic_backtest(df, 10000, risk_percent=effective_risk, log_trades=True)

# 9. النتائج
print("\n📊 نتائج المحاكاة الجافة:")
if trades:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
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

print("\n🎉 المحاكاة الجافة اكتملت. الفلاتر تتكيف ديناميكياً.")
print("=" * 60)
