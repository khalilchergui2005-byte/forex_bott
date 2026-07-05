"""
dry_run_simulator.py - برج المراقبة + وكيل الظل (Mean Reversion)
المشروع: الكيان الفيروسي - أقوى شركة تداول
يدير محفظة باستراتيجيتين: المصفوفة + الارتداد نحو المتوسط.
"""
import pandas as pd
from pathlib import Path
from data_loader import load_hourly, load_m15
from strategy_matrix import generate_signals
from strategy_mean_reversion import generate_mr_signals
from backtest_engine import dynamic_backtest
from unified_filters import apply_time_filter, apply_weekly_bias_filter
from weekly_analyzer import analyze_week, apply_weekly_rules, get_effective_risk, recalculate_filters

print("=" * 60)
print("   🦅 برج المراقبة – المحفظة ثنائية الاستراتيجية")
print("=" * 60)

# 1. تحميل البيانات
print("\n📥 تحميل البيانات من DataHub...")
df_h1 = load_hourly()
df_m15 = load_m15()
print(f"   ✅ H1: {len(df_h1)} صف | M15: {len(df_m15)} صف")

# 2. توليد الإشارات من الاستراتيجيتين
print("\n🧠 توليد الإشارات...")
df_h1 = generate_signals(df_h1)  # المصفوفة
df_m15 = generate_mr_signals(df_m15)  # وكيل الظل
print(f"   🔍 إشارات المصفوفة (H1): {(df_h1['Final']!=0).sum()}")
print(f"   🔍 إشارات وكيل الظل (M15): {(df_m15['MR_Signal']!=0).sum()}")

# 3. محاذاة التواريخ (نأخذ التاريخ المشترك)
common_index = df_h1.index.intersection(df_m15.index)
df_h1 = df_h1.loc[common_index]
df_m15 = df_m15.loc[common_index]

# 4. نسخ إشارات وكيل الظل إلى بيانات H1 لتوحيد المحرك
df_h1['MR_Signal'] = df_m15['MR_Signal']

# 5. حساب مصفوفة التأكيد للاستراتيجيتين
print("\n🛡️ حساب درجة التأكيد...")
time_ok = apply_time_filter(df_h1)
day_ok = apply_weekly_bias_filter(df_h1)
vol_ok = (df_h1['VolClass'] == 2)

df_h1['Confirmation'] = 1.0
df_h1.loc[time_ok, 'Confirmation'] += 0.5
df_h1.loc[day_ok, 'Confirmation'] += 0.5
df_h1.loc[vol_ok, 'Confirmation'] += 0.5
df_h1['Confirmation'] = df_h1['Confirmation'].clip(upper=2.0)

# 6. محاولة تحميل سجل سابق للفلاتر الديناميكية والمحلل الأسبوعي
print("\n🔄 تحليل السجل السابق...")
try:
    past_log = pd.read_csv('logs/dry_run_log.csv')
    dynamic_filters = recalculate_filters(past_log)
    good_days = dynamic_filters['good_days']
    good_hours = dynamic_filters['good_hours']
    analysis = analyze_week(past_log)
    print(f"   {analysis['message']}")
    weekly_rules = analysis['rules']
except FileNotFoundError:
    good_days = [1, 2, 4]
    good_hours = [7, 8, 12, 13]
    weekly_rules = {}
    print("   ⚠️ لا يوجد سجل سابق. بداية جديدة.")

# 7. تطبيق الفلاتر الديناميكية والقواعد الأسبوعية
df_h1['time_ok'] = df_h1['hour'].isin(good_hours)
df_h1['day_ok'] = df_h1['weekday'].isin(good_days)
if weekly_rules:
    df_h1 = apply_weekly_rules(df_h1, weekly_rules)

base_risk = 0.005
effective_risk = get_effective_risk(base_risk, weekly_rules)

# 8. تشغيل المحاكاة باستراتيجيتين معاً
print(f"\n⚙️ تشغيل المحاكاة (المخاطرة: {effective_risk*100:.2f}%)...")
trades, equity, trade_log = dynamic_backtest(
    df_h1, 10000, risk_percent=effective_risk, log_trades=True,
    strategy_cols=['Final', 'MR_Signal']
)

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

print("\n🎉 المحاكاة الجافة اكتملت. محفظة ثنائية الاستراتيجية.")
print("=" * 60)
