"""
dry_run_simulator.py - المحفظة ثنائية الاستراتيجية (المصفوفة + الزخم)
المشروع: الكيان الفيروسي - أقوى شركة تداول
"""
import pandas as pd
from pathlib import Path
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from unified_filters import apply_time_filter, apply_weekly_bias_filter
from weekly_analyzer import analyze_week, apply_weekly_rules, get_effective_risk, recalculate_filters

print("=" * 60)
print("   🦅 المحفظة ثنائية الاستراتيجية (المصفوفة + الزخم)")
print("=" * 60)

# 1. تحميل البيانات
print("\n📥 تحميل بيانات H1...")
df = load_hourly()
print(f"   ✅ تم تحميل {len(df)} صف")

# 2. توليد الإشارات من الاستراتيجيتين
print("\n🧠 توليد الإشارات...")
df = generate_signals(df)  # المصفوفة
df = generate_momentum_signals(df)  # الزخم
print(f"   🔍 إشارات المصفوفة: {(df['Final']!=0).sum()}")
print(f"   🔍 إشارات الزخم: {(df['MO_Signal']!=0).sum()}")

# 3. حساب مصفوفة التأكيد
print("\n🛡️ حساب درجة التأكيد...")
time_ok = apply_time_filter(df)
day_ok = apply_weekly_bias_filter(df)
vol_ok = (df['VolClass'] == 2)

df['Confirmation'] = 1.0
df.loc[time_ok, 'Confirmation'] += 0.5
df.loc[day_ok, 'Confirmation'] += 0.5
df.loc[vol_ok, 'Confirmation'] += 0.5
df['Confirmation'] = df['Confirmation'].clip(upper=2.0)

# 4. الفلاتر الديناميكية والمحلل الأسبوعي
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

df['time_ok'] = df['hour'].isin(good_hours)
df['day_ok'] = df['weekday'].isin(good_days)
if weekly_rules:
    df = apply_weekly_rules(df, weekly_rules)

base_risk = 0.005
effective_risk = get_effective_risk(base_risk, weekly_rules)

# 5. تشغيل المحاكاة باستراتيجيتين
print(f"\n⚙️ تشغيل المحاكاة (المخاطرة: {effective_risk*100:.2f}%)...")
trades, equity, trade_log = dynamic_backtest(
    df, 10000, risk_percent=effective_risk, log_trades=True,
    strategy_cols=['Final', 'MO_Signal']
)

# 6. النتائج
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
