"""
2.1_trade_log.py - نظام تسجيل الصفقات المؤسسي (مُحدّث)
يستخدم DataHub الموحد.
"""
import pandas as pd
from pathlib import Path
from strategy_matrix import generate_signals
from backtest_engine import dynamic_backtest
from data_loader import load_hourly

print("=" * 60)
print("   2.1 – نظام تسجيل الصفقات المؤسسي")
print("=" * 60)

# تحميل البيانات من المركز الموحد
df = load_hourly()

# توليد الإشارات
df = generate_signals(df)

# تطبيق فلاتر الزمن
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
london_ny_hours = [7, 8, 12, 13]
good_days = [1, 2, 4]
time_filter = df['hour'].isin(london_ny_hours)
day_filter = df['weekday'].isin(good_days)
df.loc[~(time_filter & day_filter), 'Final'] = 0

print(f"بيانات محملة: {len(df)} صف")
print(f"الإشارات المتبقية بعد الفلاتر: {(df['Final'] != 0).sum()}")

# تشغيل المحرك الديناميكي مع التسجيل
trades, equity, trade_log = dynamic_backtest(df, 10000, log_trades=True)

# حفظ السجل
if trade_log:
    log_df = pd.DataFrame(trade_log)
    Path("logs").mkdir(exist_ok=True)
    log_df.to_csv("logs/trade_log_detailed.csv", index=False)
    print(f"تم تسجيل {len(log_df)} صفقة في logs/trade_log_detailed.csv")
else:
    print("لا توجد صفقات للتسجيل.")

print("الذرة 2.1 اكتملت – نظام التسجيل المؤسسي جاهز.")
