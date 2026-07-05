import pandas as pd
from pathlib import Path
from strategy_matrix import generate_signals
from backtest_engine import dynamic_backtest

print('=' * 60)
print('   2.1 – نظام تسجيل الصفقات المؤسسي')
print('=' * 60)

# 1. تحميل البيانات
df = pd.read_csv('data/eurusd_hourly_dukascopy.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)
df = df[['open','high','low','close']].copy()
print(f'تم تحميل {len(df)} صف من البيانات.')

# 2. توليد الإشارات من المصدر الوحيد للحقيقة
df = generate_signals(df)

# 3. تطبيق فلاتر الزمن (Time-of-Day + Weekly Bias)
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
london_ny_hours = [7, 8, 12, 13]
good_days = [1, 2, 4]
time_filter = df['hour'].isin(london_ny_hours)
day_filter = df['weekday'].isin(good_days)
# تعطيل الإشارات خارج الأوقات المسموحة
df.loc[~(time_filter & day_filter), 'Final'] = 0
print(f'الإشارات المتبقية بعد الفلاتر: {(df["Final"] != 0).sum()}')

# 4. تشغيل المحرك الديناميكي مع تسجيل الصفقات
print('بدء المحاكاة...')
trades, equity, trade_log = dynamic_backtest(df, 10000, log_trades=True)

# 5. حفظ السجل
if trade_log:
    log_df = pd.DataFrame(trade_log)
    Path('logs').mkdir(exist_ok=True)
    log_df.to_csv('logs/trade_log_detailed.csv', index=False)
    print(f'تم تسجيل {len(log_df)} صفقة في logs/trade_log_detailed.csv')
else:
    print('لا توجد صفقات للتسجيل.')

print('الذرة 2.1 اكتملت – نظام التسجيل المؤسسي جاهز.')
