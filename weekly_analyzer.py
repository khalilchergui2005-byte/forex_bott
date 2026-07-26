"""
weekly_analyzer.py - المحلل الأسبوعي والآلي + إعادة حساب الفلاتر الديناميكية
المشروع: الكيان الفيروسي - أقوى شركة تداول
يتعلم من الأداء كل أسبوع ويعيد حساب الفلاتر كل شهر.
"""
import pandas as pd
from datetime import datetime, timedelta

def analyze_week(trade_log_df, initial_balance=10000):
    """
    تحليل سجل صفقات الأسبوع الماضي واتخاذ قرارات للأسبوع القادم.
    المدخل: DataFrame لسجل الصفقات.
    المخرج: قاموس بالقرارات.
    """
    if trade_log_df.empty or len(trade_log_df) < 5:
        return {'rules': {}, 'message': 'بيانات غير كافية لاتخاذ قرارات.'}

    # التأكد من وجود الأعمدة المطلوبة
    required_cols = ['entry_time', 'net_pnl']
    if not all(col in trade_log_df.columns for col in required_cols):
        # محاولة استخدام exit_time كبديل
        if 'exit_time' in trade_log_df.columns:
            trade_log_df['entry_time'] = trade_log_df['exit_time']
        else:
            return {'rules': {}, 'message': 'لا يوجد عمود زمني في السجل.'}

    # تحويل عمود الوقت
    trade_log_df['entry_time'] = pd.to_datetime(trade_log_df['entry_time'], utc=True)
    
    # تحديد فترة الأسبوع الماضي
    now = trade_log_df['entry_time'].max()
    week_ago = now - timedelta(days=7)
    recent = trade_log_df[trade_log_df['entry_time'] >= week_ago]

    if recent.empty:
        return {'rules': {}, 'message': 'لا توجد صفقات في الأسبوع الماضي.'}

    # تحليل الأداء حسب اليوم والساعة
    recent['day'] = recent['entry_time'].dt.dayofweek
    recent['hour'] = recent['entry_time'].dt.hour

    rules = {}
    messages = []

    # --- تحليل الأيام ---
    day_pnl = recent.groupby('day')['net_pnl'].sum()
    for day, pnl in day_pnl.items():
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
        day_name = day_names.get(day, f'Day_{day}')
        if pnl < -initial_balance * 0.005:  # خسارة > 0.5% من الرصيد
            rules[f'day_{day}'] = 'disable'
            messages.append(f"🚫 {day_name}: خسارة {pnl:.2f} - سيتم التعطيل الأسبوع القادم.")
        elif pnl > initial_balance * 0.01:  # ربح > 1% من الرصيد
            rules[f'day_{day}'] = 'boost'
            messages.append(f"🚀 {day_name}: ربح {pnl:.2f} - سيتم التعزيز الأسبوع القادم.")

    # --- تحليل الساعات ---
    if 'hour' in recent.columns:
        hour_pnl = recent.groupby('hour')['net_pnl'].sum()
        for hour, pnl in hour_pnl.items():
            if pnl < -initial_balance * 0.005:
                rules[f'hour_{hour}'] = 'disable'
                messages.append(f"🚫 ساعة {hour}:00: خسارة {pnl:.2f} - سيتم التعطيل.")
            elif pnl > initial_balance * 0.01:
                rules[f'hour_{hour}'] = 'boost'
                messages.append(f"🚀 ساعة {hour}:00: ربح {pnl:.2f} - سيتم التعزيز.")

    # --- التهدئة العامة ---
    total_weekly_pnl = recent['net_pnl'].sum()
    if total_weekly_pnl < -initial_balance * 0.01:
        rules['overall_risk'] = 0.8  # خفض المخاطرة 20%
        messages.insert(0, f"⚠️ خسارة أسبوعية {total_weekly_pnl:.2f} - تخفيض المخاطرة 20%.")
    elif total_weekly_pnl > initial_balance * 0.03:
        rules['overall_risk'] = 1.1  # زيادة المخاطرة 10%
        messages.insert(0, f"🔥 ربح أسبوعي قوي {total_weekly_pnl:.2f} - زيادة المخاطرة 10%.")

    if not rules:
        messages.append("✅ الأداء مستقر. لا تغييرات.")

    return {'rules': rules, 'message': '\n'.join(messages)}


def recalculate_filters(trade_log_df, lookback_days=30):
    """
    إعادة حساب الفلاتر الديناميكية بناءً على أداء آخر N يوم.
    المدخل: DataFrame لسجل الصفقات.
    المخرج: قاموس يحتوي على good_days, good_hours, good_vol_classes.
    """
    if trade_log_df.empty or len(trade_log_df) < 10:
        return {'good_days': [1, 2, 4], 'good_hours': [7, 8, 12, 13], 'good_vol_classes': [1, 2]}

    # التأكد من وجود عمود الوقت
    if 'entry_time' not in trade_log_df.columns:
        if 'exit_time' in trade_log_df.columns:
            trade_log_df['entry_time'] = trade_log_df['exit_time']
        else:
            return {'good_days': [1, 2, 4], 'good_hours': [7, 8, 12, 13], 'good_vol_classes': [1, 2]}

    trade_log_df['entry_time'] = pd.to_datetime(trade_log_df['entry_time'], utc=True)
    
    # تحديد فترة التحليل
    now = trade_log_df['entry_time'].max()
    start_date = now - timedelta(days=lookback_days)
    recent = trade_log_df[trade_log_df['entry_time'] >= start_date]

    if recent.empty or len(recent) < 10:
        return {'good_days': [1, 2, 4], 'good_hours': [7, 8, 12, 13], 'good_vol_classes': [1, 2]}

    # استخراج اليوم والساعة من وقت الدخول
    recent['day'] = recent['entry_time'].dt.dayofweek
    recent['hour'] = recent['entry_time'].dt.hour

    # --- تحليل الأيام ---
    day_pnl = recent.groupby('day')['net_pnl'].sum()
    good_days = [day for day, pnl in day_pnl.items() if pnl > 0]
    if not good_days:  # إذا لم تكن هناك أيام رابحة، استخدم القائمة الافتراضية
        good_days = [1, 2, 4]

    # --- تحليل الساعات ---
    hour_pnl = recent.groupby('hour')['net_pnl'].sum()
    good_hours = [hour for hour, pnl in hour_pnl.items() if pnl > 0]
    if not good_hours:
        good_hours = [7, 8, 12, 13]

    # --- تحليل أنظمة التقلب (إذا كان العمود موجوداً) ---
    good_vol_classes = [1, 2]  # افتراضي
    if 'VolClass' in recent.columns:
        vol_pnl = recent.groupby('VolClass')['net_pnl'].sum()
        good_vol_classes = [vol for vol, pnl in vol_pnl.items() if pnl > 0]
        if not good_vol_classes:
            good_vol_classes = [1, 2]

    return {
        'good_days': good_days,
        'good_hours': good_hours,
        'good_vol_classes': good_vol_classes
    }


def apply_weekly_rules(df, rules, signal_cols=None):
    """
    تطبيق القواعد الأسبوعية على DataFrame البيانات.
    """
    if not rules:
        return df

    if signal_cols is None:
        signal_cols = ['Final', 'MO_Signal']

    df = df.copy()
    
    # تطبيق قواعد الأيام
    for rule, action in rules.items():
        if rule.startswith('day_'):
            day = int(rule.split('_')[1])
            if action == 'disable':
                for col in signal_cols:
                    if col in df.columns:
                        df.loc[df['weekday'] == day, col] = 0
            elif action == 'boost':
                if 'Confirmation' in df.columns:
                    df.loc[df['weekday'] == day, 'Confirmation'] += 0.3

    # تطبيق قواعد الساعات
    for rule, action in rules.items():
        if rule.startswith('hour_'):
            hour = int(rule.split('_')[1])
            if action == 'disable':
                for col in signal_cols:
                    if col in df.columns:
                        df.loc[df['hour'] == hour, col] = 0
            elif action == 'boost':
                if 'Confirmation' in df.columns:
                    df.loc[df['hour'] == hour, 'Confirmation'] += 0.3

    return df

    df = df.copy()
    
    # تطبيق قواعد الأيام
    for rule, action in rules.items():
        if rule.startswith('day_'):
            day = int(rule.split('_')[1])
            if action == 'disable':
                df.loc[df['weekday'] == day, 'Final'] = 0
            elif action == 'boost':
                if 'Confirmation' in df.columns:
                    df.loc[df['weekday'] == day, 'Confirmation'] += 0.3

    # تطبيق قواعد الساعات
    for rule, action in rules.items():
        if rule.startswith('hour_'):
            hour = int(rule.split('_')[1])
            if action == 'disable':
                df.loc[df['hour'] == hour, 'Final'] = 0
            elif action == 'boost':
                if 'Confirmation' in df.columns:
                    df.loc[df['hour'] == hour, 'Confirmation'] += 0.3

    return df


def get_effective_risk(base_risk, rules):
    """
    حساب نسبة المخاطرة الفعلية بعد تطبيق القواعد الأسبوعية.
    """
    if 'overall_risk' in rules:
        return base_risk * rules['overall_risk']
    return base_risk


# للاختبار المباشر
if __name__ == '__main__':
    log_path = 'logs/dry_run_log.csv'
    try:
        df = pd.read_csv(log_path)
        result = analyze_week(df)
        print(result['message'])
        print("\nالقواعد المطبقة:")
        print(result['rules'])
        
        print("\n--- إعادة حساب الفلاتر ---")
        filters = recalculate_filters(df)
        print(f"الأيام الجيدة: {filters['good_days']}")
        print(f"الساعات الجيدة: {filters['good_hours']}")
        print(f"فئات التقلب الجيدة: {filters['good_vol_classes']}")
    except FileNotFoundError:
        print("ملف السجل غير موجود. قم بتشغيل المحاكاة أولاً.")

