"""
unified_filters.py - المصدر الوحيد للحقيقة لكل فلاتر التداول
المشروع: الكيان الفيروسي - أقوى شركة تداول
"""
import pandas as pd

def apply_time_filter(df):
    """
    فلتر Time-of-Day: يسمح فقط بساعات سوق لندن ونيويورك الاكثر سيولة.
    المدخل: DataFrame يحتوي على عمود 'hour' (UTC).
    المخرج: Series من booleans (True = مسموح).
    """
    allowed_hours = [7, 8, 12, 13]
    return df['hour'].isin(allowed_hours)

def apply_weekly_bias_filter(df):
    """
    فلتر Weekly Bias: يسمح فقط بالايام التي أظهرت أداء إيجابيا تاريخيا.
    المدخل: DataFrame يحتوي على عمود 'weekday' (0=الاثنين).
    المخرج: Series من booleans (True = مسموح).
    """
    good_days = [1, 2, 4]
    return df['weekday'].isin(good_days)

def apply_volatility_filter(df):
    """
    فلتر Volatility Regime: يسمح بالتداول في التقلب المرتفع فقط.
    المدخل: DataFrame يحتوي على عمود 'VolClass'.
    المخرج: Series من booleans (True = مسموح).
    """
    return df['VolClass'] == 2
