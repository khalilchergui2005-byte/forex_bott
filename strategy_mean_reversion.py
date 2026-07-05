"""
strategy_mean_reversion.py - وكيل الظل (نسخة محسنة)
المشروع: الكيان الفيروسي - أقوى شركة تداول
نسخة محسنة: فلتر إضافي لتجنب الإشارات الضعيفة.
"""
import pandas as pd
import numpy as np

def generate_mr_signals(df):
    """
    توليد إشارات ارتدادية محسنة على إطار 15 دقيقة.
    """
    df = df.copy()
    
    # 1. حساب المؤشرات الديناميكية
    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - 2.0 * df['BB_std']
    df['BB_upper'] = df['BB_mid'] + 2.0 * df['BB_std']
    
    df['SMA_50'] = df['close'].rolling(50).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    
    # 2. توليد الإشارات المحسنة
    df['MR_Signal'] = 0
    
    # إشارة شراء محسنة: لمس الشريط السفلي + RSI منخفض + السعر تحت المتوسط 50
    buy_condition = (
        (df['low'] <= df['BB_lower']) & 
        (df['RSI'] < 30) & 
        (df['close'] < df['SMA_50'])
    )
    df.loc[buy_condition, 'MR_Signal'] = 1
    
    # إشارة بيع محسنة: لمس الشريط العلوي + RSI مرتفع + السعر فوق المتوسط 50
    sell_condition = (
        (df['high'] >= df['BB_upper']) & 
        (df['RSI'] > 70) & 
        (df['close'] > df['SMA_50'])
    )
    df.loc[sell_condition, 'MR_Signal'] = -1
    
    return df
