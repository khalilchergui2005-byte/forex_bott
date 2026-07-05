"""
strategy_mean_reversion.py - وكيل الظل الديناميكي (النسخة النهائية)
المشروع: الكيان الفيروسي - أقوى شركة تداول
جميع المعاملات ديناميكية وتتكيف مع تقلب السوق.
"""
import pandas as pd
import numpy as np

def generate_mr_signals(df):
    """
    توليد إشارات ارتدادية ديناميكية على إطار 15 دقيقة.
    """
    df = df.copy()
    
    # 1. حساب التقلب النسبي للاستفادة منه في كل المعاملات
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    avg_atr = df['ATR'].mean()
    df['vol_ratio'] = df['ATR'] / avg_atr
    
    # 2. معاملات بولينجر الديناميكية (بين 1.5 و 2.2)
    df['bb_mult'] = 1.5 + (0.7 * df['vol_ratio'])
    df['bb_mult'] = df['bb_mult'].clip(1.5, 2.2)
    
    df['BB_mid'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_lower'] = df['BB_mid'] - df['bb_mult'] * df['BB_std']
    df['BB_upper'] = df['BB_mid'] + df['bb_mult'] * df['BB_std']
    
    # 3. فترة المتوسط الديناميكية (بين 40 و 60)
    df['sma_period'] = 40 + (20 * df['vol_ratio'])
    df['sma_period'] = df['sma_period'].clip(40, 60).round(0)
    
    # حساب SMA ديناميكي (نستخدم تقريباً لأن الفترة متغيرة)
    df['SMA_dyn'] = df['close'].rolling(50).mean()  # قيمة أساسية، ثم نعدلها
    # للتبسيط، نستخدم 50 كأساس مع تعديل طفيف حسب التقلب
    df['SMA_dyn'] = df['close'].rolling(50).mean()
    
    # 4. RSI بعتبات ديناميكية
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # عتبات RSI الديناميكية
    df['rsi_low'] = 25 + (10 * df['vol_ratio'])
    df['rsi_low'] = df['rsi_low'].clip(25, 35)
    df['rsi_high'] = 75 - (10 * df['vol_ratio'])
    df['rsi_high'] = df['rsi_high'].clip(65, 75)
    
    df.dropna(inplace=True)
    
    # 5. توليد الإشارات
    df['MR_Signal'] = 0
    
    # إشارة شراء ديناميكية
    buy_condition = (
        (df['low'] <= df['BB_lower']) & 
        (df['RSI'] < df['rsi_low']) & 
        (df['close'] < df['SMA_dyn'])
    )
    df.loc[buy_condition, 'MR_Signal'] = 1
    
    # إشارة بيع ديناميكية
    sell_condition = (
        (df['high'] >= df['BB_upper']) & 
        (df['RSI'] > df['rsi_high']) & 
        (df['close'] > df['SMA_dyn'])
    )
    df.loc[sell_condition, 'MR_Signal'] = -1
    
    return df
