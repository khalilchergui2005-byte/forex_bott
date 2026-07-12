"""
DataHub - المركز الموحد لبيانات EUR/USD
المشروع: الكيان الفيروسي - أقوى شركة تداول
يجمع بيانات Dukascopy و MT5 في مصدر واحد مضاد للرصاص.
"""
import pandas as pd
from pathlib import Path

UNIFIED_PATH = "data/eurusd_unified.csv"
DUKASCOPY_PATH = "data/eurusd_hourly_dukascopy.csv"
MT5_PATH = "data/eurusd_mt5_2025.csv"

def build_unified_data():
    """
    بناء ملف البيانات الموحد من مصدري Dukascopy و MT5.
    الفترة الناتجة: من أقدم تاريخ في Dukascopy إلى أحدث تاريخ في MT5.
    """
    print("🔄 بناء مركز البيانات الموحد...")
    
    # 1. تحميل Dukascopy (النظيف)
    df_duka = pd.read_csv(DUKASCOPY_PATH, index_col=0, parse_dates=True)
    df_duka.index = pd.to_datetime(df_duka.index, utc=True)
    df_duka = df_duka[['open','high','low','close']]
    print(f"   ✅ Dukascopy: {len(df_duka)} صف | {df_duka.index[0]} → {df_duka.index[-1]}")
    
    # 2. تحميل وتنظيف MT5 (ملف tab-separated)
    df_mt5 = pd.read_csv(MT5_PATH, sep='\t')
    # إعادة تسمية الأعمدة
    df_mt5.rename(columns={
        '<DATE>': 'date', '<TIME>': 'time',
        '<OPEN>': 'open', '<HIGH>': 'high',
        '<LOW>': 'low', '<CLOSE>': 'close'
    }, inplace=True)
    # دمج عمودي التاريخ والوقت
    df_mt5['Datetime'] = pd.to_datetime(df_mt5['date'] + ' ' + df_mt5['time'], utc=True)
    df_mt5.set_index('Datetime', inplace=True)
    df_mt5 = df_mt5[['open','high','low','close']]
    print(f"   ✅ MT5: {len(df_mt5)} صف | {df_mt5.index[0]} → {df_mt5.index[-1]}")
    
    # 3. الدمج: Dukascopy لما قبل MT5، ثم MT5 للفترة الأحدث
    mt5_start = df_mt5.index[0]
    df_duka_old = df_duka[df_duka.index < mt5_start]
    df_unified = pd.concat([df_duka_old, df_mt5]).sort_index()
    df_unified = df_unified[~df_unified.index.duplicated(keep='last')]
    
    # 4. حفظ الملف الموحد
    df_unified.to_csv(UNIFIED_PATH)
    print(f"   🎉 تم الدمج: {len(df_unified)} صف | {df_unified.index[0]} → {df_unified.index[-1]}")
    print(f"   💾 محفوظ في: {UNIFIED_PATH}")
    return df_unified

def load_hourly():
    """تحميل البيانات الموحدة بإطار الساعة."""
    if not Path(UNIFIED_PATH).exists():
        build_unified_data()
    df = pd.read_csv(UNIFIED_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    return df

def load_daily():
    """تحميل البيانات الموحدة بإطار اليوم."""
    df = load_hourly()
    return df.resample('D').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()

def load_m15():
    # DEPRECATED: M15 data is NOT available.
    # H1-to-M15 resample produces fake candles.
    # Use load_hourly() until real M15 is integrated.
    raise NotImplementedError(
        'M15 not available. Use load_hourly() instead.'
    )

if __name__ == "__main__":
    build_unified_data()
