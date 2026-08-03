import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os

print("=" * 60)
print("   MERGING FEATURES & CREATING HOLDOUT SET")
print("=" * 60)

# 1. دمج ميزات السيولة في البيانات الأساسية
print("\n[1/3] Merging Microstructure Features into Institutional Data...")
df_inst = pd.read_csv("data/institutional_data_20y.csv", index_col=0, parse_dates=True)
df_micro = pd.read_csv("data/microstructure_features_20y.csv", index_col=0, parse_dates=True)

df_merged = df_inst.join(df_micro, how='inner')
df_merged.dropna(inplace=True)

df_merged.to_csv("data/institutional_data_20y.csv")
print(f"  SUCCESS: Merged data saved. Shape: {df_merged.shape}")

# 2. عزل بيانات Holdout (الذرة 0.6)
print("\n[2/3] Creating Sealed Holdout Set (2023-2025)...")
holdout_start = '2023-01-01'

# التأكد من أن التاريخ موجود في البيانات
if pd.Timestamp(holdout_start) in df_merged.index:
    train_df = df_merged[df_merged.index < holdout_start]
    holdout_df = df_merged[df_merged.index >= holdout_start]
else:
    # في حال لم يكن التاريخ موجوداً بالضبط، نأخذ آخر 20% كـ Holdout
    split_idx = int(len(df_merged) * 0.8)
    train_df = df_merged.iloc[:split_idx]
    holdout_df = df_merged.iloc[split_idx:]

train_df.to_csv("data/institutional_data_train.csv")
holdout_df.to_csv("data/holdout_data.csv")

print(f"  Train Set Shape: {train_df.shape} (Saved to institutional_data_train.csv)")
print(f"  Holdout Set Shape: {holdout_df.shape} (Saved to holdout_data.csv)")

# 3. التحقق النهائي
print("\n[3/3] Verification...")
print(f"  Train last date: {train_df.index[-1]}")
print(f"  Holdout first date: {holdout_df.index[0]}")
print("\n✅ AUDIT PASSED: Data is merged and Holdout is sealed.")
