import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA_DIR_D = r"D:\forex_bot_data"
DATA_DIR_LOCAL = "data"

print("=" * 70)
print("   COMPREHENSIVE READ-ONLY AUDIT")
print("=" * 70)

def audit_csv(path, name):
    if not os.path.exists(path):
        print(f"  [{name}] ❌ FILE NOT FOUND")
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  [{name}] ✅ Shape: {df.shape}")
        print(f"    Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"    Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"  [{name}] ❌ ERROR: {e}")
        return None

# 1. Raw Tick Rule Data (The 20-Year Foundation)
print("\n[1] RAW TICK DATA (D Drive):")
df_eur_raw = audit_csv(os.path.join(DATA_DIR_D, "eurusd_h1_real.csv"), "EURUSD Raw")
df_gbp_raw = audit_csv(os.path.join(DATA_DIR_D, "gbpusd_h1_real.csv"), "GBPUSD Raw")

# 2. Institutional Merged Data
print("\n[2] INSTITUTIONAL MERGED DATA (Local):")
df_inst = audit_csv(os.path.join(DATA_DIR_LOCAL, "institutional_data_20y.csv"), "Institutional 20Y")

# 3. Train & Holdout Splits
print("\n[3] TRAIN & HOLDOUT SPLITS:")
df_train = audit_csv(os.path.join(DATA_DIR_LOCAL, "institutional_data_train.csv"), "Train Set")
df_holdout = audit_csv(os.path.join(DATA_DIR_LOCAL, "holdout_data.csv"), "Holdout Set")

# 4. Microstructure Features
print("\n[4] MICROSTRUCTURE FEATURES:")
df_micro = audit_csv(os.path.join(DATA_DIR_LOCAL, "microstructure_features_20y.csv"), "Microstructure")

# 5. Macro & COT Data
print("\n[5] MACRO & COT DATA:")
df_macro = audit_csv(os.path.join(DATA_DIR_LOCAL, "macro_data.csv"), "Macro Data")
df_cot = audit_csv(os.path.join(DATA_DIR_LOCAL, "cot_data.csv"), "COT Data")

# 6. Code Audit (Meta-Labeler)
print("\n[6] CODE AUDIT (meta_labeler.py):")
with open("meta_labeler.py", "r", encoding="utf-8") as f:
    ml_code = f.read()
checks = {
    "Focal Loss": "def focal_loss_obj" in ml_code,
    "Uniqueness Weights": "def calculate_uniqueness_weights" in ml_code,
    "Continuous Signal": "def create_base_signal" in ml_code,
    "Triple Barrier": "def create_triple_barrier_target" in ml_code
}
for name, passed in checks.items():
    status = "✅ FOUND" if passed else "❌ MISSING"
    print(f"  {status} : {name}")

print("\n" + "=" * 70)
print("   AUDIT COMPLETE")
print("=" * 70)
