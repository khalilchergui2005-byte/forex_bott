import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from feature_engineer import FeatureEngineer
from meta_labeler import MetaLabeler
from volume_shock_detector import VolumeShockDetector

print("=" * 60)
print("   20-YEAR META-LABELER TRAINING (Continuous Signal)")
print("=" * 60)

print("\n[1/5] Loading 20-Year Institutional Data...")
df = pd.read_csv("data/institutional_data_20y.csv", index_col=0, parse_dates=True)

print("[2/5] Detecting Volume Shocks & Building Features...")
detector = VolumeShockDetector()
df = detector.detect(df)
fe = FeatureEngineer()
enriched, feature_cols = fe.build_features(df)

print("[3/5] Creating Continuous Base Signal & Targets...")
ml = MetaLabeler()
enriched = ml.create_base_signal(enriched)
enriched = ml.create_triple_barrier_target(enriched)

# Filter only where base signal is active
trade_data = enriched[enriched["base_signal"] != 0].copy()
print(f"\nTotal Base Signal Trades Generated: {len(trade_data)}")

if len(trade_data) < 100:
    print("ERROR: Still not enough trades. Need to lower threshold further.")
else:
    # Split Train/Test (80% Train, 20% Test - Time Series Split)
    split_idx = int(len(trade_data) * 0.8)

    X_train = trade_data.iloc[:split_idx][feature_cols]
    y_train = trade_data.iloc[:split_idx]["target"]

    X_test = trade_data.iloc[split_idx:][feature_cols]
    y_test = trade_data.iloc[split_idx:]["target"]

    print(f"\n[4/5] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Positive Targets (TP hit) in Train: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")

    print("\n[5/5] Training XGBoost Meta-Labeler...")
    ml.train(X_train, y_train)
    ml.evaluate(X_test, y_test)
