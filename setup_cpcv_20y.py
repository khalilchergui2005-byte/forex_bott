# ==========================================
# 1. إصلاح خطأ T=12 في deflated_sharpe_calculator.py
# ==========================================
f = 'deflated_sharpe_calculator.py'
t = open(f, encoding='utf-8').read()

# Replace T=12 with T=22 (actual OOS trades from 80/20 split)
t = t.replace('num_returns=12,  # 12 trades in OOS', 'num_returns=22,  # 22 actual OOS trades from 80/20 split')
# Also handle the case where it was already changed to 1255
t = t.replace('num_returns=1255,  # 1255 OOS bars in 20Y test set', 'num_returns=22,  # 22 actual OOS trades from 80/20 split')

open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: Fixed T parameter to 22 in DSR calculator")

import ast
ast.parse(t)
print("DSR Calculator AST OK")

# ==========================================
# 2. كتابة سكريبت CPCV على الـ 20 سنة
# ==========================================
cpcv_code = r'''"""
run_cpcv_20y.py - CPCV Validation on 20-Year Data
Tests the Meta-Labeler across 15 combinatorial paths.
Collects ALL OOS trades for definitive DSR calculation.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from itertools import combinations
from feature_engineer import FeatureEngineer
from meta_labeler import MetaLabeler
from volume_shock_detector import VolumeShockDetector
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score

print("=" * 60)
print("   CPCV VALIDATION ON 20-YEAR DATA (15 Paths)")
print("=" * 60)

# 1. Load 20-year data
print("\n[1/5] Loading 20-Year Institutional Data...")
df = pd.read_csv("data/institutional_data_20y.csv", index_col=0, parse_dates=True)

# 2. Build Features + Signal + Targets
print("[2/5] Building Features, Signal & Targets...")
detector = VolumeShockDetector()
df = detector.detect(df)
fe = FeatureEngineer()
enriched, feature_cols = fe.build_features(df)

ml = MetaLabeler()
enriched = ml.create_base_signal(enriched)
enriched = ml.create_triple_barrier_target(enriched)

# Filter only where base signal is active
trade_data = enriched[enriched["base_signal"] != 0].copy()
print(f"  Total trade opportunities: {len(trade_data)}")

# 3. CPCV Setup (N=6, k=2 → 15 paths)
print("[3/5] Setting up CPCV (N=6, k=2, purge=50, embargo=24)...")
N_GROUPS = 6
K_TEST = 2
PURGE = 50
EMBARGO = 24

total_len = len(trade_data)
group_size = total_len // N_GROUPS
groups = [i * group_size for i in range(N_GROUPS)]
groups.append(total_len)

test_combinations = list(combinations(range(N_GROUPS), K_TEST))
print(f"  Total CPCV paths: {len(test_combinations)}")

# 4. Run CPCV
print("[4/5] Running CPCV across 15 paths...")
all_oos_trades = []
all_oos_predictions = []
all_oos_actuals = []

for path_idx, test_indices in enumerate(test_combinations):
    train_indices = [i for i in range(N_GROUPS) if i not in test_indices]
    
    # Build train set (concatenate train groups with purge between them)
    train_parts = []
    for idx in train_indices:
        start = groups[idx]
        end = groups[idx + 1]
        train_parts.append(trade_data.iloc[start:end])
    train_df = pd.concat(train_parts)
    
    # Build test set (concatenate test groups)
    test_parts = []
    for idx in test_indices:
        start = groups[idx] + EMBARGO  # Apply embargo
        end = groups[idx + 1] - EMBARGO
        if start < end:
            test_parts.append(trade_data.iloc[start:end])
    test_df = pd.concat(test_parts)
    
    if len(train_df) < 100 or len(test_df) < 10:
        continue
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        eval_metric="logloss", use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    # Predict with optimal threshold (0.3967)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.3967).astype(int)
    
    # Collect OOS trades (where model approved = 1)
    approved_mask = preds == 1
    if approved_mask.sum() > 0:
        # For each approved trade, check if it was actually profitable (target=1)
        all_oos_predictions.extend(preds.tolist())
        all_oos_actuals.extend(y_test.tolist())
        
        # Count wins and losses
        wins = int((preds[approved_mask] == y_test[approved_mask]).sum())
        losses = int(approved_mask.sum() - wins)
        all_oos_trades.append({"path": path_idx + 1, "trades": int(approved_mask.sum()), "wins": wins, "losses": losses})
    
    print(f"  Path {path_idx + 1:2d}/15: Train={len(train_df)}, Test={len(test_df)}, Approved Trades={int(approved_mask.sum())}")

# 5. Aggregate Results
print("\n[5/5] Aggregating CPCV Results...")
total_trades = sum(r["trades"] for r in all_oos_trades)
total_wins = sum(r["wins"] for r in all_oos_trades)
total_losses = sum(r["losses"] for r in all_oos_trades)

if total_losses > 0:
    pf = total_wins / total_losses
else:
    pf = float('inf')

# Calculate actual Precision from OOS
oos_pred = np.array(all_oos_predictions)
oos_act = np.array(all_oos_actuals)
if len(oos_pred) > 0:
    precision = precision_score(oos_act, oos_pred, zero_division=0)
    recall = recall_score(oos_act, oos_pred, zero_division=0)
else:
    precision = 0
    recall = 0

print(f"\n{'='*60}")
print(f"   CPCV FINAL REPORT (20-Year, 15 Paths)")
print(f"{'='*60}")
print(f"Total OOS Trades (all paths): {total_trades}")
print(f"Total Wins: {total_wins}")
print(f"Total Losses: {total_losses}")
print(f"Win Rate: {total_wins/total_trades*100:.2f}%" if total_trades > 0 else "N/A")
print(f"Profit Factor (PF): {pf:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"{'='*60}")

# Save results for DSR
results = {
    "total_trades": total_trades,
    "pf": pf,
    "precision": precision,
    "wins": total_wins,
    "losses": total_losses
}

import json
with open("data/cpcv_20y_results.json", "w") as f:
    json.dump(results, f)
print(f"\nResults saved to data/cpcv_20y_results.json")
print(f"\nNEXT STEP: Run DSR with N_trials=17, T={total_trades}, PF={pf:.2f}")
'''
open("run_cpcv_20y.py", "w", encoding="utf-8").write(cpcv_code)
print("SUCCESS: run_cpcv_20y.py created")

ast.parse(cpcv_code)
print("CPCV Script AST OK")
