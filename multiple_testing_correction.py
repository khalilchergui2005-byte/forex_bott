"""
multiple_testing_correction.py - Bonferroni Correction (Council Atom 8.3)
Calculates the total number of trials across all project history,
and applies strict correction to the temporal edge discovery.
"""
import pandas as pd
import numpy as np
from scipy import stats

print("=" * 60)
print("   MULTIPLE TESTING CORRECTION - Temporal Edge")
print("=" * 60)

# 1. Estimate total number of independent trials across project history
print("\n[1/3] Estimating total trials across project history...")

# Known trials from our documented history:
# - 7 parameters tested in Bayesian Optimization (100 trials each)
# - 5 Walk-Forward windows per run
# - Number of distinct optimizer runs
optimizer_runs = 3  # optimizer_safe (original), optimizer_foundry (x2: normal + normalized)
trials_per_run = 100
params_per_trial = 7
walk_forward_windows = 5
meta_labeling_combinations = 3  # (min_smart_score, max_fear, require_shock)

# Rough estimate: total independent hypotheses tested
total_trials = (optimizer_runs * trials_per_run * params_per_trial) + \
               (meta_labeling_combinations * walk_forward_windows) + \
               24  # hours tested in temporal edge (conservative)
print(f"   Estimated total independent trials: {total_trials}")

# 2. Original p-value for the temporal edge (Hour 00:00)
print("\n[2/3] Applying Bonferroni correction...")
original_p_value = 0.002  # From temporal_edge_test.py for Hour 00:00
corrected_p_value = min(original_p_value * total_trials, 1.0)

print(f"   Original p-value (Hour 00:00): {original_p_value}")
print(f"   Corrected p-value (after {total_trials} trials): {corrected_p_value}")
print(f"   Still significant at 0.05? {'YES' if corrected_p_value < 0.05 else 'NO'}")

# 3. Apply to Hour 13:00
original_p_value_13 = 0.049
corrected_p_value_13 = min(original_p_value_13 * total_trials, 1.0)
print(f"\n   Original p-value (Hour 13:00): {original_p_value_13}")
print(f"   Corrected p-value (after {total_trials} trials): {corrected_p_value_13}")
print(f"   Still significant at 0.05? {'YES' if corrected_p_value_13 < 0.05 else 'NO'}")

# 4. Final verdict
print(f"\n{'='*60}")
print(f"   FINAL VERDICT:")
if corrected_p_value < 0.05:
    print(f"   ✅ Hour 00:00 edge SURVIVES Bonferroni correction.")
    print(f"   This edge is statistically robust against multiple testing.")
    print(f"   Proceed to Atom 8.4 (Robustness ±15/±30 min).")
else:
    print(f"   ❌ Hour 00:00 edge FAILS Bonferroni correction.")
    print(f"   The discovery is likely a statistical artifact.")
    print(f"   Consider this path CLOSED. Move to Atom 8.4 for verification.")
print(f"{'='*60}")
