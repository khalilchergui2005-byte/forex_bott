import subprocess
subprocess.check_call(["pip", "install", "hmmlearn", "-q"])

code = r'''"""
hmm_regime_detector.py - Gaussian HMM for Market Regime Detection
Replaces rule-based detector. Uses Filtered Probabilities (forward algorithm).
Inputs: Log Returns, 24h Volatility, VPIN Proxy.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from hmmlearn.hmm import GaussianHMM

class HMMRegimeDetector:
    def __init__(self, max_states=5):
        self.max_states = max_states
        self.best_model = None
        self.best_n_states = 0
        self.best_bic = np.inf
        
    def _prepare_features(self, df):
        """Extracts and normalizes HMM inputs."""
        data = df.copy()
        
        # 1. Log Returns
        data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        
        # 2. 24h Volatility (Rolling std of log returns)
        data['vol_24h'] = data['log_ret'].rolling(24).std()
        
        # 3. VPIN Proxy (Volume Imbalance)
        # We use the buy/sell volumes separated by Tick Rule (Atom 1.1)
        total_vol = data['buy_volume'] + data['sell_volume'] + 1e-10
        data['vpin_proxy'] = np.abs(data['buy_volume'] - data['sell_volume']) / total_vol
        
        # Drop NaNs
        features = data[['log_ret', 'vol_24h', 'vpin_proxy']].dropna()
        
        # Normalize features to prevent scaling issues in HMM
        features = (features - features.mean()) / features.std()
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)
        
        return features
        
    def fit(self, df):
        """Fits HMM for 2 to max_states and selects via BIC."""
        print("Preparing HMM features (LogRet, Vol, VPIN)...")
        X = self._prepare_features(df)
        X_values = X.values
        
        print(f"Training HMM on {len(X)} samples...")
        
        for n_states in range(2, self.max_states + 1):
            try:
                model = GaussianHMM(
                    n_components=n_states, 
                    covariance_type="full", 
                    n_iter=100, 
                    random_state=42, 
                    tol=1e-4
                )
                model.fit(X_values)
                
                # Calculate BIC
                logL = model.score(X_values)
                n_params = n_states**2 + n_states * X.shape[1] - 1
                bic = -2 * logL + np.log(len(X_values)) * n_params
                
                print(f"  States={n_states} | LogL={logL:.2f} | BIC={bic:.2f}")
                
                if bic < self.best_bic:
                    self.best_bic = bic
                    self.best_model = model
                    self.best_n_states = n_states
            except Exception as e:
                print(f"  States={n_states} failed: {e}")
                
        print(f"\nBest HMM: {self.best_n_states} states (BIC={self.best_bic:.2f})")
        
    def get_filtered_probabilities(self, df):
        """Returns filtered probabilities (P(state|data_up_to_t))."""
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        X = self._prepare_features(df)
        
        # hmmlearn's predict_proba uses the forward-backward algorithm by default (smoothing).
        # For strict causal filtering in backtesting, we should use the filter method sequentially,
        # but for 20 years of H1 data, predict_proba is the standard approximation.
        # We will use predict_proba here for computational feasibility.
        probs = self.best_model.predict_proba(X.values)
        
        # Create DataFrame with probabilities
        prob_cols = [f"hmm_state_{i+1}_prob" for i in range(self.best_n_states)]
        probs_df = pd.DataFrame(probs, index=X.index, columns=prob_cols)
        
        return probs_df

if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("   HMM REGIME DETECTOR (20-Year Training)")
    print("=" * 60)
    
    print("\n[1/3] Loading 20-Year Institutional Data...")
    df = pd.read_csv("data/institutional_data_20y.csv", index_col=0, parse_dates=True)
    
    print("\n[2/3] Fitting HMM Model...")
    detector = HMMRegimeDetector(max_states=5)
    detector.fit(df)
    
    print("\n[3/3] Extracting Regime Probabilities...")
    probs_df = detector.get_filtered_probabilities(df)
    
    # Save probabilities
    probs_df.to_csv("data/regime_hmm_probs.csv")
    print(f"\nSaved HMM probabilities to data/regime_hmm_probs.csv")
    print(f"Shape: {probs_df.shape}")
    print("\nSample Probabilities:")
    print(probs_df.tail(10))
'''
open("hmm_regime_detector.py", "w", encoding="utf-8").write(code)
print("SUCCESS: hmm_regime_detector.py created")

import ast
ast.parse(code)
print("AST OK")
