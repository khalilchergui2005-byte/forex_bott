"""
regime_detector_gmm.py - Gaussian Mixture Model for Market Regime Detection
Replaces HMM to avoid C++ build dependencies. Uses scikit-learn.
Trained on an expanding window to prevent look-ahead bias.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture
import os

class GMMRegimeDetector:
    def __init__(self, n_states=4):
        self.n_states = n_states
        self.model = None
        
    def _prepare_features(self, df):
        """Extracts and normalizes inputs."""
        data = df.copy()
        
        # 1. Log Returns
        data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        
        # 2. 24h Volatility
        data['vol_24h'] = data['log_ret'].rolling(24).std()
        
        # 3. VPIN Proxy (with Tick Rule fallback)
        if 'buy_volume' not in data.columns or 'sell_volume' not in data.columns:
            # FIX: Apply H1 Tick Rule proxy if tick-level separation is missing
            direction = np.sign(data['close'].diff().fillna(0))
            direction.replace(0, np.nan, inplace=True)
            direction.ffill(inplace=True)
            direction.fillna(0, inplace=True)
            data['buy_volume'] = data['volume'].where(direction > 0, 0)
            data['sell_volume'] = data['volume'].where(direction < 0, 0)
            
        total_vol = data['buy_volume'] + data['sell_volume'] + 1e-10
        data['vpin_proxy'] = np.abs(data['buy_volume'] - data['sell_volume']) / total_vol
        
        features = data[['log_ret', 'vol_24h', 'vpin_proxy']].dropna()
        
        # Normalize
        features = (features - features.mean()) / features.std()
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.dropna(inplace=True)
        
        return features
        
    def fit_predict_causal(self, df):
        """
        Trains GMM on an expanding window (year by year) to prevent look-ahead bias.
        Predicts probabilities for each bar based ONLY on past data.
        """
        print("Preparing Regime features (LogRet, Vol, VPIN)...")
        X = self._prepare_features(df)
        
        print(f"Running Causal GMM on {len(X)} samples...")
        
        # Group by year for expanding window refit
        X['year'] = X.index.year
        years = sorted(X['year'].unique())
        
        all_probs = []
        
        for i, year in enumerate(years):
            # Train on all data up to the previous year
            train_data = X[X['year'] < year].drop(columns=['year'])
            test_data = X[X['year'] == year].drop(columns=['year'])
            
            if len(train_data) < 1000 or len(test_data) == 0:
                continue
                
            # Fit GMM
            model = GaussianMixture(
                n_components=self.n_states, 
                covariance_type='full', 
                random_state=42, 
                n_init=1
            )
            model.fit(train_data.values)
            
            # Predict probabilities for current year
            probs = model.predict_proba(test_data.values)
            prob_cols = [f"regime_state_{j+1}_prob" for j in range(self.n_states)]
            probs_df = pd.DataFrame(probs, index=test_data.index, columns=prob_cols)
            all_probs.append(probs_df)
            
            print(f"  Trained up to {year-1}, predicted {year} ({len(test_data)} bars)")
            
        if not all_probs:
            return pd.DataFrame()
            
        final_probs = pd.concat(all_probs)
        final_probs.sort_index(inplace=True)
        return final_probs

if __name__ == "__main__":
    print("=" * 60)
    print("   GMM REGIME DETECTOR (Causal Expanding Window)")
    print("=" * 60)
    
    print("\n[1/3] Loading 20-Year Institutional Data...")
    df = pd.read_csv("data/institutional_data_20y.csv", index_col=0, parse_dates=True)
    
    print("\n[2/3] Running Causal GMM (4 States)...")
    detector = GMMRegimeDetector(n_states=4)
    probs_df = detector.fit_predict_causal(df)
    
    if not probs_df.empty:
        print(f"\n[3/3] Saved Regime probabilities to data/regime_gmm_probs.csv")
        print(f"Shape: {probs_df.shape}")
        print("\nSample Probabilities:")
        print(probs_df.tail(10))
        probs_df.to_csv("data/regime_gmm_probs.csv")
    else:
        print("ERROR: No probabilities generated.")
