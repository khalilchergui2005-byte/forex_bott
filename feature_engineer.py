"""
feature_engineer.py V4 - Institutional Feature Purge
Hard-caps features at 13 economically motivated variables to prevent Curse of Dimensionality.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class FeatureEngineer:
    def __init__(self, ffd_d=0.4):
        self.ffd_d = ffd_d

    def _frac_diff(self, series, d, thresh=1e-5):
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < thresh:
                break
            w.append(w_k)
            k += 1
            if k > 100:
                break
        w = np.array(w[::-1]).reshape(-1, 1)
        width = len(w)
        df = pd.Series(index=series.index, dtype=float)
        values = series.values
        for i in range(width, len(values)):
            window = values[i-width:i].reshape(-1, 1)
            df.iloc[i] = np.sum(window * w)
        return df

    def build_features(self, df):
        data = df.copy()
        
        # 1. Price: Fractional Differentiation (Memory-preserving stationarity)
        data['ffd_close'] = self._frac_diff(data['close'], self.ffd_d)
        
        # 2. Define the exact 13 features to keep
        features_to_keep = [
            'ffd_close',
            'vpin', 'vpin_z', 'ofi_z', 'spread_z', 'tick_freq_proxy', 'adv_selection_proxy',
            'regime_state_1_prob', 'regime_state_2_prob', 'regime_state_3_prob', 'regime_state_4_prob',
            'cot_z_score', 
            'short_end_spread'
        ]
        
        # Ensure all features exist
        missing = [f for f in features_to_keep if f not in data.columns]
        if missing:
            raise ValueError(f"Missing required features in data: {missing}")
            
        enriched = data[features_to_keep].copy()
        enriched.dropna(inplace=True)
        
        print(f"Feature Engineering V4 (Purged) Complete: {len(features_to_keep)} features.")
        return enriched, features_to_keep

if __name__ == "__main__":
    print("=" * 60)
    print("   FEATURE ENGINEER V4 (PURGE)")
    print("=" * 60)
    
    print("\nLoading Train Data...")
    df = pd.read_csv("data/institutional_data_train.csv", index_col=0, parse_dates=True)
    
    print("Building Purged Features...")
    fe = FeatureEngineer()
    enriched, features = fe.build_features(df)
    
    print(f"\nTotal rows: {len(enriched)}")
    print(f"Total features: {len(features)}")
    print(f"Features: {features}")
