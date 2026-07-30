"""
feature_engineer.py - Institutional Feature Engineering V2
Integrates Macro Data (FRED), Cyclical Time, and Fractional Differentiation.
"""
import pandas as pd
import numpy as np
import os
from volume_profile import VolumeProfile

class FeatureEngineer:
    def __init__(self, lookback=14, ffd_d=0.4):
        self.lookback = lookback
        self.ffd_d = ffd_d

    def _frac_diff(self, series, d, thresh=1e-5):
        """Fractional Differentiation (López de Prado FFD)"""
        width = 10
        weights = [1.0]
        for k in range(1, width):
            w = weights[-1] * (1 - (d + k - 1) / k)
            if abs(w) < thresh: break
            weights.append(w)
        weights = np.array(weights[::-1])
        
        # Vectorized convolution
        df_val = series.values
        out = np.full(len(df_val), np.nan)
        for i in range(len(weights)-1, len(df_val)):
            window = df_val[i-len(weights)+1:i+1]
            if not np.isnan(window).any():
                out[i] = np.dot(weights, window)
        return pd.Series(out, index=series.index)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        
        # ==========================================
        # 1. MACRO FEATURES (V3 - Short-End Carry)
        # ==========================================
        macro_path = "data/macro_data.csv"
        if os.path.exists(macro_path):
            macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
            # FIX-INST: Use Short-End Rates (DFF, ECBDFR) instead of 10Y proxy
            cols_to_fill = ['US_10Y', 'DE_10Y', 'US_2Y', 'DXY', 'VIX', 'US_FFR', 'ECB_DF', 'US_DE_Spread', 'Short_End_Spread']
            existing_cols = [c for c in cols_to_fill if c in macro_df.columns]
            
            data = data.join(macro_df[existing_cols], how='left')
            data[existing_cols] = data[existing_cols].ffill()
            
            # Macro Derivatives
            data['DXY_change'] = data['DXY'].pct_change(1)
            data['VIX_change'] = data['VIX'].pct_change(1)
            
            # True Carry Trade Features (Short-End)
            if 'Short_End_Spread' in data.columns:
                data['Carry_Change'] = data['Short_End_Spread'].diff(1)
                data['Carry_Z_Score'] = (data['Short_End_Spread'] - data['Short_End_Spread'].rolling(50).mean()) / (data['Short_End_Spread'].rolling(50).std() + 1e-10)
            
            # 2Y-10Y Term Structure (Yield Curve Slope)
            if 'US_10Y' in data.columns and 'US_2Y' in data.columns:
                data['US_Term_Structure'] = data['US_10Y'] - data['US_2Y']
                data['Term_Structure_Change'] = data['US_Term_Structure'].diff(1)
        # COT POSITIONING FEATURES (Institutional)
        # ==========================================
        cot_path = "data/cot_data.csv"
        if os.path.exists(cot_path):
            cot_df = pd.read_csv(cot_path, index_col=0, parse_dates=True)
            data = data.join(cot_df, how='left')
            # Forward-fill weekly COT data to H1
            data[['COT_Net', 'COT_Z_Score', 'COT_Change']] = data[['COT_Net', 'COT_Z_Score', 'COT_Change']].ffill()
            
        # 2. FRACTIONAL DIFFERENTIATION (New)
        # ==========================================
        data['ffd_close'] = self._frac_diff(data['close'], self.ffd_d)
        data['ffd_close_change'] = data['ffd_close'].diff(1)

        # ==========================================
        # 3. PRICE & VOLATILITY FEATURES
        # ==========================================
        data["ret_1"] = data["close"].pct_change(1)
        data["ret_5"] = data["close"].pct_change(5)
        data["ret_10"] = data["close"].pct_change(10)
        data["volatility_10"] = data["ret_1"].rolling(10).std()
        data["volatility_20"] = data["ret_1"].rolling(20).std()
        data["atr_14"] = (data["high"] - data["low"]).rolling(14).mean()
        
        # ==========================================
        # VOLUME PROFILE FEATURES (New)
        # ==========================================
        vp = VolumeProfile(lookback=24, num_bins=20)
        data = vp.calculate_features(data)
        
        # ==========================================
        # 4. VOLUME & ORDER FLOW FEATURES
        # ==========================================
        if "volume" in data.columns:
            data["volume_z_20"] = (data["volume"] - data["volume"].rolling(20).mean()) / (data["volume"].rolling(20).std() + 1e-10)
        if "cvd" in data.columns:
            data["cvd_change"] = data["cvd"].diff(1)
            data["cvd_z_20"] = (data["cvd"] - data["cvd"].rolling(20).mean()) / (data["cvd"].rolling(20).std() + 1e-10)
        if "delta" in data.columns:
            data["delta_ma_5"] = data["delta"].rolling(5).mean()

        # ==========================================
        # 5. REGIME & DIVERGENCE FEATURES
        # ==========================================
        if "regime_name" in data.columns:
            regime_dummies = pd.get_dummies(data["regime_name"], prefix="regime")
            for col in regime_dummies.columns:
                data[col] = regime_dummies[col].astype(int)
        if "rolling_corr" in data.columns:
            data["corr_ma_5"] = data["rolling_corr"].rolling(5).mean()

        # ==========================================
        # 6. CYCLICAL TIME FEATURES (New)
        # ==========================================
        if isinstance(data.index, pd.DatetimeIndex):
            data["hour"] = data.index.hour
            data["day_of_week"] = data.index.dayofweek
            # Cyclical encoding
            data["hour_sin"] = np.sin(2 * np.pi * data["hour"]/24.0)
            data["hour_cos"] = np.cos(2 * np.pi * data["hour"]/24.0)
            data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"]/7.0)
            data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"]/7.0)

        # Drop NaN rows
        data.dropna(inplace=True)
        
        # Define feature columns (exclude raw price columns and strings)
        exclude_cols = ["open", "high", "low", "close", "volume", 
                        "EURUSD_open", "EURUSD_high", "EURUSD_low", "EURUSD_close", "EURUSD_volume",
                        "GBPUSD_open", "GBPUSD_high", "GBPUSD_low", "GBPUSD_close", "GBPUSD_volume",
                        "regime_name", "spread", "avg_spread", "TR", "+DM", "-DM", "+DI", "-DI", "ADX",
                        "MA_Fast", "MA_Slow", "ATR_Pct", "ATR",
                        "base_signal", "target", "future_ret"]
        
        feature_cols = [c for c in data.columns if c not in exclude_cols and data[c].dtype != bool]
        
        print(f"Feature Engineering V2 Complete: {len(feature_cols)} features created.")
        return data, feature_cols

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from institutional_data_handler import prepare_institutional_data
    from volume_shock_detector import VolumeShockDetector
    
    print("Loading Institutional Data...")
    df = prepare_institutional_data()
    
    print("Detecting Volume Shocks...")
    detector = VolumeShockDetector()
    df = detector.detect(df)
    
    print("\nBuilding Features V2...")
    fe = FeatureEngineer()
    enriched, features = fe.build_features(df)
    
    print(f"\nTotal rows: {len(enriched)}")
    print(f"Total features: {len(features)}")
    print(f"Features: {features}")
