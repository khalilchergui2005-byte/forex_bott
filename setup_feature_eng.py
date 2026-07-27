code = r'''"""
feature_engineer.py - Institutional Feature Engineering
Creates 50+ features from enriched data for ML Meta-Labeling.
"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, lookback=14):
        self.lookback = lookback

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds 50+ features from institutional enriched data.
        Input must contain: open, high, low, close, volume, 
        regime_name, cvd, delta, rolling_corr, bull_div, bear_div, 
        exhaustion, vol_shock, shock_dir
        """
        data = df.copy()
        
        # ==========================================
        # 1. PRICE FEATURES (10)
        # ==========================================
        data["ret_1"] = data["close"].pct_change(1)
        data["ret_2"] = data["close"].pct_change(2)
        data["ret_3"] = data["close"].pct_change(3)
        data["ret_5"] = data["close"].pct_change(5)
        data["ret_10"] = data["close"].pct_change(10)
        data["ret_20"] = data["close"].pct_change(20)
        data["momentum_5"] = data["close"] / data["close"].shift(5) - 1
        data["momentum_10"] = data["close"] / data["close"].shift(10) - 1
        data["high_low_range"] = (data["high"] - data["low"]) / data["close"]
        data["close_position"] = (data["close"] - data["low"]) / (data["high"] - data["low"] + 1e-10)

        # ==========================================
        # 2. VOLATILITY FEATURES (8)
        # ==========================================
        data["volatility_5"] = data["ret_1"].rolling(5).std()
        data["volatility_10"] = data["ret_1"].rolling(10).std()
        data["volatility_20"] = data["ret_1"].rolling(20).std()
        data["vol_ratio_5_20"] = data["volatility_5"] / (data["volatility_20"] + 1e-10)
        data["atr_14"] = (data["high"] - data["low"]).rolling(14).mean()
        data["atr_pct"] = data["atr_14"] / data["close"]
        data["atr_ratio"] = data["atr_14"] / (data["atr_14"].rolling(50).mean() + 1e-10)
        data["range_expansion"] = (data["high"] - data["low"]) / (data["atr_14"] + 1e-10)

        # ==========================================
        # 3. VOLUME FEATURES (8)
        # ==========================================
        data["volume_change"] = data["volume"].pct_change(1)
        data["volume_ma_5"] = data["volume"].rolling(5).mean()
        data["volume_ma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio"] = data["volume_ma_5"] / (data["volume_ma_20"] + 1e-10)
        data["volume_z_20"] = (data["volume"] - data["volume"].rolling(20).mean()) / (data["volume"].rolling(20).std() + 1e-10)
        data["volume_z_50"] = (data["volume"] - data["volume"].rolling(50).mean()) / (data["volume"].rolling(50).std() + 1e-10)
        data["cum_volume_5"] = data["volume"].rolling(5).sum()
        data["volume_trend"] = data["volume"].rolling(5).mean() / (data["volume"].rolling(20).mean() + 1e-10)

        # ==========================================
        # 4. ORDER FLOW FEATURES (8)
        # ==========================================
        if "cvd" in data.columns:
            data["cvd_change"] = data["cvd"].diff(1)
            data["cvd_ma_5"] = data["cvd"].rolling(5).mean()
            data["cvd_ma_20"] = data["cvd"].rolling(20).mean()
            data["cvd_divergence"] = data["cvd"] - data["cvd_ma_20"]
            data["cvd_slope_5"] = data["cvd"].diff(5)
            data["cvd_slope_10"] = data["cvd"].diff(10)
            data["cvd_ratio"] = data["cvd_ma_5"] / (data["cvd_ma_20"] + 1e-10)
            data["cvd_z_20"] = (data["cvd"] - data["cvd"].rolling(20).mean()) / (data["cvd"].rolling(20).std() + 1e-10)

        if "delta" in data.columns:
            data["delta_ma_5"] = data["delta"].rolling(5).mean()
            data["delta_positive_ratio"] = (data["delta"] > 0).rolling(20).mean()

        # ==========================================
        # 5. REGIME FEATURES (6)
        # ==========================================
        if "regime" in data.columns:
            # One-hot encode regime (5 categories: TREND_UP, TREND_DOWN, RANGE, VOLATILE, QUIET)
            regime_dummies = pd.get_dummies(data["regime_name"], prefix="regime")
            for col in regime_dummies.columns:
                data[col] = regime_dummies[col].astype(int)
            
        if "regime_confidence" in data.columns:
            data["regime_conf_ma_5"] = data["regime_confidence"].rolling(5).mean()
            data["regime_conf_change"] = data["regime_confidence"].diff(1)

        if "adx" in data.columns:
            data["adx_ma_5"] = data["adx"].rolling(5).mean()
            data["adx_change"] = data["adx"].diff(1)

        # ==========================================
        # 6. CORRELATION & DIVERGENCE FEATURES (6)
        # ==========================================
        if "rolling_corr" in data.columns:
            data["corr_ma_5"] = data["rolling_corr"].rolling(5).mean()
            data["corr_change"] = data["rolling_corr"].diff(1)
            data["corr_z_20"] = (data["rolling_corr"] - data["rolling_corr"].rolling(20).mean()) / (data["rolling_corr"].rolling(20).std() + 1e-10)

        if "bull_div" in data.columns:
            data["bull_div_count_20"] = data["bull_div"].rolling(20).sum()
        if "bear_div" in data.columns:
            data["bear_div_count_20"] = data["bear_div"].rolling(20).sum()

        # ==========================================
        # 7. SHOCK/NEWS FEATURES (4)
        # ==========================================
        if "vol_shock" in data.columns:
            data["vol_shock"] = data["vol_shock"].astype(int)
            data["shock_count_20"] = data["vol_shock"].rolling(20).sum()
        if "shock_dir" in data.columns:
            data["shock_dir_ma_5"] = data["shock_dir"].rolling(5).mean()

        # ==========================================
        # 8. TIME FEATURES (4)
        # ==========================================
        if isinstance(data.index, pd.DatetimeIndex):
            data["hour"] = data.index.hour
            data["day_of_week"] = data.index.dayofweek
            data["is_london"] = ((data["hour"] >= 7) & (data["hour"] < 16)).astype(int)
            data["is_ny"] = ((data["hour"] >= 12) & (data["hour"] < 21)).astype(int)

        # ==========================================
        # 9. MOVING AVERAGE FEATURES (6)
        # ==========================================
        data["ma_5"] = data["close"].rolling(5).mean()
        data["ma_10"] = data["close"].rolling(10).mean()
        data["ma_20"] = data["close"].rolling(20).mean()
        data["ma_50"] = data["close"].rolling(50).mean()
        data["ma_5_20_cross"] = data["ma_5"] / (data["ma_20"] + 1e-10)
        data["price_ma_20_ratio"] = data["close"] / (data["ma_20"] + 1e-10)

        # Drop NaN rows
        data.dropna(inplace=True)
        
        # Count features
        feature_cols = [c for c in data.columns if c not in ["open", "high", "low", "close", "volume", 
                                                              "EURUSD_open", "EURUSD_high", "EURUSD_low", 
                                                              "EURUSD_close", "EURUSD_volume",
                                                              "GBPUSD_open", "GBPUSD_high", "GBPUSD_low",
                                                              "GBPUSD_close", "GBPUSD_volume",
                                                              "regime_name", "spread", "avg_spread",
                                                              "TR", "+DM", "-DM", "+DI", "-DI", "ADX",
                                                              "MA_Fast", "MA_Slow", "ATR_Pct", "ATR",
                                                              "BB_mid", "BB_std", "BB_lower", "BB_upper",
                                                              "Asian_High", "Asian_Low", "Range_Size",
                                                              "Body_Ratio", "Raw_Signal", "dayofweek",
                                                              "hedge_ratio", "coint_p_value", "coint_valid",
                                                              "Pairs_Signal", "News_Signal"]]
        
        print(f"Feature Engineering Complete: {len(feature_cols)} features created.")
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
    
    print("\nBuilding Features...")
    fe = FeatureEngineer()
    enriched, features = fe.build_features(df)
    
    print(f"\nTotal rows after feature engineering: {len(enriched)}")
    print(f"Total features: {len(features)}")
    print(f"\nFeature List:\n{features}")
'''
open("feature_engineer.py", "w", encoding="utf-8").write(code)
print("SUCCESS: feature_engineer.py created")

import ast
ast.parse(code)
print("AST OK")
