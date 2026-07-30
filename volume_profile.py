"""
volume_profile.py - Institutional Volume Profile Calculator
Calculates Rolling POC, Value Area, HVN, LVN from real tick volume.
Outputs features for ML Meta-Labeler.
"""
import pandas as pd
import numpy as np

class VolumeProfile:
    def __init__(self, lookback=24, num_bins=20, va_pct=0.70):
        self.lookback = lookback  # 24 hours for daily profile
        self.num_bins = num_bins
        self.va_pct = va_pct
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches dataframe with Volume Profile features.
        Requires 'high', 'low', 'close', 'volume', 'atr_14' columns.
        """
        data = df.copy()
        
        # Initialize columns
        data["dist_poc"] = np.nan
        data["in_value_area"] = 0
        data["at_hvn"] = 0
        data["at_lvn"] = 0
        data["poc_strength"] = np.nan  # POC volume relative to average
        
        for i in range(self.lookback, len(data)):
            # 1. Extract rolling window
            window = data.iloc[i-self.lookback:i]
            curr_close = data["close"].iloc[i]
            curr_atr = data["atr_14"].iloc[i] if "atr_14" in data.columns else 0.0010
            
            # 2. Calculate Typical Price and assign volume
            typical_prices = (window["high"] + window["low"] + window["close"]) / 3
            volumes = window["volume"].values
            
            # 3. Bin prices into ranges
            price_min = window["low"].min()
            price_max = window["high"].max()
            
            if price_max == price_min:
                continue
                
            bins = np.linspace(price_min, price_max, self.num_bins + 1)
            # Use digitize to assign each price to a bin
            bin_indices = np.digitize(typical_prices, bins) - 1
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
            
            # 4. Aggregate volume by bin
            vol_by_bin = np.zeros(self.num_bins)
            for b_idx, vol in zip(bin_indices, volumes):
                vol_by_bin[b_idx] += vol
                
            # 5. Find POC (Point of Control)
            poc_bin_idx = np.argmax(vol_by_bin)
            poc_price = (bins[poc_bin_idx] + bins[poc_bin_idx + 1]) / 2
            poc_vol = vol_by_bin[poc_bin_idx]
            
            # 6. Calculate Value Area (70% of volume)
            total_vol = np.sum(vol_by_bin)
            target_va_vol = total_vol * self.va_pct
            
            # Expand from POC outward until we capture 70% volume
            va_indices = [poc_bin_idx]
            current_vol = poc_vol
            low_idx = poc_bin_idx - 1
            high_idx = poc_bin_idx + 1
            
            while current_vol < target_va_vol and (low_idx >= 0 or high_idx < self.num_bins):
                low_vol = vol_by_bin[low_idx] if low_idx >= 0 else -1
                high_vol = vol_by_bin[high_idx] if high_idx < self.num_bins else -1
                
                if high_vol > low_vol:
                    va_indices.append(high_idx)
                    current_vol += high_vol
                    high_idx += 1
                else:
                    va_indices.append(low_idx)
                    current_vol += low_vol
                    low_idx -= 1
                    
            va_low = bins[min(va_indices)]
            va_high = bins[max(va_indices) + 1]
            
            # 7. Identify HVN and LVN
            avg_vol = total_vol / self.num_bins
            hvn_threshold = avg_vol * 1.5
            lvn_threshold = avg_vol * 0.5
            
            # Find which bin the current close falls into
            curr_bin_idx = np.digitize([curr_close], bins)[0] - 1
            curr_bin_idx = max(0, min(curr_bin_idx, self.num_bins - 1))
            curr_bin_vol = vol_by_bin[curr_bin_idx]
            
            # 8. Assign Features
            data.iloc[i, data.columns.get_loc("dist_poc")] = (curr_close - poc_price) / (curr_atr + 1e-10)
            data.iloc[i, data.columns.get_loc("in_value_area")] = 1 if va_low <= curr_close <= va_high else 0
            data.iloc[i, data.columns.get_loc("at_hvn")] = 1 if curr_bin_vol >= hvn_threshold else 0
            data.iloc[i, data.columns.get_loc("at_lvn")] = 1 if curr_bin_vol <= lvn_threshold else 0
            data.iloc[i, data.columns.get_loc("poc_strength")] = poc_vol / (avg_vol + 1e-10)
            
        return data

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import os
    from regime_detector import RegimeDetector
    
    DATA_DIR = r"D:\forex_bot_data"
    print("Testing Volume Profile on EURUSD H1 (Last 1000 bars)...")
    
    file_path = os.path.join(DATA_DIR, "eurusd_h1_real.csv")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True).tail(1000)
    
    # Need ATR for distance normalization
    rd = RegimeDetector()
    df = rd._compute_indicators(df)
    
    vp = VolumeProfile(lookback=24, num_bins=20)
    enriched = vp.calculate_features(df)
    
    # Drop initial NaNs
    enriched.dropna(subset=["dist_poc"], inplace=True)
    
    print(f"\nTotal rows: {len(enriched)}")
    print(f"POC Distance Stats: Min={enriched['dist_poc'].min():.2f}, Max={enriched['dist_poc'].max():.2f}")
    print(f"In Value Area: {enriched['in_value_area'].sum()}/{len(enriched)} ({enriched['in_value_area'].mean()*100:.1f}%)")
    print(f"At HVN: {enriched['at_hvn'].sum()}/{len(enriched)} ({enriched['at_hvn'].mean()*100:.1f}%)")
    print(f"At LVN: {enriched['at_lvn'].sum()}/{len(enriched)} ({enriched['at_lvn'].mean()*100:.1f}%)")
    
    print("\nSample Data:")
    print(enriched[["close", "volume", "dist_poc", "in_value_area", "at_hvn", "at_lvn", "poc_strength"]].tail(10))
