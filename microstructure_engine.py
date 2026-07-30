"""
microstructure_engine.py - H1 Microstructure Feature Extractor
Calculates VPIN, OFI, Spread Dynamics, Tick Frequency, and Adverse Selection.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class MicrostructureEngine:
    def __init__(self, vpin_window=10, ofi_window=50, spread_window=50):
        self.vpin_window = vpin_window
        self.ofi_window = ofi_window
        self.spread_window = spread_window

    def calculate_features(self, df):
        data = df.copy()
        
        # 1. VPIN (Volume-Synchronized Probability of Informed Trading)
        total_vol = data['buy_volume'] + data['sell_volume'] + 1e-10
        vpin_raw = np.abs(data['buy_volume'] - data['sell_volume']) / total_vol
        data['vpin'] = vpin_raw.ewm(span=self.vpin_window).mean()
        data['vpin_z'] = (data['vpin'] - data['vpin'].rolling(100).mean()) / (data['vpin'].rolling(100).std() + 1e-10)
        
        # 2. OFI (Order Flow Imbalance)
        data['ofi_bar'] = data['buy_volume'] - data['sell_volume']
        data['ofi_cumulative'] = data['ofi_bar'].cumsum()
        data['ofi_z'] = (data['ofi_bar'] - data['ofi_bar'].rolling(self.ofi_window).mean()) / (data['ofi_bar'].rolling(self.ofi_window).std() + 1e-10)
        
        # 3. Spread Dynamics
        if 'avg_spread' in data.columns:
            data['spread_z'] = (data['avg_spread'] - data['avg_spread'].rolling(self.spread_window).mean()) / (data['avg_spread'].rolling(self.spread_window).std() + 1e-10)
        else:
            data['spread_z'] = 0.0
            
        # 4. Tick Frequency Proxy (Volatility of returns)
        data['tick_freq_proxy'] = data['close'].pct_change().rolling(24).std()
        
        # 5. Adverse Selection Proxy (Spread Z-Score * Volatility)
        data['adv_selection_proxy'] = data['spread_z'] * data['tick_freq_proxy']
        
        return data[['vpin', 'vpin_z', 'ofi_bar', 'ofi_cumulative', 'ofi_z', 'spread_z', 'tick_freq_proxy', 'adv_selection_proxy']]

if __name__ == "__main__":
    print("=" * 60)
    print("   MICROSTRUCTURE ENGINE (20-Year H1 Features)")
    print("=" * 60)
    
    print("\n[1/2] Loading 20-Year Institutional Data...")
    df = pd.read_csv("data/institutional_data_20y.csv", index_col=0, parse_dates=True)
    
    print("[2/2] Calculating Microstructure Features (VPIN, OFI, Spread, Freq, AdvSel)...")
    engine = MicrostructureEngine()
    micro_df = engine.calculate_features(df)
    
    # Save features
    micro_df.to_csv("data/microstructure_features_20y.csv")
    print(f"\nSaved to data/microstructure_features_20y.csv")
    print(f"Shape: {micro_df.shape}")
    print("\nSample Features:")
    print(micro_df.tail(10))
