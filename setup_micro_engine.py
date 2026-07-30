code = r'''"""
microstructure_engine.py - H1 Microstructure Feature Extractor
Calculates VPIN, OFI, and Spread Dynamics using the Tick Rule proxy.
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
        
        # 1. Apply H1 Tick Rule (Lee-Ready Proxy) to separate volume
        direction = np.sign(data['close'].diff().fillna(0))
        direction.replace(0, np.nan, inplace=True)
        direction.ffill(inplace=True)
        direction.fillna(0, inplace=True)
        
        data['buy_volume'] = data['volume'].where(direction > 0, 0)
        data['sell_volume'] = data['volume'].where(direction < 0, 0)
        
        # 2. VPIN (Volume-Synchronized Probability of Informed Trading)
        # Fraction of toxic (directional) volume
        total_vol = data['buy_volume'] + data['sell_volume'] + 1e-10
        vpin_raw = np.abs(data['buy_volume'] - data['sell_volume']) / total_vol
        data['vpin'] = vpin_raw.ewm(span=self.vpin_window).mean()
        data['vpin_z'] = (data['vpin'] - data['vpin'].rolling(100).mean()) / (data['vpin'].rolling(100).std() + 1e-10)
        
        # 3. OFI (Order Flow Imbalance)
        data['ofi_bar'] = data['buy_volume'] - data['sell_volume']
        data['ofi_cumulative'] = data['ofi_bar'].cumsum()
        data['ofi_z'] = (data['ofi_bar'] - data['ofi_bar'].rolling(self.ofi_window).mean()) / (data['ofi_bar'].rolling(self.ofi_window).std() + 1e-10)
        
        # 4. Spread Dynamics (Relative Spread Widening)
        if 'avg_spread' in data.columns:
            data['spread_z'] = (data['avg_spread'] - data['avg_spread'].rolling(self.spread_window).mean()) / (data['avg_spread'].rolling(self.spread_window).std() + 1e-10)
        else:
            data['spread_z'] = 0.0
            
        return data[['vpin', 'vpin_z', 'ofi_bar', 'ofi_cumulative', 'ofi_z', 'spread_z']]

if __name__ == "__main__":
    print("=" * 60)
    print("   MICROSTRUCTURE ENGINE (20-Year H1 Features)")
    print("=" * 60)
    
    print("\n[1/2] Loading 20-Year Institutional Data...")
    df = pd.read_csv("data/institutional_data_20y.csv", index_col=0, parse_dates=True)
    
    print("[2/2] Calculating Microstructure Features (VPIN, OFI, Spread)...")
    engine = MicrostructureEngine()
    micro_df = engine.calculate_features(df)
    
    # Save features
    micro_df.to_csv("data/microstructure_features_20y.csv")
    print(f"\nSaved to data/microstructure_features_20y.csv")
    print(f"Shape: {micro_df.shape}")
    print("\nSample Features:")
    print(micro_df.tail(10))
'''
open("microstructure_engine.py", "w", encoding="utf-8").write(code)
print("SUCCESS: microstructure_engine.py created")

import ast
ast.parse(code)
print("AST OK")
