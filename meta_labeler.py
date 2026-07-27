"""
meta_labeler.py - Institutional ML Meta-Labeler (Triple Barrier Method)
Trains an XGBoost Binary Classifier to predict if a trade will hit TP before SL.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

class MetaLabeler:
    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.05):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            use_label_encoder=False
        )
        
    def create_base_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a logical base signal (Trend + Order Flow Confirmation)."""
        data = df.copy()
        data["base_signal"] = 0
        
        # Buy: Trend is UP and Order Flow (CVD) is positive
        buy_cond = (data["regime_name"] == "TREND_UP") & (data["cvd"] > 0)
        data.loc[buy_cond, "base_signal"] = 1
        
        # Sell: Trend is DOWN and Order Flow (CVD) is negative
        sell_cond = (data["regime_name"] == "TREND_DOWN") & (data["cvd"] < 0)
        data.loc[sell_cond, "base_signal"] = -1
        
        return data
        
    def create_triple_barrier_target(self, df: pd.DataFrame, tp_mult=2.0, sl_mult=1.5, forward_period=5) -> pd.DataFrame:
        """
        Creates the target variable using Triple Barrier Method.
        1 if TP is hit first, 0 if SL is hit first or timeout.
        """
        data = df.copy()
        targets = []
        
        for i in range(len(data) - forward_period):
            signal = data["base_signal"].iloc[i]
            entry = data["close"].iloc[i]
            atr = data["atr_14"].iloc[i]
            
            # FIX-INST: Widen TP barrier by transaction costs to prevent learning false positives
            # Round-trip cost = Spread (entry+exit) + Slippage (entry+exit) + Commission
            # Assuming avg ATR ~20 pips, 2.7 pips cost = 0.135 ATR (using a safe fixed 0.15 ATR proxy)
            # Or better: calculate from avg_spread if available
            cost_in_price = data["avg_spread"].iloc[i] if "avg_spread" in data.columns else 0.00010  # Fallback 1.0 pip
            cost_in_price += 0.00005  # Slippage 0.5 pip
            cost_in_price += 0.00007  # Commission ~0.7 pip
            cost_in_atr = cost_in_price / (atr + 1e-10)
            
            if signal == 1:  # Long
                tp = entry + ((tp_mult + cost_in_atr) * atr)
                sl = entry - (sl_mult * atr)  # SL unchanged, realized loss is worse by cost
            elif signal == -1:  # Short
                tp = entry - ((tp_mult + cost_in_atr) * atr)
                sl = entry + (sl_mult * atr)
            else:
                targets.append(0)
                continue
                
            hit_tp = False
            hit_sl = False
            
            for j in range(1, forward_period + 1):
                high = data["high"].iloc[i + j]
                low = data["low"].iloc[i + j]
                
                if signal == 1:
                    if high >= tp:
                        hit_tp = True
                        break
                    if low <= sl:
                        hit_sl = True
                        break
                else: # Short
                    if low <= tp:
                        hit_tp = True
                        break
                    if high >= sl:
                        hit_sl = True
                        break
                        
            targets.append(1 if hit_tp else 0)
            
        # Pad the end with 0s
        targets.extend([0] * forward_period)
        data["target"] = targets
        data.dropna(subset=["target"], inplace=True)
        return data
        
    def train(self, X_train, y_train):
        print("Training XGBoost Meta-Labeler (Triple Barrier)...")
        self.model.fit(X_train, y_train)
        print("Training complete.")
        
    def predict(self, X):
        """Returns 1 (approve trade) or 0 (reject trade)."""
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        print(f"\n=== Model Evaluation (Triple Barrier) ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f} (When model says 'Take Trade', it hits TP {prec*100:.1f}% of the time)")
        print(f"Recall: {rec:.4f} (Model caught {rec*100:.1f}% of all TP hits)")
        return acc, prec, rec

if __name__ == "__main__":
    from feature_engineer import FeatureEngineer
    from institutional_data_handler import prepare_institutional_data
    from volume_shock_detector import VolumeShockDetector
    
    print("Loading Institutional Data...")
    df = prepare_institutional_data()
    
    print("Detecting Volume Shocks...")
    detector = VolumeShockDetector()
    df = detector.detect(df)
    
    print("Building Features...")
    fe = FeatureEngineer()
    enriched, feature_cols = fe.build_features(df)
    
    print("Creating Base Signal...")
    ml = MetaLabeler()
    enriched = ml.create_base_signal(enriched)
    
    print("Calculating Triple Barrier Targets (TP=2.0 ATR, SL=1.5 ATR, Hold=5h)...")
    enriched = ml.create_triple_barrier_target(enriched)
    
    # Filter only where base signal is active
    trade_data = enriched[enriched["base_signal"] != 0].copy()
    
    # Split Train/Test (80% Train, 20% Test - Time Series Split)
    split_idx = int(len(trade_data) * 0.8)
    
    X_train = trade_data.iloc[:split_idx][feature_cols]
    y_train = trade_data.iloc[:split_idx]["target"]
    
    X_test = trade_data.iloc[split_idx:][feature_cols]
    y_test = trade_data.iloc[split_idx:]["target"]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Positive Targets (TP hit) in Train: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    
    # Train and Evaluate
    ml.train(X_train, y_train)
    ml.evaluate(X_test, y_test)
