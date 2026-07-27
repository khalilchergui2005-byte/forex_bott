import subprocess
subprocess.check_call(["pip", "install", "xgboost", "scikit-learn", "-q"])

code = r'''"""
meta_labeler.py - Institutional ML Meta-Labeler
Trains an XGBoost Binary Classifier to predict if a base signal will be profitable.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

class MetaLabeler:
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.05):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            use_label_encoder=False
        )
        
    def create_base_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a simple momentum base signal (Buy if ret_5 > 0, Sell if ret_5 < 0)."""
        data = df.copy()
        data["base_signal"] = 0
        data.loc[data["ret_5"] > 0, "base_signal"] = 1
        data.loc[data["ret_5"] < 0, "base_signal"] = -1
        return data
        
    def create_target(self, df: pd.DataFrame, forward_period=5) -> pd.DataFrame:
        """Creates the target variable: 1 if future return is positive, 0 otherwise."""
        data = df.copy()
        data["future_ret"] = data["close"].shift(-forward_period) / data["close"] - 1
        data["target"] = (data["future_ret"] > 0).astype(int)
        data.dropna(subset=["target"], inplace=True)
        return data
        
    def train(self, X_train, y_train):
        print("Training XGBoost Meta-Labeler...")
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
        print(f"\n=== Model Evaluation ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f} (When model says 'Buy', it's right {prec*100:.1f}% of the time)")
        print(f"Recall: {rec:.4f} (Model caught {rec*100:.1f}% of all profitable trades)")
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
    
    print("Creating Base Signal & Target...")
    ml = MetaLabeler()
    enriched = ml.create_base_signal(enriched)
    enriched = ml.create_target(enriched)
    
    # Filter only where base signal is active (we only want to label trades we would take)
    trade_data = enriched[enriched["base_signal"] != 0].copy()
    
    # Split Train/Test (80% Train, 20% Test - Time Series Split)
    split_idx = int(len(trade_data) * 0.8)
    
    X_train = trade_data.iloc[:split_idx][feature_cols]
    y_train = trade_data.iloc[:split_idx]["target"]
    
    X_test = trade_data.iloc[split_idx:][feature_cols]
    y_test = trade_data.iloc[split_idx:]["target"]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train and Evaluate
    ml.train(X_train, y_train)
    ml.evaluate(X_test, y_test)
'''
open("meta_labeler.py", "w", encoding="utf-8").write(code)
print("SUCCESS: meta_labeler.py created")

import ast
ast.parse(code)
print("AST OK"
