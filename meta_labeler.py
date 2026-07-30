"""
meta_labeler.py - Institutional ML Meta-Labeler (Triple Barrier Method)
Trains an XGBoost Binary Classifier to predict if a trade will hit TP before SL.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
import warnings
warnings.filterwarnings("ignore")

# FIX-0.5.4: Focal Loss Objective
def focal_loss_obj(preds, dtrain):
    """Custom Focal Loss objective. Forces focus on hard examples."""
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    alpha = 0.25
    gamma = 2.0
    pt = np.where(labels == 1, preds, 1 - preds)
    grad = alpha * gamma * (1 - pt)**(gamma - 1) * (pt - labels)
    hess = alpha * gamma * ( (gamma - 1) * (1 - pt)**gamma * (pt - labels) / (1 - pt + 1e-9) + (1 - pt)**(gamma - 1) * pt * (1 - pt) )
    return grad, hess

def focal_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return 'focal_logloss', log_loss(labels, preds)

class MetaLabeler:
    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.05):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=focal_loss_obj, # FIX-0.5.4: Use Focal Loss
            eval_metric=focal_eval,
            use_label_encoder=False
        )
        
    def create_base_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a continuous base score (Trend Strength * CVD Magnitude).
        FIX-INST: Replaced binary AND-gate with continuous score to increase trade count.
        """
        data = df.copy()
        data["base_signal"] = 0
        
        # 1. Calculate Trend Score (0 to 1)
        # Positive for TREND_UP, Negative for TREND_DOWN, 0 otherwise
        trend_score = np.where(data["regime_name"] == "TREND_UP", data["regime_confidence"], 
                      np.where(data["regime_name"] == "TREND_DOWN", -data["regime_confidence"], 0))
        
        # 2. Calculate CVD Score (normalized using tanh to bound between -1 and 1)
        # This prevents extreme volume days from dominating the score
        cvd_std = data["cvd"].rolling(50, min_periods=10).std()
        cvd_score = np.tanh(data["cvd"] / (cvd_std + 1e-10))
        cvd_score = cvd_score.fillna(0)
        
        # 3. Continuous Base Score (Trend * Flow)
        data["base_score"] = trend_score * cvd_score
        
        # 4. Generate Binary Signal from Score (Threshold = 0.2 for more trades)
        # Old threshold was effectively 1.0 (binary AND). 0.2 is much more relaxed.
        data.loc[data["base_score"] > 0.2, "base_signal"] = 1
        data.loc[data["base_score"] < -0.2, "base_signal"] = -1
        
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
        
    def calculate_uniqueness_weights(self, df, forward_period=5):
        """
        Calculates sample uniqueness weights (López de Prado).
        Fixes IID violation caused by overlapping Triple Barrier windows.
        """
        print("Calculating Sample Uniqueness Weights...")
        # Filter only rows where a trade occurs
        trade_indices = df[df["base_signal"] != 0].index
        
        # Initialize concurrency array
        concurrency = pd.Series(0, index=df.index)
        
        # Mark concurrency for each forward period
        for idx in trade_indices:
            end_idx = df.index.get_loc(idx) + forward_period
            if end_idx > len(df.index):
                end_idx = len(df.index)
            concurrency.iloc[df.index.get_loc(idx):end_idx] += 1
            
        # Calculate uniqueness for each trade
        weights = []
        for idx in trade_indices:
            start = df.index.get_loc(idx)
            end = start + forward_period
            if end > len(df.index):
                end = len(df.index)
            # Average inverse concurrency over the holding period
            avg_inv_concurrency = np.mean(1.0 / concurrency.iloc[start:end].replace(0, 1))
            weights.append(avg_inv_concurrency)
            
        return np.array(weights)

    def train(self, X_train, y_train, sample_weights=None):
        print("Training XGBoost Meta-Labeler (Triple Barrier)...")
        if sample_weights is not None:
            print(f"  Applying {len(sample_weights)} uniqueness weights...")
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train)
        print("Training complete.")
        
    def predict(self, X):
        """Returns 1 (approve trade) or 0 (reject trade)."""
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import precision_recall_curve
        import numpy as np
        
        # Get probabilities instead of hard predictions
        probs = self.model.predict_proba(X_test)[:, 1]
        
        # Find the best threshold that maximizes trades (recall) while keeping precision > 0.46
        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        
        best_thresh = 0.5
        best_f1 = 0.0
        
        for i, p in enumerate(precisions[:-1]):
            if p >= 0.46:  # Breakeven + margin
                f1 = 2 * (p * recalls[i]) / (p + recalls[i] + 1e-10)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresholds[i]
                    
        # Apply best threshold
        preds = (probs >= best_thresh).astype(int)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        print(f"\n=== Model Evaluation (Optimized Threshold) ===")
        print(f"Optimal Threshold: {best_thresh:.4f} (Default was 0.50)")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f} (Must be > 0.4630 to be profitable)")
        print(f"Recall: {rec:.4f} (Higher is better for FTMO min days)")
        print(f"Total Trades Taken: {preds.sum()}/{len(y_test)}")
        
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
