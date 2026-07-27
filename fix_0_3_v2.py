import os

# ==========================================
# 1. إصلاح optimizer_foundry.py - إضافة Purge (مع تجاهل BOM)
# ==========================================
f = 'optimizer_foundry.py'
# قراءة الملف مع utf-8-sig لتجاهل علامة BOM
t = open(f, encoding='utf-8-sig').read()

old_split = "    start_test = end_train\n    end_test = start_test + window_size"
new_split = "    PURGE_BARS = 10  # FIX-INST: Prevent label leakage across train/test boundary\n    start_test = end_train + PURGE_BARS\n    end_test = start_test + window_size"

if old_split in t:
    t = t.replace(old_split, new_split, 1)
    # الكتابة بدون BOM
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Added Purge (10 bars) to optimizer_foundry.py")
else:
    print("WARN: Pattern not found in optimizer_foundry.py (may already be fixed)")

import ast
ast.parse(t)
print("optimizer_foundry.py AST OK")

# ==========================================
# 2. إعادة كتابة walk_forward_optimizer.py بالكامل
# ==========================================
wf_code = r'''"""
walk_forward_optimizer.py - Institutional Walk-Forward Validator
Non-overlapping windows with Purge (10 bars) and Embargo (24 bars).
Generic: accepts any strategy column name.
"""
import pandas as pd
import numpy as np
from backtest_engine import dynamic_backtest
import warnings
warnings.filterwarnings("ignore")

def run_walk_forward(data, strategy_cols, risk_percent=0.005, num_windows=5, train_size=2000, test_size=1000, purge=10, embargo=24):
    """
    Runs Walk-Forward test with non-overlapping windows and Purge/Embargo.
    """
    print("=" * 60)
    print("   WALK-FORWARD VALIDATION (Purged & Embargoed)")
    print("=" * 60)
    
    gap = purge + embargo
    fold_span = train_size + gap + test_size
    total_bars = len(data)
    
    print(f"Total bars: {total_bars} (~{total_bars/24/30:.1f} months)")
    print(f"Fold span: {fold_span} bars (Train={train_size}, Gap={gap}, Test={test_size})")
    print(f"Windows: {num_windows}")
    print()
    
    results = []
    
    for w in range(num_windows):
        start_train = w * fold_span
        end_train = start_train + train_size
        start_test = end_train + purge
        end_test = start_test + test_size
        
        if end_test > total_bars:
            print(f"   Window {w+1}: Skipped (insufficient data)")
            break
        
        test_data = data.iloc[start_test:end_test].copy()
        
        res = dynamic_backtest(
            data=test_data,
            strategy_cols=strategy_cols,
            risk_percent=risk_percent,
            log_trades=False
        )
        
        trades = res[0] if isinstance(res, tuple) else res
        
        if trades and len(trades) > 5:
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t <= 0]
            pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0.0
            results.append({
                'window': w + 1,
                'pf': pf,
                'trades': len(trades),
                'win_rate': len(wins) / len(trades) * 100
            })
            print(f"   Window {w+1}: PF={pf:.2f}, Trades={len(trades)}, WinRate={len(wins)/len(trades)*100:.1f}%")
        else:
            results.append({
                'window': w + 1,
                'pf': 0.0,
                'trades': 0,
                'win_rate': 0.0
            })
            print(f"   Window {w+1}: No trades!")
    
    if results:
        pfs = [r['pf'] for r in results]
        avg_pf = np.mean(pfs)
        std_pf = np.std(pfs)
        total_trades = sum(r['trades'] for r in results)
        
        print(f"\n{'='*60}")
        print(f"   WALK-FORWARD FINAL REPORT")
        print(f"{'='*60}")
        print(f"   Windows tested: {len(results)}")
        print(f"   Total trades: {total_trades}")
        print(f"   Average PF: {avg_pf:.2f} (±{std_pf:.2f})")
        print(f"   Stability: {'STABLE' if avg_pf > 1.0 and std_pf < 0.5 else 'UNSTABLE'}")
        print(f"{'='*60}")
        
        return {
            'results': results,
            'avg_pf': avg_pf,
            'std_pf': std_pf,
            'stable': avg_pf > 1.0 and std_pf < 0.5,
            'total_trades': total_trades
        }
    
    return None

if __name__ == "__main__":
    from institutional_data_handler import prepare_institutional_data
    from volume_shock_detector import VolumeShockDetector
    from strategy_news_deviation import generate_signals as gen_news
    
    print("Loading Institutional Data...")
    df = prepare_institutional_data()
    
    print("Detecting Shocks...")
    detector = VolumeShockDetector()
    df = detector.detect(df)
    
    print("Generating News Signals...")
    df = gen_news(df)
    
    print("\nRunning Walk-Forward Test on News Strategy...\n")
    run_walk_forward(df, strategy_cols=['News_Signal'])
'''
open("walk_forward_optimizer.py", "w", encoding="utf-8").write(wf_code)
print("SUCCESS: walk_forward_optimizer.py rewritten with Purge/Embargo")

ast.parse(wf_code)
print("walk_forward_optimizer.py AST OK")
