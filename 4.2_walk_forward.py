# ============================================================
# الذرة 4.2: Walk-Forward Optimization لمنع Overfitting
# ============================================================
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import random
from pathlib import Path

print("📥 تحميل بيانات الساعة...")
df = pd.read_csv("data/eurusd_hourly_dukascopy.csv", index_col=0, parse_dates=True)
print(f"✅ {len(df)} شمعة ساعة")

# 1. تقسيم البيانات إلى نوافذ متحركة (Walk-Forward)
# نافذة تدريب: 120 يوم. نافذة اختبار: 30 يوم. تحريك: 30 يوم.
train_window = 120 * 24  # 120 يوم ساعة
test_window  = 30 * 24   # 30 يوم ساعة
step         = 30 * 24   # تحريك 30 يوم

results = []
windows_tested = 0

start = 0
while start + train_window + test_window <= len(df):
    train_data = df.iloc[start:start + train_window].copy()
    test_data  = df.iloc[start + train_window:start + train_window + test_window].copy()
    
    # 2. تحضير الميزات (نفس منطق الجين)
    def prepare(data):
        d = data.copy()
        d["Return"] = d["close"].pct_change()
        d["Range"] = (d["high"] - d["low"]) / d["close"]
        d["Volatility"] = d["Return"].rolling(20).std()
        d["Momentum"] = d["close"].pct_change(10)
        d.dropna(inplace=True)
        if len(d) < 100:
            return d
        gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
        d["Regime"] = gmm.fit_predict(d[["Return","Range","Volatility","Momentum"]])
        sr = d.groupby("Regime")["Return"].mean().sort_values()
        d["Regime"] = d["Regime"].map({sr.index[0]:-1, sr.index[1]:0, sr.index[2]:1})
        return d

    train_data = prepare(train_data)
    test_data  = prepare(test_data)
    
    if len(train_data) < 100 or len(test_data) < 20:
        start += step
        continue

    # 3. بحث سريع عن أفضل جين في هذه النافذة
    def evaluate(gene, data):
        sma_s, sma_l, rsi_p, rsi_low, rsi_high, atr_m, reg_f = gene
        d = data.copy()
        d["SMA_S"] = d["close"].rolling(int(sma_s)).mean()
        d["SMA_L"] = d["close"].rolling(int(sma_l)).mean()
        d["ATR_d"] = (d["high"] - d["low"]).rolling(14).mean()
        delta = d["close"].diff()
        d["RSI_d"] = 100 - (100 / (1 + delta.clip(lower=0).rolling(int(rsi_p)).mean() / (-delta.clip(upper=0)).rolling(int(rsi_p)).mean()))
        d.dropna(inplace=True)
        d["Sig"] = 0
        d.loc[(d["SMA_S"]>d["SMA_L"]) & (d["SMA_S"].shift(1)<=d["SMA_L"].shift(1)), "Sig"] = 1
        d.loc[(d["SMA_S"]<d["SMA_L"]) & (d["SMA_S"].shift(1)>=d["SMA_L"].shift(1)), "Sig"] = -1
        d["RSI_Sig"] = 0
        d.loc[(d["RSI_d"]<rsi_low)&(d["close"]>d["SMA_L"]), "RSI_Sig"] = 1
        d.loc[(d["RSI_d"]>rsi_high)&(d["close"]<d["SMA_L"]), "RSI_Sig"] = -1
        d["Final"] = 0
        d.loc[(d["Sig"]==1)|(d["RSI_Sig"]==1), "Final"] = 1
        d.loc[(d["Sig"]==-1)|(d["RSI_Sig"]==-1), "Final"] = -1
        if reg_f:
            d.loc[(d["Final"]==1)&(d["Regime"]<0), "Final"] = 0
            d.loc[(d["Final"]==-1)&(d["Regime"]>0), "Final"] = 0
        cap = 10_000
        pos = 0
        entry = 0.0
        units = 0.0
        trades = []
        for i in range(len(d)):
            p = d["close"].iloc[i]
            a = d["ATR_d"].iloc[i]
            s = d["Final"].iloc[i]
            if pos and s and s != pos:
                pnl = (p - entry)*pos*units
                cost = (units/100_000)*7
                cap += (pnl-cost)
                trades.append(pnl-cost)
                pos = 0
                continue
            if pos == 0 and s != 0:
                risk = cap*0.01
                stop_d = atr_m*a
                if stop_d>0.0001:
                    units = min(risk/stop_d, 50000)
                    pos = s
                    entry = p
        if not trades:
            return -1000, 0, 0, 0, 0, 0
        wins = [t for t in trades if t>0]
        losses = [t for t in trades if t<=0]
        n = len(trades)
        wr = len(wins)/n*100
        total = sum(trades)
        dd_pct = (pd.Series([10_000]+trades).cumsum()/10_000-1).min()*100
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
        penalty = max(0, abs(dd_pct) - 10) * 100
        fitness = total + n*3 + pf*50 - penalty
        return fitness, total, n, wr, dd_pct, pf

    # بحث سريع: 10 جينات عشوائية، نختار الأفضل
    best_fit = -np.inf
    best_gene = None
    best_result = None
    for _ in range(10):
        gene = [
            random.randint(5,30), random.randint(20,100),
            random.randint(5,21), random.randint(20,40),
            random.randint(60,80), round(random.uniform(1,3),1),
            random.choice([True,False])
        ]
        fit, tot, n, wr, dd, pf = evaluate(gene, train_data)
        if fit > best_fit:
            best_fit = fit
            best_gene = gene
            best_result = (tot, n, wr, dd, pf)

    # اختبار الجين على بيانات الاختبار (Out-of-Sample لهذه النافذة)
    if best_gene:
        fit_test, tot_test, n_test, wr_test, dd_test, pf_test = evaluate(best_gene, test_data)
        results.append({
            'window': windows_tested,
            'train_return': best_result[0],
            'train_trades': best_result[1],
            'train_wr': best_result[2],
            'train_dd': best_result[3],
            'train_pf': best_result[4],
            'test_return': tot_test,
            'test_trades': n_test,
            'test_wr': wr_test,
            'test_dd': dd_test,
            'test_pf': pf_test,
            'gene': best_gene.copy()
        })
        windows_tested += 1
        if windows_tested <= 3 or windows_tested % 5 == 0:
            print(f"   نافذة {windows_tested}: تدريب={best_result[0]:.1f}$ | اختبار={tot_test:.1f}$ | DD={dd_test:.1f}% | PF={pf_test:.1f}")

    start += step

if results:
    df_results = pd.DataFrame(results)
    print(f"\n📊 نتائج Walk-Forward ({len(results)} نافذة):")
    print(f"   متوسط عائد التدريب: {df_results['train_return'].mean():.1f}$")
    print(f"   متوسط عائد الاختبار: {df_results['test_return'].mean():.1f}$")
    print(f"   متوسط DD الاختبار: {df_results['test_dd'].mean():.1f}%")
    print(f"   متوسط PF الاختبار: {df_results['test_pf'].mean():.2f}")
    
    # الجين الأكثر استقراراً (أفضل متوسط اختبار)
    best_idx = df_results['test_return'].idxmax()
    best = df_results.iloc[best_idx]
    print(f"\n🏆 أفضل نافذة (#{int(best['window'])+1}):")
    print(f"   تدريب: {best['train_return']:.1f}$ | اختبار: {best['test_return']:.1f}$")
    print(f"   DD: {best['test_dd']:.1f}% | PF: {best['test_pf']:.1f} | صفقات: {int(best['test_trades'])}")
    
    Path("models").mkdir(exist_ok=True)
    df_results.to_csv("models/walk_forward_results.csv", index=False)
    print("✅ النتائج محفوظة في models/walk_forward_results.csv")
else:
    print("⚠️ لم نتمكن من إنشاء نوافذ كافية.")

print("🎉 الذرة 4.2 اكتملت.")
