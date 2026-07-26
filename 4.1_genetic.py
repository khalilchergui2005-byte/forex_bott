import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import random
from pathlib import Path

df = pd.read_csv("data/train_hourly.csv", index_col=0, parse_dates=True)
df["Return"] = df["close"].pct_change()
df["Range"] = (df["high"] - df["low"]) / df["close"]
df["Volatility"] = df["Return"].rolling(24).std()
df["Momentum"] = df["close"].pct_change(10)
df.dropna(inplace=True)
gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
df["Regime"] = gmm.fit_predict(df[["Return","Range","Volatility","Momentum"]])
sr = df.groupby("Regime")["Return"].mean().sort_values()
df["Regime"] = df["Regime"].map({sr.index[0]:-1, sr.index[1]:0, sr.index[2]:1})
df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
delta = df["close"].diff()
df["RSI"] = 100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()))
df.dropna(inplace=True)

def evaluate(params):
    sma_s, sma_l, rsi_p, rsi_low, rsi_high, atr_m, reg_f = params
    d = df.copy()
    d["SMA_S"] = d["close"].rolling(int(sma_s)).mean()
    d["SMA_L"] = d["close"].rolling(int(sma_l)).mean()
    d["ATR_d"] = (d["high"] - d["low"]).rolling(14).mean()
    dd = d["close"].diff()
    d["RSI_d"] = 100 - (100 / (1 + dd.clip(lower=0).rolling(int(rsi_p)).mean() / (-dd.clip(upper=0)).rolling(int(rsi_p)).mean()))
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
    dd = (pd.Series([10_000]+trades).cumsum()/10_000-1).min()*100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    if n < 5 or pf < 0.8:
        return -1000, 0, 0, 0, 0, 0
    fitness = total + n*2 + pf*100 - abs(dd)*2
    return fitness, total, n, wr, dd, pf

POP = 40
GENS = 15
def rand_gene():
    return [random.randint(5,30), random.randint(20,100), random.randint(5,21),
            random.randint(20,40), random.randint(60,80), round(random.uniform(1,3),1),
            random.choice([True,False])]
def mutate(g):
    i = random.randint(0,6)
    if i==0: g[0]=max(3,g[0]+random.randint(-5,5))
    elif i==1: g[1]=max(g[0]+5,g[1]+random.randint(-15,15))
    elif i==2: g[2]=max(2,g[2]+random.randint(-3,3))
    elif i==3: g[3]=max(10,min(g[4]-5,g[3]+random.randint(-5,5)))
    elif i==4: g[4]=max(g[3]+5,min(90,g[4]+random.randint(-5,5)))
    elif i==5: g[5]=round(max(0.5,min(4,g[5]+random.uniform(-0.5,0.5))),1)
    elif i==6: g[6]=random.choice([True,False])
    return g
def cross(p1,p2):
    i = random.randint(1,5)
    return p1[:i]+p2[i:]

pop = [rand_gene() for _ in range(POP)]
best_g = None
best_f = -np.inf
best_r = None
for gen in range(GENS):
    scores = []
    for g in pop:
        f, tot, n, wr, dd, pf = evaluate(g)
        scores.append((f, g, tot, n, wr, dd, pf))
    scores.sort(key=lambda x: x[0], reverse=True)
    if scores[0][0] > best_f:
        best_f, best_g, best_r = scores[0][0], scores[0][1], scores[0]
        print(f"Gen {gen+1}: ${best_r[2]:.1f} | Trades={best_r[3]} | Win={best_r[4]:.1f}% | DD={best_r[5]:.1f}% | PF={best_r[6]:.1f}")
        print(f"  SMA({best_g[0]},{best_g[1]}) RSI({best_g[2]},{best_g[3]},{best_g[4]}) ATRx{best_g[5]} Filter={best_g[6]}")
    else:
        print(f"Gen {gen+1}: No improvement")
    surv = [g for _,g,_,_,_,_,_ in scores[:8]]
    new_pop = surv.copy()
    while len(new_pop) < POP:
        p1,p2 = random.sample(surv,2)
        child = cross(p1,p2)
        if random.random()<0.3:
            child = mutate(child)
        new_pop.append(child)
    pop = new_pop

Path("models").mkdir(exist_ok=True)
with open("models/best_genome.txt","w") as f:
    f.write(f"SMA({best_g[0]},{best_g[1]}) RSI({best_g[2]},{best_g[3]},{best_g[4]}) ATRx{best_g[5]} Filter={best_g[6]}\n")
    if best_r:
        f.write(f"PnL=${best_r[2]:.2f} Trades={best_r[3]} Win={best_r[4]:.1f}% DD={best_r[5]:.1f}% PF={best_r[6]:.1f}\n")
print("Best genome saved. 4.1 Done.")
