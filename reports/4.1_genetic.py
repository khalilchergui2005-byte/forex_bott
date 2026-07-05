# ============================================================
# المشروع: الكيان الفيروسي – الذرة 4.1: الخوارزميات الجينية
# ============================================================
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import random
from pathlib import Path
import matplotlib.pyplot as plt

print("Preparing hourly data...")
df = pd.read_csv("data/train_hourly.csv", index_col=0, parse_dates=True)

# HMM Regime
df["Return"] = df["close"].pct_change()
df["Range"] = (df["high"] - df["low"]) / df["close"]
df["Volatility"] = df["Return"].rolling(24).std()
df["Momentum"] = df["close"].pct_change(10)
df.dropna(inplace=True)

gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=5)
df["Regime"] = gmm.fit_predict(df[["Return", "Range", "Volatility", "Momentum"]])
state_returns = df.groupby("Regime")["Return"].mean().sort_values()
df["Regime"] = df["Regime"].map({state_returns.index[0]: -1, state_returns.index[1]: 0, state_returns.index[2]: 1})

# Indicators
df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
df["RSI"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
df.dropna(inplace=True)

def evaluate_strategy(params):
    sma_short, sma_long, rsi_period, rsi_low, rsi_high, atr_mult, regime_filter = params
    
    data = df.copy()
    data["SMA_S"] = data["close"].rolling(int(sma_short)).mean()
    data["SMA_L"] = data["close"].rolling(int(sma_long)).mean()
    data["ATR_dyn"] = (data["high"] - data["low"]).rolling(14).mean()
    
    delta_local = data["close"].diff()
    gain_local = delta_local.clip(lower=0)
    loss_local = -delta_local.clip(upper=0)
    data["RSI_dyn"] = 100 - (100 / (1 + gain_local.rolling(int(rsi_period)).mean() / loss_local.rolling(int(rsi_period)).mean()))
    data.dropna(inplace=True)
    
    # Signals
    data["Signal"] = 0
    data.loc[(data["SMA_S"] > data["SMA_L"]) & (data["SMA_S"].shift(1) <= data["SMA_L"].shift(1)), "Signal"] = 1
    data.loc[(data["SMA_S"] < data["SMA_L"]) & (data["SMA_S"].shift(1) >= data["SMA_L"].shift(1)), "Signal"] = -1
    
    data["RSI_Sig"] = 0
    data.loc[(data["RSI_dyn"] < rsi_low) & (data["close"] > data["SMA_L"]), "RSI_Sig"] = 1
    data.loc[(data["RSI_dyn"] > rsi_high) & (data["close"] < data["SMA_L"]), "RSI_Sig"] = -1
    
    data["Final"] = 0
    data.loc[(data["Signal"] == 1) | (data["RSI_Sig"] == 1), "Final"] = 1
    data.loc[(data["Signal"] == -1) | (data["RSI_Sig"] == -1), "Final"] = -1
    
    if regime_filter:
        data.loc[(data["Final"] == 1) & (data["Regime"] < 0), "Final"] = 0
        data.loc[(data["Final"] == -1) & (data["Regime"] > 0), "Final"] = 0
    
    # Prevent trade clustering
    last_trade = -50
    for i in range(len(data)):
        if data["Final"].iloc[i] != 0:
            if i - last_trade < 24:
                data["Final"].iloc[i] = 0
            else:
                last_trade = i
    
    # Backtest
    capital = 10_000
    pos = 0
    entry = 0.0
    units = 0.0
    trades = []
    
    for i in range(len(data)):
        price = data["close"].iloc[i]
        atr_val = data["ATR_dyn"].iloc[i]
        sig = data["Final"].iloc[i]
        
        if pos != 0 and sig != 0 and sig != pos:
            pnl = (price - entry) * pos * units
            cost = (units / 100_000) * 7
            net = pnl - cost
            capital += net
            trades.append(net)
            pos = 0
            units = 0.0
            continue
        
        if pos == 0 and sig != 0:
            risk = capital * 0.01
            stop_dist = atr_mult * atr_val
            if stop_dist > 0.0001:
                units = min(risk / stop_dist, 50000)
                pos = sig
                entry = price
    
    if not trades:
        return -1000, 0, 0, 0, 0, 0
    
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins) / n * 100
    total = sum(trades)
    dd = (pd.Series([10_000] + trades).cumsum() / 10_000 - 1).min() * 100
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else (10 if wins else 0)
    
    if n < 20 or wr < 50 or pf < 1.3:
        return -1000, 0, 0, 0, 0, 0
    
    fitness = (total / 100) * (n / 50) * pf * (1 + abs(dd) / 100)
    return fitness, total, n, wr, dd, pf

# Genetic Algorithm
POP_SIZE = 60
GENERATIONS = 25
MUTATION_RATE = 0.25

def random_gene():
    return [
        random.randint(5, 30),
        random.randint(20, 100),
        random.randint(5, 21),
        random.randint(20, 40),
        random.randint(60, 80),
        round(random.uniform(1.0, 3.0), 1),
        random.choice([True, False])
    ]

def mutate(gene):
    i = random.randint(0, 6)
    if i == 0: gene[0] = max(3, gene[0] + random.randint(-5, 5))
    elif i == 1: gene[1] = max(gene[0] + 5, gene[1] + random.randint(-15, 15))
    elif i == 2: gene[2] = max(2, gene[2] + random.randint(-3, 3))
    elif i == 3: gene[3] = max(10, min(gene[4] - 5, gene[3] + random.randint(-5, 5)))
    elif i == 4: gene[4] = max(gene[3] + 5, min(90, gene[4] + random.randint(-5, 5)))
    elif i == 5: gene[5] = round(max(0.5, min(4.0, gene[5] + random.uniform(-0.5, 0.5))), 1)
    elif i == 6: gene[6] = random.choice([True, False])
    return gene

def crossover(p1, p2):
    i = random.randint(1, 5)
    return p1[:i] + p2[i:]

print("Starting genetic evolution...")
population = [random_gene() for _ in range(POP_SIZE)]
best_gene = None
best_fitness = -np.inf
best_result = None

for gen in range(GENERATIONS):
    scores = []
    for gene in population:
        fit, tot, n, wr, dd, pf = evaluate_strategy(gene)
        scores.append((fit, gene, tot, n, wr, dd, pf))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:10]
    
    if scores[0][0] > best_fitness:
        best_fitness = scores[0][0]
        best_gene = scores[0][1]
        best_result = scores[0]
        print(f"\nGen {gen+1}: PnL=${scores[0][2]:.1f} | Trades={scores[0][3]} | Win={scores[0][4]:.1f}% | DD={scores[0][5]:.1f}% | PF={scores[0][6]:.1f}")
        print(f"   Genes: SMA({best_gene[0]},{best_gene[1]}) RSI({best_gene[2]},{best_gene[3]},{best_gene[4]}) ATRx{best_gene[5]} Filter={best_gene[6]}")
    else:
        print(f"Gen {gen+1}: No improvement.")
    
    survivors = [g for _, g, _, _, _, _, _ in top[:10]]
    new_pop = survivors.copy()
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(survivors, 2)
        child = crossover(p1, p2)
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        new_pop.append(child)
    population = new_pop

Path("models").mkdir(exist_ok=True)
with open("models/best_genome.txt", "w") as f:
    f.write(f"SMA_short={best_gene[0]}, SMA_long={best_gene[1]}, RSI_period={best_gene[2]}, RSI_low={best_gene[3]}, RSI_high={best_gene[4]}, ATR_mult={best_gene[5]}, regime_filter={best_gene[6]}\n")
    if best_result:
        f.write(f"Return=${best_result[2]:.2f}, Trades={best_result[3]}, WinRate={best_result[4]:.1f}%, DD={best_result[5]:.1f}%, PF={best_result[6]:.1f}\n")

print(f"\nBest genome saved to models/best_genome.txt")
print("4.1 Done.")