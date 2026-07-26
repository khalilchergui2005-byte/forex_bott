# ============================================================
# المشروع: الكيان الفيروسي – الذرة 3.2: تدريب SAC (مصحح)
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from pathlib import Path

class LatentTradingEnv(gym.Env):
    def __init__(self, df, window_size=24):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_size
        self.capital = 10_000
        self.pos = 0.0
        self.entry = 0.0
        self.units = 0.0
        self.trades = []
        self.equity = [10_000]
        self.idx = self.window
        
        self.latent_cols = [c for c in df.columns if c.startswith("z_")]
        latent_dim = len(self.latent_cols) * self.window + 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)

    def _obs(self):
        i = self.idx
        z_vals = []
        for c in self.latent_cols:
            z_vals.append(self.df[c].iloc[i-self.window:i].values)
        obs = np.concatenate(z_vals + [np.array([self.pos])])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.window
        self.capital = 10_000
        self.pos = 0.0
        self.entry = 0.0
        self.units = 0.0
        self.trades = []
        self.equity = [10_000]
        return self._obs(), {}

    def step(self, action):
        if self.idx >= len(self.df) - 1:
            return self._obs(), 0.0, True, False, {}
        
        desired = float(np.clip(action[0], -1, 1))
        reward = 0.0
        ret = self.df["return"].iloc[self.idx] if "return" in self.df.columns else 0.0
        
        if self.pos != 0 and abs(desired - self.pos) > 0.3:
            pnl = ret * self.pos * self.units
            cost = abs(self.pos) * 0.0001
            net = pnl - cost
            self.capital += net
            self.trades.append(net)
            reward = net / 10_000 * 100
            self.pos = 0.0
            self.units = 0.0
        
        if self.pos == 0.0 and abs(desired) > 0.3:
            self.units = min(50000, self.capital * 0.01 / 0.001)
            self.pos = 1.0 if desired > 0 else -1.0
        
        self.idx += 1
        self.equity.append(self.capital)
        done = self.idx >= len(self.df) - 1
        
        if done and self.pos != 0:
            pnl = ret * self.pos * self.units
            cost = abs(self.pos) * 0.0001
            self.capital += (pnl - cost)
            self.trades.append(pnl - cost)
            self.pos = 0.0
        
        return self._obs(), reward, done, False, {}

print("Loading data...")
real_df = pd.read_csv("data/train_latent.csv", index_col=0, parse_dates=True)
real_df["return"] = real_df["close"].pct_change() if "close" in real_df.columns else 0.0
real_df.dropna(inplace=True)

synth_df = pd.read_csv("data/synthetic_latent.csv")
synth_df["return"] = np.random.normal(0, 0.002, len(synth_df))

combined = pd.concat([real_df, synth_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Training data: {len(real_df)} real + {len(synth_df)} synthetic = {len(combined)} total")

train_env = DummyVecEnv([lambda: LatentTradingEnv(combined)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

print("Training SAC on full data...")
model = SAC("MlpPolicy", train_env, learning_rate=0.0003, buffer_size=200_000,
            batch_size=512, tau=0.005, gamma=0.99, train_freq=16,
            gradient_steps=16, ent_coef="auto", verbose=1)
model.learn(total_timesteps=100_000)
print("Training done")

Path("models").mkdir(exist_ok=True)
model.save("models/sac_beast_final")
train_env.save("models/sac_beast_normalize.pkl")

print("Testing on real data...")
test_df = pd.read_csv("data/test_latent.csv", index_col=0, parse_dates=True)
test_df["return"] = test_df["close"].pct_change() if "close" in test_df.columns else 0.0
test_df.dropna(inplace=True)

eval_env = DummyVecEnv([lambda: LatentTradingEnv(test_df)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
eval_env.obs_rms = train_env.obs_rms
eval_env.ret_rms = train_env.ret_rms

obs = eval_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = eval_env.step(action)
    done = done[0]

inst = eval_env.envs[0]
ret = (inst.capital / 10_000 - 1) * 100
n = len(inst.trades)
print(f"\nResults: Capital={inst.capital:,.2f} | Return={ret:.2f}% | Trades={n}")
if n > 0:
    wins = [t for t in inst.trades if t > 0]
    losses = [t for t in inst.trades if t <= 0]
    print(f"Win rate: {len(wins)/n*100:.1f}%")
    if wins: print(f"Avg win: {np.mean(wins):.2f}")
    if losses: print(f"Avg loss: {np.mean(losses):.2f}")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(inst.equity, color="green", lw=2)
axes[0].axhline(y=10_000, color="gray", ls="--")
axes[0].set_title(f"SAC - Return={ret:.2f}%", fontsize=14)
dd = (pd.Series(inst.equity).cummax() - pd.Series(inst.equity)) / 10_000 * 100
axes[1].fill_between(range(len(dd)), 0, dd, color="red", alpha=0.4)
axes[1].plot(dd, color="red", lw=1)
axes[1].set_title("Drawdown", fontsize=14)
plt.tight_layout()
Path("reports").mkdir(exist_ok=True)
fig.savefig("reports/3.2_sac_final.png", dpi=150)
plt.close()
print("Report saved. 3.2 Done.")
