# ============================================================
# ???????: ?????? ???????? ??????? ?????? ??????
# ????? 3.2: ????? ???? PPO (???? ?????)
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
import matplotlib.pyplot as plt

class ForexTradingEnv(gym.Env):
    def __init__(self, df, window_size=20, initial_capital=10_000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_size
        self.initial_capital = initial_capital
        self.current_step = self.window
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.units = 0.0
        self.trades = []
        self.equity = [initial_capital]
        self.action_space = spaces.Discrete(3)
        # 20 ??? + 7 ????? = 27
        obs_dim = window_size + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        idx = self.current_step
        prices = self.df['close'].iloc[idx - self.window:idx].values
        sma_20 = self.df['SMA_20'].iloc[idx]
        sma_50 = self.df['SMA_50'].iloc[idx]
        atr = self.df['ATR'].iloc[idx]
        rsi = self.df['RSI'].iloc[idx]
        regime = self.df['Regime'].iloc[idx]
        cap_ratio = self.capital / self.initial_capital
        obs = np.concatenate([prices, [sma_20, sma_50, atr, rsi, regime, float(self.position), cap_ratio]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.units = 0.0
        self.trades = []
        self.equity = [self.initial_capital]
        return self._get_obs(), {}

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0.0, True, False, {}
        price = self.df['close'].iloc[self.current_step]
        atr = self.df['ATR'].iloc[self.current_step]
        desired = action - 1
        reward = 0.0

        if self.position != 0:
            if desired != 0 and desired != self.position:
                pnl = (price - self.entry_price) * self.position * self.units
                cost = (self.units / 100_000) * 7
                net_pnl = pnl - cost
                self.capital += net_pnl
                self.trades.append(net_pnl)
                reward = net_pnl / self.initial_capital * 100
                self.position = 0
                self.units = 0.0

        if self.position == 0 and desired != 0:
            risk = self.capital * 0.01
            stop_dist = 2.0 * atr
            if stop_dist > 0.0001:
                self.units = min(risk / stop_dist, 50000)
                self.position = desired
                self.entry_price = price

        self.current_step += 1
        self.equity.append(self.capital)
        done = self.current_step >= len(self.df) - 1
        if done and self.position != 0:
            pnl = (price - self.entry_price) * self.position * self.units
            cost = (self.units / 100_000) * 7
            self.capital += (pnl - cost)
            self.trades.append(pnl - cost)
            self.position = 0
        return self._get_obs(), reward, done, False, {}

# ============================================================
# 1. ????? ????????
# ============================================================
train_df = pd.read_csv("data/train_data.csv", index_col="Date", parse_dates=True)
test_df  = pd.read_csv("data/test_data.csv", index_col="Date", parse_dates=True)
print(f"? ?????: {len(train_df)} ?? | ??????: {len(test_df)} ??")

# ============================================================
# 2. ????? PPO
# ============================================================
train_env = DummyVecEnv([lambda: ForexTradingEnv(train_df)])
model = PPO("MlpPolicy", train_env, learning_rate=0.0003, n_steps=128, batch_size=32, n_epochs=10, gamma=0.99, verbose=1)
print("?? ????? PPO...")
model.learn(total_timesteps=5_000)
print("? ?? ???????")

Path("models").mkdir(exist_ok=True)
model.save("models/ppo_forex_agent")
print("?? ?????? ?????")

# ============================================================
# 3. ??????
# ============================================================
test_env = DummyVecEnv([lambda: ForexTradingEnv(test_df)])
obs = test_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    done = done[0]

env_inst = test_env.envs[0]
ret = (env_inst.capital / 10_000 - 1) * 100
n = len(env_inst.trades)

print(f"\n?? ????? PPO ??? ????????:")
print(f"?? ??? ????? ???????: {env_inst.capital:,.2f} $")
print(f"?? ??????: {ret:.2f}%")
print(f"?? ???????: {n}")
if n > 0:
    wins = [t for t in env_inst.trades if t > 0]
    losses = [t for t in env_inst.trades if t <= 0]
    print(f"? ???? ?????: {len(wins)/n*100:.1f}%")
    print(f"?? ????? ?????: {np.mean(wins):.2f}" if wins else "?? -")
    print(f"?? ????? ???????: {np.mean(losses):.2f}" if losses else "?? -")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(env_inst.equity, color="green", linewidth=2)
axes[0].axhline(y=10_000, color="gray", linestyle="--")
axes[0].set_title(f"????? ?????? ? PPO (????={ret:.2f}%)", fontsize=14)
dd = (np.array(env_inst.equity) / np.maximum.accumulate(env_inst.equity) - 1) * 100
axes[1].fill_between(range(len(dd)), 0, dd, color="red", alpha=0.4)
axes[1].plot(dd, color="red", linewidth=1)
axes[1].set_title("Drawdown", fontsize=14)
plt.tight_layout()
fig.savefig("reports/3.2_ppo_agent_test.png", dpi=150)
plt.close()
print("? ??????? ????? ?? reports/3.2_ppo_agent_test.png")
print("?? ????? 3.2 ??????.")
