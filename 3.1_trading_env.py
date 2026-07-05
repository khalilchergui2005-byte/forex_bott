# ============================================================
# المشروع: الكيان الفيروسي للتداول الذاتي التطور
# الذرة 3.1: بيئة تداول Gym احترافية للفوركس
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path

class ForexTradingEnv(gym.Env):
    """بيئة تداول فوركس تحاكي وسيط ECN حقيقي."""
    def __init__(self, df, window_size=20, initial_capital=10_000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_size
        self.initial_capital = initial_capital
        self.action_space = spaces.Discrete(3)
        obs_dim = window_size + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.current_step = self.window
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.units = 0.0
        self.trades = []
        self.equity = [initial_capital]

    def _get_obs(self):
        idx = self.current_step
        prices = self.df["close"].iloc[idx - self.window : idx].values
        sma_20 = self.df["SMA_20"].iloc[idx]
        sma_50 = self.df["SMA_50"].iloc[idx]
        atr = self.df["ATR"].iloc[idx]
        rsi = self.df["RSI"].iloc[idx]
        regime = self.df["Regime"].iloc[idx] if "Regime" in self.df.columns else 0.0
        pos = float(self.position)
        cap_ratio = self.capital / self.initial_capital
        feats = np.array([sma_20, sma_50, atr, rsi, regime, pos, cap_ratio])
        obs = np.concatenate([prices, feats])
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
        price = self.df["close"].iloc[self.current_step]
        atr = self.df["ATR"].iloc[self.current_step]
        desired = action - 1
        reward = 0.0
        spread_cost = 0.0001
        commission_per_lot = 5.0
        slippage = np.random.uniform(-0.00005, 0.00005)

        if self.position != 0 and desired != 0 and desired != self.position:
            exec_price = price + slippage
            pnl = (exec_price - self.entry_price) * self.position * self.units
            cost = (self.units / 100_000) * commission_per_lot * 2
            cost += spread_cost * self.units
            net_pnl = pnl - cost
            self.capital += net_pnl
            self.trades.append(net_pnl)
            reward = net_pnl / self.initial_capital * 100
            self.position = 0
            self.units = 0.0

        if self.position == 0 and desired != 0:
            risk_amount = self.capital * 0.01
            stop_distance = 2.0 * atr
            if stop_distance > 0.0001:
                self.units = min(risk_amount / stop_distance, 50000)
                self.position = desired
                self.entry_price = price + slippage

        self.current_step += 1
        self.equity.append(self.capital)
        done = self.current_step >= len(self.df) - 1

        if done and self.position != 0:
            exec_price = price + slippage
            pnl = (exec_price - self.entry_price) * self.position * self.units
            cost = (self.units / 100_000) * commission_per_lot * 2
            cost += spread_cost * self.units
            net_pnl = pnl - cost
            self.capital += net_pnl
            self.trades.append(net_pnl)
            self.position = 0

        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass


# ============================================================
# تحميل البيانات وتجهيزها
# ============================================================
import yfinance as yf
from sklearn.mixture import GaussianMixture

print("Loading EUR/USD hourly data...")
end = pd.Timestamp.now()
start = end - pd.DateOffset(days=730)
data = yf.download("EURUSD=X", start=start, end=end, interval="1h", progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df = data[["Open", "High", "Low", "Close"]].copy()
df.columns = ["open", "high", "low", "close"]
df.dropna(inplace=True)

df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
df["RSI"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
df["Return"] = df["close"].pct_change()
df["Volatility"] = df["Return"].rolling(24).std()

df_temp = df.dropna(subset=["Return", "Volatility"])
gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
df.loc[df_temp.index, "Regime"] = gmm.fit_predict(df_temp[["Return", "Volatility"]].values)
df["Regime"] = df["Regime"].fillna(0)
df.dropna(inplace=True)

print(f"Data ready: {len(df)} hourly bars")

train_size = int(0.7 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

Path("data").mkdir(exist_ok=True)
train_df.to_csv("data/train_hourly.csv")
test_df.to_csv("data/test_hourly.csv")

env = ForexTradingEnv(df, window_size=20)
obs, _ = env.reset()
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n}")
print("3.1 Done - Trading environment ready.")
