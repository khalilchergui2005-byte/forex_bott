# ============================================================
# الذرة 3.2 – الحكم النهائي: RL مع تقسيم 70/30
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.mixture import GaussianMixture
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 1. تحميل بيانات Wavelet
df = pd.read_csv("data/eurusd_wavelet_features.csv", index_col=0, parse_dates=True)
df['SMA_20'] = df['close'].rolling(20).mean()
df['SMA_50'] = df['close'].rolling(50).mean()
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
df['RSI'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
df['Return'] = df['close'].pct_change()
df['Volatility'] = df['Return'].rolling(24).std()
df.dropna(inplace=True)
gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
df['Regime'] = gmm.fit_predict(df[['Return','Volatility']].values)

# 2. تقسيم 70/30
split = int(0.7 * len(df))
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()
print(f"✅ تدريب: {len(train_df)} صف | اختبار Out-of-Sample: {len(test_df)} صف")

# 3. بيئة التداول
class WaveletEnv(gym.Env):
    def __init__(self, df, window=50):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window
        self.capital = 10_000
        self.pos = 0.0
        self.entry = 0.0
        self.units = 0.0
        self.trades = []
        self.equity = [10_000]
        self.idx = window
        obs_dim = window + 11
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _obs(self):
        i = self.idx
        prices = self.df['close'].iloc[i-self.window:i].values
        feats = np.array([
            self.df['SMA_20'].iloc[i], self.df['SMA_50'].iloc[i],
            self.df['ATR'].iloc[i], self.df['RSI'].iloc[i],
            self.df['Return'].iloc[i], self.df['Volatility'].iloc[i],
            self.df.get('w_energy_0', pd.Series([0]*len(self.df))).iloc[i],
            self.df.get('w_energy_1', pd.Series([0]*len(self.df))).iloc[i],
            self.df.get('w_energy_2', pd.Series([0]*len(self.df))).iloc[i],
            self.df.get('w_detail_ratio', pd.Series([0]*len(self.df))).iloc[i],
            float(self.pos)
        ])
        return np.concatenate([prices, feats]).astype(np.float32)

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
        if self.idx >= len(self.df)-1:
            return self._obs(), 0.0, True, False, {}
        price = self.df['close'].iloc[self.idx]
        atr = self.df['ATR'].iloc[self.idx]
        desired = np.clip(action[0], -1, 1)
        reward = 0.0
        spread = 0.0001
        if self.pos != 0 and abs(desired - self.pos) > 0.3:
            pnl = (price - self.entry) * self.pos * self.units
            cost = (self.units/100_000)*7 + spread*self.units
            net = pnl - cost
            self.capital += net
            self.trades.append(net)
            reward = net / 10_000 * 100
            self.pos = 0.0
        if self.pos == 0 and abs(desired) > 0.3:
            risk = self.capital * 0.01
            stop_d = 2.0 * atr
            if stop_d > 0.0001:
                self.units = min(risk/stop_d, 50000)
                self.pos = 1.0 if desired > 0 else -1.0
                self.entry = price
        self.idx += 1
        self.equity.append(self.capital)
        done = self.idx >= len(self.df)-1
        if done and self.pos != 0:
            pnl = (price - self.entry) * self.pos * self.units
            cost = (self.units/100_000)*7 + spread*self.units
            self.capital += (pnl - cost)
            self.trades.append(pnl - cost)
            self.pos = 0.0
        return self._obs(), reward, done, False, {}

# 4. تدريب على جزء التدريب فقط
print("🧠 تدريب PPO على بيانات التدريب فقط...")
train_env = DummyVecEnv([lambda: WaveletEnv(train_df)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", train_env, learning_rate=0.0003, n_steps=512, batch_size=128, n_epochs=20, gamma=0.99, verbose=1)
model.learn(total_timesteps=50_000)
print("✅ تم التدريب")

# 5. اختبار على بيانات Out-of-Sample (لم يرها الوكيل)
print("🧪 اختبار على بيانات Out-of-Sample (لم يرها الوكيل)...")
eval_env = DummyVecEnv([lambda: WaveletEnv(test_df)])
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
print(f"\n📊 نتيجة الاختبار Out-of-Sample:")
print(f"💰 رأس المال: {inst.capital:,.2f} $ | عائد: {ret:.2f}% | صفقات: {n}")
if n > 0:
    wins = [t for t in inst.trades if t > 0]
    losses = [t for t in inst.trades if t <= 0]
    wr = len(wins)/n*100
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 0
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else np.inf
    print(f"✅ فوز: {wr:.1f}% | 💚 ربح وسطي: {avg_w:.2f} | ❤️ خسارة وسطي: {avg_l:.2f}")
    print(f"⚖️ Profit Factor: {pf:.2f}")

    if ret > 5 and pf > 1.2:
        print("\n🎉 RL يعمل! الوكيل يربح على بيانات لم يرها!")
    elif ret > 0 and pf > 1.0:
        print("\n⚠️ RL يعمل لكنه ضعيف. يحتاج تحسين.")
    else:
        print("\n❌ RL لم يثبت جدواه.")
else:
    print("❌ لا صفقات. RL لم يتعلم التداول.")

print("🎉 الحكم النهائي صدر.")
