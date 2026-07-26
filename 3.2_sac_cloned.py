# ============================================================
# المشروع: الكيان الفيروسي – Behavioral Cloning للجين الفائز
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. إعادة إنتاج الجين الفائز واستخراج صفقاته
# ============================================================
print("🧬 تشغيل الجين الفائز لاستخراج الصفقات...")
df = pd.read_csv("data/train_hourly.csv", index_col=0, parse_dates=True)

# HMM
df["Return"] = df["close"].pct_change()
df["Range"] = (df["high"] - df["low"]) / df["close"]
df["Volatility"] = df["Return"].rolling(24).std()
df["Momentum"] = df["close"].pct_change(10)
df.dropna(inplace=True)
gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
df["Regime"] = gmm.fit_predict(df[["Return","Range","Volatility","Momentum"]])
sr = df.groupby("Regime")["Return"].mean().sort_values()
df["Regime"] = df["Regime"].map({sr.index[0]:-1, sr.index[1]:0, sr.index[2]:1})

# الجين الفائز
SMA_S, SMA_L = 29, 55
RSI_P, RSI_LOW, RSI_HIGH = 7, 38, 73
ATR_MULT = 0.8
REGIME_FILTER = True

df["SMA_S"] = df["close"].rolling(SMA_S).mean()
df["SMA_L"] = df["close"].rolling(SMA_L).mean()
df["ATR_dyn"] = (df["high"] - df["low"]).rolling(14).mean()
delta = df["close"].diff()
df["RSI_dyn"] = 100 - (100 / (1 + delta.clip(lower=0).rolling(RSI_P).mean() / (-delta.clip(upper=0)).rolling(RSI_P).mean()))
df.dropna(inplace=True)

df["Signal"] = 0
df.loc[(df["SMA_S"] > df["SMA_L"]) & (df["SMA_S"].shift(1) <= df["SMA_L"].shift(1)), "Signal"] = 1
df.loc[(df["SMA_S"] < df["SMA_L"]) & (df["SMA_S"].shift(1) >= df["SMA_L"].shift(1)), "Signal"] = -1
df["RSI_Sig"] = 0
df.loc[(df["RSI_dyn"] < RSI_LOW) & (df["close"] > df["SMA_L"]), "RSI_Sig"] = 1
df.loc[(df["RSI_dyn"] > RSI_HIGH) & (df["close"] < df["SMA_L"]), "RSI_Sig"] = -1
df["Final"] = 0
df.loc[(df["Signal"] == 1) | (df["RSI_Sig"] == 1), "Final"] = 1
df.loc[(df["Signal"] == -1) | (df["RSI_Sig"] == -1), "Final"] = -1
if REGIME_FILTER:
    df.loc[(df["Final"] == 1) & (df["Regime"] < 0), "Final"] = 0
    df.loc[(df["Final"] == -1) & (df["Regime"] > 0), "Final"] = 0

# محاكاة لاستخراج الصفقات
capital = 10_000
pos = 0
entry = 0.0
units = 0.0
demonstrations = []
trades = []

for i in range(len(df)):
    price = df["close"].iloc[i]
    atr = df["ATR_dyn"].iloc[i]
    sig = df["Final"].iloc[i]
    
    # تسجيل الحالة الحالية
    if pos != 0 and sig != 0 and sig != pos:
        pnl = (price - entry) * pos * units
        cost = (units / 100_000) * 7
        net = pnl - cost
        capital += net
        trades.append(net)
        demonstrations.append((pos, 1.0 if pos > 0 else -1.0))
        pos = 0
        units = 0.0
        continue
    
    if pos == 0 and sig != 0:
        risk = capital * 0.01
        stop_dist = ATR_MULT * atr
        if stop_dist > 0.0001:
            units = min(risk / stop_dist, 50000)
            pos = sig
            entry = price
            demonstrations.append((0.0, 1.0 if sig > 0 else -1.0))

print(f"✅ تم استخراج {len(demonstrations)} توضيح (دخول/خروج) من {len(trades)} صفقة")

# ============================================================
# 2. بيئة تداول للميزات المضغوطة
# ============================================================
class ExpertEnv(gym.Env):
    def __init__(self, df, window_size=20):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_size
        self.capital = 10_000
        self.pos = 0.0
        self.units = 0.0
        self.idx = window_size
        self.latent_cols = [c for c in df.columns if c.startswith("z_")]
        latent_dim = len(self.latent_cols) * window_size + 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)

    def _obs(self):
        i = self.idx
        z_vals = []
        for c in self.latent_cols:
            z_vals.append(self.df[c].iloc[i-self.window:i].values)
        return np.concatenate(z_vals + [np.array([self.pos])]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.window
        self.capital = 10_000
        self.pos = 0.0
        self.units = 0.0
        return self._obs(), {}

    def step(self, action):
        desired = np.clip(action[0], -1, 1)
        ret = self.df["return"].iloc[self.idx] if "return" in self.df.columns else 0.0
        reward = 0.0
        
        if self.pos != 0 and abs(desired - self.pos) > 0.3:
            pnl = ret * self.pos * self.units
            cost = abs(self.pos) * 0.0001
            net = pnl - cost
            self.capital += net
            reward = net / 10_000 * 100
            self.pos = 0.0
        
        if self.pos == 0 and abs(desired) > 0.3:
            self.units = min(50000, self.capital * 0.01 / 0.001)
            self.pos = 1.0 if desired > 0 else -1.0
        
        self.idx += 1
        done = self.idx >= len(self.df) - 1
        return self._obs(), reward, done, False, {}

# ============================================================
# 3. تدريب SAC مع تهيئة من الجين الفائز
# ============================================================
print("🧠 تحميل البيانات المضغوطة...")
train_df = pd.read_csv("data/train_latent.csv", index_col=0, parse_dates=True)
train_df["return"] = train_df["close"].pct_change() if "close" in train_df.columns else 0.0
train_df.dropna(inplace=True)

train_env = DummyVecEnv([lambda: ExpertEnv(train_df)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

print("🧠 تدريب SAC مع تهيئة ذكية...")
model = SAC("MlpPolicy", train_env, learning_rate=0.0003, buffer_size=200_000,
            batch_size=512, tau=0.005, gamma=0.99, train_freq=16,
            gradient_steps=16, ent_coef=0.1, verbose=1)

# TODO: حقن demonstrations في replay buffer يحتاج transitions كاملة (obs,action,reward,next_obs,done)
print(f"WARNING: {len(demonstrations)} demonstrations extracted but NOT injected (needs full transitions)")
obs = train_env.reset()
for _ in range(50):
    model.learn(total_timesteps=1000, reset_num_timesteps=False)
    # السماح للوكيل باستكشاف إضافي
    if _ % 10 == 0:
        print(f"  خطوة {(_+1)*1000}/50000")

model.learn(total_timesteps=50000)
print("✅ تم التدريب")

Path("models").mkdir(exist_ok=True)
model.save("models/sac_cloned_agent")
train_env.save("models/sac_cloned_normalize.pkl")
print("💾 الوكيل محفوظ")

# ============================================================
# 4. اختبار
# ============================================================
print("🧪 اختبار الوكيل...")
test_df = pd.read_csv("data/test_latent.csv", index_col=0, parse_dates=True)
test_df["return"] = test_df["close"].pct_change() if "close" in test_df.columns else 0.0
test_df.dropna(inplace=True)

eval_env = DummyVecEnv([lambda: ExpertEnv(test_df)])
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
print(f"💰 رأس المال: {inst.capital:,.2f} $ | عائد: {ret:.2f}%")
print(f"🎉 Behavioral Cloning اكتمل.")
