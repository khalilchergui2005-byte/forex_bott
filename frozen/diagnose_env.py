# ============================================================
# تشخيص بيئة التداول – فحص القيم الشاذة
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

# نسخة معدلة من البيئة مع Assertions
class DiagnoseEnv(gym.Env):
    def __init__(self, df, window_size=20):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_size
        self.capital = 10_000
        self.pos = 0
        self.entry = 0.0
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
        obs = np.concatenate(z_vals + [np.array([self.pos])])
        
        # 🔍 فحص المجلس
        assert not np.isnan(obs).any(), f"NaN at step {self.idx}"
        assert obs.max() < 100 and obs.min() > -100, f"Extreme values: [{obs.min():.3f}, {obs.max():.3f}]"
        
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.window
        self.capital = 10_000
        self.pos = 0
        self.entry = 0.0
        self.units = 0.0
        return self._obs(), {}

    def step(self, action):
        desired = np.clip(action[0], -1, 1)
        ret = self.df["return"].iloc[self.idx] if "return" in self.df.columns else 0.0
        
        if self.pos != 0 and abs(desired - self.pos) > 0.3:
            pnl = ret * self.pos * self.units
            cost = abs(self.pos) * 0.0001
            self.capital += (pnl - cost)
            self.pos = 0.0
        
        if self.pos == 0 and abs(desired) > 0.3:
            self.units = min(50000, self.capital * 0.01 / 0.001)
            self.pos = 1.0 if desired > 0 else -1.0
        
        self.idx += 1
        done = self.idx >= len(self.df) - 1
        if done and self.pos != 0:
            ret = self.df["return"].iloc[-1]
            pnl = ret * self.pos * self.units
            self.capital += (pnl - abs(self.pos) * 0.0001)
            self.pos = 0.0
        
        # 🔍 فحص المجلس بعد كل خطوة
        obs = self._obs()
        return obs, 0.0, done, False, {}

# تحميل البيانات المضغوطة
df = pd.read_csv("data/train_latent.csv", index_col=0, parse_dates=True)
df["return"] = df["close"].pct_change() if "close" in df.columns else 0.0
df.dropna(inplace=True)

# 1. فحص البيانات الخام أولاً
print("=== فحص البيانات الخام ===")
z_cols = [c for c in df.columns if c.startswith("z_")]
for c in z_cols[:5]:
    print(f"{c}: min={df[c].min():.3f}, max={df[c].max():.3f}, NaN={df[c].isna().sum()}")
print(f"Total NaN in latent: {df[z_cols].isna().sum().sum()}")

# 2. فحص البيئة
print("\n=== فحص البيئة ===")
env = DiagnoseEnv(df)
obs, _ = env.reset()
print(f"Obs shape: {obs.shape}, min: {obs.min():.3f}, max: {obs.max():.3f}, NaN: {np.isnan(obs).sum()}")

# 3. تشغيل 100 خطوة عشوائية
print("\n=== اختبار 100 خطوة عشوائية ===")
try:
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            break
    print(f"✅ اكتمل 100 خطوة بدون أخطاء")
except AssertionError as e:
    print(f"❌ فشل في الخطوة {step}: {e}")

print("\n🎉 اكتمل التشخيص.")
