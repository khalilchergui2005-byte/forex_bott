# ============================================================
# المشروع: الكيان الفيروسي – الذرة 3.3: ضاغط الميزات (مُصحح)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# ============================================================
# 1. تحميل البيانات وبناء الميزات
# ============================================================
print("📥 تحميل البيانات وبناء الميزات...")
df = pd.read_csv("data/train_hourly.csv", index_col=0, parse_dates=True)

def build_features(data):
    d = data.copy()
    d["return"] = d["close"].pct_change()
    d["range"] = (d["high"] - d["low"]) / d["close"]
    d["sma_5"] = d["close"].rolling(5).mean()
    d["sma_10"] = d["close"].rolling(10).mean()
    d["sma_20"] = d["close"].rolling(20).mean()
    d["sma_50"] = d["close"].rolling(50).mean()
    d["ema_12"] = d["close"].ewm(span=12).mean()
    d["ema_26"] = d["close"].ewm(span=26).mean()
    d["macd"] = d["ema_12"] - d["ema_26"]
    d["atr_14"] = (d["high"] - d["low"]).rolling(14).mean()
    d["atr_ratio"] = d["atr_14"] / d["close"]
    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for period in [7, 14, 21]:
        d[f"rsi_{period}"] = 100 - (100 / (1 + gain.rolling(period).mean() / loss.rolling(period).mean()))
    d["bb_width"] = (2 * d["close"].rolling(20).std()) / d["close"]
    d["volume_ratio"] = d["high"].rolling(20).std() / d["close"]
    for period in [5, 10, 20]:
        d[f"momentum_{period}"] = d["close"].pct_change(period)
        d[f"volatility_{period}"] = d["return"].rolling(period).std()
    return d.dropna()

df = build_features(df)
exclude = ["open", "high", "low", "close", "Regime"]
feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32]]
print(f"✅ عدد الميزات: {len(feature_cols)}")

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Path("models").mkdir(exist_ok=True)
joblib.dump(scaler, "models/feature_scaler.pkl")
pd.DataFrame({"features": feature_cols}).to_csv("models/feature_names.csv", index=False)

# ============================================================
# 2. Variational Autoencoder
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64)
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)
    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    def compress(self, x):
        return self.encode(x)[0]

# ============================================================
# 3. تدريب VAE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ التدريب على: {device}")

input_dim = X_scaled.shape[1]
vae = VAE(input_dim, 32).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
X_tensor = torch.FloatTensor(X_scaled).to(device)
dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

print(f"🧠 تدريب VAE ({input_dim} -> 32)...")
vae.train()
for epoch in range(100):
    total_loss = 0
    for (batch,) in loader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch)
        recon_loss = nn.MSELoss()(recon, batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
        loss = recon_loss + 0.001 * kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/100 - Loss: {total_loss/len(loader):.4f}")
print("✅ تم تدريب VAE")

torch.save(vae.state_dict(), "models/vae_encoder.pth")
print("💾 تم حفظ VAE في models/vae_encoder.pth")

# ============================================================
# 4. تحويل البيانات إلى الفضاء الكامن
# ============================================================
print("📦 تحويل البيانات إلى الفضاء الكامن...")
vae.eval()
for split_name, file_name in [("train", "data/train_hourly.csv"), ("test", "data/test_hourly.csv")]:
    data = pd.read_csv(file_name, index_col=0, parse_dates=True)
    data = build_features(data)
    X_split = scaler.transform(data[feature_cols].values)
    X_tensor = torch.FloatTensor(X_split).to(device)
    with torch.no_grad():
        latent = vae.compress(X_tensor).cpu().numpy()
    latent_cols = [f"z_{i}" for i in range(32)]
    latent_df = pd.DataFrame(latent, index=data.index, columns=latent_cols)
    combined = pd.concat([data[["open","high","low","close","Regime"]], latent_df], axis=1)
    combined.to_csv(f"data/{split_name}_latent.csv")
    print(f"  ✅ {split_name}: {len(combined)} صف مضغوط")

print("🎉 الذرة 3.3 اكتملت.")
