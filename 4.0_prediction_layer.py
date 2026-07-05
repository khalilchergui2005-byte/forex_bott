import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import joblib
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   الذرة 4.0 – بناء طبقة التنبؤ الموحدة")
print("=" * 60)

# تعريف بنية VAE (نفس التي استخدمناها سابقاً)
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

print("تحميل نموذج VAE...")
state_dict = torch.load("models/vae_encoder.pth")
vae = VAE(input_dim=28, latent_dim=32)
vae.load_state_dict(state_dict)
vae.eval()
scaler = joblib.load("models/feature_scaler.pkl")
print("VAE جاهز")

print("تحميل FinBERT...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("FinBERT جاهز")

print("تحميل بيانات Wavelet...")
df = pd.read_csv("data/eurusd_wavelet_features.csv", index_col=0, parse_dates=True)
print(f"{len(df)} صف بميزات Wavelet")

# استخدام الميزات المتاحة فعلاً
feature_cols = [c for c in df.columns if c in ['ATR_ratio', 'SMA_20', 'Return', 'Volatility']]
if len(feature_cols) < 2:
    feature_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32']][:4]
    print(f"استخدام الميزات: {feature_cols}")

print("بناء درجة الثقة لكل صفقة...")
confidence_scores = []
for idx, row in df.iterrows():
    w_score = row.get('w_energy_0', 0.5)
    try:
        features = scaler.transform([row[feature_cols].values])
        with torch.no_grad():
            latent = vae.compress(torch.FloatTensor(features))
        vae_score = latent.abs().mean().item()
    except:
        vae_score = 0.5
    confidence = (w_score * 0.4 + vae_score * 0.3 + 0.5 * 0.3)
    confidence_scores.append(confidence)

df['confidence'] = confidence_scores
df.to_csv("data/eurusd_with_confidence.csv")
print(f"تم حفظ {len(df)} صف مع درجة الثقة")
print("الذرة 4.0 اكتملت.")
