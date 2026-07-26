# ============================================================
# المشروع: الكيان الفيروسي – الذرة 3.4: مولد TimeGAN
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import joblib

# ============================================================
# 1. تحميل الميزات المضغوطة من الذرة 3.3
# ============================================================
print("📥 تحميل البيانات المضغوطة...")
df = pd.read_csv("data/train_latent.csv", index_col=0, parse_dates=True)
latent_cols = [c for c in df.columns if c.startswith("z_")]
X = df[latent_cols].values.astype(np.float32)
print(f"✅ {X.shape[0]} صف × {X.shape[1]} بعداً كامناً")

# ============================================================
# 2. TimeGAN
# ============================================================
class TimeGAN(nn.Module):
    """TimeGAN: شبكة توليدية للسلاسل الزمنية"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedder
        self.embedder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Generator
        self.generator = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Discriminator
        self.discriminator = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layers
        self.embed_fc = nn.Linear(hidden_dim, input_dim)
        self.gen_fc = nn.Linear(hidden_dim, input_dim)
        self.disc_fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def embed(self, x, h=None):
        out, h = self.embedder(x, h)
        return self.embed_fc(out), h
    
    def generate(self, x, h=None):
        out, h = self.generator(x, h)
        return self.gen_fc(out), h
    
    def discriminate(self, x, h=None):
        out, h = self.discriminator(x, h)
        return self.disc_fc(out[:, -1, :]), h

# ============================================================
# 3. تدريب TimeGAN
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ التدريب على: {device}")

# تحضير البيانات على شكل تسلسلات (seq_len=24, step=1)
seq_len = 24
sequences = []
for i in range(len(X) - seq_len):
    sequences.append(X[i:i+seq_len])
X_seq = np.array(sequences)
print(f"📊 {len(X_seq)} تسلسل (كل تسلسل: {seq_len} خطوة)")

# تحويل إلى Tensor
X_tensor = torch.FloatTensor(X_seq).to(device)
dataset = torch.utils.data.TensorDataset(X_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

input_dim = X.shape[1]
model = TimeGAN(input_dim).to(device)
optimizer_G = optim.Adam(
    list(model.embedder.parameters()) + list(model.generator.parameters()) + 
    list(model.embed_fc.parameters()) + list(model.gen_fc.parameters()), lr=0.001
)
optimizer_D = optim.Adam(
    list(model.discriminator.parameters()) + list(model.disc_fc.parameters()), lr=0.001
)
criterion = nn.BCELoss()

print(f"🧠 تدريب TimeGAN...")
for epoch in range(200):
    g_loss_total = 0
    d_loss_total = 0
    
    for (real,) in loader:
        batch_size = real.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Generator
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, seq_len, input_dim).to(device)
        fake_embedded, _ = model.embed(noise)
        fake_generated, _ = model.generate(fake_embedded)
        fake_pred, _ = model.discriminate(fake_generated)
        g_loss = criterion(fake_pred, real_labels) + 0.1 * nn.MSELoss()(fake_embedded, real)
        g_loss.backward()
        optimizer_G.step()
        g_loss_total += g_loss.item()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        real_embedded, _ = model.embed(real)
        real_pred, _ = model.discriminate(real_embedded)
        d_real_loss = criterion(real_pred, real_labels)
        noise = torch.randn(batch_size, seq_len, input_dim).to(device)
        fake_embedded, _ = model.embed(noise.detach())
        fake_generated, _ = model.generate(fake_embedded.detach())
        fake_pred, _ = model.discriminate(fake_generated)
        d_fake_loss = criterion(fake_pred, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        d_loss_total += d_loss.item()
    
    if (epoch + 1) % 40 == 0:
        print(f"  Epoch {epoch+1}/200 - G Loss: {g_loss_total/len(loader):.4f} - D Loss: {d_loss_total/len(loader):.4f}")

print("✅ تم تدريب TimeGAN")

# حفظ النموذج
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/timegan.pth")

# ============================================================
# 4. توليد 10,000 سيناريو صناعي
# ============================================================
print("🎲 توليد 10,000 سيناريو صناعي...")
model.eval()
synthetic_data = []

with torch.no_grad():
    for _ in range(200):  # 200 دفعة × 50 = 10,000
        noise = torch.randn(50, seq_len, input_dim).to(device)
        fake_embedded, _ = model.embed(noise)
        fake_generated, _ = model.generate(fake_embedded)
        synthetic_data.append(fake_generated.cpu().numpy())

synthetic = np.concatenate(synthetic_data, axis=0)
synthetic = synthetic.reshape(-1, input_dim)

# حفظ البيانات الصناعية
synthetic_df = pd.DataFrame(synthetic, columns=latent_cols)
synthetic_df.to_csv("data/synthetic_latent.csv", index=False)
print(f"✅ تم توليد {len(synthetic_df)} صف صناعي")
print("🎉 الذرة 3.4 اكتملت.")
