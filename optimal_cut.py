import pandas as pd
df = pd.read_csv("logs/trade_log_detailed.csv")
pnl = "net_pnl"
bad = (df["day"]==2) & (df["hour"]==13)
clean = df[~bad].copy()
w=clean[clean[pnl]>0][pnl]; l=clean[clean[pnl]<0][pnl].abs()
print(f"الصفقات: {len(df)} -> {len(clean)}")
print(f"PF: {df[df[pnl]>0][pnl].sum()/df[df[pnl]<0][pnl].abs().sum():.3f} -> {w.sum()/l.sum():.3f}")
print(f"فوز: {(df[pnl]>0).mean()*100:.1f}% -> {(clean[pnl]>0).mean()*100:.1f}%")
print(f"الربح الكلي: {df[pnl].sum():.2f} -> {clean[pnl].sum():.2f}")
