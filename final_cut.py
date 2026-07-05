import pandas as pd
df = pd.read_csv("logs/trade_log_detailed.csv")
pnl = "net_pnl"
print("=== تحليل الاربعاء بعمق ===")
wed = df[df["day"]==2]
print(f"صفقات الاربعاء: {len(wed)}")
print(f"ربح الاربعاء: {wed[pnl].sum():.2f}")
print(f"فوز الاربعاء: {(wed[pnl]>0).mean()*100:.1f}%")
print(wed.groupby("hour")[pnl].agg(["sum","count","mean"]).round(2).to_string())
print()
print("=== تحليل الجمعة ===")
fri = df[df["day"]==4]
print(f"صفقات الجمعة: {len(fri)}")
print(f"ربح الجمعة: {fri[pnl].sum():.2f}")
print(fri.groupby("hour")[pnl].agg(["sum","count","mean"]).round(2).to_string())
print()
print("=== سيناريو: الجمعة فقط ===")
fri_only = df[df["day"]==4]
w=fri_only[fri_only[pnl]>0][pnl]; l=fri_only[fri_only[pnl]<0][pnl].abs()
if len(l)>0: print(f"PF الجمعة فقط: {w.sum()/l.sum():.3f}")
print(f"فوز: {len(w)/len(fri_only)*100:.1f}% | صفقات: {len(fri_only)}")
print()
print("=== سيناريو: بدون الاربعاء ===")
no_wed = df[df["day"]!=2]
w2=no_wed[no_wed[pnl]>0][pnl]; l2=no_wed[no_wed[pnl]<0][pnl].abs()
if len(l2)>0: print(f"PF بدون الاربعاء: {w2.sum()/l2.sum():.3f}")
print(f"فوز: {len(w2[w2>0])/len(no_wed)*100:.1f}% | صفقات: {len(no_wed)}")
print()
print("=== الساعة 13 — هل نبقيها؟ ===")
h13 = df[df["hour"]==13]
print(f"صفقات ساعة 13: {len(h13)}")
print(f"ربح ساعة 13: {h13[pnl].sum():.2f}")
print(f"فوز ساعة 13: {(h13[pnl]>0).mean()*100:.1f}%")
print(h13.groupby("day")[pnl].agg(["sum","count"]).round(2).to_string())
