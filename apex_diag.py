import pandas as pd
import numpy as np
df = pd.read_csv("logs/trade_log_detailed.csv")
pnl = "net_pnl"
winners = df[df[pnl]>0][pnl]
losers = df[df[pnl]<0][pnl].abs()
print("TOTAL:", len(df))
print("WIN:", len(winners))
print("LOSE:", len(losers))
print("AVG_WIN:", round(winners.mean(),2))
print("AVG_LOSE:", round(losers.mean(),2))
print("RR:", round(winners.mean()/losers.mean(),3))
print("PF:", round(winners.sum()/losers.sum(),3))
print("HOURS:", df.groupby("hour")[pnl].sum().round(2).to_dict())
print("DAYS:", df.groupby("day")[pnl].sum().round(2).to_dict())
print("REGIME:", df.groupby("entry_regime")[pnl].sum().round(2).to_dict())
worst = df.nsmallest(5,pnl)[["entry_time","hour","day","entry_regime","net_pnl","rr_ratio"]]
print("WORST5:"); print(worst.to_string())
best = df.nlargest(5,pnl)[["entry_time","hour","day","entry_regime","net_pnl","rr_ratio"]]
print("BEST5:"); print(best.to_string())
