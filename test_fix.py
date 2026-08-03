import lzma, struct, requests
from datetime import datetime

d = datetime(2024, 5, 15)
hour = 10
url = f"https://datafeed.dukascopy.com/datafeed/EURUSD/{d.year}/{d.month-1:02d}/{d.day:02d}/{hour:02d}h_ticks.bi5"
print("URL:", url)
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
print("Status:", r.status_code, "| Bytes:", len(r.content))
if r.status_code == 200 and len(r.content) > 0:
    raw = lzma.decompress(r.content)
    for t in list(struct.iter_unpack(">iiiff", raw))[:5]:
        ms, ask, bid, askvol, bidvol = t
        print(ms, round(ask/100000.0,5), round(bid/100000.0,5), askvol, bidvol)
else:
    print("رد غير صالح - جرب ساعة أخرى")
