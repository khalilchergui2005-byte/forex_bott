f = 'macro_data_fetcher.py'
t = open(f, encoding='utf-8').read()

# إدراج مفتاح الـ API
t = t.replace('API_KEY = "YOUR_FRED_API_KEY_HERE"', 'API_KEY = "180ef006ed27e1d219e49daae4ff31ad"')
open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: FRED API Key inserted")

# تشغيل الأداة لجلب البيانات
import subprocess
import sys
process = subprocess.run([sys.executable, "macro_data_fetcher.py"], capture_output=True, text=True)
print(process.stdout)
if process.stderr:
    print("Errors/Warnings:")
    print(process.stderr)
