f = 'alpha_vantage_news.py'
t = open(f, encoding='utf-8').read()

# إدراج مفتاح الـ API
t = t.replace('API_KEY = "YOUR_API_KEY_HERE"', 'API_KEY = "HKV29AUFFRK26P61"')
open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: Alpha Vantage API Key inserted")

# تشغيل الأداة لجلب البيانات
import subprocess
import sys
process = subprocess.run([sys.executable, "alpha_vantage_news.py"], capture_output=True, text=True)
print(process.stdout)
if process.stderr:
    print("Errors/Warnings:")
    print(process.stderr)
