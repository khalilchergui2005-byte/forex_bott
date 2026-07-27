f = 'economic_calendar_fetcher.py'
t = open(f, encoding='utf-8').read()

# إدراج مفتاح الـ API
t = t.replace('API_KEY = "YOUR_API_KEY_HERE"', 'API_KEY = "d9j7dcpr01qg3bn386j0d9j7dcpr01qg3bn386jg"')
open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: API Key inserted")

# تشغيل السكريبت لجلب البيانات
import subprocess
import sys
process = subprocess.run([sys.executable, "economic_calendar_fetcher.py"], capture_output=True, text=True)
print(process.stdout)
if process.stderr:
    print("Errors/Warnings:")
    print(process.stderr)
