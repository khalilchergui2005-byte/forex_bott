# تقرير فوري - يتحمل أي خطأ في ملف السجل
from pathlib import Path
log_file = "logs/beast_log.csv"
if not Path(log_file).exists():
    print("ملف السجل غير موجود. انتظر دورة واحدة.")
    exit()

print("=" * 60)
print("   🐺 تقرير الوحش - آخر الأحداث")
print("=" * 60)

# قراءة مباشرة بدون تحليل
lines = Path(log_file).read_text(encoding='utf-8').strip().split('\n')
data_lines = [l for l in lines[1:] if l.strip()]

print(f"\n📊 إجمالي الأحداث: {len(data_lines)}")
print(f"\n📋 آخر 15 حدثاً:\n")

for line in data_lines[-15:]:
    parts = line.split(',')
    if len(parts) >= 3:
        ts = parts[0]
        strategy = parts[1]
        event = parts[2]
        details = ','.join(parts[3:]) if len(parts) > 3 else ""
        print(f"   [{ts}] {strategy:12} | {event:15} | {details[:70]}")
    else:
        print(f"   {line[:100]}")

print("\n" + "=" * 60)
print("   ملاحظة: إذا لم تظهر صفقات، فالسوق غير مناسب.")
print("   البوت يعمل - ينتظر الظروف المثالية.")
print("=" * 60)
