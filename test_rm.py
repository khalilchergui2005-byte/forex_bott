import sys
sys.path.insert(0, r'C:\Users\GTINF\Desktop\forex_bot')
from risk_manager import RiskManager, CircuitBreakerException

print("=== اختبار 1: إنشاء مدير المخاطر ===")
rm = RiskManager(10000)
print(rm.status_report())

print("\n=== اختبار 2: حساب حجم الصفقة (وقف 20 نقطة) ===")
size = rm.calculate_position_size(1.1000, 1.0980, 1.0)
print(f"حجم الصفقة المحسوب: {size} وحدة")

print("\n=== اختبار 3: محاكاة خسارة 100$ (أقل من الحد اليومي) ===")
rm.update_pnl(-100)
try:
    rm.check_circuit_breaker()
    print("نجاح: قاطع الخسارة لم يتوقف (الخسارة تحت الحد)")
except CircuitBreakerException as e:
    print(f"فشل: {e}")

print("\n=== اختبار 4: محاكاة خسارة إضافية 250$ (يتجاوز الحد اليومي) ===")
rm.update_pnl(-250)
try:
    rm.check_circuit_breaker()
    print("فشل: كان يجب أن يتوقف قاطع الخسارة!")
except CircuitBreakerException as e:
    print(f"نجاح: قاطع الخسارة أوقف التداول. السبب: {e}")

print("\n=== اختبار 5: وضع شركة التمويل FTMO ===")
rm2 = RiskManager(10000)
rm2.enable_prop_firm_mode('FTMO')
print(rm2.status_report())

print("\n=== جميع الاختبارات انتهت ===")