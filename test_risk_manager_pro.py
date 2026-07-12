"""
اختبارات مؤسسية شاملة لـ RiskManager
المستوى: إنتاجي. أي فشل هنا يعني خللاً قد يمس رأس المال.
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from risk_manager import RiskManager, CircuitBreakerException

@pytest.fixture
def rm():
    """إعداد مدير مخاطر افتراضي للاختبارات."""
    return RiskManager(10000.0)

class TestInitialization:
    """اختبارات التهيئة والمنطق الأساسي."""
    
    def test_reject_zero_or_negative_balance(self):
        """يجب رفض رصيد أولي صفر أو سالب."""
        with pytest.raises(ValueError):
            RiskManager(0)
        with pytest.raises(ValueError):
            RiskManager(-500)

    def test_default_values(self, rm):
        """التحقق من القيم الافتراضية."""
        assert rm.current_balance == Decimal('10000')
        assert rm.base_risk_percent == Decimal('0.005')
        assert rm.max_daily_loss_pct == Decimal('0.02')
        assert rm.max_total_drawdown_pct == Decimal('0.10')

class TestPositionSizing:
    """اختبارات حساب حجم الصفقة."""
    
    def test_normal_calculation(self, rm):
        """حساب طبيعي بوقف 20 نقطة."""
        size = rm.calculate_position_size(1.1000, 1.0980, 1.0)
        # 50$ مخاطرة / 0.0020 وقف = 25000 وحدة، لكن الحد 5% من 10000 = 500$ / 1.1000 = 454
        assert size == 454

    def test_very_tight_stop(self, rm):
        """يجب رفض وقف خسارة أقل من 0.00001."""
        size = rm.calculate_position_size(1.1000, 1.100001, 1.0)
        assert size == 0

    def test_entry_equals_stop(self, rm):
        """يجب رفض صفقة بدون مسافة وقف."""
        size = rm.calculate_position_size(1.1000, 1.1000, 1.0)
        assert size == 0

    def test_max_position_value_limit(self, rm):
        """لا تتجاوز قيمة الصفقة 5% من الرصيد."""
        # نجعل مسافة الوقف صغيرة جداً لنرى تأثير الحد الأعلى
        size = rm.calculate_position_size(1.1000, 1.0900, 1.0)
        # المخاطرة = 50$ / 0.0100 = 5000 وحدة. قيمة الصفقة = 5000 * 1.1000 = 5500 > 500 (5% من 10000)
        # الحد الأعلى سيطبق: 500$ / 1.1000 = 454
        assert size == 454

class TestCircuitBreaker:
    """اختبارات قاطع الخسارة."""
    
    def test_daily_loss_triggers_breaker(self, rm):
        """خسارة تتجاوز الحد اليومي يجب أن توقف التداول."""
        # الحد اليومي 2% من 10000 = 200$
        rm.update_pnl(-250)
        with pytest.raises(CircuitBreakerException, match="قاطع الخسارة اليومي"):
            rm.check_circuit_breaker()

    def test_daily_reset_on_new_day(self, rm):
        """عداد الخسارة اليومية يجب أن يعيد تعيين نفسه في يوم جديد."""
        rm.update_pnl(-150)
        # لا يجب أن يتعطل بعد
        rm.check_circuit_breaker()
        # نحاكي يوم جديد يدوياً
        rm.last_trade_date = date.today() - timedelta(days=1)
        rm.daily_pnl = Decimal('-150') # خسارة الأمس
        # الآن يجب أن يعيد التعيين ويسمح بالتداول
        try:
            rm.check_circuit_breaker()
            assert rm.daily_pnl == Decimal('0')
        except CircuitBreakerException:
            pytest.fail("Circuit breaker should have reset on a new day")

    def test_total_drawdown_breaker(self, rm):
        """خسارة كلية 10% من رأس المال الأولي يجب أن توقف كل شيء."""
        # خسارة 1000$ = 10% من 10000$
        rm.update_pnl(-1000)
        # الرصيد الحالي = 9000، والهبوط الكلي = 1000 = 10% من 10000
        with pytest.raises(CircuitBreakerException, match="قاطع الخسارة الكلي"):
            rm.check_circuit_breaker()

class TestPropFirmMode:
    """اختبارات وضع شركات التمويل."""
    
    def test_ftmo_mode_changes_limits(self, rm):
        """تفعيل FTMO يجب أن يغير الحد اليومي إلى 5% والكلي إلى 10%."""
        rm.enable_prop_firm_mode('FTMO')
        assert rm.max_daily_loss_pct == Decimal('0.05')
        assert rm.max_total_drawdown_pct == Decimal('0.10')
        assert rm.prop_firm_mode == 'FTMO'

    def test_ftmo_daily_limit_respected(self, rm):
        """FTMO: خسارة 400$ (4%) لا توقف، لكن 600$ (6%) توقف."""
        rm.enable_prop_firm_mode('FTMO')
        # الحد اليومي FTMO = 5% من 10000 = 500$
        rm.update_pnl(-400)
        # لا يجب أن يتوقف
        try:
            rm.check_circuit_breaker()
        except CircuitBreakerException:
            pytest.fail("FTMO breaker tripped too early (400$ loss should be allowed)")
        
        rm.update_pnl(-200) # المجموع -600$
        with pytest.raises(CircuitBreakerException, match="قاطع الخسارة اليومي"):
            rm.check_circuit_breaker()

class TestScalingAndUtility:
    """اختبارات التدرج والوظائف المساعدة."""
    
    def test_scaling_factor_increases_with_growth(self):
        """عامل التدرج يجب أن يزيد عند نمو الرصيد."""
        rm = RiskManager(10000.0)
        assert rm.scaling_factor == Decimal('1.0')
        # محاكاة ربح 30% (الرصيد = 13000$)
        rm.update_balance(13000.0)
        # يجب أن يزيد scaling_factor عن 1.0
        assert rm.scaling_factor > Decimal('1.0')

    def test_status_report_runs(self, rm):
        """التحقق أن تقرير الحالة يعمل ويعيد نصاً."""
        report = rm.status_report()
        assert "Balance:" in report
        assert "Trading Allowed:" in report