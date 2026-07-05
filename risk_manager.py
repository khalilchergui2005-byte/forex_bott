"""
مدير المخاطر المؤسسي - المشروع: الكيان الفيروسي
=================================================
وحدة تحكم آمنة تماماً: لا تسمح بتمرير أي صفقة دون المرور عبر قواعدها.
كل الحدود نسب مئوية من الرصيد الحالي، تتكيف تلقائياً مع نمو رأس المال أو انكماشه.
تدعم قوالب شركات التمويل (Prop Firms) للامتثال التلقائي.
المستوى: إنتاج مؤسسي حقيقي.
"""

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Optional, Tuple, Dict

# إعداد التسجيل الاحترافي
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RiskManager")

class CircuitBreakerException(Exception):
    """استثناء خاص: يُرفع عندما يمنع قاطع الخسارة التداول."""
    pass

class RiskManager:
    """
    عقل المخاطر الديناميكي.
    يجب إنشاء كائن واحد فقط من هذا الصنف لكل حساب تداول.
    """

    def __init__(self, initial_balance: float, config: Optional[Dict] = None):
        """
        تهيئة مدير المخاطر.
        
        :param initial_balance: رأس المال الابتدائي (لحساب الحد الأقصى للهبوط الكلي).
        :param config: قاموس اختياري لتجاوز الإعدادات الافتراضية.
        """
        if initial_balance <= 0:
            raise ValueError("رأس المال الأولي يجب أن يكون أكبر من صفر.")

        # إعدادات افتراضية مؤسسية (نسب مئوية)
        self.initial_balance = Decimal(str(initial_balance))
        self.current_balance = Decimal(str(initial_balance))
        
        # القيم الأساسية (نسب)
        self.base_risk_percent = Decimal('0.005')      # 0.5% مخاطرة لكل صفقة
        self.max_daily_loss_pct = Decimal('0.02')      # 2% خسارة يومية قصوى
        self.max_weekly_loss_pct = Decimal('0.04')     # 4% خسارة أسبوعية
        self.max_total_drawdown_pct = Decimal('0.10')  # 10% هبوط كلي من رأس المال الأولي
        
        # النمو الآمن: مع زيادة الرصيد، تزيد المخاطرة ببطء
        self.scaling_factor = Decimal('1.0')
        self.max_scale_factor = Decimal('2.0')         # أقصى مضاعفة للمخاطرة (1% كحد أقصى)
        self.growth_threshold = Decimal('1.20')         # عند 20% ربح نبدأ بزيادة المخاطرة
        
        # القيود المطلقة للسلامة (تطبق مهما كانت الظروف)
        self.min_order_value = Decimal('500')           # الحد الأدنى لقيمة الصفقة
        self.max_position_pct = Decimal('0.05')        # لا تتجاوز الصفقة 5% من الرصيد كقيمة افتراضية
        
        # حالة التداول اليومية
        self.daily_pnl = Decimal('0')
        self.weekly_pnl = Decimal('0')
        self.last_trade_date = None
        self.week_start_date = None
        self.trading_allowed = True
        
        # وضع شركات التمويل
        self.prop_firm_mode = None
        
        # تجاوز الإعدادات من قاموس المستخدم
        if config:
            self._apply_config(config)
            
        logger.info(f"RiskManager initialized. Balance: ${self.current_balance:,.2f}, "
                    f"Max Daily Loss: {self.max_daily_loss_pct*100:.1f}%, "
                    f"Max Total DD: {self.max_total_drawdown_pct*100:.1f}%")

    def _apply_config(self, config: Dict):
        """تطبيق إعدادات مخصصة من المستخدم."""
        mapping = {
            'base_risk_percent': 'base_risk_percent',
            'max_daily_loss_pct': 'max_daily_loss_pct',
            'max_weekly_loss_pct': 'max_weekly_loss_pct',
            'max_total_drawdown_pct': 'max_total_drawdown_pct',
            'initial_balance': 'initial_balance',
        }
        for key, attr in mapping.items():
            if key in config:
                value = config[key]
                if isinstance(value, (int, float)):
                    setattr(self, attr, Decimal(str(value)))
                else:
                    setattr(self, attr, Decimal(value))
        # إعادة تعيين الرصيد الحالي إذا تغير الرصيد الأولي
        if 'initial_balance' in config:
            self.current_balance = self.initial_balance

    def enable_prop_firm_mode(self, template: str = "FTMO"):
        """
        تفعيل وضع الالتزام بقواعد شركات التمويل.
        القوالب المدعومة: FTMO (افتراضي)، TFT.
        """
        templates = {
            "FTMO": {
                'max_daily_loss_pct': Decimal('0.05'),  # 5% يومي
                'max_total_drawdown_pct': Decimal('0.10'), # 10% كلي
                'profit_target_pct': Decimal('0.10')     # هدف 10% (للتحدي)
            },
            "TFT": {  # مثال: The Funded Trader
                'max_daily_loss_pct': Decimal('0.04'),
                'max_total_drawdown_pct': Decimal('0.08'),
                'profit_target_pct': Decimal('0.08')
            }
        }
        if template.upper() not in templates:
            raise ValueError(f"قالب غير معروف: {template}. القوالب المتاحة: {list(templates.keys())}")
        
        settings = templates[template.upper()]
        self.max_daily_loss_pct = settings['max_daily_loss_pct']
        self.max_total_drawdown_pct = settings['max_total_drawdown_pct']
        self.prop_firm_mode = template.upper()
        logger.warning(f"Prop Firm Mode ENABLED: {self.prop_firm_mode}. "
                       f"Daily Loss Limit: {self.max_daily_loss_pct*100}%, "
                       f"Max Total DD: {self.max_total_drawdown_pct*100}%")

    def update_balance(self, new_balance: float):
        """تحديث الرصيد الحالي (يُستدعى من حساب الوسيط الحي)."""
        if new_balance <= 0:
            raise ValueError("الرصيد لا يمكن أن يكون صفراً أو أقل.")
        self.current_balance = Decimal(str(new_balance))
        self._update_scaling()

    def _update_scaling(self):
        """ضبط عامل النمو التدريجي الآمن."""
        growth_ratio = self.current_balance / self.initial_balance
        if growth_ratio >= self.growth_threshold:
            extra = min(growth_ratio - self.growth_threshold, Decimal('1.0'))
            self.scaling_factor = Decimal('1.0') + (extra * Decimal('0.5'))
            self.scaling_factor = min(self.scaling_factor, self.max_scale_factor)
        else:
            self.scaling_factor = Decimal('1.0')

    def calculate_position_size(self, entry_price: float, stop_loss_price: float,
                                 signal_quality: float = 1.0) -> int:
        """
        حساب حجم الصفقة الآمن (بالوحدات).
        
        :param entry_price: سعر الدخول.
        :param stop_loss_price: سعر وقف الخسارة.
        :param signal_quality: جودة الإشارة (0.0 إلى 1.0). 1.0 = ثقة كاملة.
        :return: عدد الوحدات (int).
        """
        if not self.trading_allowed:
            raise CircuitBreakerException("التداول معلق بواسطة قاطع الخسارة.")
        
        entry = Decimal(str(entry_price))
        stop = Decimal(str(stop_loss_price))
        quality = Decimal(str(max(0.0, min(1.0, signal_quality))))
        
        # 1. حساب مسافة الوقف
        stop_distance = abs(entry - stop)
        if stop_distance < Decimal('0.00001'):
            logger.error("مسافة وقف الخسارة صغيرة جداً (أقل من 0.00001). تم الرفض.")
            return 0
        
        # 2. المبلغ المخاطر = الرصيد × نسبة المخاطرة × معامل النمو × جودة الإشارة
        risk_percent = self.base_risk_percent * self.scaling_factor * quality
        risk_amount = self.current_balance * risk_percent
        
        # 3. حجم الصفقة = المبلغ المخاطر / مسافة الوقف
        try:
            position_size = risk_amount / stop_distance
        except (InvalidOperation, ZeroDivisionError):
            logger.error("خطأ في حساب حجم الصفقة.")
            return 0
        
        # 4. تطبيق الحدود العليا (لا تتجاوز نسبة معينة من الرصيد)
        max_value = self.current_balance * self.max_position_pct
        max_size = max_value / entry
        position_size = min(position_size, max_size)
        
        # 5. التأكد من أن قيمة الصفقة لا تقل عن الحد الأدنى
        position_value = position_size * entry
        if position_value < self.min_order_value:
            logger.warning(f"قيمة الصفقة المحسوبة (${position_value:,.2f}) أقل من الحد الأدنى (${self.min_order_value:,.2f}).")
            return 0
        
        # 6. إرجاع الحجم بعد تقريبه إلى عدد صحيح آمن
        final_size = int(position_size.to_integral_value(rounding=ROUND_DOWN))
        logger.info(f"حجم الصفقة المحسوب: {final_size} وحدة. المخاطرة: ${risk_amount:,.2f} "
                    f"({risk_percent*100:.2f}% من الرصيد)")
        return final_size

    def check_circuit_breaker(self) -> bool:
        """
        فحص حالة قاطع الخسارة. يُستدعى قبل كل دورة تداول.
        يعيد True إذا كان التداول مسموحاً، وإلا يرفع استثناء أو يعيد False.
        """
        today = date.today()
        # إعادة تعيين العدادات اليومية
        if self.last_trade_date != today:
            self.daily_pnl = Decimal('0')
            self.last_trade_date = today
            logger.info("تمت إعادة تعيين عداد الخسارة اليومية.")
        
        # إعادة تعيين الأسبوعي (يبدأ الأسبوع يوم الاثنين)
        week_start = today - timedelta(days=today.weekday())
        if self.week_start_date != week_start:
            self.weekly_pnl = Decimal('0')
            self.week_start_date = week_start
            logger.info("تمت إعادة تعيين عداد الخسارة الأسبوعية.")
        
        # حساب الحدود الفعلية
        daily_limit = self.current_balance * self.max_daily_loss_pct
        weekly_limit = self.current_balance * self.max_weekly_loss_pct
        total_dd_limit = self.initial_balance * self.max_total_drawdown_pct
        
        # فحص الخسارة الكلية من رأس المال الأولي
        total_drawdown = self.initial_balance - self.current_balance
        if total_drawdown >= total_dd_limit:
            self.trading_allowed = False
            msg = (f"قاطع الخسارة الكلي: الهبوط ${total_drawdown:,.2f} "
                   f"تجاوز الحد الأقصى ${total_dd_limit:,.2f}")
            logger.critical(msg)
            raise CircuitBreakerException(msg)
        
        # فحص الخسارة اليومية
        if self.daily_pnl <= -daily_limit:
            self.trading_allowed = False
            msg = (f"قاطع الخسارة اليومي: الخسارة ${abs(self.daily_pnl):,.2f} "
                   f"تجاوزت الحد ${daily_limit:,.2f}")
            logger.critical(msg)
            raise CircuitBreakerException(msg)
        
        # فحص الخسارة الأسبوعية
        if self.weekly_pnl <= -weekly_limit:
            self.trading_allowed = False
            msg = (f"قاطع الخسارة الأسبوعي: الخسارة ${abs(self.weekly_pnl):,.2f} "
                   f"تجاوزت الحد ${weekly_limit:,.2f}")
            logger.critical(msg)
            raise CircuitBreakerException(msg)
        
        self.trading_allowed = True
        return True

    def update_pnl(self, profit_amount: float):
        """
        تحديث سجلات الأرباح والخسائر اليومية والأسبوعية.
        يُستدعى بعد إغلاق كل صفقة.
        """
        pnl = Decimal(str(profit_amount))
        if self.last_trade_date is None:
            self.last_trade_date = date.today()
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.current_balance += pnl
        self._update_scaling()
        logger.info(f"تم تحديث الأرباح/الخسائر: ${pnl:,.2f}. الرصيد الحالي: ${self.current_balance:,.2f}")

    def reset_for_testing(self):
        """إعادة تعيين كاملة للاختبارات الخلفية."""
        self.current_balance = self.initial_balance
        self.daily_pnl = Decimal('0')
        self.weekly_pnl = Decimal('0')
        self.last_trade_date = None
        self.week_start_date = None
        self.trading_allowed = True
        self.scaling_factor = Decimal('1.0')
        logger.info("تمت إعادة تعيين مدير المخاطر للاختبار.")

    def status_report(self) -> str:
        """تقرير نصي بالحالة الراهنة."""
        total_dd = self.initial_balance - self.current_balance
        dd_pct = (total_dd / self.initial_balance) * 100
        report = (
            f"--- Risk Manager Status ---\n"
            f"Balance: ${self.current_balance:,.2f}\n"
            f"Initial: ${self.initial_balance:,.2f}\n"
            f"Total DD: ${total_dd:,.2f} ({dd_pct:.2f}%)\n"
            f"Daily P&L: ${self.daily_pnl:,.2f}\n"
            f"Weekly P&L: ${self.weekly_pnl:,.2f}\n"
            f"Trading Allowed: {self.trading_allowed}\n"
            f"Base Risk: {self.base_risk_percent*100:.2f}% (Scaled: {self.base_risk_percent*self.scaling_factor*100:.2f}%)\n"
            f"Prop Firm Mode: {self.prop_firm_mode or 'None'}\n"
            f"----------------------------"
        )
        return report