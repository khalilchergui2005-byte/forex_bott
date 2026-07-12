"""
risk_manager.py - درع المخاطر الديناميكي
المشروع: الكيان الفيروسي - أقوى شركة تداول
يدعم قوالب شركات التمويل (Prop Firms) وحجم صفقة ديناميكي.
"""
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("RiskManager")

class CircuitBreakerException(Exception):
    pass

class RiskManager:
    def __init__(self, initial_balance: float, config: Optional[Dict] = None):
        if initial_balance <= 0:
            raise ValueError("رأس المال الأولي يجب أن يكون أكبر من صفر.")
        self.initial_balance = Decimal(str(initial_balance))
        self.current_balance = Decimal(str(initial_balance))
        self.base_risk_percent = Decimal('0.005')
        self.max_daily_loss_pct = Decimal('0.02')
        self.max_weekly_loss_pct = Decimal('0.04')
        self.max_total_drawdown_pct = Decimal('0.10')
        self.min_order_pct = Decimal('0.005')  # 0.5% of balance as dynamic min
        self.min_order_value_abs = Decimal('500')  # Absolute floor from broker constraints
        self.max_position_pct = Decimal('0.05')  # 5% notional cap
        self.scaling_factor = Decimal('1.0')
        self.daily_pnl = Decimal('0')
        self.weekly_pnl = Decimal('0')
        self.last_trade_date = date.today()
        self.week_start_date = date.today() - timedelta(days=date.today().weekday())
        self.trading_allowed = True
        self.prop_firm_mode = None
        if config:
            self._apply_config(config)
        logger.info(f"RiskManager initialized. Balance: ${self.current_balance:,.2f}, Max Daily Loss: {self.max_daily_loss_pct*100:.1f}%, Max Total DD: {self.max_total_drawdown_pct*100:.1f}%")

    def _apply_config(self, config: Dict):
        mapping = {
            'base_risk_percent': 'base_risk_percent',
            'max_daily_loss_pct': 'max_daily_loss_pct',
            'max_total_drawdown_pct': 'max_total_drawdown_pct',
            'initial_balance': 'initial_balance',
            'min_order_pct': 'min_order_pct',
            'min_order_value_abs': 'min_order_value_abs',
            'max_position_pct': 'max_position_pct',
        }
        for key, attr in mapping.items():
            if key in config:
                setattr(self, attr, Decimal(str(config[key])))
        if 'initial_balance' in config:
            self.current_balance = self.initial_balance

    def calculate_position_size(self, entry_price: float, stop_loss_price: float, signal_quality: float = 1.0, risk_fraction: float = None) -> int:
        try:
            entry = Decimal(str(entry_price))
            stop = Decimal(str(stop_loss_price))
        except Exception:
            return 0
        quality = Decimal(str(max(0.0, min(1.0, signal_quality))))
        stop_distance = abs(entry - stop)
        
        # Defensive gate: reject invalid stop distances
        if pd.isna(float(stop_distance)) or np.isinf(float(stop_distance)) or stop_distance < Decimal('0.00001'):
            return 0
        
        base = Decimal(str(risk_fraction)) if risk_fraction is not None else self.base_risk_percent
        risk_percent = base * self.scaling_factor * quality
        risk_amount = self.current_balance * risk_percent
        try:
            position_size = risk_amount / stop_distance
        except Exception:
            return 0
        min_order_value = max(self.min_order_value_abs, self.current_balance * self.min_order_pct)
        position_value = position_size * entry
        max_position_value = self.current_balance * self.max_position_pct
        if position_value > max_position_value:
            position_size = max_position_value / entry
        if position_value < min_order_value:
            logger.warning(f"قيمة الصفقة المحسوبة (${position_value:,.2f}) أقل من الحد الأدنى (${min_order_value:,.2f}).")
            return 0
        final_size = int(position_size.to_integral_value(rounding=ROUND_DOWN))
        logger.info(f"حجم الصفقة المحسوب: {final_size} وحدة. المخاطرة: ${risk_amount:,.2f} ({risk_percent*100:.2f}% من الرصيد)")
        return final_size

    def update_balance(self, new_balance: float):
        """تحديث الرصيد وإعادة حساب عامل التدرج - صيغة الجذر التربيعي (المعيار الصناعي).
        الجذر التربيعي يوفر نمواً مستداماً مع حماية من التضخم المفرط للمخاطرة.
        """
        if new_balance <= 0:
            raise ValueError("الرصيد يجب أن يكون أكبر من صفر.")
        self.current_balance = Decimal(str(new_balance))
        ratio = float(self.current_balance / self.initial_balance)
        self.scaling_factor = Decimal(str(max(0.5, ratio ** 0.5))).quantize(
            Decimal('0.0001'), rounding=ROUND_DOWN
        )
        logger.info(f'تم تحديث الرصيد: ${self.current_balance:,.2f}, عامل التدرج: {self.scaling_factor}')

    def check_circuit_breaker(self, current_time=None) -> bool:
        if current_time is None:
            today = datetime.utcnow().date()
        elif hasattr(current_time, 'date'):
            today = current_time.date()
        else:
            today = current_time
        if self.last_trade_date is None:
            self.last_trade_date = today
        elif self.last_trade_date != today:
            self.daily_pnl = Decimal('0')
            self.last_trade_date = today
            logger.info("تمت إعادة تعيين عداد الخسارة اليومية.")
        week_start = today - timedelta(days=today.weekday())
        if self.week_start_date is None:
            self.week_start_date = week_start
        elif self.week_start_date != week_start:
            self.weekly_pnl = Decimal('0')
            self.week_start_date = week_start
            logger.info("تمت إعادة تعيين عداد الخسارة الأسبوعية.")
        daily_limit = self.current_balance * self.max_daily_loss_pct
        total_dd_limit = self.initial_balance * self.max_total_drawdown_pct
        total_drawdown = self.initial_balance - self.current_balance
        if total_drawdown >= total_dd_limit:
            self.trading_allowed = False
            msg = f"قاطع الخسارة الكلي: الهبوط ${total_drawdown:,.2f} تجاوز الحد الأقصى ${total_dd_limit:,.2f}"
            logger.critical(msg)
            raise CircuitBreakerException(msg)
        if self.daily_pnl <= -daily_limit:
            self.trading_allowed = False
            msg = f"قاطع الخسارة اليومي: الخسارة ${abs(self.daily_pnl):,.2f} تجاوزت الحد ${daily_limit:,.2f}"
            logger.critical(msg)
            raise CircuitBreakerException(msg)
        self.trading_allowed = True
        return True

    def update_pnl(self, profit_amount: float):
        pnl = Decimal(str(profit_amount))
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.current_balance += pnl
        logger.info(f"تم تحديث الأرباح/الخسائر: ${pnl:,.2f}. الرصيد الحالي: ${self.current_balance:,.2f}")

    def reset_for_testing(self):
        self.current_balance = self.initial_balance
        self.daily_pnl = Decimal('0')
        self.weekly_pnl = Decimal('0')
        self.last_trade_date = None
        self.week_start_date = None
        self.trading_allowed = True
        self.scaling_factor = Decimal('1.0')
        logger.info("تمت إعادة تعيين مدير المخاطر للاختبار.")

    def status_report(self) -> str:
        d = chr(36)
        lines = []
        lines.append("Balance: " + d + str(self.current_balance) + " / " + d + str(self.initial_balance))
        lines.append("Trading Allowed: " + str(self.trading_allowed))
        return "\n".join(lines)
    def enable_prop_firm_mode(self, preset_name):
        presets = {
            'FTMO': {
                'max_daily_loss_pct': Decimal('0.05'),
                'max_total_drawdown_pct': Decimal('0.10')
            }
        }
        name = preset_name.upper()
        if name not in presets:
            raise ValueError(f'Unknown prop firm: {preset_name}')
        p = presets[name]
        self.max_daily_loss_pct = p['max_daily_loss_pct']
        self.max_total_drawdown_pct = p['max_total_drawdown_pct']
        self.prop_firm_mode = name
        self.last_trade_date = date.today()
        self.week_start_date = date.today() - timedelta(days=date.today().weekday())
        logger.info(f'Prop firm mode enabled: {name}')