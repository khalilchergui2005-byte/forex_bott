"""
parameter_controller.py - Dynamic Parameter Hub for the Viral Entity.
All tunable knobs in one place. Ready for Bayesian Optimization, Walk-Forward,
and per-Prop-Firm scenario profiles.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class StrategyParameters:
    """Core trading parameters - starting point for institutional optimization."""
    
    # --- Account / Prop Firm Settings ---
    initial_capital: float = 10000.0         # Default starting balance (overridable)
    min_order_value: float = 50.0            # Absolute min notional value. TEMPORARY: Reduced for 10k testing. Use 500 for 100k prop firm.
    min_order_pct: float = 0.005            # Min order as fraction of capital (0.5%). Dynamically scales with account size.

    # --- Market Awareness (Fear Sensor) ---
    fear_threshold: float = 0.02            # ATR% threshold (2% daily). Risk is halved above this.
    
    # --- Position Sizing & Risk (confirmed by user: base values) ---
    base_risk_percent: float = 0.005        # 0.5% per trade
    max_daily_loss_pct: float = 0.02        # 2% daily loss circuit breaker
    max_total_dd_pct: float = 0.10          # 10% max total drawdown
    
    # --- Sniping (increased size on high-confidence signals) ---
    sniping_multiplier: float = 1.5         # Multiplier when Confirmation >= 2.0
    sniping_threshold: float = 2.0          # Confirmation score to trigger sniping
    
    # --- Partial Exit ---
    partial_exit_ratio: float = 0.5         # Close 50% at first target
    
    # --- Dynamic Stop & Take Profit ---
    stop_min_mult: float = 1.5              # Minimum stop ATR multiplier
    stop_max_mult: float = 2.5              # Maximum stop ATR multiplier
    stop_volatility_factor: float = 0.5     # How much vol_ratio reduces stop_mult
    
    # --- Turbo Volatility (VolClass == 2) ---
    turbo_tp_mult: float = 2.0              # Take profit multiplier in high vol
    turbo_sl_mult: float = 2.5              # Stop loss multiplier in high vol
    
    # --- Normal Regime ---
    normal_tp_mult: float = 3.0             # Take profit multiplier in normal vol
    
    # --- Second Target (after partial exit) ---
    second_target_atr_mult: float = 6.0     # ATR multiplier for second target
    
    # --- Strategy Matrix Regime Rules ---
    regime_rules: Dict[int, int] = field(default_factory=lambda: {
        0: -1, 1: 0, 2: 1,
        3: -1, 4: 0, 5: 1,
        6: -1, 7: 0, 8: 1,
    })
    
    # --- Entry Indicator Thresholds ---
    rsi_buy_threshold: float = 47.0
    rsi_sell_threshold: float = 50.0
    bb_multiplier: float = 1.2              # Bollinger Band width
    
    # --- Momentum Agent (MACD/ADX) ---
    macd_fast: int = 9
    macd_slow: int = 34
    macd_signal: int = 9
    adx_period: int = 16
    adx_threshold: int = 27

    # --- Target Behavior ---
    dynamic_target: bool = False  # False = fixed at entry (Prop Firm safe)

    # --- Kelly Criterion (Dynamic Position Sizing) ---
    # FROZEN - Disabled, caused risk over-shrinkage, see report section 3.
    kelly_window: int = 50              # Rolling window for Kelly calculation
    # FROZEN - Disabled, caused risk over-shrinkage, see report section 3.
    kelly_min_fraction: float = 0.25    # Minimum Kelly multiplier (safety floor)
    # FROZEN - Disabled, caused risk over-shrinkage, see report section 3.
    kelly_max_fraction: float = 2.0     # Maximum Kelly multiplier (safety cap)

    # --- Shock & Awe (Anti-Martingale) ---
    shock_awe_threshold: int = 3        # Consecutive wins to trigger
    shock_awe_multiplier: float = 2.0   # Multiply risk by this when triggered

    # --- Optimization Guards ---
    min_trades_optimizer: int = 30   # Minimum trades to accept a parameter set

    def apply_preset(self, preset_name: str):
        """Load a named preset for specific Prop Firm challenges."""
        presets = {
            'FTMO': {'max_daily_loss_pct': 0.05, 'max_total_dd_pct': 0.10},
            'THE_FUNDED_TRADER': {'max_daily_loss_pct': 0.05, 'max_total_dd_pct': 0.10},
            'DEFAULT': {},
        }
        if preset_name in presets:
            for k, v in presets[preset_name].items():
                setattr(self, k, v)

# Global instance (singleton pattern for now)
params = StrategyParameters()
