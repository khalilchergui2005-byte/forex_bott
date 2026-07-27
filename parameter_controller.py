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
    initial_capital: float = 100000.0        # Default starting balance for FTMO Challenge
    min_order_value: float = 500.0           # Absolute min notional value for 100k prop firm.
    min_order_pct: float = 0.005            # Min order as fraction of capital (0.5%). Dynamically scales with account size.

    # --- Execution Costs (Realistic Backtest) ---
    spread_pips: float = 1.0                # Spread in pips (1.0 = normal EUR/USD)
    pip_size: float = 0.0001                # Pip value in price units (4-digit pairs)
    commission_per_lot: float = 7.0           # Commission per standard lot round-turn ($)
    slippage_pips: float = 0.5                # Average slippage in pips
    slippage_model: str = 'fixed'             # fixed or volatility

    # --- Market Awareness (Fear Sensor) ---
    fear_threshold: float = 0.02            # ATR% threshold (2% daily). Risk is halved above this.
    
    # --- Position Sizing & Risk (FTMO Defaults) ---
    base_risk_percent: float = 0.005        # 0.5% per trade
    max_daily_loss_pct: float = 0.05        # 5% daily loss circuit breaker (FTMO)
    max_total_drawdown_pct: float = 0.10          # 10% max total drawdown (FTMO)
    
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
    turbo_tp_mult: float = 4.0  # FIX: Adjusted RRR to 1:2              # Take profit multiplier in high vol
    turbo_sl_mult: float = 2.0  # FIX: Adjusted RRR to 1:2              # Stop loss multiplier in high vol
    
    # --- Normal Regime ---
    normal_tp_mult: float = 3.0             # Take profit multiplier in normal vol
    
    # --- Second Target (after partial exit) ---
    second_target_atr_mult: float = 6.0     # ATR multiplier for second target
    
    # --- Strategy-Specific Trade Management (ARCH-02 Fix) ---
    # Maps column name to specific SL/TP multipliers. Overrides global TP/SL if found.
    strategy_sl_tp: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Signal': {'tp_mult': 2.0, 'sl_mult': 2.0, 'second_target_mult': 4.0},  # Mean Reversion
        'Asian_Breakout_Signal': {'tp_mult': 2.5, 'sl_mult': 2.0, 'second_target_mult': 5.0},  # Session Breakout
        'Absorption_Signal': {'tp_mult': 3.0, 'sl_mult': 1.5, 'second_target_mult': 6.0},  # Smart Money
        'Pairs_Signal': {'tp_mult': 0, 'sl_mult': 0, 'second_target_mult': 0},  # Pairs Trading
        'News_Signal': {'tp_mult': 3.0, 'sl_mult': 2.0, 'second_target_mult': 6.0}  # News Deviation
    })
    
    # --- Strategy Matrix Regime Rules ---
    regime_rules: Dict[int, int] = field(default_factory=lambda: {
        0: -1, 1: 0, 2: 1,
        3: -1, 4: 0, 5: 1,
        6: -1, 7: 0, 8: 1,
    })
    
    # --- Entry Indicator Thresholds ---
    rsi_buy_threshold: float = 41.0
    rsi_sell_threshold: float = 60.0
    bb_multiplier: float = 1.5              # Bollinger Band width
    
    # --- Momentum Agent (MACD/ADX) ---
    macd_fast: int = 14
    macd_slow: int = 25
    macd_signal: int = 9
    adx_period: int = 14
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

    # --- Meta-Labeling Filter Thresholds (Council Recommendation) ---
    min_smart_score_filter: float = 1.0      # Minimum SmartScore to accept a trade
    max_fear_level_filter: float = 1.0       # Fear multiplier must be >= this to accept (1.0 = no fear)
    require_shock_awe_filter: bool = False   # If True, only take trades when Shock&Awe is active

    # --- Shock & Awe (Anti-Martingale) ---
    # FROZEN - Failed statistical runs test (Atom 7.3). Rule increases risk without proven momentum.
    shock_awe_threshold: int = 3        # Consecutive wins to trigger
    # FROZEN - Failed statistical runs test (Atom 7.3). Rule increases risk without proven momentum.
    shock_awe_multiplier: float = 2.0   # Multiply risk by this when triggered

    # --- Temporal Edge Overlay (Council Atom 7.4 Discovery) ---
    allowed_hours: list = field(default_factory=lambda: [0])       # Hours to allow trading (UTC). 0 = midnight
    blocked_hours: list = field(default_factory=lambda: [13])      # Hours to block trading (UTC). 13 = 1 PM

    # --- Optimization Guards ---
    min_trades_optimizer: int = 30   # Minimum trades to accept a parameter set


    # --- Cost Scenarios (Atom 4.4) ---
    cost_scenario: str = "normal"

    # --- Order Execution Simulation (Atom 4.5) ---
    rejection_rate: float = 0.0
    partial_fill_probability: float = 0.0
    partial_fill_min_ratio: float = 0.5

    def apply_preset(self, preset_name: str):
        """Load a named preset for specific Prop Firm challenges."""
        presets = {
            'FTMO': {'max_daily_loss_pct': 0.05, 'max_total_drawdown_pct': 0.10},
            'THE_FUNDED_TRADER': {'max_daily_loss_pct': 0.05, 'max_total_drawdown_pct': 0.10},
            'DEFAULT': {},
        }
        if preset_name in presets:
            for k, v in presets[preset_name].items():
                setattr(self, k, v)

    COST_SCENARIOS = {
        "normal": {
            "spread_pips": 1.0,
            "slippage_pips": 0.5,
            "commission_per_lot": 7.0,
            "rejection_rate": 0.0,
            "partial_fill_probability": 0.0,
        },
        "bad": {
            "spread_pips": 3.0,
            "slippage_pips": 2.0,
            "commission_per_lot": 10.0,
            "rejection_rate": 0.02,
            "partial_fill_probability": 0.05,
        },
        "crisis": {
            "spread_pips": 10.0,
            "slippage_pips": 5.0,
            "commission_per_lot": 15.0,
            "rejection_rate": 0.05,
            "partial_fill_probability": 0.10,
        },
    }

    def apply_cost_scenario(self, scenario_name: str):
        """Apply a cost scenario preset to execution cost parameters."""
        if scenario_name not in self.COST_SCENARIOS:
            raise ValueError(
                f"Unknown cost scenario: {scenario_name}. "
                f"Available: {list(self.COST_SCENARIOS.keys())}"
            )
        for k, v in self.COST_SCENARIOS[scenario_name].items():
            setattr(self, k, v)
        self.cost_scenario = scenario_name

# Global instance (singleton pattern for now)
params = StrategyParameters()
