"""
deflated_sharpe_calculator.py - Deflated Sharpe Ratio Calculator
Implements Bailey & López de Prado (2014) DSR methodology.
Tests whether observed Sharpe is significant after multiple testing correction.
"""
import numpy as np
from scipy import stats

class DeflatedSharpeCalculator:
    EULER_GAMMA = 0.5772156649
    
    def __init__(self, num_trials, sharpe_estimates):
        self.N = num_trials
        self.sharpe_estimates = np.array(sharpe_estimates)
        
    def expected_max_sharpe(self):
        """E[max SR] under null hypothesis (all strategies have zero true SR)."""
        sigma_sr = np.std(self.sharpe_estimates)
        
        # Expected maximum of N draws from standard normal (Bailey & López de Prado)
        z = ((1 - self.EULER_GAMMA) * stats.norm.ppf(1 - 1/self.N) + 
             self.EULER_GAMMA * stats.norm.ppf(1 - 1/(self.N * np.e)))
        
        return sigma_sr * z
    
    def compute_dsr(self, observed_sr, num_returns, skewness=0, kurtosis=3):
        """
        Computes the Deflated Sharpe Ratio.
        
        Returns:
            DSR probability (0 to 1). >0.95 means significant at 5% level.
        """
        T = num_returns
        SR = observed_sr
        gamma3 = skewness
        gamma4 = kurtosis
        
        e_max_sr = self.expected_max_sharpe()
        
        numerator = (SR - e_max_sr) * np.sqrt(T - 1)
        denominator = np.sqrt(1 - gamma3 * SR + ((gamma4 - 1) / 4) * SR**2)
        
        if denominator == 0:
            return 0.0, e_max_sr
        
        dsr_z = numerator / denominator
        dsr_prob = stats.norm.cdf(dsr_z)
        
        return dsr_prob, e_max_sr
    
    def compute_min_trl(self, observed_sr, skewness=0, kurtosis=3, confidence=0.95):
        """Minimum Track Record Length — minimum observations needed to confirm edge."""
        SR = observed_sr
        gamma3 = skewness
        gamma4 = kurtosis
        z_alpha = stats.norm.ppf(confidence)
        
        if SR <= 0:
            return float('inf')
        
        numerator = (1 - gamma3 * SR + ((gamma4 - 1) / 4) * SR**2) * z_alpha**2
        T_min = 1 + numerator / SR**2
        
        return int(np.ceil(T_min))


if __name__ == "__main__":
    print("=" * 60)
    print("   DEFLATED SHARPE RATIO CALCULATOR")
    print("=" * 60)
    
    # All 17 strategy trials with their PF values and actual OOS trade counts
    strategies = [
        ("Range Hunter", 0.77, 20),
        ("Asian Breakout", 0.81, 20),
        ("Pairs Trading", 0.60, 20),
        ("News Deviation", 0.21, 20),
        ("Liquidity Absorption", 0.63, 20),
        ("Long Lower Shadow", 1.25, 20),
        ("High ATR", 1.07, 20),
        ("Hour00", 1.02, 20),
        ("Matrix", 0.91, 20),
        ("Momentum", 0.88, 20),
        ("Donchian", 0.98, 20),
        ("MA Crossover", 1.00, 20),
        ("Vol Compression", 0.89, 20),
        ("Weekend Gap Fade", 1.23, 20),
        ("Liquidity Sweep", 0.76, 20),
        ("ML Meta-Labeler (5Y)", 1.05, 12),
        ("ML Continuous (20Y)", 2.33, 22),  # 22 actual OOS trades
    ]
    
    # Convert PF to approximate Sharpe ratios
    # SR ≈ ln(PF) * sqrt(n_trades) — standard conversion
    print("\nConverting PF to approximate Sharpe ratios...")
    print("(Using ln(PF) * sqrt(n_trades), n_trades=12 for OOS test)")
    print()
    
    sharpe_estimates = []
    for name, pf, n_trades in strategies:
        sr = np.log(pf) * np.sqrt(n_trades) if pf > 0 else -2.0
        sharpe_estimates.append(sr)
        print(f"  {name:25s}: PF={pf:.2f}  N={n_trades}  SR≈{sr:+.4f}")
    
    # Our best strategy
    best_idx = np.argmax(sharpe_estimates)
    best_name = strategies[best_idx][0]
    best_sr = sharpe_estimates[best_idx]
    
    print(f"\nBest strategy: {best_name} (SR≈{best_sr:.4f})")
    
    # Compute DSR
    calculator = DeflatedSharpeCalculator(
        num_trials=len(strategies),
        sharpe_estimates=sharpe_estimates
    )
    
    # Typical assumptions for H1 Forex trend-following:
    # - Moderate negative skewness (trend strategies cut winners short, let losers run on reversal)
    # - Fat tails (excess kurtosis)
    skewness = -0.3
    kurtosis = 4.0  # Raw kurtosis (excess = 1.0)
    
    dsr_prob, e_max_sr = calculator.compute_dsr(
        observed_sr=best_sr,
        num_returns=22,  # 22 actual OOS trades for the best model
        skewness=skewness,
        kurtosis=kurtosis
    )
    
    min_trl = calculator.compute_min_trl(
        observed_sr=best_sr,
        skewness=skewness,
        kurtosis=kurtosis,
        confidence=0.95
    )
    
    print(f"\n{'='*60}")
    print(f"   DSR ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Number of trials (N): {len(strategies)}")
    print(f"Std of Sharpe estimates: {np.std(sharpe_estimates):.4f}")
    print(f"Expected max SR under null: {e_max_sr:+.4f}")
    print(f"Observed best SR: {best_sr:+.4f}")
    print(f"Skewness assumed: {skewness}")
    print(f"Kurtosis assumed: {kurtosis}")
    print(f"Number of returns (T): 12")
    print(f"\nDeflated Sharpe Ratio (probability): {dsr_prob:.4f}")
    print(f"  (>0.95 = significant at 5% level)")
    print(f"  (>0.99 = significant at 1% level)")
    print(f"\nMinimum Track Record Length: {min_trl} trades")
    print(f"(Need this many OOS trades to confirm edge at 95% confidence)")
    
    print(f"\n{'='*60}")
    if dsr_prob < 0.95:
        print(f"VERDICT: NOT SIGNIFICANT (DSR={dsr_prob:.4f} < 0.95)")
        print(f"\nThe observed edge CANNOT be distinguished from the")
        print(f"best result of {len(strategies)} random trials.")
        print(f"\nRequired action: Collect {min_trl} OOS trades before trusting this edge.")
    else:
        print(f"VERDICT: SIGNIFICANT (DSR={dsr_prob:.4f} >= 0.95)")
        print(f"\nThe observed edge is statistically significant even after")
        print(f"multiple testing correction across {len(strategies)} trials.")
    print(f"{'='*60}")
