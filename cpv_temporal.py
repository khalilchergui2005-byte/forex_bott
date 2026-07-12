"""
Combinatorial Purged Cross-Validation (CPCV)
Temporal Engine

Reference: Lopez de Prado (2018)
Advances in Financial Machine Learning, Ch.7
"""


import numpy as np
from itertools import combinations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CPCVSplit:
    """Single train/test split with metadata."""
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: Tuple[int, ...]
    test_groups: Tuple[int, ...]
    split_id: int

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)

class TemporalCPCV:
    """
    Combinatorial Purged Cross-Validation.

    n_groups: temporal groups (default 6)
    n_test_groups: test groups per combo (default 2)
    purge_gap: bars to purge (default 24)
    embargo_pct: extra embargo fraction (default 0.01)
    """

    def __init__(self, n_groups=6, n_test_groups=2,
                 purge_gap=24, embargo_pct=0.01):
        if n_test_groups >= n_groups:
            raise ValueError(
                f"n_test_groups ({n_test_groups}) must be < "
                f"n_groups ({n_groups})")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self._test_combos = list(
            combinations(range(n_groups), n_test_groups))
        self.n_splits = len(self._test_combos)

    def generate_splits(self, n_bars):
        """Generate CPCV splits for n_bars data points."""
        group_ids = self._assign_groups(n_bars)
        splits = []
        for sid, tg in enumerate(self._test_combos):
            tr_g = tuple(
                g for g in range(self.n_groups)
                if g not in tg)
            tr_mask = np.isin(group_ids, tr_g)
            te_mask = np.isin(group_ids, tg)
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]
            tr_idx, te_idx = self._apply_purge(
                tr_idx, te_idx)
            tr_idx = self._apply_embargo(
                tr_idx, te_idx)
            if len(tr_idx) < 100 or len(te_idx) < 30:
                continue
            splits.append(CPCVSplit(
                train_indices=tr_idx,
                test_indices=te_idx,
                train_groups=tr_g,
                test_groups=tg,
                split_id=sid))
        if not splits:
            raise ValueError(
                f"No valid splits. n_bars={n_bars}")
        return splits

    def _assign_groups(self, n_bars):
        """Assign each bar to a temporal group."""
        gids = np.empty(n_bars, dtype=int)
        gs = n_bars // self.n_groups
        rem = n_bars % self.n_groups
        start = 0
        for g in range(self.n_groups):
            sz = gs + (1 if g < rem else 0)
            gids[start:start + sz] = g
            start += sz
        return gids

    def _apply_purge(self, train_idx, test_idx):
        """Remove train bars near any test bar."""
        if (self.purge_gap == 0 or
                len(train_idx) == 0 or
                len(test_idx) == 0):
            return train_idx, test_idx
        n = max(train_idx.max(), test_idx.max()) + 1
        tm = np.zeros(n, dtype=bool)
        tm[test_idx] = True
        dl = tm.copy()
        for _ in range(self.purge_gap):
            p = dl.copy()
            dl[1:] |= p[:-1]
            dl[:-1] |= p[1:]
        keep = ~dl[train_idx]
        return train_idx[keep], test_idx

    def _apply_embargo(self, train_idx, test_idx):
        """Extra embargo for autocorrelation."""
        if self.embargo_pct <= 0 or len(test_idx) == 0:
            return train_idx
        eb = max(1, int(len(test_idx) * self.embargo_pct))
        tmin = test_idx.min()
        cutoff = tmin - self.purge_gap - eb
        return train_idx[train_idx < cutoff]

    def summary(self):
        """Return configuration summary."""
        return (
            f"CPCV: {self.n_groups} groups, "
            f"{self.n_test_groups} test, "
            f"{self.n_splits} splits, "
            f"purge={self.purge_gap}, "
            f"embargo={self.embargo_pct}")
