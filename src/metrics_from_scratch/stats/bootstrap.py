"""Bootstrap confidence intervals: how much a metric would move on another sample of the same size."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float  # the statistic on the original sample
    low: float
    high: float
    confidence: float
    undefined_resamples: int = 0  # resamples where the statistic was nan, left out of the interval

    def excludes(self, value: float) -> bool:
        return not self.low <= value <= self.high


@dataclass(frozen=True)
class Bootstrap:
    """Percentile bootstrap: resample examples with replacement, recompute the statistic, take the middle share.

    All arrays passed to `interval` are resampled with the same row indices, so paired data stays paired. That is
    what makes a paired comparison (model A vs model B on one test set) tighter than comparing two separate CIs.
    A statistic can be undefined on some resamples (kappa when one rater uses a single label). Those resamples are
    dropped and counted in `Interval.undefined_resamples`, so one of them can't turn the whole interval into nan.
    """

    n_resamples: int = 2_000
    confidence: float = 0.95
    seed: int = 0

    def __post_init__(self):
        if not 0 < self.confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")

    def interval(self, statistic: Callable[..., float], *arrays: np.ndarray) -> Interval:
        arrays = tuple(np.asarray(a) for a in arrays)
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("all arrays must have one row per example")
        rng = np.random.default_rng(self.seed)
        # One resample's row indices at a time, so memory stays O(n) however many resamples there are.
        replicates = np.array([statistic(*(a[rows] for a in arrays))
                               for rows in (rng.integers(0, n, size=n) for _ in range(self.n_resamples))])
        defined = replicates[~np.isnan(replicates)]
        tail = (1 - self.confidence) / 2
        low, high = np.quantile(defined, [tail, 1 - tail]) if defined.size else (np.nan, np.nan)
        return Interval(estimate=float(statistic(*arrays)), low=float(low), high=float(high),
                        confidence=self.confidence, undefined_resamples=int(replicates.size - defined.size))
