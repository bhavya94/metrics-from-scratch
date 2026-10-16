"""Synthetic scored data for a binary classifier, where positives can be rare."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoredLabels:
    """True labels and a classifier's scores for the same examples."""

    y_true: np.ndarray  # 0 or 1 per example
    scores: np.ndarray  # higher means "more likely positive", in (0, 1)

    def predict(self, threshold: float = 0.5) -> np.ndarray:
        """Hard 0/1 predictions: positive when the score is at or above the threshold."""
        return (self.scores >= threshold).astype(int)


@dataclass(frozen=True)
class ImbalancedBinary:
    """A recipe for a binary task with a chosen positive rate and class separation.

    Each example gets a latent value: positives from N(separation, 1), negatives from N(0, 1).
    The score is the sigmoid of (latent - separation / 2), so 0.5 sits halfway between the classes.
    """

    n: int = 10_000
    positive_rate: float = 0.05
    separation: float = 1.5
    seed: int = 0

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")
        if not 0 < self.positive_rate < 1:
            raise ValueError(f"positive_rate must be in (0, 1), got {self.positive_rate}")

    def sample(self) -> ScoredLabels:
        rng = np.random.default_rng(self.seed)
        y_true = (rng.random(self.n) < self.positive_rate).astype(int)
        latent = rng.normal(loc=y_true * self.separation, scale=1.0)
        scores = 1.0 / (1.0 + np.exp(-(latent - self.separation / 2)))
        return ScoredLabels(y_true=y_true, scores=scores)


@dataclass(frozen=True)
class Observations:
    """One input and one real-valued target per example."""

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class NoisyLinear:
    """A recipe for y = intercept + slope * x + Gaussian noise, with a share of large one-sided outliers.

    The defaults read as delivery times: 10 minutes plus 3 per km, over 1-15 km. Set `outlier_rate=0.02` to delay
    2% of orders by an hour.
    """

    n: int = 2_000
    intercept: float = 10.0
    slope: float = 3.0
    noise: float = 4.0
    outlier_rate: float = 0.0
    outlier_size: float = 60.0
    seed: int = 0

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")
        if not 0 <= self.outlier_rate < 1:
            raise ValueError(f"outlier_rate must be in [0, 1), got {self.outlier_rate}")

    def line(self, x: np.ndarray) -> np.ndarray:
        """The noise-free value of y at x: the best prediction anyone can make."""
        return self.intercept + self.slope * x

    def sample(self) -> Observations:
        rng = np.random.default_rng(self.seed)
        x = rng.uniform(1.0, 15.0, self.n)
        delayed = rng.random(self.n) < self.outlier_rate
        y = self.line(x) + rng.normal(0.0, self.noise, self.n) + delayed * self.outlier_size
        return Observations(x=x, y=y)
