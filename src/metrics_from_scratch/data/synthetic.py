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
