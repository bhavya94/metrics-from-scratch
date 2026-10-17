"""Threshold-free metrics: the ROC and precision-recall curves, and the areas under them."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ThresholdSweep:
    """Cumulative true and false positives as the decision threshold sweeps from the highest score down.

    Entry k holds the counts when every example scoring at or above `thresholds[k]` is predicted positive.
    Tied scores share one threshold, so they enter the positive set together. Both curves read from this.
    """

    thresholds: np.ndarray  # distinct scores, highest first
    true_positives: np.ndarray
    false_positives: np.ndarray
    positives: int
    negatives: int

    @classmethod
    def from_scores(cls, y_true: Sequence[int], scores: Sequence[float]) -> "ThresholdSweep":
        y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
        if y_true.shape != scores.shape:
            raise ValueError(f"y_true and scores differ in shape: {y_true.shape} vs {scores.shape}")
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("y_true must be 0 or 1, with 1 the positive class")
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite; NaN or infinite scores have no place in the ranking")
        positives = int(y_true.sum())
        if positives in (0, len(y_true)):
            raise ValueError("both classes must be present to sweep a threshold")
        order = np.argsort(-scores, kind="stable")
        sorted_scores, sorted_true = scores[order], y_true[order]
        # The last index of each run of tied scores: where the threshold can stop.
        cuts = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
        true_positives = np.cumsum(sorted_true)[cuts]
        return cls(thresholds=sorted_scores[cuts], true_positives=true_positives,
                   false_positives=cuts + 1 - true_positives, positives=positives,
                   negatives=len(y_true) - positives)

    def roc_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """(false positive rate, true positive rate) per threshold, starting at (0, 0): nothing predicted positive."""
        return (np.r_[0.0, self.false_positives / self.negatives], np.r_[0.0, self.true_positives / self.positives])

    def roc_auc(self) -> float:
        """Area under the ROC curve by the trapezoid rule.

        It equals the probability that a random positive outscores a random negative, counting ties as one half.
        """
        fpr, tpr = self.roc_curve()
        return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

    def pr_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """(recall, precision) per threshold, highest threshold first."""
        predicted_positive = self.true_positives + self.false_positives
        return self.true_positives / self.positives, self.true_positives / predicted_positive

    def average_precision(self) -> float:
        """Precision at each threshold, weighted by the recall gained there: sum of (R_k - R_k-1) * P_k.

        This is a step sum, not a trapezoid. Joining PR points with straight lines overstates the area, because
        precision does not change linearly between thresholds.
        """
        recall, precision = self.pr_curve()
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
