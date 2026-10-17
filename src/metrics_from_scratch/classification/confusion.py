"""The confusion matrix, and the point metrics computed from it."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts of (true label, predicted label) pairs.

    `counts[i, j]` is the number of examples whose true label is `labels[i]` and whose prediction is `labels[j]`.
    Rows are the truth, columns are the prediction, the same layout as sklearn.
    A metric whose denominator is zero (no predicted positives for precision) returns 0.0, like sklearn's
    `zero_division=0`. Unlike sklearn, a label in the data but missing from `labels` is an error, not ignored.
    """

    labels: tuple
    counts: np.ndarray

    @classmethod
    def from_labels(cls, y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> "ConfusionMatrix":
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true and y_pred differ in shape: {y_true.shape} vs {y_pred.shape}")
        if y_true.size == 0:
            raise ValueError("no examples")
        labels = tuple(np.unique(np.concatenate([y_true, y_pred])).tolist()) if labels is None else tuple(labels)
        index = {label: i for i, label in enumerate(labels)}
        unknown = (set(y_true.tolist()) | set(y_pred.tolist())) - set(index)
        if unknown:
            raise ValueError(f"labels {sorted(map(str, unknown))} are not in {labels}")
        counts = np.zeros((len(labels), len(labels)), dtype=int)
        # One increment per example at (row of its true label, column of its predicted label).
        np.add.at(counts, ([index[t] for t in y_true.tolist()], [index[p] for p in y_pred.tolist()]), 1)
        return cls(labels=labels, counts=counts)

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def accuracy(self) -> float:
        return _ratio(np.trace(self.counts), self.total)

    def _index(self, label) -> int:
        if label not in self.labels:
            raise ValueError(f"label {label!r} is not in {self.labels}")
        return self.labels.index(label)

    def true_positives(self, label) -> int:
        i = self._index(label)
        return int(self.counts[i, i])

    def precision(self, label=1) -> float:
        """Of the examples predicted as `label`, the share that truly are `label`: TP / (TP + FP)."""
        return _ratio(self.true_positives(label), self.counts[:, self._index(label)].sum())

    def recall(self, label=1) -> float:
        """Of the examples that truly are `label`, the share predicted as `label`: TP / (TP + FN)."""
        return _ratio(self.true_positives(label), self.counts[self._index(label), :].sum())

    def f1(self, label=1) -> float:
        """Harmonic mean of precision and recall, written as 2TP / (2TP + FP + FN)."""
        i = self._index(label)
        tp = self.counts[i, i]
        return _ratio(2 * tp, self.counts[:, i].sum() + self.counts[i, :].sum())

    def macro_f1(self) -> float:
        """Unweighted mean of per-label F1, so a rare label counts as much as a common one."""
        return float(np.mean([self.f1(label) for label in self.labels]))


def _ratio(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else 0.0
