"""Agreement between two raters, such as an LLM judge and a human, beyond what chance alone would give."""

from typing import Sequence

import numpy as np

from metrics_from_scratch.classification.confusion import ConfusionMatrix


def cohen_kappa(rater_a: Sequence, rater_b: Sequence, labels: Sequence | None = None) -> float:
    """Cohen's kappa: (observed agreement - chance agreement) / (1 - chance agreement).

    Chance agreement is what two raters would reach by labeling independently at their own label rates:
    the sum over labels of P(a says it) * P(b says it). 1 is perfect agreement, 0 is chance level, below 0 is
    worse than chance. When both raters use one label for everything, chance agreement is 1 and kappa is
    undefined, so this returns nan, like sklearn.
    """
    matrix = ConfusionMatrix.from_labels(rater_a, rater_b, labels)
    shares = matrix.counts / matrix.total
    observed = np.trace(shares)
    chance = float(shares.sum(axis=1) @ shares.sum(axis=0))
    if chance == 1:
        return float("nan")
    return float((observed - chance) / (1 - chance))
