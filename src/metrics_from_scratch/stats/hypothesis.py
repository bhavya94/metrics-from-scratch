"""Welch's two-sample t-test: is the difference between two group means bigger than noise?"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TTestResult:
    mean_difference: float  # mean(a) - mean(b)
    statistic: float
    degrees_of_freedom: float
    p_value: float  # two-sided


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> TTestResult:
    """Compare two independent groups without assuming equal variances.

    t = (mean_a - mean_b) / sqrt(var_a / n_a + var_b / n_b), with sample variances (ddof = 1).
    Degrees of freedom come from the Welch-Satterthwaite formula. Only the tail probability of the t distribution
    is looked up in scipy, the way a printed t table would be.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least 2 values to estimate its variance")
    var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)  # squared standard error of each mean
    if var_a + var_b == 0:
        raise ValueError("both groups are constant, so the t statistic is undefined")
    difference = a.mean() - b.mean()
    t = difference / np.sqrt(var_a + var_b)
    df = (var_a + var_b) ** 2 / (var_a**2 / (len(a) - 1) + var_b**2 / (len(b) - 1))
    return TTestResult(mean_difference=float(difference), statistic=float(t), degrees_of_freedom=float(df),
                       p_value=float(2 * stats.t.sf(abs(t), df)))
