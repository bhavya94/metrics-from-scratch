"""Error metrics for real-valued predictions."""

from typing import Sequence

import numpy as np


def _residuals(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred differ in shape: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("no examples")
    return y_true - y_pred


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean squared error. Squaring makes one large miss cost more than many small ones."""
    return float(np.mean(_residuals(y_true, y_pred) ** 2))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error: MSE back in the units of y."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error. Every unit of error costs the same, so outliers pull it less than MSE."""
    return float(np.mean(np.abs(_residuals(y_true, y_pred))))


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination: 1 - SS_residual / SS_total.

    1 is a perfect fit, 0 is no better than always predicting the mean of y_true, and below 0 is worse than that.
    Constant y_true has no variance to explain; like sklearn, return 1.0 for a perfect fit and 0.0 otherwise.
    With a single example R2 is undefined, so return nan, also like sklearn.
    """
    residuals = _residuals(y_true, y_pred)
    if residuals.size < 2:
        return float("nan")
    y_true = np.asarray(y_true, dtype=float)
    ss_residual = float(np.sum(residuals**2))
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else 0.0
    return 1.0 - ss_residual / ss_total
