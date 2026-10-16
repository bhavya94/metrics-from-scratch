import numpy as np
import pytest
from sklearn import metrics as sk

from metrics_from_scratch.data.synthetic import NoisyLinear
from metrics_from_scratch.regression.metrics import mae, mse, r2, rmse

# Delivery times in minutes: what happened, and what a model predicted.
ACTUAL = [22.0, 35.5, 18.0, 41.0, 27.5]
PREDICTED = [25.0, 33.0, 18.0, 39.0, 30.5]


def test_hand_computed_values():
    errors = np.array([-3.0, 2.5, 0.0, 2.0, -3.0])
    assert mse(ACTUAL, PREDICTED) == pytest.approx(np.mean(errors**2))
    assert rmse(ACTUAL, PREDICTED) == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert mae(ACTUAL, PREDICTED) == pytest.approx(np.mean(np.abs(errors)))


@pytest.mark.parametrize("outlier_rate", [0.0, 0.02, 0.2])
def test_matches_sklearn(outlier_rate):
    recipe = NoisyLinear(outlier_rate=outlier_rate, seed=5)
    data = recipe.sample()
    predicted = recipe.line(data.x)
    assert mse(data.y, predicted) == pytest.approx(sk.mean_squared_error(data.y, predicted), abs=1e-12)
    assert rmse(data.y, predicted) == pytest.approx(sk.root_mean_squared_error(data.y, predicted), abs=1e-12)
    assert mae(data.y, predicted) == pytest.approx(sk.mean_absolute_error(data.y, predicted), abs=1e-12)
    assert r2(data.y, predicted) == pytest.approx(sk.r2_score(data.y, predicted), abs=1e-12)


def test_r2_is_zero_for_the_mean_and_negative_for_worse():
    data = NoisyLinear(seed=1).sample()
    assert r2(data.y, np.full_like(data.y, data.y.mean())) == pytest.approx(0.0, abs=1e-12)
    assert r2(data.y, np.full_like(data.y, data.y.mean() + 10)) < 0


def test_outliers_move_mse_more_than_mae():
    def errors(outlier_rate):
        recipe = NoisyLinear(outlier_rate=outlier_rate, seed=2)
        data = recipe.sample()
        return mse(data.y, recipe.line(data.x)), mae(data.y, recipe.line(data.x))

    (clean_mse, clean_mae), (delayed_mse, delayed_mae) = errors(0.0), errors(0.02)
    assert delayed_mse / clean_mse > 2 * (delayed_mae / clean_mae)


def test_degenerate_targets_match_sklearn_conventions():
    assert r2([30.0, 30.0, 30.0], [30.0, 30.0, 30.0]) == 1.0
    assert r2([30.0, 30.0, 30.0], [30.0, 30.0, 31.0]) == 0.0
    assert np.isnan(r2([30.0], [28.0]))


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="differ in shape"):
        mse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="no examples"):
        mae([], [])
