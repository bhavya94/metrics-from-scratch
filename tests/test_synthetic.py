import numpy as np
import pytest

from metrics_from_scratch.data.synthetic import ImbalancedBinary


def test_same_seed_gives_identical_samples():
    a, b = ImbalancedBinary(seed=3).sample(), ImbalancedBinary(seed=3).sample()
    assert np.array_equal(a.y_true, b.y_true) and np.array_equal(a.scores, b.scores)


def test_positive_rate_and_score_range():
    data = ImbalancedBinary(n=50_000, positive_rate=0.02, seed=1).sample()
    assert data.y_true.mean() == pytest.approx(0.02, abs=0.003)
    assert ((data.scores > 0) & (data.scores < 1)).all()
    assert data.scores[data.y_true == 1].mean() > data.scores[data.y_true == 0].mean()


def test_predict_uses_the_threshold_inclusively():
    data = ImbalancedBinary(n=100, seed=0).sample()
    assert data.predict(threshold=0.0).all()
    assert data.predict(threshold=float(data.scores.max())).sum() == 1


@pytest.mark.parametrize("bad", [{"positive_rate": 0.0}, {"positive_rate": 1.0}, {"n": 0}])
def test_invalid_recipes_are_rejected(bad):
    with pytest.raises(ValueError):
        ImbalancedBinary(**bad)
