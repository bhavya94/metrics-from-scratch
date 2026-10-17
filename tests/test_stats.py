import numpy as np
import pytest
from scipy import stats

from metrics_from_scratch.stats.bootstrap import Bootstrap
from metrics_from_scratch.stats.hypothesis import welch_t_test

rng = np.random.default_rng(42)
# Minutes per session in an A/B test: the new layout (b) has longer, more spread-out sessions.
SESSIONS_A = rng.normal(12.0, 4.0, 300)
SESSIONS_B = rng.normal(13.0, 6.0, 250)


def test_welch_matches_scipy():
    ours = welch_t_test(SESSIONS_A, SESSIONS_B)
    theirs = stats.ttest_ind(SESSIONS_A, SESSIONS_B, equal_var=False)
    assert ours.statistic == pytest.approx(theirs.statistic, abs=1e-12)
    assert ours.p_value == pytest.approx(theirs.pvalue, abs=1e-12)
    assert ours.degrees_of_freedom == pytest.approx(theirs.df, abs=1e-9)
    assert ours.mean_difference == pytest.approx(SESSIONS_A.mean() - SESSIONS_B.mean())


def test_welch_rejects_groups_it_cannot_test():
    with pytest.raises(ValueError, match="at least 2"):
        welch_t_test([12.0], [11.0, 14.0])
    with pytest.raises(ValueError, match="constant"):
        welch_t_test([12.0, 12.0, 12.0], [14.0, 14.0])


def test_bootstrap_mean_interval_is_close_to_scipy():
    ours = Bootstrap(n_resamples=5_000, seed=1).interval(np.mean, SESSIONS_A)
    theirs = stats.bootstrap((SESSIONS_A,), np.mean, n_resamples=5_000, method="percentile",
                             rng=np.random.default_rng(1)).confidence_interval
    assert ours.estimate == pytest.approx(SESSIONS_A.mean())
    assert ours.low == pytest.approx(theirs.low, abs=0.05) and ours.high == pytest.approx(theirs.high, abs=0.05)


def test_95_percent_intervals_cover_the_true_mean_about_95_percent_of_the_time():
    bootstrap = Bootstrap(n_resamples=400, seed=0)
    covered = [not bootstrap.interval(np.mean, np.random.default_rng(trial).normal(12.0, 4.0, 80)).excludes(12.0)
               for trial in range(300)]
    assert 0.90 <= np.mean(covered) <= 0.98


def test_arrays_are_resampled_with_the_same_rows():
    km = np.arange(1.0, 51.0)
    minutes = 10 + 3 * km
    always_paired = Bootstrap(n_resamples=200).interval(lambda x, y: float(np.all(y == 10 + 3 * x)), km, minutes)
    assert always_paired.low == always_paired.high == 1.0


def test_same_seed_same_interval_and_bad_inputs_rejected():
    assert Bootstrap(seed=3).interval(np.median, SESSIONS_B) == Bootstrap(seed=3).interval(np.median, SESSIONS_B)
    with pytest.raises(ValueError, match="one row per example"):
        Bootstrap().interval(lambda a, b: 0.0, SESSIONS_A, SESSIONS_B)
    with pytest.raises(ValueError, match="confidence"):
        Bootstrap(confidence=95)


def test_undefined_resamples_are_dropped_and_counted():
    # The median of positive values is undefined (nan) when a resample has none: rare with 3 of 20 positive.
    minutes = np.array([-2.0] * 17 + [4.0, 6.0, 9.0])
    interval = Bootstrap(n_resamples=500, seed=0).interval(
        lambda x: float(np.median(x[x > 0])) if (x > 0).any() else float("nan"), minutes)
    assert 0 < interval.undefined_resamples < 100
    assert not np.isnan(interval.low) and not np.isnan(interval.high)
