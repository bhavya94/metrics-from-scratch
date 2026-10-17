import numpy as np
import pytest

from metrics_from_scratch.stats.hypothesis import welch_t_test
from metrics_from_scratch.stats.power import PowerPlan


def test_the_sixteen_over_effect_squared_rule_of_thumb():
    # alpha 0.05, power 0.8, equal unit variances: n = 2 * (1.96 + 0.84)^2 / d^2, about 16 / d^2.
    assert PowerPlan().n_for_means(sd_a=1.0, sd_b=1.0, mde=1.0) == 16
    assert PowerPlan().n_for_means(sd_a=1.0, sd_b=1.0, mde=0.5) == 63


def test_halving_the_effect_roughly_quadruples_the_sample():
    plan = PowerPlan()
    assert plan.n_for_means(4.0, 6.0, 0.5) / plan.n_for_means(4.0, 6.0, 1.0) == pytest.approx(4, rel=0.01)


def test_power_at_the_planned_sample_is_the_target():
    plan = PowerPlan(power=0.9)
    n = plan.n_for_means(4.0, 6.0, 1.0)
    assert plan.power_for_means(n, 4.0, 6.0, 1.0) == pytest.approx(0.9, abs=0.002)


def test_simulated_tests_at_the_planned_sample_reject_about_80_percent_of_the_time():
    plan = PowerPlan()
    n = plan.n_for_means(sd_a=4.0, sd_b=6.0, mde=1.0)
    rng = np.random.default_rng(0)
    rejected = [welch_t_test(rng.normal(13.0, 6.0, n), rng.normal(12.0, 4.0, n)).p_value < 0.05
                for _ in range(1_000)]
    assert np.mean(rejected) == pytest.approx(0.8, abs=0.04)


def test_simulated_conversion_tests_reach_the_planned_power():
    plan = PowerPlan()
    n = plan.n_for_proportions(baseline=0.10, mde=0.02)
    rng = np.random.default_rng(1)
    rejected = [welch_t_test(rng.random(n) < 0.12, rng.random(n) < 0.10).p_value < 0.05 for _ in range(500)]
    assert np.mean(rejected) == pytest.approx(0.8, abs=0.05)


def test_invalid_plans_are_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        PowerPlan().n_for_means(4.0, 6.0, 0.0)
    with pytest.raises(ValueError, match="rates"):
        PowerPlan().n_for_proportions(baseline=0.99, mde=0.02)
    with pytest.raises(ValueError, match="alpha and power"):
        PowerPlan(power=80)
