import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from metrics_from_scratch.agreement.kappa import cohen_kappa
from metrics_from_scratch.data.synthetic import JudgedAnswers
from metrics_from_scratch.stats.bootstrap import Bootstrap

# A human and an LLM judge label 10 chatbot answers as pass or fail.
HUMAN = ["pass", "pass", "fail", "pass", "fail", "pass", "pass", "fail", "pass", "pass"]
JUDGE = ["pass", "pass", "pass", "pass", "fail", "pass", "fail", "fail", "pass", "pass"]


def test_hand_computed_kappa():
    # Observed agreement 8/10. Human says pass 7/10, judge 7/10: chance = 0.7 * 0.7 + 0.3 * 0.3 = 0.58.
    assert cohen_kappa(HUMAN, JUDGE) == pytest.approx((0.8 - 0.58) / (1 - 0.58))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn_with_three_labels(seed):
    rng = np.random.default_rng(seed)
    human = rng.choice(["correct", "partly correct", "wrong"], size=400, p=[0.6, 0.25, 0.15])
    judge = np.where(rng.random(400) < 0.7, human, rng.choice(["correct", "partly correct", "wrong"], size=400))
    assert cohen_kappa(human, judge) == pytest.approx(cohen_kappa_score(human, judge), abs=1e-12)


def test_a_judge_that_always_says_pass_agrees_often_but_scores_zero():
    human = ["pass"] * 90 + ["fail"] * 10
    always_pass = ["pass"] * 100
    assert np.mean(np.array(human) == np.array(always_pass)) == 0.9
    assert cohen_kappa(human, always_pass) == pytest.approx(0.0)


def test_kappa_is_symmetric_and_can_be_negative():
    assert cohen_kappa(HUMAN, JUDGE) == pytest.approx(cohen_kappa(JUDGE, HUMAN))
    flipped = ["fail" if label == "pass" else "pass" for label in HUMAN]
    assert cohen_kappa(HUMAN, flipped) < 0


def test_one_label_for_everything_is_undefined():
    assert math.isnan(cohen_kappa(["pass"] * 5, ["pass"] * 5))


def test_judge_recipe_hits_its_true_and_false_rates():
    ratings = JudgedAnswers(n=20_000, pass_rate=0.8, tpr=0.95, tnr=0.85, seed=0).sample()
    human_pass, judge_pass = ratings.human == "pass", ratings.judge == "pass"
    assert judge_pass[human_pass].mean() == pytest.approx(0.95, abs=0.01)
    assert (~judge_pass[~human_pass]).mean() == pytest.approx(0.85, abs=0.02)


def test_kappa_interval_on_a_small_skewed_set_is_not_nan():
    # 30 answers, 95% pass: some resamples contain no human fail, where kappa is undefined.
    ratings = JudgedAnswers(n=30, pass_rate=0.95, seed=0).sample()
    interval = Bootstrap(n_resamples=500, seed=0).interval(cohen_kappa, ratings.human, ratings.judge)
    assert interval.undefined_resamples > 0
    assert not math.isnan(interval.low) and not math.isnan(interval.high)
