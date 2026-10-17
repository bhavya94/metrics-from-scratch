import numpy as np
import pytest
from sklearn import metrics as sk

from metrics_from_scratch.classification.curves import ThresholdSweep
from metrics_from_scratch.data.synthetic import ImbalancedBinary

# Six support tickets scored for "urgent" (1); two tickets tie at 0.8, one urgent and one not.
TICKETS_TRUE = [0, 1, 1, 0, 1, 0]
TICKETS_SCORE = [0.1, 0.9, 0.8, 0.8, 0.4, 0.2]


def test_tied_scores_enter_the_positive_set_together():
    sweep = ThresholdSweep.from_scores(TICKETS_TRUE, TICKETS_SCORE)
    assert sweep.thresholds.tolist() == [0.9, 0.8, 0.4, 0.2, 0.1]
    assert sweep.true_positives.tolist() == [1, 2, 3, 3, 3]
    assert sweep.false_positives.tolist() == [0, 1, 1, 2, 3]


def test_roc_auc_is_the_chance_a_positive_outscores_a_negative():
    y, s = np.array(TICKETS_TRUE), np.array(TICKETS_SCORE)
    pairs = [(p, n) for p in s[y == 1] for n in s[y == 0]]
    by_counting = np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs])
    assert ThresholdSweep.from_scores(y, s).roc_auc() == pytest.approx(by_counting)


@pytest.mark.parametrize("positive_rate", [0.5, 0.05, 0.01])
def test_matches_sklearn_on_imbalanced_data(positive_rate):
    data = ImbalancedBinary(n=5_000, positive_rate=positive_rate, seed=11).sample()
    sweep = ThresholdSweep.from_scores(data.y_true, data.scores)

    fpr, tpr = sweep.roc_curve()
    sk_fpr, sk_tpr, _ = sk.roc_curve(data.y_true, data.scores, drop_intermediate=False)
    assert np.allclose(fpr, sk_fpr, atol=1e-12) and np.allclose(tpr, sk_tpr, atol=1e-12)
    assert sweep.roc_auc() == pytest.approx(sk.roc_auc_score(data.y_true, data.scores), abs=1e-12)

    recall, precision = sweep.pr_curve()
    sk_precision, sk_recall, _ = sk.precision_recall_curve(data.y_true, data.scores)
    # sklearn lists thresholds lowest first and appends a (recall 0, precision 1) end point.
    assert np.allclose(recall, sk_recall[-2::-1], atol=1e-12)
    assert np.allclose(precision, sk_precision[-2::-1], atol=1e-12)
    assert sweep.average_precision() == pytest.approx(sk.average_precision_score(data.y_true, data.scores),
                                                      abs=1e-12)


def test_roc_auc_ignores_imbalance_but_average_precision_does_not():
    balanced = ImbalancedBinary(n=50_000, positive_rate=0.5, seed=2).sample()
    rare = ImbalancedBinary(n=50_000, positive_rate=0.01, seed=2).sample()
    sweeps = [ThresholdSweep.from_scores(d.y_true, d.scores) for d in (balanced, rare)]
    assert sweeps[0].roc_auc() == pytest.approx(sweeps[1].roc_auc(), abs=0.02)
    assert sweeps[1].average_precision() < sweeps[0].average_precision() - 0.4


def test_a_perfect_ranking_scores_one():
    sweep = ThresholdSweep.from_scores([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.95])
    assert sweep.roc_auc() == 1.0 and sweep.average_precision() == 1.0


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="both classes"):
        ThresholdSweep.from_scores([1, 1, 1], [0.2, 0.5, 0.9])
    with pytest.raises(ValueError, match="0 or 1"):
        ThresholdSweep.from_scores([1, 2, 2, 1], [0.2, 0.5, 0.9, 0.4])
    # Log-probabilities where the model assigned probability 0.
    with pytest.raises(ValueError, match="finite"):
        ThresholdSweep.from_scores([1, 0, 0, 1], [-np.inf, -np.inf, -0.7, -0.1])
