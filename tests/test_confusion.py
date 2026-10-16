import numpy as np
import pytest
from sklearn import metrics as sk

from metrics_from_scratch.classification.confusion import ConfusionMatrix
from metrics_from_scratch.data.synthetic import ImbalancedBinary

# A spam filter on 10 emails (1 = spam): 3 caught, 2 missed, 1 good email flagged, 4 good emails passed.
SPAM_TRUE = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
SPAM_PRED = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]


def test_counts_follow_rows_truth_columns_prediction():
    matrix = ConfusionMatrix.from_labels(SPAM_TRUE, SPAM_PRED)
    assert matrix.labels == (0, 1)
    assert matrix.counts.tolist() == [[4, 1], [2, 3]]  # [[TN, FP], [FN, TP]]


def test_point_metrics_match_hand_computation():
    matrix = ConfusionMatrix.from_labels(SPAM_TRUE, SPAM_PRED)
    assert matrix.accuracy() == pytest.approx(7 / 10)
    assert matrix.precision() == pytest.approx(3 / 4)
    assert matrix.recall() == pytest.approx(3 / 5)
    assert matrix.f1() == pytest.approx(2 * 3 / (2 * 3 + 1 + 2))


@pytest.mark.parametrize("positive_rate", [0.5, 0.05, 0.01])
def test_matches_sklearn_on_imbalanced_data(positive_rate):
    data = ImbalancedBinary(n=5_000, positive_rate=positive_rate, seed=7).sample()
    y_pred = data.predict()
    matrix = ConfusionMatrix.from_labels(data.y_true, y_pred, labels=(0, 1))
    assert np.array_equal(matrix.counts, sk.confusion_matrix(data.y_true, y_pred, labels=[0, 1]))
    assert matrix.accuracy() == pytest.approx(sk.accuracy_score(data.y_true, y_pred), abs=1e-12)
    assert matrix.precision() == pytest.approx(sk.precision_score(data.y_true, y_pred), abs=1e-12)
    assert matrix.recall() == pytest.approx(sk.recall_score(data.y_true, y_pred), abs=1e-12)
    assert matrix.f1() == pytest.approx(sk.f1_score(data.y_true, y_pred), abs=1e-12)
    assert matrix.macro_f1() == pytest.approx(sk.f1_score(data.y_true, y_pred, average="macro"), abs=1e-12)


def test_multiclass_matches_sklearn():
    y_true = ["billing", "billing", "outage", "refund", "refund", "refund", "outage", "billing"]
    y_pred = ["billing", "refund", "outage", "refund", "billing", "refund", "billing", "billing"]
    matrix = ConfusionMatrix.from_labels(y_true, y_pred)
    assert np.array_equal(matrix.counts, sk.confusion_matrix(y_true, y_pred, labels=list(matrix.labels)))
    assert matrix.precision("refund") == pytest.approx(sk.precision_score(y_true, y_pred, labels=["refund"],
                                                                          average="macro"))
    assert matrix.macro_f1() == pytest.approx(sk.f1_score(y_true, y_pred, average="macro"))


def test_no_predicted_positives_gives_zero_precision_not_an_error():
    matrix = ConfusionMatrix.from_labels([1, 0, 0, 0], [0, 0, 0, 0], labels=(0, 1))
    assert matrix.precision() == 0.0
    assert matrix.f1() == 0.0
    assert matrix.accuracy() == pytest.approx(0.75)  # high accuracy, zero recall: the imbalance trap


def test_bad_inputs_are_rejected_with_clear_errors():
    with pytest.raises(ValueError, match="differ in shape"):
        ConfusionMatrix.from_labels([1, 0, 1], [1, 0])
    with pytest.raises(ValueError, match="no examples"):
        ConfusionMatrix.from_labels([], [])
    # An LLM judge that answers "1" as a string next to integer gold labels.
    with pytest.raises(ValueError, match="not in"):
        ConfusionMatrix.from_labels([1, 0, 1], ["1", "0", "1"], labels=(0, 1))
    with pytest.raises(ValueError, match="label 1 is not in"):
        ConfusionMatrix.from_labels(["billing", "refund"], ["billing", "billing"]).precision()
