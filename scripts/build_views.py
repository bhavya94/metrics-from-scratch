"""Build docs/views/index.html: one chart per metric, from the same seeded runs as the README.

Run `uv run python scripts/build_views.py`, then open the page in a browser. The page is self-contained: this script
puts the chart data into `docs/views/template.html` as JSON, and the page draws the charts from it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from metrics_from_scratch.agreement.kappa import cohen_kappa
from metrics_from_scratch.classification.confusion import ConfusionMatrix
from metrics_from_scratch.classification.curves import ThresholdSweep
from metrics_from_scratch.data.synthetic import ImbalancedBinary, JudgedAnswers, NoisyLinear
from metrics_from_scratch.regression.metrics import mae, mse
from metrics_from_scratch.stats.bootstrap import Bootstrap
from metrics_from_scratch.stats.power import PowerPlan

VIEWS = Path(__file__).resolve().parents[1] / "docs" / "views"
PLACEHOLDER = "/*VIEW_DATA*/null"


def _round(values, digits=4) -> list[float]:
    return [round(float(v), digits) for v in values]


def _thin(x: np.ndarray, y: np.ndarray, points: int = 200) -> list[list[float]]:
    """Keep about `points` points of a long curve: evenly spaced by index and by x (x must not decrease).

    Spacing by x as well keeps the shape where the curve moves fast in x over few points, like a precision-recall
    curve with 0.1% positives, whose first rise in recall happens within its first few hundred thresholds.
    """
    by_index = np.linspace(0, len(x) - 1, points).astype(int)
    by_x = np.searchsorted(x, np.linspace(x[0], x[-1], points)).clip(max=len(x) - 1)
    keep = np.unique(np.r_[by_index, by_x, len(x) - 1])
    return [_round(x[keep]), _round(y[keep])]


def confusion_view() -> dict:
    """`mfs classify --positive-rate 0.01` at every threshold from 0.05 to 0.95."""
    data = ImbalancedBinary(positive_rate=0.01).sample()
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        matrix = ConfusionMatrix.from_labels(data.y_true, data.predict(threshold), labels=(0, 1))
        (tn, fp), (fn, tp) = matrix.counts.tolist()
        rows.append({"threshold": float(threshold), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                     "accuracy": matrix.accuracy(), "precision": matrix.precision(), "recall": matrix.recall(),
                     "f1": matrix.f1()})
    return {"n": len(data.y_true), "positive_share": float(data.y_true.mean()), "rows": rows}


def curves_view() -> dict:
    """`mfs curves`: ROC and precision-recall curves as positives get rarer."""
    series = []
    for rate in [0.5, 0.1, 0.01, 0.001]:
        data = ImbalancedBinary(n=100_000, positive_rate=rate).sample()
        sweep = ThresholdSweep.from_scores(data.y_true, data.scores)
        fpr, tpr = sweep.roc_curve()
        recall, precision = sweep.pr_curve()
        series.append({"rate": rate, "roc_auc": sweep.roc_auc(), "average_precision": sweep.average_precision(),
                       "positive_share": float(data.y_true.mean()), "roc": _thin(fpr, tpr),
                       "pr": _thin(recall, precision)})
    return {"series": series}


def regression_view() -> dict:
    """`mfs regress`: the true line's residuals, and which of them MSE and MAE are made of, with and without delays."""
    edges = np.arange(-16, 77, 4)
    runs = []
    for rate in [0.0, 0.02]:
        recipe = NoisyLinear(outlier_rate=rate)
        data = recipe.sample()
        predicted = recipe.line(data.x)
        residuals = data.y - predicted
        # Per bin: how many deliveries, and how much of the squared and the absolute error they carry.
        counts, squared, absolute = (np.histogram(residuals, bins=edges, weights=w)[0]
                                     for w in (None, residuals**2, np.abs(residuals)))
        runs.append({"outlier_rate": rate, "mse": mse(data.y, predicted), "mae": mae(data.y, predicted),
                     "counts": counts.astype(int).tolist(), "squared_share": _round(squared / squared.sum()),
                     "absolute_share": _round(absolute / absolute.sum())})
    return {"edges": edges.tolist(), "runs": runs}


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ConfusionMatrix.from_labels(y_true, y_pred, labels=(0, 1)).f1()


def bootstrap_view() -> dict:
    """`mfs compare`: separate F1 intervals for two models, and the paired interval of their gap."""
    a = ImbalancedBinary(n=2_000, positive_rate=0.1, separation=1.5).sample()
    b = ImbalancedBinary(n=2_000, positive_rate=0.1, separation=1.7).sample()
    bootstrap = Bootstrap(n_resamples=1_000)
    rows = [("Model A F1", bootstrap.interval(_f1, a.y_true, a.predict())),
            ("Model B F1", bootstrap.interval(_f1, a.y_true, b.predict())),
            ("B − A, paired", bootstrap.interval(lambda y, pa, pb: _f1(y, pb) - _f1(y, pa),
                                                 a.y_true, a.predict(), b.predict()))]
    return {"rows": [{"label": label, "estimate": ci.estimate, "low": ci.low, "high": ci.high}
                     for label, ci in rows]}


def power_view() -> dict:
    """`mfs power`: the chance of detecting a 1-minute lift (sd 4 vs 6) at each sample size per arm."""
    plan = PowerPlan()
    sizes = list(range(20, 1001, 20))
    return {"sizes": sizes, "power": _round(plan.power_for_means(n, 4.0, 6.0, 1.0) for n in sizes),
            "needed": plan.n_for_means(4.0, 6.0, 1.0), "at_200": plan.power_for_means(200, 4.0, 6.0, 1.0)}


def kappa_view() -> dict:
    """`mfs kappa`: raw agreement against Cohen's kappa for three judges on the same human labels."""
    judges = {"Always pass": (1.0, 0.0), "Random 80/20": (0.8, 0.2), "Careful": (0.95, 0.85)}
    bootstrap = Bootstrap(n_resamples=1_000)
    rows = []
    for name, (tpr, tnr) in judges.items():
        ratings = JudgedAnswers(tpr=tpr, tnr=tnr).sample()
        matrix = ConfusionMatrix.from_labels(ratings.human, ratings.judge, labels=("fail", "pass"))
        ci = bootstrap.interval(cohen_kappa, ratings.human, ratings.judge)
        rows.append({"judge": name, "agreement": matrix.accuracy(), "tpr": matrix.recall("pass"),
                     "tnr": matrix.recall("fail"), "kappa": ci.estimate, "low": ci.low, "high": ci.high})
    return {"human_pass": float((ratings.human == "pass").mean()), "rows": rows}


def build() -> dict:
    return {"confusion": confusion_view(), "curves": curves_view(), "regression": regression_view(),
            "bootstrap": bootstrap_view(), "power": power_view(), "kappa": kappa_view()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=VIEWS / "index.html")
    args = parser.parse_args()
    template = (VIEWS / "template.html").read_text()
    if PLACEHOLDER not in template:
        raise SystemExit(f"{PLACEHOLDER} is missing from the template")
    args.out.write_text(template.replace(PLACEHOLDER, json.dumps(build())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
