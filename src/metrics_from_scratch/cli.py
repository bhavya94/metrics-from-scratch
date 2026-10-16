"""The `mfs` command: run the metrics on synthetic data and print the results."""

import argparse

from metrics_from_scratch.classification.confusion import ConfusionMatrix
from metrics_from_scratch.classification.curves import ThresholdSweep
from metrics_from_scratch.data.synthetic import ImbalancedBinary


def _classify(args: argparse.Namespace) -> None:
    data = ImbalancedBinary(n=args.n, positive_rate=args.positive_rate, separation=args.separation,
                            seed=args.seed).sample()
    matrix = ConfusionMatrix.from_labels(data.y_true, data.predict(args.threshold), labels=(0, 1))
    (tn, fp), (fn, tp) = matrix.counts.tolist()
    print(f"{args.n} examples, {data.y_true.mean():.1%} positive, threshold {args.threshold}")
    print(f"confusion matrix (rows = truth, columns = prediction): TN={tn} FP={fp} FN={fn} TP={tp}")
    for name, value in [("accuracy", matrix.accuracy()), ("precision", matrix.precision()),
                        ("recall", matrix.recall()), ("f1", matrix.f1()), ("macro_f1", matrix.macro_f1())]:
        print(f"  {name:<10} {value:.4f}")


def _curves(args: argparse.Namespace) -> None:
    print(f"{args.n} examples per row, separation {args.separation}")
    print(f"  {'positive rate':<14} {'ROC-AUC':>8} {'avg precision':>14} {'random AP':>10}")
    for rate in args.positive_rates:
        data = ImbalancedBinary(n=args.n, positive_rate=rate, separation=args.separation, seed=args.seed).sample()
        sweep = ThresholdSweep.from_scores(data.y_true, data.scores)
        # A random ranking has average precision equal to the share of positives.
        print(f"  {rate:<14} {sweep.roc_auc():>8.4f} {sweep.average_precision():>14.4f} {data.y_true.mean():>10.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mfs", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    classify = commands.add_parser("classify", help="point metrics from a confusion matrix")
    classify.add_argument("--n", type=int, default=10_000, help="number of examples")
    classify.add_argument("--positive-rate", type=float, default=0.05, help="share of true positives")
    classify.add_argument("--separation", type=float, default=1.5, help="distance between class means")
    classify.add_argument("--threshold", type=float, default=0.5, help="score at or above which we predict 1")
    classify.add_argument("--seed", type=int, default=0)
    classify.set_defaults(run=_classify)

    curves = commands.add_parser("curves", help="ROC-AUC and average precision as positives get rarer")
    curves.add_argument("--positive-rates", type=float, nargs="+", default=[0.5, 0.1, 0.01, 0.001])
    curves.add_argument("--n", type=int, default=100_000, help="number of examples per positive rate")
    curves.add_argument("--separation", type=float, default=1.5, help="distance between class means")
    curves.add_argument("--seed", type=int, default=0)
    curves.set_defaults(run=_curves)

    args = parser.parse_args(argv)
    args.run(args)
