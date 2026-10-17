"""The `mfs` command: run the metrics on synthetic data and print the results."""

import argparse

import numpy as np

from metrics_from_scratch.classification.confusion import ConfusionMatrix
from metrics_from_scratch.classification.curves import ThresholdSweep
from metrics_from_scratch.data.synthetic import ImbalancedBinary, NoisyLinear, TwoArms
from metrics_from_scratch.regression.metrics import mae, mse, r2, rmse
from metrics_from_scratch.stats.bootstrap import Bootstrap
from metrics_from_scratch.stats.hypothesis import welch_t_test
from metrics_from_scratch.stats.power import PowerPlan


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


def _regress(args: argparse.Namespace) -> None:
    print(f"{args.n} delivery times (minutes); the true line is 10 + 3 * km")
    print(f"  {'delayed':<8} {'prediction':<15} {'MSE':>8} {'RMSE':>6} {'MAE':>6} {'R2':>7}")
    for rate in args.outlier_rates:
        recipe = NoisyLinear(n=args.n, outlier_rate=rate, seed=args.seed)
        data = recipe.sample()
        for name, predicted in [("true line", recipe.line(data.x)), ("mean of y", np.full_like(data.y, data.y.mean())),
                                ("mean + 10 min", np.full_like(data.y, data.y.mean() + 10))]:
            print(f"  {rate:<8} {name:<15} {mse(data.y, predicted):>8.1f} {rmse(data.y, predicted):>6.2f} "
                  f"{mae(data.y, predicted):>6.2f} {r2(data.y, predicted):>7.3f}")


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ConfusionMatrix.from_labels(y_true, y_pred, labels=(0, 1)).f1()


def _compare(args: argparse.Namespace) -> None:
    # The same seed gives both models the same labels and noise; model B just separates the classes better.
    a = ImbalancedBinary(n=args.n, positive_rate=args.positive_rate, separation=1.5, seed=args.seed).sample()
    b = ImbalancedBinary(n=args.n, positive_rate=args.positive_rate, separation=args.separation_b,
                         seed=args.seed).sample()
    bootstrap = Bootstrap(n_resamples=args.resamples, seed=args.seed)
    print(f"one test set of {args.n} examples, {a.y_true.mean():.1%} positive; 95% percentile bootstrap CIs")
    rows = [("model A F1", bootstrap.interval(_f1, a.y_true, a.predict())),
            ("model B F1", bootstrap.interval(_f1, a.y_true, b.predict())),
            ("B - A, paired", bootstrap.interval(lambda y, pa, pb: _f1(y, pb) - _f1(y, pa),
                                                 a.y_true, a.predict(), b.predict()))]
    for name, interval in rows:
        print(f"  {name:<14} {interval.estimate:.4f}  [{interval.low:.4f}, {interval.high:.4f}]")
    (_, ci_a), (_, ci_b), (_, gap) = rows
    print(f"  separate CIs overlap: {ci_a.high >= ci_b.low and ci_b.high >= ci_a.low}; "
          f"paired CI excludes 0: {gap.excludes(0.0)}")


def _t_test(args: argparse.Namespace) -> None:
    arms = TwoArms(n=args.n, lift=args.lift, seed=args.seed).sample()
    result = welch_t_test(arms.b, arms.a)
    print(f"minutes per session, {args.n} users per arm: A ~ N(12, 4), B ~ N({12 + args.lift}, 6)")
    print(f"  B - A = {result.mean_difference:.3f} min, t = {result.statistic:.3f}, "
          f"df = {result.degrees_of_freedom:.1f}, p = {result.p_value:.4f}")


def _power(args: argparse.Namespace) -> None:
    plan = PowerPlan(alpha=args.alpha, power=args.power)
    rows = [("session minutes, sd 4 vs 6, lift 1 min", plan.n_for_means(4.0, 6.0, 1.0)),
            ("  power with only 200 per arm", f"{plan.power_for_means(200, 4.0, 6.0, 1.0):.0%}"),
            ("conversion 10% -> 12%", plan.n_for_proportions(0.10, 0.02)),
            ("conversion 10% -> 11%", plan.n_for_proportions(0.10, 0.01))]
    print(f"two-sided test, alpha {args.alpha}, power {args.power}; users per arm")
    for label, value in rows:
        print(f"  {label:<40} {value:>6}")


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

    regress = commands.add_parser("regress", help="MSE, MAE and R2 with and without outliers")
    regress.add_argument("--outlier-rates", type=float, nargs="+", default=[0.0, 0.02],
                         help="share of orders delayed by an hour")
    regress.add_argument("--n", type=int, default=2_000, help="number of deliveries")
    regress.add_argument("--seed", type=int, default=0)
    regress.set_defaults(run=_regress)

    compare = commands.add_parser("compare", help="paired bootstrap CI for the F1 gap between two models")
    compare.add_argument("--n", type=int, default=2_000, help="test set size")
    compare.add_argument("--positive-rate", type=float, default=0.1, help="share of true positives")
    compare.add_argument("--separation-b", type=float, default=1.7, help="model B's class separation (A has 1.5)")
    compare.add_argument("--resamples", type=int, default=1_000, help="bootstrap resamples")
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(run=_compare)

    t_test = commands.add_parser("t-test", help="Welch's t-test on a simulated A/B test")
    t_test.add_argument("--n", type=int, default=200, help="users per arm")
    t_test.add_argument("--lift", type=float, default=1.0, help="true extra minutes per session in arm B")
    t_test.add_argument("--seed", type=int, default=0)
    t_test.set_defaults(run=_t_test)

    power = commands.add_parser("power", help="sample size per arm for an A/B test")
    power.add_argument("--alpha", type=float, default=0.05, help="false positive rate (two-sided)")
    power.add_argument("--power", type=float, default=0.8, help="chance of detecting a real effect")
    power.set_defaults(run=_power)

    args = parser.parse_args(argv)
    args.run(args)
