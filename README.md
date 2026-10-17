# metrics-from-scratch

Evaluation metrics and statistics, implemented from scratch in numpy and checked against sklearn and scipy.

The point is to know exactly what each number means and when it misleads. For example, on data where 1% of
examples are positive, a classifier can score 99% accuracy while it finds fewer than 1 in 10 positives:

```console
$ uv run mfs classify --positive-rate 0.01 --threshold 0.9
10000 examples, 0.9% positive, threshold 0.9
confusion matrix (rows = truth, columns = prediction): TN=9896 FP=15 FN=82 TP=7
  accuracy   0.9903
  precision  0.3182
  recall     0.0787
  f1         0.1261
  macro_f1   0.5606
```

ROC-AUC has the opposite blind spot. It stays the same as positives get rarer, while average precision
(the area under the precision-recall curve) falls toward the positive rate:

```console
$ uv run mfs curves
100000 examples per row, separation 1.5
  positive rate   ROC-AUC  avg precision  random AP
  0.5              0.8540         0.8524     0.5010
  0.1              0.8547         0.4792     0.1018
  0.01             0.8529         0.1112     0.0099
  0.001            0.8482         0.0383     0.0009
```

For regression, an hour's delay on 2% of deliveries multiplies MSE by 5 but moves MAE by about one
minute. R2 is 0 for always predicting the mean, and negative for anything worse:

```console
$ uv run mfs regress
2000 delivery times (minutes); the true line is 10 + 3 * km
  delayed  prediction           MSE   RMSE    MAE      R2
  0.0      true line           15.7   3.97   3.18   0.904
  0.0      mean of y          164.5  12.83  10.93   0.000
  0.0      mean + 10 min      264.5  16.26  13.25  -0.608
  0.02     true line           79.6   8.92   4.19   0.637
  0.02     mean of y          219.6  14.82  11.75   0.000
  0.02     mean + 10 min      319.6  17.88  14.30  -0.455
```

A metric needs an interval before a difference means anything. Two models scored on the same test set have
overlapping confidence intervals, yet a paired bootstrap (resampling the same rows for both) shows B is better:

```console
$ uv run mfs compare
one test set of 2000 examples, 10.5% positive; 95% percentile bootstrap CIs
  model A F1     0.4026  [0.3585, 0.4450]
  model B F1     0.4420  [0.3964, 0.4855]
  B - A, paired  0.0394  [0.0277, 0.0533]
  separate CIs overlap: True; paired CI excludes 0: True
```

And a real effect can hide in noise when the sample is small. Arm B truly adds 1 minute per session, but with
200 users per arm Welch's t-test can't tell:

```console
$ uv run mfs t-test
minutes per session, 200 users per arm: A ~ N(12, 4), B ~ N(13.0, 6)
  B - A = 0.408 min, t = 0.793, df = 333.9, p = 0.4282
```

A power analysis says so before the test runs. That design had a 50% chance of detecting the lift; it needed
409 users per arm. Conversion tests need far more, and halving the effect quadruples the sample:

```console
$ uv run mfs power
two-sided test, alpha 0.05, power 0.8; users per arm
  session minutes, sd 4 vs 6, lift 1 min      409
    power with only 200 per arm               50%
  conversion 10% -> 12%                      3841
  conversion 10% -> 11%                     14751
```

## What is here
- **Synthetic data** (`data/`): a binary task with a chosen positive rate and class separation, seeded.
- **Classification metrics** (`classification/`): a confusion matrix, and accuracy, precision, recall, F1 and
  macro-F1 computed from it. Binary and multiclass. A threshold sweep gives the ROC and precision-recall
  curves, ROC-AUC and average precision.
- **Regression metrics** (`regression/`): MSE, RMSE, MAE and R2.
- **Statistics** (`stats/`): percentile bootstrap confidence intervals (paired across arrays), Welch's
  two-sample t-test, and sample sizes for A/B tests (means and conversion rates).

Every metric and test has a parity test against the sklearn or scipy function with the same meaning
(`tests/`). The
t-test computes its statistic and degrees of freedom here and looks up only the t distribution's tail in scipy.
The power formulas have no reference here; `tests/test_power.py` checks them by simulation instead.

## Run it
```console
uv sync
uv run pytest
uv run mfs classify --help
uv run mfs curves --help
uv run mfs regress --help
uv run mfs compare --help
uv run mfs t-test --help
uv run mfs power --help
```

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py`
enforces both rules.

1. `data`: synthetic datasets.
2. `classification`: metrics on labels and scores.
3. `regression`: metrics on real-valued predictions.
4. `stats`: intervals and tests on any statistic.
5. `cli.py`: the `mfs` command.
