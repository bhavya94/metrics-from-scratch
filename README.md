# metrics-from-scratch

Evaluation metrics and statistics, implemented from scratch in numpy and checked against sklearn and scipy.

Each metric here can make a weak model look good under the right conditions. The project shows those conditions
with seeded synthetic data, so every number below can be re-run with one command. Every metric matches
sklearn or scipy in tests; the power formulas are checked by simulation.

## When each number misleads

| Number | Misleads when | Check instead |
|---|---|---|
| Accuracy | Positives are rare: predicting "negative" scores high | Precision, recall, F1 for the rare class |
| ROC-AUC | Positives are rare: it ignores how many false positives each true positive costs | Average precision, against the positive rate |
| MSE | A few large errors dominate the average | MAE next to MSE; look at the residuals |
| R2 | Read as "share explained" when it is negative or computed on a different set | The mean-prediction baseline (R2 = 0) |
| A single score | Two models differ by less than the noise | A paired bootstrap CI of the difference |
| p > 0.05 | The test had low power, so "no effect" means "can't tell" | A power analysis before the test |
| Raw agreement | One label dominates, so agreeing by chance is easy | Cohen's kappa, plus TPR and TNR |

## Evidence

On data where 1% of examples are positive, a classifier can score 99% accuracy while it finds fewer than 1 in 10
positives:

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

The same trap applies to LLM judges. A judge that passes everything agrees with the human 72% of the time here,
because most answers deserve a pass. Cohen's kappa removes the agreement that chance alone would give, and the
true positive and true negative rates show which side a judge errs on:

```console
$ uv run mfs kappa
200 answers labeled by a human (72% pass) and three judges
  judge          agree    TPR    TNR  kappa  95% CI
  always pass     0.72   1.00   0.00   0.00  [0.00, 0.00]
  random 80/20    0.62   0.79   0.18  -0.03  [-0.16, 0.11]
  careful         0.93   0.96   0.84   0.81  [0.71, 0.90]
```

## Charts
`docs/views/index.html` draws one chart per section above, from the same seeded runs. Open it in a browser. To
rebuild it after a code change, run `uv run python scripts/build_views.py`.

## What is here
- **Synthetic data** (`data/`): a binary task with a chosen positive rate and class separation, seeded.
- **Classification metrics** (`classification/`): a confusion matrix, and accuracy, precision, recall, F1 and
  macro-F1 computed from it. Binary and multiclass. A threshold sweep gives the ROC and precision-recall
  curves, ROC-AUC and average precision.
- **Regression metrics** (`regression/`): MSE, RMSE, MAE and R2.
- **Statistics** (`stats/`): percentile bootstrap confidence intervals (paired across arrays), Welch's
  two-sample t-test, and sample sizes for A/B tests (means and conversion rates).
- **Agreement** (`agreement/`): Cohen's kappa between two raters, such as an LLM judge and a human.

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
uv run mfs kappa --help
```

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py`
enforces both rules.

1. `data`: synthetic datasets.
2. `classification`: metrics on labels and scores.
3. `regression`: metrics on real-valued predictions.
4. `stats`: intervals and tests on any statistic.
5. `agreement`: rater agreement, built on the confusion matrix.
6. `cli.py`: the `mfs` command.

## Limits
- **Synthetic data.** The datasets are simple by design: Gaussian classes, a straight line, normal sessions.
  Real data has skew, drift and label noise, which make every interval here wider.
- **Normal approximations.** The power formulas assume large samples. For small samples or rare conversions,
  check the plan by simulation, as `tests/test_power.py` does.
- **Percentile bootstrap.** It can undercover for small samples and for skewed statistics. BCa intervals
  correct for that; they are not implemented here.
- **Peeking.** Each test assumes you look at the result once, at the planned sample size. Checking daily and
  stopping at the first p < 0.05 inflates the false positive rate. Sequential tests handle that; not covered.
- **Kappa depends on prevalence.** With one label at 95%, even a good judge can get a low kappa. Report TPR and
  TNR next to it.
