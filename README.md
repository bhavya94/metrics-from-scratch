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

## What is here
- **Synthetic data** (`data/`): a binary task with a chosen positive rate and class separation, seeded.
- **Classification metrics** (`classification/`): a confusion matrix, and accuracy, precision, recall, F1 and
  macro-F1 computed from it. Binary and multiclass. A threshold sweep gives the ROC and precision-recall
  curves, ROC-AUC and average precision.

Every metric has a parity test against the sklearn function with the same meaning (`tests/`).

## Run it
```console
uv sync
uv run pytest
uv run mfs classify --help
uv run mfs curves --help
```

## Layers
A package never imports a package listed after it, and only `cli.py` prints. `tests/test_layers.py`
enforces both rules.

1. `data`: synthetic datasets.
2. `classification`: metrics on labels and scores.
3. `cli.py`: the `mfs` command.
