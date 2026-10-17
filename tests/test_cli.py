from metrics_from_scratch.cli import main


def test_classify_prints_the_confusion_matrix_and_metrics(capsys):
    main(["classify", "--n", "2000", "--positive-rate", "0.05", "--seed", "4"])
    out = capsys.readouterr().out
    assert "2000 examples" in out and "TP=" in out
    assert all(name in out for name in ["accuracy", "precision", "recall", "f1", "macro_f1"])


def test_curves_prints_one_row_per_positive_rate(capsys):
    main(["curves", "--positive-rates", "0.5", "0.02", "--n", "5000"])
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[2:]]
    assert [row[0] for row in rows] == ["0.5", "0.02"]


def test_regress_prints_three_predictions_per_outlier_rate(capsys):
    main(["regress", "--outlier-rates", "0.0", "0.05", "--n", "500"])
    rows = capsys.readouterr().out.splitlines()[2:]
    assert len(rows) == 6 and sum("true line" in row for row in rows) == 2


def test_compare_reports_a_paired_interval(capsys):
    main(["compare", "--n", "1000", "--resamples", "200"])
    out = capsys.readouterr().out
    assert "B - A, paired" in out and "paired CI excludes 0:" in out


def test_t_test_reports_a_p_value(capsys):
    main(["t-test", "--n", "100", "--lift", "0"])
    assert "p = " in capsys.readouterr().out


def test_power_prints_sample_sizes_per_arm(capsys):
    main(["power"])
    out = capsys.readouterr().out
    assert "409" in out and "3841" in out and "50%" in out
