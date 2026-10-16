from metrics_from_scratch.cli import main


def test_classify_prints_the_confusion_matrix_and_metrics(capsys):
    main(["classify", "--n", "2000", "--positive-rate", "0.05", "--seed", "4"])
    out = capsys.readouterr().out
    assert "2000 examples" in out and "TP=" in out
    assert all(name in out for name in ["accuracy", "precision", "recall", "f1", "macro_f1"])
