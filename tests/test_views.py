import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_views.py"


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    """One fresh build of the page, shared by every test here."""
    out = tmp_path_factory.mktemp("views") / "index.html"
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], check=True, capture_output=True)
    return out.read_text()


def _data(page: str) -> dict:
    start = page.index("const DATA = ") + len("const DATA = ")
    return json.loads(page[start:page.index(";\n", start)])


def test_page_carries_the_readme_numbers(page):
    data = _data(page)
    at_09 = next(row for row in data["confusion"]["rows"] if row["threshold"] == 0.9)
    assert (at_09["tn"], at_09["fp"], at_09["fn"], at_09["tp"]) == (9896, 15, 82, 7)
    assert [round(s["roc_auc"], 4) for s in data["curves"]["series"]] == [0.854, 0.8547, 0.8529, 0.8482]
    assert [round(run["mse"], 1) for run in data["regression"]["runs"]] == [15.7, 79.6]
    assert data["bootstrap"]["rows"][2]["low"] > 0  # the paired gap excludes zero
    assert data["power"]["needed"] == 409
    assert [round(row["kappa"], 2) for row in data["kappa"]["rows"]] == [0.0, -0.03, 0.81]


def test_residual_bands_hold_every_delivery(page):
    for run in _data(page)["regression"]["runs"]:
        assert sum(run["counts"]) == 2_000


def test_committed_page_matches_a_fresh_build(page):
    committed = SCRIPT.parents[1] / "docs" / "views" / "index.html"
    assert committed.read_text() == page, "run `uv run python scripts/build_views.py` and commit the page"
