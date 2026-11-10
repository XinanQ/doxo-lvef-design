"""Smoke tests: the milestone scripts still run end to end, and W1's numbers didn't move.

The proposal wants W1.py to keep working while the package grows. W1 runs once here with
its output folder pointed at a temp dir (so the committed files in outputs/ don't get
touched), and the tests below look at what it wrote. At the end each script also runs
once as a real `python scripts/<name>.py` command, on a copy of the repo.
"""
import csv
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")      # no windows popping up during tests
import matplotlib.pyplot as plt
import pytest

REPO = Path(__file__).resolve().parents[1]
STATUSES = {"identified", "disjoint", "lower bound only", "upper bound only", "not identified"}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_rows(path):
    """The data rows of a W1 csv (the '#' lines at the top are skipped)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return list(csv.DictReader(line for line in lines if not line.startswith("#")))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def w1_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("w1")
    w1 = load_script("W1")
    w1.OUT_DIR = out
    w1.main()
    return w1, out


def test_w1_runs_and_writes_its_outputs(w1_run):
    w1, out = w1_run
    assert (out / "W1_fits.png").exists()
    assert len(csv_rows(out / "W1_fits.csv")) == len(w1.CASES) * w1.N_REPS * len(w1.SCHEDULES)
    # The fresh file should have plain "\n" line endings and the versions on line 2. (Only the
    # fresh one: git may turn the committed copy's line endings into "\r\n" on Windows.)
    raw = (out / "W1_fits.csv").read_bytes()
    assert b"\r" not in raw and raw.split(b"\n")[1].startswith(b"# env: python=")


def test_w1_output_matches_the_committed_csv(w1_run):
    # Same seed + same code should give exactly the committed numbers. Careful: this compares with
    # the file in your working tree, so run pytest BEFORE rerunning W1. After a change you meant to
    # make, rerun W1, look at `git diff outputs/` and explain the diff in docs/weekly/.
    # The "# env:" line (python/numpy/scipy versions) is skipped on purpose, so a different patch
    # version doesn't fail this for no reason. It's printed if anything else differs; if the
    # versions don't match, try `pip install -r requirements.txt` first.
    # This only catches changes. It can't tell if the numbers are right; test_w1_fit.py checks the
    # labels on cases where we know the answer.
    _, out = w1_run
    new = (out / "W1_fits.csv").read_text(encoding="utf-8").splitlines()
    old = (REPO / "outputs" / "W1_fits.csv").read_text(encoding="utf-8").splitlines()
    env_lines = [line for line in new + old if line.startswith("# env:")]
    assert [x for x in new if not x.startswith("# env:")] == [x for x in old if not x.startswith("# env:")], env_lines


def test_w1_rows_follow_the_reporting_rules(w1_run):
    # These should hold whatever the numbers are.
    w1, out = w1_run
    b_R_zero = w1.LOWER[1] + 1e-9 * (w1.UPPER[1] - w1.LOWER[1])      # same tolerance fit_grid uses
    for r in csv_rows(out / "W1_fits.csv"):
        assert r["half_life_status"] in STATUSES
        # A half-life only when it's identified, and then both interval ends are there too.
        assert (r["half_life_hat"] != "") == (r["half_life_status"] == "identified")
        if r["half_life_status"] == "identified":
            assert r["half_life_ci_low"] and r["half_life_ci_high"]
        # b_R = 0 always gets its note, and never a grid edge or a grid half-life. (The seeded run
        # happens to have no b_R = 0 fit, so here this only catches the note on the wrong row; the
        # rule itself is tested in test_w1_fit.py.)
        assert ("b_R = 0" in r["notes"]) == (float(r["b_R_hat"]) <= b_R_zero)
        if "b_R = 0" in r["notes"]:
            assert r["k_at_grid_edge"] == "False" and r["half_life_grid_raw"] == ""
        # The ranking loss is the proposal's formula, worked out again from the row itself.
        a = float(r["a"])
        errors = [(float(r[f"{b}_hat"]) - float(r[f"{b}_true"])) / a for b in ("b_R", "b_P")]
        assert float(r["norm_amp_loss"]) == pytest.approx(sum(e ** 2 for e in errors) / 2, rel=1e-3, abs=1e-6)


def test_w1_labels_match_what_the_notes_say(w1_run):
    # docs/weekly/W1.md quotes these, so if they change the doc has to change too.
    _, out = w1_run
    rows = csv_rows(out / "W1_fits.csv")
    labeled = {(r["case"], r["rep"], r["schedule"]): r["half_life_status"]
               for r in rows if r["half_life_status"] != "not identified"}
    assert labeled == {("2", "0", "uniform"): "identified", ("1", "1", "uniform"): "lower bound only",
                       ("1", "2", "late"): "upper bound only"}
    assert sum(r["notes"] == "short side counted open (first ruled-out point is on a cap)" for r in rows) == 7


def test_w2_runs_and_writes_its_figure(tmp_path):
    w2 = load_script("W2")
    w2.OUT_DIR = tmp_path
    w2.main()
    assert (tmp_path / "W2_trajectories.png").exists()


def test_w3_runs_and_the_two_paths_agree(tmp_path, capsys):
    w3 = load_script("W3")
    w3.OUT_DIR = tmp_path
    w3.main()
    assert (tmp_path / "W3_check.png").exists()
    printed = capsys.readouterr().out
    assert "so they agree" in printed and "so the class is identified" in printed


@pytest.mark.parametrize("name, files", [("W1", ["W1_fits.csv", "W1_fits.png"]),
                                         ("W2", ["W2_trajectories.png"]),
                                         ("W3", ["W3_check.png"])], ids=["W1", "W2", "W3"])
def test_script_runs_as_a_command(name, files, tmp_path):
    # Run the real `python scripts/<name>.py`, so the `if __name__ == "__main__"` part and the
    # default output folder get tested too. It runs on a copy of scripts/, src/ and configs/, so the
    # committed outputs/ don't get touched. pytest puts src/ on sys.path for every test (see
    # pyproject.toml), which would hide a script that forgot to set up its own import path, so use a
    # brand-new Python: -I ignores PYTHONPATH and doesn't add any folder to the path, and -B means
    # this run writes no .pyc files.
    copy = tmp_path / "repo"
    for folder in ("scripts", "src", "configs"):
        shutil.copytree(REPO / folder, copy / folder, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run([sys.executable, "-I", "-B", str(copy / "scripts" / f"{name}.py")],
                   cwd=tmp_path, check=True, timeout=120)
    for f in files:
        assert (copy / "outputs" / f).exists()
