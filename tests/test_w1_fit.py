"""Tests for the labels W1's fit_grid puts on each fit.

The CSV regression test in test_scripts.py only tells us the output didn't change,
not that the labels are right (it would happily accept wrong labels once the file is
regenerated). So these use cases where we know the answer, mostly noiseless data.
The expected numbers were worked out once with a scratch script and then checked by
hand against the profile.
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")      # no windows popping up during tests
import numpy as np
import pytest
from matplotlib.axes import Axes

from doxo_lvef.inputs import visit_times

REPO = Path(__file__).resolve().parents[1]
TREATMENT = np.array([0.0, 2.0, 6.0, 8.0, 20.0])


@pytest.fixture(scope="module")
def w1():
    # Load scripts/W1.py as a module without running its main().
    spec = importlib.util.spec_from_file_location("W1", REPO / "scripts" / "W1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def noiseless(w1, schedule, a, f_P, half_life):
    t = np.array(w1.SCHEDULES[schedule])
    return t, w1.lvef(t, **w1.true_params(a, f_P, half_life))


# ---------- labels ----------

def test_b_R_zero_means_k_is_not_in_the_model(w1):
    # y = 75 - 2H + 0.8G would need a NEGATIVE b_R, so the bounded fit puts b_R at 0. Then k drops
    # out and every k fits exactly as well: no grid edge and no grid half-life, just "not identified".
    H, G = w1.basis(TREATMENT, np.log(2) / 2)
    fit = w1.fit_grid(TREATMENT, 75 - 2 * H + 0.8 * G, 1.0)
    assert fit["b_R"] == 0.0 and "b_R" in fit["coef_at_bound"].split()
    assert fit["profile_range"] == 0.0                                  # exactly flat
    assert fit["half_life_status"] == "not identified"
    assert fit["k_at_grid_edge"] is False and fit["half_life_grid_raw"] is None
    assert fit["notes"] == "b_R = 0, so k drops out of the model"


def test_a_tiny_b_R_counts_as_zero(w1):
    # A true b_R of 1e-11 comes back from bvls as a tiny positive number at every grid k (not
    # exactly 0), so it's the tolerance in fit_grid that makes this a b_R = 0 fit.
    fit = w1.fit_grid(TREATMENT, w1.lvef(TREATMENT, 75.0, 1e-11, 1.0, np.log(2) / 4), 1.0)
    assert 0.0 < fit["b_R"] <= 1e-9 * (w1.UPPER[1] - w1.LOWER[1])
    assert fit["notes"] == "b_R = 0, so k drops out of the model"


def test_half_life_below_the_grid_lands_on_the_fast_edge(w1):
    # True half-life 0.05 weeks, below our 0.1-week grid end. Visits 0.02 weeks after three of the
    # doses so the fast recovery actually shows. b_R isn't 0 here, so this one is a real grid edge.
    t = np.array([0.0, 2.02, 6.02, 7.02, 20.0])
    fit = w1.fit_grid(t, w1.lvef(t, 75.0, 3.0, 1.0, np.log(2) / 0.05), 1.0)
    assert fit["k"] == w1.K_GRID[0] and fit["b_R"] == pytest.approx(2.36, abs=0.01)
    assert fit["k_at_grid_edge"] is True
    assert fit["half_life_status"] == "not identified" and fit["half_life_hat"] is None


def test_half_life_above_the_grid_lands_on_the_slow_edge(w1):
    # True half-life 100 weeks, past our 52-week grid end. Over 20 weeks that barely recovers,
    # so the best grid point is the 52-week edge. b_R isn't 0, so it's a real edge again.
    fit = w1.fit_grid(*noiseless(w1, "uniform", 4, 0.3, 100), 1.0)
    assert fit["half_life_grid_raw"] == w1.HALF_LIVES[-1] and fit["b_R"] == pytest.approx(1.55, abs=0.01)
    assert fit["k_at_grid_edge"] is True
    assert fit["half_life_status"] == "not identified" and fit["half_life_hat"] is None


def test_every_coefficient_on_a_bound_gets_reported(w1):
    # LVEF of 100 everywhere: L0 hits its upper bound of 95 and both amplitudes sit at 0.
    assert w1.fit_grid(TREATMENT, np.full(5, 100.0), 1.0)["coef_at_bound"] == "L0 b_R b_P"


def test_which_bounds_count_as_caps(w1):
    # Upper bounds on the amplitudes and both L0 bounds are caps (our search box).
    # An amplitude at 0 isn't: amplitudes >= 0 is part of the model.
    coefs = np.array([[75.0, 6.0, 1.0], [75.0, 1.0, 6.0], [50.0, 1.0, 1.0], [95.0, 1.0, 1.0],
                      [75.0, 0.0, 0.0], [75.0, 5.99, 5.99]])
    assert w1.stuck_on_a_cap(coefs).tolist() == [True, True, True, True, False, False]


def test_profile_range_is_in_units_of_sigma_squared(w1):
    y = w1.lvef(TREATMENT, 75.0, 1.4, 0.6, np.log(2) / 2) + np.array([0.3, -0.5, 0.2, 0.4, -0.1])
    wide = w1.fit_grid(TREATMENT, y, 1.0)["profile_range"]
    assert wide == pytest.approx(4.0 * w1.fit_grid(TREATMENT, y, 2.0)["profile_range"])


# Noiseless a = 4, f_P = 0.3, half-life 2. The best grid point is 1.945 weeks (the grid value
# closest to 2), and the ends are the first grid points outside the 95% set.
@pytest.mark.parametrize("schedule, low, high", [("uniform", 1.273, 3.305), ("treatment", 1.030, 5.050),
                                                 ("late", 1.145, 3.305)])
def test_noiseless_data_with_small_noise_sd_is_identified(w1, schedule, low, high):
    fit = w1.fit_grid(*noiseless(w1, schedule, 4, 0.3, 2), 0.25)
    assert fit["half_life_status"] == "identified"
    assert fit["half_life_hat"] == pytest.approx(1.945, abs=1e-3)
    assert (fit["half_life_ci_low"], fit["half_life_ci_high"]) == pytest.approx((low, high), abs=1e-3)
    assert fit["half_life_ci_low"] < 2.0 < fit["half_life_ci_high"]


@pytest.mark.parametrize("schedule, high", [("uniform", 27.53), ("treatment", 30.61), ("late", 24.76)])
def test_same_data_with_noise_sd_1_only_gives_an_upper_bound(w1, schedule, high):
    # At sigma = 1 the first fast half-life that gets ruled out has b_R stuck at its cap of 6.
    # That side counts as open, and the note says why.
    fit = w1.fit_grid(*noiseless(w1, schedule, 4, 0.3, 2), 1.0)
    assert fit["half_life_status"] == "upper bound only" and fit["half_life_hat"] is None
    assert fit["half_life_ci_low"] is None and fit["half_life_ci_high"] == pytest.approx(high, abs=0.01)
    assert fit["notes"] == "short side counted open (first ruled-out point is on a cap)"


def test_rising_past_the_cutoff_somewhere_is_not_enough(w1):
    # Noiseless a = 4, f_P = 0.7, half-life 8 on the treatment schedule. The profile rises by
    # 4.65 sigma^2, so the looser check in W1.md (does it rise by 3.84 anywhere?) would pass it.
    # But the 95% set runs out to the 52-week grid end, and on the short side the first
    # ruled-out point is on a cap, so neither side counts as closed.
    fit = w1.fit_grid(*noiseless(w1, "treatment", 4, 0.7, 8), 1.0)
    assert fit["profile_range"] == pytest.approx(4.65, abs=0.01)
    assert fit["half_life_status"] == "not identified"
    assert fit["notes"] == "short side counted open (first ruled-out point is on a cap)"


def test_a_cap_on_the_long_side_gets_its_own_note(w1, monkeypatch):
    # No fit in our runs has had this yet, so fake it: also flag the first ruled-out point ABOVE
    # the set as capped. Then the long side should count as open, with the long-side note.
    t, y = noiseless(w1, "uniform", 4, 0.3, 2)
    high = w1.fit_grid(t, y, 0.25)["half_life_ci_high"]
    real_check = w1.stuck_on_a_cap

    def also_flag_high(coefs):
        flags = real_check(coefs)
        flags[w1.HALF_LIVES.tolist().index(high)] = True
        return flags

    monkeypatch.setattr(w1, "stuck_on_a_cap", also_flag_high)
    fit = w1.fit_grid(t, y, 0.25)
    assert fit["half_life_status"] == "lower bound only" and fit["half_life_ci_high"] is None
    assert fit["notes"] == "long side counted open (first ruled-out point is on a cap)"


def test_a_gap_in_the_95_set_means_no_half_life(w1, monkeypatch):
    # No fit has come out disjoint yet either, so take the real result for an identified fit and
    # pretend the set had a gap. W1 shouldn't hand out a half-life then, and should say why.
    real_interval = w1.profile_interval

    def with_a_gap(*args, **kwargs):
        return real_interval(*args, **kwargs) | {"status": "disjoint", "disjoint": True}

    monkeypatch.setattr(w1, "profile_interval", with_a_gap)
    fit = w1.fit_grid(*noiseless(w1, "uniform", 4, 0.3, 2), 0.25)
    assert fit["half_life_status"] == "disjoint" and fit["half_life_hat"] is None
    assert fit["notes"] == "95% set has a gap"


# ---------- what the table and the figure say ----------

@pytest.mark.parametrize("status, hat, low, high, text", [
    ("identified", 6.242006895586503, 1.4153011984486359, 22.270589033743462, "6.24 [1.42, 22.3]"),
    ("lower bound only", None, 0.833065404601962, None, ">= 0.833"),
    ("upper bound only", None, None, 7.715998846573208, "<= 7.72"),
    ("disjoint", None, 1.0, 20.0, "--"),
    ("not identified", None, None, None, "--"),
], ids=["identified", "lower bound only", "upper bound only", "disjoint", "not identified"])
def test_half_life_text_for_each_label(w1, status, hat, low, high, text):
    # The first three are fits from the committed run (W1.md quotes them).
    r = dict(half_life_status=status, half_life_hat=hat, half_life_ci_low=low, half_life_ci_high=high)
    assert w1.half_life_text(r) == text


def test_printed_table_says_what_the_fit_says(w1, capsys):
    # W1.md quotes this table, so check a row with a half-life and a row with every kind of flag.
    # Only replicate 0 gets printed.
    identified = dict(case=2, rep=0, schedule="uniform", b_R_true=0.7, b_R_hat=4.21019, b_P_true=0.3,
                      b_P_hat=0.0, half_life_true=8, half_life_status="identified",
                      half_life_hat=6.242006895586503, half_life_ci_low=1.4153011984486359,
                      half_life_ci_high=22.270589033743462, norm_amp_loss=6.20582,
                      k_at_grid_edge=False, coef_at_bound="b_P", notes="")
    flagged = identified | dict(case=1, half_life_status="not identified", half_life_hat=None,
                                half_life_ci_low=None, half_life_ci_high=None, k_at_grid_edge=True,
                                coef_at_bound="b_R",
                                notes="short side counted open (first ruled-out point is on a cap)")
    w1.print_summary([identified, flagged, identified | dict(rep=1)])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3                                     # the header and the two rep 0 rows
    assert "6.24 [1.42, 22.3]" in lines[1] and lines[1].endswith("6.206  at bound: b_P")
    assert lines[2].endswith("not identified, best k at grid edge, at bound: b_R, "
                             "short side counted open (first ruled-out point is on a cap)")


def test_figure_titles_say_what_the_fit_says(w1, monkeypatch, tmp_path):
    # No test looks at the PNG itself, so at least check every panel title says the right thing.
    titles = []
    real_set_title = Axes.set_title

    def spy(ax, label, *args, **kwargs):
        titles.append(label)
        return real_set_title(ax, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", spy)
    monkeypatch.setattr(w1, "OUT_DIR", tmp_path)
    H, G = w1.basis(TREATMENT, np.log(2) / 2)
    fast = np.array([0.0, 2.02, 6.02, 7.02, 20.0])
    data = {(0, "uniform"): (*noiseless(w1, "uniform", 4, 0.3, 2), 0.25),                   # identified
            (0, "late"): (*noiseless(w1, "late", 4, 0.3, 100), 0.25),                        # >= bound, slow edge
            (1, "treatment"): (TREATMENT, 75 - 2 * H + 0.8 * G, 1.0),                        # b_R = 0
            (2, "late"): (fast, w1.lvef(fast, 75.0, 3.0, 1.0, np.log(2) / 0.05), 1.0)}       # fast edge
    truth = w1.true_params(4, 0.3, 2)                    # only used to draw the black curve
    fits = {key: w1.fit_grid(t, y, sigma) for key, (t, y, sigma) in data.items()}
    w1.plot({key: (truth, data[key][0], data[key][1], fit) for key, fit in fits.items()})
    identified, bound = w1.half_life_text(fits[0, "uniform"]), w1.half_life_text(fits[0, "late"])
    assert identified[0].isdigit() and bound.startswith(">=")
    assert titles == [f"case 0 {w1.CASES[0]}, uniform\nfitted half-life (wk): {identified}",
                      f"case 0 {w1.CASES[0]}, late\nfitted half-life (wk): {bound}, best k at grid edge",
                      f"case 1 {w1.CASES[1]}, treatment\nfitted half-life (wk): not identified (b_R = 0)",
                      f"case 2 {w1.CASES[2]}, late\nfitted half-life (wk): not identified, best k at grid edge"]


# ---------- visit offset ----------

def test_visit_offset_reaches_the_fit(w1, monkeypatch, tmp_path):
    # Shift the follow-ups by one day and check fit_grid really gets the shifted times, with the
    # baseline left alone, exactly the way inputs.visit_times does it.
    seen = []
    real_fit = w1.fit_grid

    def spy(t, y, sigma):
        seen.append(t)
        return real_fit(t, y, sigma)

    monkeypatch.setattr(w1, "VISIT_OFFSET", 1 / 7)
    monkeypatch.setattr(w1, "N_REPS", 1)
    monkeypatch.setattr(w1, "OUT_DIR", tmp_path)
    monkeypatch.setattr(w1, "fit_grid", spy)
    w1.main()
    schedules = list(w1.SCHEDULES.values())
    assert len(seen) == len(w1.CASES) * len(schedules)
    for i, t in enumerate(seen):                          # order: case, then schedule
        assert np.array_equal(t, visit_times(schedules[i % len(schedules)], 1 / 7))


# ---------- failing loudly ----------

@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan])
def test_bad_sigma_stops_the_fit(w1, sigma):
    with pytest.raises(ValueError):
        w1.fit_grid(TREATMENT, np.full(5, 70.0), sigma)


@pytest.mark.parametrize("t, y", [(TREATMENT, [75.0, 70.0, np.nan, 69.0, 72.0]),
                                  ([0.0, 2.0, np.nan, 8.0, 20.0], [75.0, 70.0, 70.0, 69.0, 72.0]),
                                  (TREATMENT, [75.0, 70.0, 69.0])], ids=["NaN in y", "NaN in t", "too short"])
def test_bad_data_stops_the_fit(w1, t, y):
    # W1's basis() would treat a NaN time as "before every dose", so this check is the only thing
    # stopping it. match= makes sure it's W1's own check that fires, not some error inside scipy.
    with pytest.raises(ValueError, match="t and y"):
        w1.fit_grid(np.array(t), np.array(y), 1.0)


@pytest.mark.parametrize("how", ["status 0", "NaN cost"])
def test_one_failed_solve_stops_the_run(w1, monkeypatch, how):
    # Pretend lsq_linear gave up at just ONE k on the grid (status 0 = ran out of iterations), or
    # gave back a NaN cost. W1 doesn't have a "failed" label yet, so it should stop instead of
    # letting one bad point into the profile, where it could pass for a ruled-out half-life.
    real_solver = w1.lsq_linear
    calls = []

    def fails_once(*args, **kwargs):
        sol = real_solver(*args, **kwargs)
        calls.append(sol)
        if len(calls) == 30:
            if how == "status 0":
                sol.status = 0
            else:
                sol.cost = np.nan
        return sol

    monkeypatch.setattr(w1, "lsq_linear", fails_once)
    with pytest.raises(RuntimeError):
        w1.fit_grid(TREATMENT, np.full(5, 70.0), 1.0)


def test_bvls_gets_enough_iterations(w1):
    # With bvls's default max_iter (3 here, the number of unknowns), three of the k's for these data
    # stop exactly at the optimum but still report status 0, so W1 would stop with an error. Found by
    # a random search; it never happened with our pilot data. max_iter=50 lets bvls finish.
    t = np.array([0.0, 2.0, 6.0, 10.0, 16.0])
    fit = w1.fit_grid(t, np.array([93.58, 85.77, 64.96, 62.31, 80.23]), 1.0)
    assert fit["b_R"] == pytest.approx(6.0) and fit["coef_at_bound"] == "b_R"
