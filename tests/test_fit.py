"""Tests for the bounded fitting interface (fit.py).

The first one is the test the Oct 29 reply asked for: on a noiseless case,
refinement moves the half-life by less than one grid step. The second one makes
sure the plain grid gives W1's fits, and that the 95% set and its class come out
the same when worked out from W1's own profile. The others each cover one rule:
  - no refinement at the grid end, with b_R = 0 or on a cap;
  - keeping whichever RSS is lower;
  - the second-minimum rule;
  - the 95% set measured from the refined minimum;
  - the nadir depth, clipping, "failed" and bad input.

Some use a made-up profile in place of solve_at_k, because none of our real fits
has a second minimum or a failed solve.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from doxo_lvef import fit as fit_module
from doxo_lvef.fit import fit_lvef, fit_settings, local_minima, nadir_depth
from doxo_lvef.inputs import dosing_from_config, load_config
from doxo_lvef.models import params_from_case, response_basis, simulate_lvef

REPO = Path(__file__).resolve().parents[1]
CFG = load_config()
DOSING = dosing_from_config(CFG)
SETTINGS = fit_settings(CFG)
HALF_LIVES = SETTINGS["half_lives"]
GRID_STEP = np.log(HALF_LIVES[1] / HALF_LIVES[0])        # one grid step, in log half-life
CUTOFF = chi2.ppf(0.95, 1)


def noiseless(schedule, a, f_P, half_life):
    t = np.array(CFG["schedules"][schedule])
    return t, simulate_lvef(t, dosing=DOSING, **params_from_case(a, f_P, half_life))


def fake_profile(rss_of_half_life):
    """A stand-in for solve_at_k that returns a made-up RSS (and coefficients nowhere near a bound)."""
    def solve(t, y, k, dosing, lower, upper):
        return np.array([75.0, 1.0, 1.0]), float(rss_of_half_life(np.log(2) / k)), True
    return solve


# ---------- the test the Oct 29 reply asked for ----------

def test_refinement_moves_the_half_life_less_than_one_grid_step():
    # Noiseless a = 4, f_P = 0.3, half-life 2, with sigma = 0.25 (identified on all three
    # schedules). The best grid point is 1.945 weeks.
    for schedule in ("uniform", "treatment", "late"):
        t, y = noiseless(schedule, 4, 0.3, 2)
        grid = fit_lvef(t, y, DOSING, 0.25, **SETTINGS)
        refined = fit_lvef(t, y, DOSING, 0.25, refine=True, **SETTINGS)
        assert grid["half_life"] == pytest.approx(1.945, abs=1e-3) and refined["refined"]
        assert abs(np.log(refined["half_life"] / grid["half_life"])) < GRID_STEP
        # "Less than one step" can hardly fail, since the search stays between the two neighbours
        # anyway. So also check that it found the real minimum: a lower RSS and the true half-life.
        assert refined["rss"] <= grid["rss"]
        assert refined["half_life"] == pytest.approx(2.0, rel=1e-3)


# ---------- the other rules ----------

@pytest.fixture(scope="module")
def w1():
    # Load scripts/W1.py as a module without running its main().
    spec = importlib.util.spec_from_file_location("W1", REPO / "scripts" / "W1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def w1_datasets(w1):
    """The 27 datasets W1's main() fits: same seed, same order, same paired noise."""
    rng = np.random.default_rng(w1.SEED)
    union = np.unique(np.concatenate([np.array(ts) for ts in w1.SCHEDULES.values()]))
    for a, f_P, half_life, sigma in w1.CASES:
        clean = w1.lvef(union, **w1.true_params(a, f_P, half_life))
        for _ in range(w1.N_REPS):
            noisy = clean + rng.normal(0.0, sigma, union.size)
            for times in w1.SCHEDULES.values():
                idx = np.searchsorted(union, times)
                yield union[idx], noisy[idx], sigma


def test_plain_grid_gives_the_same_fits_as_w1(w1):
    # With refine=False, fit_lvef is W1's grid fit moved into the package. On W1's 27 datasets it
    # should give the same profile, estimates and flags (W1 keeps its Week 1 labels, so those aren't
    # compared). The 95% set and its class are also checked against W1's profile directly, and the
    # cap flag against W1's own stuck_on_a_cap.
    assert np.array_equal(HALF_LIVES, w1.HALF_LIVES)
    assert np.array_equal(SETTINGS["lower"], w1.LOWER) and np.array_equal(SETTINGS["upper"], w1.UPPER)
    classes = []
    for t, y, sigma in w1_datasets(w1):
        old = w1.fit_grid(t, y, sigma)
        new = fit_lvef(t, y, DOSING, sigma, **SETTINGS)
        assert new["status"] == "ok" and not new["refined"]
        assert new["profile"]["rss"] == pytest.approx(old["profile"], rel=1e-10)
        assert [new[n] for n in ("L0", "b_R", "b_P", "k")] == pytest.approx([old[n] for n in ("L0", "b_R", "b_P", "k")],
                                                                            rel=1e-10, abs=1e-12)
        assert (new["half_life_grid"], new["at_grid_end"], new["coef_at_bound"]) == \
               (old["half_life_grid_raw"], old["k_at_grid_edge"], old["coef_at_bound"])
        assert new["on_cap"] == w1.stuck_on_a_cap(np.array([[old["L0"], old["b_R"], old["b_P"]]]))[0]
        inside = (old["profile"] - old["profile"].min()) / sigma ** 2 <= CUTOFF
        assert np.array_equal(new["profile"]["in_set"], inside)
        assert (new["set_low"], new["set_high"]) == (HALF_LIVES[inside][0], HALF_LIVES[inside][-1])
        expected = {(False, False): "identified", (False, True): "open above", (True, False): "open below",
                    (True, True): "uninformative"}[bool(inside[0]), bool(inside[-1])]
        assert new["half_life_class"] == expected
        classes.append(expected)
        coefs = new["profile"]["coefs"]
        assert np.all(coefs >= SETTINGS["lower"]) and np.all(coefs <= SETTINGS["upper"])
    # docs/weekly/W3.md quotes these counts.
    assert {c: classes.count(c) for c in set(classes)} == {"identified": 3, "open above": 6, "uninformative": 18}


def test_no_refinement_at_the_grid_end():
    # True half-life 100 weeks, past the 52-week end. The best point is the end, so it stays there
    # and gets flagged as a bound hit instead. With sigma = 0.25 short half-lives get ruled out, but
    # the 95% set runs to the top of the grid: open above. (At sigma = 1 it's uninformative.)
    t, y = noiseless("uniform", 4, 0.3, 100)
    result = fit_lvef(t, y, DOSING, 0.25, refine=True, **SETTINGS)
    assert result["half_life"] == HALF_LIVES[-1] and result["at_grid_end"] and not result["refined"]
    assert result["half_life_class"] == "open above" and result["set_high"] == HALF_LIVES[-1]
    assert result["notes"] == "bound hit: half-life at the grid end"


def test_no_refinement_on_a_cap():
    # Noiseless data with b_R = 9, above the cap of 6. The fit stops at b_R = 6 and makes up for it
    # with a slower recovery (1.42 weeks instead of 1): the b_R / k ridge from the Nov 8 reply, where
    # the cap acts as a lower bound on the half-life. A fit on a cap counts as "at the bound", so it
    # isn't refined and gets flagged.
    t = np.array(CFG["schedules"]["uniform"])
    y = simulate_lvef(t, 75.0, 9.0, 1.0, np.log(2) / 1.0, DOSING)
    result = fit_lvef(t, y, DOSING, 1.0, refine=True, **SETTINGS)
    assert result["b_R"] == 6.0 and result["on_cap"] and not result["refined"]
    assert result["half_life"] == pytest.approx(1.415, abs=1e-3)
    assert result["notes"] == "bound hit: b_R at 6"
    # L0 on either of its bounds counts too: LVEF of 100 everywhere would need L0 above 95, and
    # 40 everywhere would need it below 50.
    for level, bound in [(100.0, 95), (40.0, 50)]:
        result = fit_lvef(t, np.full(t.size, level), DOSING, 1.0, **SETTINGS)
        assert result["on_cap"] and f"bound hit: L0 at {bound}" in result["notes"]


def test_no_refinement_when_b_R_is_zero(monkeypatch):
    # y = 75 - 2H + 0.8G would need a negative b_R, so the bounded fit puts b_R at 0 and the profile
    # is flat. No refinement, no half-life, and the note says why. The flat profile is one minimum,
    # and its 95% set is the whole grid. The nadir depth is still there: it doesn't need k.
    t = np.array(CFG["schedules"]["treatment"])
    H, G = response_basis(t, np.log(2) / 2, DOSING)
    y = 75 - 2 * H + 0.8 * G
    result = fit_lvef(t, y, DOSING, 1.0, refine=True, **SETTINGS)
    assert result["b_R"] == 0.0 and not result["refined"]
    assert result["half_life"] is None and result["half_life_class"] == "uninformative"
    assert result["notes"] == "b_R = 0, so k drops out of the model"
    assert len(result["near_optimal"]) == 1 and result["nadir_depth"] > 0
    # The flat profile's lowest point is the grid end there, which wouldn't be refined anyway. So also
    # fake b_R = 0 with a minimum in the middle of the grid: it still shouldn't be refined.
    monkeypatch.setattr(fit_module, "solve_at_k", lambda t, y, k, dosing, lower, upper: (
        np.array([75.0, 0.0, 1.0]), 10 + np.log(np.log(2) / k / 2) ** 2, True))
    result = fit_lvef(t, y, DOSING, 1.0, refine=True, **SETTINGS)
    assert not result["refined"] and result["half_life"] is None


def test_refinement_keeps_the_grid_point_if_that_is_lower(monkeypatch):
    # "Keep whichever value is lower": if the real minimum sits right on a grid point, the bounded
    # search can't beat it, so the grid point stays and the fit isn't marked as refined.
    on_grid = HALF_LIVES[30]
    monkeypatch.setattr(fit_module, "solve_at_k", fake_profile(lambda h: 10 + 5 * np.log(h / on_grid) ** 2))
    t, y = noiseless("uniform", 4, 0.3, 2)
    result = fit_lvef(t, y, DOSING, 1.0, refine=True, **SETTINGS)
    assert result["half_life"] == on_grid and result["rss"] == 10.0 and not result["refined"]


def test_a_second_minimum_within_2_is_refined_and_kept(monkeypatch):
    # None of the W1 fits has a second minimum, so fake the profile: minima at 1.0 week (RSS 11),
    # 7.3 weeks (RSS 10) and 30 weeks (RSS 14), none of them on a grid point. With sigma = 1 the
    # first two are within 2 units of RSS/sigma^2 of each other and the third isn't. (1.0 is just
    # below its nearest grid point and 7.3 just above, so both halves of the search get used.)
    levels = {1.0: 11.0, 7.3: 10.0, 30.0: 14.0}
    monkeypatch.setattr(fit_module, "solve_at_k", fake_profile(
        lambda h: min(level + 20 * np.log(h / center) ** 2 for center, level in levels.items())))
    t, y = noiseless("uniform", 4, 0.3, 2)         # the fake ignores the data
    result = fit_lvef(t, y, DOSING, 1.0, refine=True, **SETTINGS)
    kept = result["near_optimal"]                   # best first
    assert [e["half_life"] for e in kept] == pytest.approx([7.3, 1.0], rel=1e-6)
    assert all(e["refined"] for e in kept)
    assert [e["delta"] for e in kept] == pytest.approx([0.0, 1.0], abs=1e-9)
    assert result["half_life"] == kept[0]["half_life"]


def test_the_95_set_is_measured_from_the_refined_minimum(monkeypatch):
    # The Oct 29 reply lists profile-likelihood intervals as a place where the refined minimum
    # should be used. Fake a smooth profile RSS = 10 + c ln(h/2)^2 (sigma = 1): the real minimum is
    # 10 at 2 weeks, between grid points. c is picked so that the grid point 3 steps above the best
    # one is just inside the 95% set when the rise is measured from the grid's minimum, and just
    # outside when it's measured from the refined one.
    i = int(np.argmin(np.abs(np.log(HALF_LIVES / 2.0))))
    c = CUTOFF / (np.log(HALF_LIVES[i + 3] / 2) ** 2 - np.log(HALF_LIVES[i] / 2) ** 2 / 2)
    monkeypatch.setattr(fit_module, "solve_at_k", fake_profile(lambda h: 10 + c * np.log(h / 2) ** 2))
    t, y = noiseless("uniform", 4, 0.3, 2)
    plain = fit_lvef(t, y, DOSING, 1.0, **SETTINGS)
    refined = fit_lvef(t, y, DOSING, 1.0, refine=True, **SETTINGS)
    assert refined["rss"] == pytest.approx(10.0) and refined["half_life"] == pytest.approx(2.0, rel=1e-6)
    assert plain["set_high"] == HALF_LIVES[i + 3]
    assert refined["set_high"] == HALF_LIVES[i + 2]


def test_nadir_depth_of_the_known_truth_fit():
    # Noiseless a = 4, f_P = 0.3, half-life 2: the refined fit gets the truth back, so its nadir
    # depth should be the true one, 10.8 (the lowest point, 64.2, is in the middle panel of W2's
    # figure). nadir_depth only looks on a grid of times, so it's also checked with a 1-D minimizer.
    # The true lowest point is just after the last dose, between weeks 7 and 8.
    t, y = noiseless("treatment", 4, 0.3, 2)
    result = fit_lvef(t, y, DOSING, 0.25, refine=True, **SETTINGS)
    p = params_from_case(4, 0.3, 2)
    assert result["nadir_depth"] == pytest.approx(nadir_depth(*p.values(), DOSING, 0.0, 20.0), rel=1e-6)
    lowest = minimize_scalar(lambda s: simulate_lvef(s, dosing=DOSING, **p)[0], bounds=(7.0, 8.0),
                             method="bounded", options={"xatol": 1e-9})
    assert result["nadir_depth"] == pytest.approx(75.0 - lowest.fun, abs=1e-4)
    assert result["nadir_depth"] == pytest.approx(10.8, abs=0.01)


def test_coefficients_are_clipped_into_the_bounds(monkeypatch):
    # bvls can hand back something like -6e-16 for a coefficient on its bound. Fake that for b_P
    # (which is exactly 0 at every k for these data) and check it comes out as exactly 0.
    real = fit_module.lsq_linear

    def slightly_off(*args, **kwargs):
        sol = real(*args, **kwargs)
        sol.x = sol.x - np.array([0.0, 0.0, 6e-16])
        return sol

    monkeypatch.setattr(fit_module, "lsq_linear", slightly_off)
    t = np.array(CFG["schedules"]["uniform"])
    H, G = response_basis(t, np.log(2) / 2, DOSING)
    result = fit_lvef(t, 75 - 2 * G + 0.5 * H, DOSING, 1.0, **SETTINGS)     # would need b_P < 0
    assert result["b_P"] == 0.0 and np.all(result["profile"]["coefs"][:, 2] == 0.0)


def test_a_failed_solve_makes_the_whole_fit_failed(monkeypatch):
    # bvls hasn't failed on any of our data, so fake one failure at one grid point.
    real = fit_module.solve_at_k

    def fail_at_point_10(t, y, k, dosing, lower, upper):
        coefs, rss, ok = real(t, y, k, dosing, lower, upper)
        return coefs, rss, ok and not np.isclose(k, np.log(2) / HALF_LIVES[10])

    monkeypatch.setattr(fit_module, "solve_at_k", fail_at_point_10)
    t, y = noiseless("uniform", 4, 0.3, 2)
    result = fit_lvef(t, y, DOSING, 0.25, refine=True, **SETTINGS)
    assert result["status"] == "failed" and result["half_life_class"] == "failed"
    assert all(np.isnan(result[n]) for n in ("L0", "b_R", "b_P", "k", "rss", "half_life", "nadir_depth"))
    assert result["profile"]["solver_ok"].sum() == HALF_LIVES.size - 1


def test_local_minima_on_made_up_profiles():
    assert local_minima([3, 2, 1, 2, 3]) == [2]
    assert local_minima([2, 1, 2, 1.5, 2]) == [1, 3]              # two separate minima
    assert local_minima([1, 2, 3]) == [0] and local_minima([3, 2, 1]) == [2]     # a grid end can be one
    assert local_minima([5, 5, 5, 5]) == [0]                       # flat (b_R = 0): one minimum, not four
    assert local_minima([3, 1, 1 + 1e-12, 1, 3]) == [1]            # a tiny wiggle in a flat stretch: still one


def test_bad_input_is_rejected():
    t, y = noiseless("uniform", 4, 0.3, 2)
    with pytest.raises(ValueError, match="sigma"):
        fit_lvef(t, y, DOSING, 0.0, **SETTINGS)
    with pytest.raises(ValueError, match="same length"):
        fit_lvef(t, y[:-1], DOSING, 1.0, **SETTINGS)
    with pytest.raises(ValueError, match="NaN"):
        fit_lvef(t, np.r_[y[:-1], np.nan], DOSING, 1.0, **SETTINGS)
    with pytest.raises(ValueError, match="half_lives"):
        fit_lvef(t, y, DOSING, 1.0, **{**SETTINGS, "half_lives": HALF_LIVES[::-1]})
