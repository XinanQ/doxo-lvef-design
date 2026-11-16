"""Tests for the pilot runner (evaluate.py): the config and seed tests and the failure
logging from the proposal's Week 4, plus the cap refit and the summaries.
"""
import copy

import numpy as np
import pytest

from doxo_lvef import fit as fit_module
from doxo_lvef.evaluate import (FAILED_LOSS, noise_stream, pick_visits, pilot_data, pilot_times, run_pilot,
                                summarize)
from doxo_lvef.inputs import dosing_from_config, load_config
from doxo_lvef.models import simulate_lvef

CFG = load_config()
DOSING = dosing_from_config(CFG)


def edited(**changes):
    """A copy of the config with some entries of case 0 changed."""
    cfg = copy.deepcopy(CFG)
    cfg["cases"][0].update(changes)
    return cfg


# ---------- configuration ----------

def test_pilot_settings_come_from_the_config():
    # 20 replicates per case (proposal), and the refit cap is wider than the declared cap of 6.
    assert CFG["pilot"]["n_reps"] == 20
    assert CFG["fit"]["b_R_bounds"][1] == 6.0 and CFG["fit"]["b_R_refit_cap"] == 12.0
    # The noise is drawn on the baseline plus the nine candidate weeks.
    assert pilot_times(CFG, DOSING).tolist() == [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]


def test_run_pilot_follows_the_config():
    # A small config: 2 replicates. Every case x replicate x schedule should be there once at
    # the declared cap, and every row should say which cap it was fitted with.
    cfg = copy.deepcopy(CFG)
    cfg["pilot"]["n_reps"] = 2
    rows = run_pilot(cfg)
    declared = [r for r in rows if r["b_R_cap"] == 6.0]
    assert len(declared) == 3 * 2 * 3
    assert {(r["case"], r["rep"], r["schedule"]) for r in declared} == \
           {(c, r, s) for c in range(3) for r in range(2) for s in cfg["schedules"]}
    assert {r["b_R_cap"] for r in rows} <= {6.0, 12.0}


# ---------- seeds ----------

def test_same_seed_gives_the_same_data():
    t1, y1, _ = pilot_data(CFG, DOSING, 1, 3)
    t2, y2, _ = pilot_data(CFG, DOSING, 1, 3)
    assert np.array_equal(t1, t2) and np.array_equal(y1, y2)


def test_every_case_and_replicate_has_its_own_stream():
    draws = {(c, r): noise_stream(CFG["seeds"]["development"], c, r).normal(size=4).tolist()
             for c in range(3) for r in range(3)}
    assert len({tuple(d) for d in draws.values()}) == 9          # all different
    other_seed = noise_stream(CFG["seeds"]["development"] + 1, 0, 0).normal(size=4).tolist()
    assert other_seed != draws[0, 0]


def test_changing_one_case_leaves_the_others_alone():
    # In W1 one stream fed everything, so editing a case or the number of replicates moved
    # everybody's noise. Here it shouldn't.
    cfg = edited(noise_sd=5.0, a=3.0)
    for case_id in (1, 2):
        assert np.array_equal(pilot_data(cfg, DOSING, case_id, 0)[1], pilot_data(CFG, DOSING, case_id, 0)[1])
    few, many = copy.deepcopy(CFG), copy.deepcopy(CFG)
    few["pilot"]["n_reps"], many["pilot"]["n_reps"] = 1, 2
    first = [r for r in run_pilot(many) if r["rep"] == 0]
    assert first == run_pilot(few)


def test_noise_has_the_sd_from_the_config_and_is_shared_by_the_schedules():
    # The noise is sigma times the same standard normal draws, so doubling sigma doubles it.
    t, y, truth = pilot_data(CFG, DOSING, 0, 0)
    clean = simulate_lvef(t, dosing=DOSING, **truth)
    doubled = pilot_data(edited(noise_sd=2.0), DOSING, 0, 0)[1]
    assert doubled - clean == pytest.approx(2.0 * (y - clean))
    # Paired: uniform and late both visit weeks 16 and 20, and they see the same values there.
    t_u, y_u = pick_visits(CFG, DOSING, t, y, "uniform")
    t_l, y_l = pick_visits(CFG, DOSING, t, y, "late")
    assert np.array_equal(y_u[np.isin(t_u, [16.0, 20.0])], y_l[np.isin(t_l, [16.0, 20.0])])


def test_a_visit_off_the_candidate_grid_is_an_error():
    cfg = copy.deepcopy(CFG)
    cfg["schedules"]["odd"] = [0.0, 3.0, 6.0, 8.0, 20.0]         # week 3 isn't a candidate week
    t, y, _ = pilot_data(cfg, DOSING, 0, 0)
    with pytest.raises(ValueError, match="candidate"):
        pick_visits(cfg, DOSING, t, y, "odd")


# ---------- failures ----------

def test_failed_fits_stay_in_with_loss_100(monkeypatch):
    # bvls hasn't failed on our data, so make every solve report a failure. The fits should still
    # be there as rows: status "failed", loss 100, no estimate, and counted in the denominator.
    real = fit_module.solve_at_k
    monkeypatch.setattr(fit_module, "solve_at_k", lambda *args: (*real(*args)[:2], False))
    cfg = copy.deepcopy(CFG)
    cfg["pilot"]["n_reps"] = 1
    rows = run_pilot(cfg)
    assert len(rows) == 9 and all(r["status"] == "failed" for r in rows)
    assert all(r["norm_amp_loss"] == FAILED_LOSS and r["half_life_hat"] is None and r["covered"] is None
               for r in rows)
    for s in summarize(rows, 6.0):
        assert (s["n"], s["failed"], s["covered_of"], s["mean_norm_amp_loss"]) == (1, 1, 0, FAILED_LOSS)
        assert s["identified"] + s["open_above"] + s["open_below"] + s["uninformative"] == 0


# ---------- the cap on b_R ----------

def test_a_fit_on_the_cap_is_fitted_again_with_the_wider_cap():
    # One case with a true b_R of 9 (a = 10, f_P = 0.1), above the declared cap of 6, and almost
    # no noise. Every fit ends on the cap, so each one gets a second row with the cap of 12.
    cfg = copy.deepcopy(CFG)
    cfg["cases"] = [dict(a=10.0, f_P=0.1, half_life_weeks=1.0, noise_sd=0.05)]
    cfg["pilot"]["n_reps"] = 1
    rows = run_pilot(cfg)
    assert [r["b_R_cap"] for r in rows] == [6.0, 12.0] * 3
    for at_6, at_12 in zip(rows[0::2], rows[1::2]):
        assert at_6["b_R_hat"] == 6.0 and at_6["on_cap"]
        assert 6.0 < at_12["b_R_hat"] <= 12.0
        # With room up to 12 the fit can reach the true b_R of 9, and the low end of the set moves.
        changed = (at_6["half_life_class"] != at_12["half_life_class"]) or (at_6["set_low"] != at_12["set_low"])
        assert at_6["cap_dependent"] == at_12["cap_dependent"] == changed
    # The summaries only use the declared cap.
    assert all(s["n"] == 1 and s["b_R_on_cap"] == 1 for s in summarize(rows, 6.0))


def test_fits_off_the_cap_have_one_row_and_no_cap_label():
    cfg = copy.deepcopy(CFG)
    cfg["cases"] = [dict(a=4.0, f_P=0.3, half_life_weeks=2.0, noise_sd=0.05)]
    cfg["pilot"]["n_reps"] = 1
    rows = run_pilot(cfg)
    assert [r["b_R_cap"] for r in rows] == [6.0] * 3 and all(r["cap_dependent"] is None for r in rows)


# ---------- summaries ----------

def test_summaries_on_made_up_rows():
    def row(cls, low, high, estimate, covered, cap=6.0, status="ok", loss=1.0, cap_dependent=None):
        return dict(case=0, schedule="uniform", b_R_cap=cap, status=status, half_life_true=2.0,
                    half_life_class=cls, set_low=low, set_high=high, half_life_hat=estimate,
                    covered=covered, norm_amp_loss=loss, cap_dependent=cap_dependent)
    rows = [row("identified", 1.0, 4.0, 4.0, True),                       # error log2(4/2) = 1
            row("identified", 1.0, 8.0, 1.0, False),                      # error |log2(1/2)| = 1
            row("open above", 2.0, 52.0, 30.0, True, cap_dependent=True),
            row("open above", 0.5, 52.0, 30.0, True, cap=12.0, cap_dependent=True),   # the refit: not counted
            row("uninformative", 0.1, 52.0, None, True, cap_dependent=False),
            row("failed", None, None, None, None, status="failed", loss=FAILED_LOSS)]
    (s,) = summarize(rows, 6.0)
    assert (s["n"], s["failed"]) == (5, 1)
    assert (s["identified"], s["open_above"], s["open_below"], s["uninformative"]) == (2, 1, 0, 1)
    assert (s["covered"], s["covered_of"]) == (3, 4)                       # counts, not a percentage
    # Widths in log2: 2, 3, log2(26), log2(520) -> the median is between 3 and log2(26).
    assert s["median_width_log2"] == pytest.approx((3 + np.log2(26)) / 2)
    # The half-life error only looks at the identified fits.
    assert (s["n_identified"], s["median_abs_log2_error"]) == (2, 1.0)
    assert s["mean_norm_amp_loss"] == pytest.approx((4 * 1.0 + FAILED_LOSS) / 5)
    assert (s["b_R_on_cap"], s["cap_dependent"]) == (2, 1)
