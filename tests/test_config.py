"""Checks that configs/pilot.toml and the Week 1 script still agree.

W1 keeps its own settings, simulator and grid fit on purpose, so the Week 1
numbers stay a fixed reference. If someone changes the config or W1 and forgets
the other, these tests should fail.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from doxo_lvef.inputs import dosing_from_config, load_config, visit_times
from doxo_lvef.models import params_from_case, response_basis

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def w1():
    # Load scripts/W1.py as a module without running its main().
    spec = importlib.util.spec_from_file_location("W1", REPO / "scripts" / "W1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visit_convention_is_written_in_the_config(cfg):
    # An Oct 29 follow-up: the convention is written in the config, offset 0 = just before the dose.
    assert cfg["visits"]["offset_weeks"] == 0.0


def test_dosing_matches_w1(cfg, w1):
    d = dosing_from_config(cfg)
    assert np.array_equal(d.times, w1.DOSE_TIMES)
    assert np.allclose(d.q, w1.DOSES)
    assert d.tau == w1.TAU


def test_schedules_cases_and_settings_match_w1(cfg, w1):
    assert cfg["seeds"]["development"] == w1.SEED
    assert cfg["visits"]["offset_weeks"] == w1.VISIT_OFFSET
    assert cfg["schedules"] == w1.SCHEDULES
    cases = [(c["a"], c["f_P"], c["half_life_weeks"], c["noise_sd"]) for c in cfg["cases"]]
    assert cases == [tuple(float(x) for x in c) for c in w1.CASES]
    assert cfg["truth"]["L0"] == w1.L0_TRUE
    assert cfg["fit"]["profile_ci_level"] == w1.PROFILE_CI_LEVEL


def test_fit_bounds_and_grid_match_w1(cfg, w1):
    f = cfg["fit"]
    assert [f["L0_bounds"][0], f["b_R_bounds"][0], f["b_P_bounds"][0]] == list(w1.LOWER)
    assert [f["L0_bounds"][1], f["b_R_bounds"][1], f["b_P_bounds"][1]] == list(w1.UPPER)
    half_lives = np.geomspace(f["half_life_min_weeks"], f["half_life_max_weeks"], f["n_grid"])
    assert np.array_equal(half_lives, w1.HALF_LIVES)
    assert np.array_equal(np.log(2) / half_lives, w1.K_GRID)


def test_package_model_matches_w1(cfg, w1):
    # The Week 2 simulator and W1's basis() use the same overflow-safe formula, so they should
    # agree bit for bit, including far out (week 300) and at huge k, where the old W1 formula
    # used to give NaN.
    d = dosing_from_config(cfg)
    t = np.linspace(0.0, 300.0, 3001)
    for k in list(w1.K_GRID) + [0.0, 4.0, 40.0, 1e3]:
        H_pkg, G_pkg = response_basis(t, k, d)
        H_w1, G_w1 = w1.basis(t, k)
        assert np.array_equal(H_pkg, H_w1) and np.array_equal(G_pkg, G_w1)


def test_params_from_case_matches_w1(cfg, w1):
    for c in cfg["cases"]:
        pkg = params_from_case(c["a"], c["f_P"], c["half_life_weeks"], cfg["truth"]["L0"])
        old = w1.true_params(c["a"], c["f_P"], c["half_life_weeks"])
        assert pkg == pytest.approx(old)


def test_load_config_reads_the_file_you_give_it(tmp_path):
    # pilot.toml happens to use the same values as the function defaults (tau 0.25,
    # interval 1, reference 2), so it can't tell if a field gets ignored. Write a weird
    # little config instead and make sure every number actually makes it through.
    path = tmp_path / "weird.toml"
    path.write_text("[dosing]\ndose_mg_per_kg = 3.0\nn_doses = 3\nfirst_dose_week = 1.0\n"
                    "interval_weeks = 2.0\nreference_mg_per_kg = 1.5\ntau_weeks = 0.5\n")
    d = dosing_from_config(load_config(path))
    assert np.allclose(d.times, [1.0, 3.0, 5.0])
    assert np.allclose(d.q, 2.0)          # 3.0 / 1.5
    assert d.tau == 0.5


def test_every_schedule_is_valid(cfg):
    allowed = set(cfg["visits"]["candidate_weeks"])
    first_dose = dosing_from_config(cfg).times[0]
    for name, schedule in cfg["schedules"].items():
        t = visit_times(schedule, cfg["visits"]["offset_weeks"], first_dose_week=first_dose)
        assert len(t) == 5, f"{name}: should be baseline + four follow-ups"
        assert schedule[0] == cfg["visits"]["baseline_week"], f"{name}: should start at the baseline"
        assert set(schedule[1:]) <= allowed, f"{name}: follow-up not on the candidate grid"


def test_candidate_weeks_match_the_proposal(cfg):
    # Nine weeks, so C(9, 4) = 126 four-visit schedules, the proposal's "126 four-visit combinations".
    assert cfg["visits"]["candidate_weeks"] == [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
