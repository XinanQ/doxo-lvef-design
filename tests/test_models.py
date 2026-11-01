"""Unit tests for the baseline simulator (models.py).

The two the proposal asks for in Week 2 are the first two: zero dose and zero
amplitudes. The rest are cheap checks I added because each one would catch a
specific mistake I could see myself making (wrong sign, off-by-one dose, the
k*tau = 1 limit, the visit convention...).

Run with:  python -m pytest
"""
import numpy as np
import pytest

from doxo_lvef.inputs import Dosing, effective_input, no_dosing, weekly_dosing
from doxo_lvef.models import params_from_case, response_basis, simulate_lvef, simulate_states

# The pilot dosing: 8 weekly doses of 1.25 mg/kg -> q = 0.625 each, total 5.
PILOT = weekly_dosing(1.25, 8)
T = np.linspace(0.0, 40.0, 801)        # 0 to 40 weeks, every 0.05 weeks


# ---------- the two tests the proposal asks for ----------

@pytest.mark.parametrize("dosing", [no_dosing(), weekly_dosing(0.0, 8)], ids=["no doses", "eight 0 mg/kg doses"])
def test_zero_dose_gives_flat_baseline(dosing):
    # No treatment -> no deficit -> LVEF should sit exactly at L0 the whole time.
    # Checked two ways: an empty schedule, and the normal schedule with 0 mg/kg each time.
    L = simulate_lvef(T, L0=75.0, b_R=3.0, b_P=2.0, k=0.3, dosing=dosing)
    assert np.array_equal(L, np.full(T.shape, 75.0))


def test_zero_amplitudes_give_flat_baseline():
    # Treatment happens, but b_R = b_P = 0 means it does nothing to the heart.
    L = simulate_lvef(T, L0=75.0, b_R=0.0, b_P=0.0, k=0.3, dosing=PILOT)
    assert np.array_equal(L, np.full(T.shape, 75.0))


# ---------- shapes (Week 2 also asks us to document these) ----------

def test_output_shapes_match_input_times():
    H, G = response_basis(T, 0.3, PILOT)
    states = simulate_states(T, 75.0, 1.4, 0.6, 0.3, PILOT)
    assert H.shape == G.shape == T.shape
    assert all(v.shape == T.shape for v in states.values())


def test_scalar_and_list_times_work():
    # A single float should come back as a length-1 array, and lists should work too.
    assert simulate_lvef(10.0, 75.0, 1.4, 0.6, 0.3, PILOT).shape == (1,)
    assert simulate_lvef([0, 4, 10], 75.0, 1.4, 0.6, 0.3, PILOT).shape == (3,)


@pytest.mark.parametrize("bad_t, message", [(np.zeros((5, 1)), "1-D"), ([1.0, np.nan], "NaN"), ([np.inf], "inf")],
                         ids=["2-D", "NaN", "inf"])
def test_weird_times_are_rejected(bad_t, message):
    # A (5, 1) column of times would quietly broadcast into a (5, 8) answer without the check,
    # and NaN would look like "before every dose" (so L = L0). Both should just be errors.
    with pytest.raises(ValueError, match=message):
        response_basis(bad_t, 0.3, PILOT)


def test_k_has_to_be_one_plain_number():
    # A float32 k (say, from an optimizer) should give exactly the float64 answer,
    # and an array of k's should be a clear error.
    _, G32 = response_basis(T, np.float32(0.3), PILOT)
    _, G64 = response_basis(T, float(np.float32(0.3)), PILOT)
    assert np.array_equal(G32, G64)
    with pytest.raises(ValueError):
        response_basis(T, np.array([0.3, 0.5]), PILOT)


# ---------- basic behaviour ----------

def test_nothing_happens_before_the_first_dose():
    dosing = weekly_dosing(1.25, 8, first_dose_week=2.0)
    H, G = response_basis(np.array([0.0, 1.0, 1.999]), 0.3, dosing)
    assert np.all(H == 0) and np.all(G == 0)


def test_doses_far_in_the_future_dont_overflow():
    # Doses at week 300+ and we look at weeks 0 and 10: exp(+300/tau) would overflow
    # if the future doses weren't zeroed out first. Overflow/NaN warnings become errors here.
    late = weekly_dosing(1.25, 8, first_dose_week=300.0)
    with np.errstate(over="raise", invalid="raise"):   # underflow to 0 is fine
        H, G = response_basis([0.0, 10.0], 0.3, late)
    assert np.all(H == 0) and np.all(G == 0)


def test_G_matches_the_textbook_formula_for_one_dose():
    # Everything else checks G at special k values (0, tiny, 1/tau). This one checks
    # ordinary k's against the formula written out by hand, including k*tau > 1.
    one_dose = Dosing(times=[0.0], q=[0.625], tau=0.25)
    s = np.array([0.1, 0.5, 1.0, 3.0, 10.0])
    for k in (0.3, 1.0, 10.0):
        _, G = response_basis(s, k, one_dose)
        expected = 0.625 * (np.exp(-k * s) - np.exp(-s / 0.25)) / (1 - k * 0.25)
        assert G == pytest.approx(expected, rel=1e-12)


def test_ode_is_satisfied_between_doses():
    # Another independent check: away from the dose times, the closed form should
    # actually solve dH/dt = E and dG/dt = E - k*G. Use a central difference.
    k, h = 0.3, 1e-5
    t = np.array([0.5, 2.3, 6.7, 7.5, 12.0])      # not on any dose time
    H_up, G_up = response_basis(t + h, k, PILOT)
    H_dn, G_dn = response_basis(t - h, k, PILOT)
    _, G = response_basis(t, k, PILOT)
    E = effective_input(t, PILOT)
    assert (H_up - H_dn) / (2 * h) == pytest.approx(E, rel=1e-6, abs=1e-8)
    assert (G_up - G_dn) / (2 * h) == pytest.approx(E - k * G, rel=1e-6, abs=1e-8)


def test_states_add_up_to_lvef():
    s = simulate_states(T, 75.0, 1.4, 0.6, 0.3, PILOT)
    assert np.allclose(s["L"], 75.0 - s["R"] - s["P"])
    assert np.allclose(s["L"], simulate_lvef(T, 75.0, 1.4, 0.6, 0.3, PILOT))


def test_R_and_P_are_not_swapped():
    # Long after dosing with a 2-week half-life, the recovering part R is basically gone
    # and the persistent part P is b_P * total dose = 0.6 * 5 = 3. (Adding them up, like the
    # test above does, can't tell if the two got swapped.)
    s = simulate_states([60.0], 75.0, 1.4, 0.6, np.log(2) / 2, PILOT)
    assert s["R"][0] < 1e-6
    assert s["P"][0] == pytest.approx(3.0)


def test_deficits_are_never_negative():
    s = simulate_states(T, 75.0, 1.4, 0.6, 0.3, PILOT)
    assert np.all(s["R"] >= 0) and np.all(s["P"] >= 0)


def test_H_ends_up_at_the_total_dose():
    # Long after the last dose, H should be the sum of all doses (5 for the pilot).
    H, _ = response_basis([60.0], 0.3, PILOT)
    assert H[0] == pytest.approx(PILOT.total, rel=1e-12)


def test_persistent_only_plateau():
    # b_R = 0: LVEF should settle at L0 - b_P * total dose and stay there.
    L = simulate_lvef([30.0, 40.0], 75.0, 0.0, 2.0, 0.3, PILOT)
    assert L == pytest.approx([75.0 - 2.0 * 5.0] * 2, abs=1e-9)


def test_recovery_only_goes_back_to_baseline():
    # b_P = 0 with a 2-week half-life: after 40 weeks basically everything has recovered.
    L = simulate_lvef([40.0], 75.0, 3.0, 0.0, np.log(2) / 2, PILOT)
    assert L[0] == pytest.approx(75.0, abs=1e-3)


def test_proposal_figure2_mixed_plateau_is_69():
    # Proposal Fig. 2: a = 4, f_P = 0.3, half-life 2 weeks -> plateau at 69%.
    p = params_from_case(4.0, 0.3, 2.0)
    L = simulate_lvef([60.0], dosing=PILOT, **p)
    assert L[0] == pytest.approx(69.0, abs=1e-6)


def test_more_dose_means_lower_lvef():
    weak, strong = weekly_dosing(1.0, 8), weekly_dosing(2.0, 8)
    L_weak = simulate_lvef(T, 75.0, 1.4, 0.6, 0.3, weak)
    L_strong = simulate_lvef(T, 75.0, 1.4, 0.6, 0.3, strong)
    assert np.all(L_strong <= L_weak)


# ---------- the tricky limits ----------

def test_k_zero_makes_G_equal_H():
    # With no recovery at all, the "recovering" part behaves just like the persistent part.
    H, G = response_basis(T, 0.0, PILOT)
    assert np.allclose(G, H, atol=1e-12)


def test_small_k_approaches_H():
    # As k -> 0, G_k -> H. This is why slow recovery looks like persistence.
    H, G = response_basis(T, 1e-8, PILOT)
    assert np.allclose(G, H, atol=1e-6)


@pytest.mark.parametrize("eps", [-1e-8, -1e-9, 1e-9, 1e-8])
def test_repeated_root_is_continuous(eps):
    # k * tau = 1 is the 0/0 case. Step just off it on BOTH sides (these go through the
    # expm1 branch, not the limit branch) and compare with the value right at it.
    # Near the root G only changes by about 0.05 * |eps|, so I allow 0.1 * |eps|.
    # The naive formula is off by about 2e-8 at |eps| = 1e-8 and about 2e-7 at 1e-9,
    # which is way more than that.
    k_star = 1.0 / PILOT.tau
    _, G_at = response_basis(T, k_star, PILOT)
    _, G_near = response_basis(T, k_star + eps, PILOT)
    assert np.max(np.abs(G_near - G_at)) <= 0.1 * abs(eps)


def test_repeated_root_matches_proposal_limit_formula():
    # Proposal: at k*tau = 1 one dose contributes q * (s/tau) * exp(-s/tau).
    one_dose = Dosing(times=[0.0], q=[0.625], tau=0.25)
    s = np.array([0.1, 0.25, 1.0, 3.0])
    _, G = response_basis(s, 4.0, one_dose)
    assert G == pytest.approx(0.625 * (s / 0.25) * np.exp(-s / 0.25), rel=1e-12)


@pytest.mark.parametrize("k", [40.0, np.log(2) / 0.02, 1e3])
def test_huge_k_stays_finite(k):
    # An earlier version gave NaN here: for k > 1/tau one exp() overflowed to inf and got
    # multiplied by another one that had underflowed to 0. Look far out to be sure.
    t = np.linspace(0.0, 300.0, 3001)
    with np.errstate(over="raise", invalid="raise"):   # underflow to 0 is fine
        H, G = response_basis(t, k, PILOT)
    assert np.all(np.isfinite(G)) and np.all(G >= 0)


@pytest.mark.parametrize("bad_k", [-0.1, np.nan, np.inf])
def test_bad_k_is_rejected(bad_k):
    with pytest.raises(ValueError):
        response_basis(T, bad_k, PILOT)


def test_params_from_case_directly():
    # Check the conversion itself, with a non-default L0 so a hard-coded 75 would get caught.
    p = params_from_case(4.0, 0.3, 2.0, L0=60.0)
    assert p == pytest.approx({"L0": 60.0, "b_R": 2.8, "b_P": 1.2, "k": np.log(2) / 2.0})


@pytest.mark.parametrize("a, f_P, half_life", [(4.0, -0.1, 2.0), (4.0, 1.5, 2.0),     # f_P outside [0, 1]
                                               (-1.0, 0.3, 2.0), (np.nan, 0.3, 2.0),   # bad amplitude
                                               (np.inf, 0.3, 2.0),
                                               (4.0, 0.3, 0.0), (4.0, 0.3, -2.0), (4.0, 0.3, np.nan),
                                               (4.0, 0.3, np.inf)])
def test_params_from_case_rejects_bad_cases(a, f_P, half_life):
    # a = -1 would give negative amplitudes (LVEF going UP after doxorubicin), and half-life 0
    # would give k = inf with just a warning. An infinite a or half-life isn't a real case either
    # (no recovery at all is written as f_P = 1). Better to stop right at the typo.
    with pytest.raises(ValueError):
        params_from_case(a, f_P, half_life)


# ---------- visit convention (Prof. Wei, Oct 29) ----------

def test_a_dose_adds_nothing_at_the_exact_dose_time():
    # "Just before the dose" and "exactly at t_j" should give the same value.
    for k in (0.0, 0.3, 4.0):
        H_at, G_at = response_basis([6.0], k, PILOT)
        H_before, G_before = response_basis([6.0 - 1e-9], k, PILOT)
        assert H_at[0] == pytest.approx(H_before[0], abs=1e-8)
        assert G_at[0] == pytest.approx(G_before[0], abs=1e-8)


def test_one_day_after_a_dose_adds_about_44_percent():
    # Prof. Wei's number: with tau = 0.25, one day later adds ~44% of that dose's H term.
    # With a single dose it's exactly 1 - exp(-(1/7)/0.25) = 0.435.
    one_dose = Dosing(times=[0.0], q=[1.0], tau=0.25)
    H_day, _ = response_basis([1 / 7], 0.3, one_dose)
    assert H_day[0] == pytest.approx(1 - np.exp(-(1 / 7) / 0.25), rel=1e-12)
    # In the real pilot schedule the earlier doses' tails add a tiny bit more (~0.44).
    H_at, _ = response_basis([6.0], 0.3, PILOT)
    H_later, _ = response_basis([6.0 + 1 / 7], 0.3, PILOT)
    assert 0.43 < (H_later[0] - H_at[0]) / 0.625 < 0.45
