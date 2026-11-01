"""Unit tests for inputs.py: dosing, E(t), visit times."""
import numpy as np
import pytest
from scipy.integrate import quad

from doxo_lvef.inputs import Dosing, effective_input, no_dosing, visit_times, weekly_dosing


def test_pilot_doses_are_normalized_by_2_mg_per_kg():
    d = weekly_dosing(1.25, 8)
    assert np.allclose(d.q, 0.625)
    assert np.allclose(d.times, np.arange(8.0))
    assert d.total == pytest.approx(5.0)


def test_no_dosing_has_zero_doses():
    d = no_dosing()
    assert d.times.shape == (0,) and d.total == 0.0


@pytest.mark.parametrize("bad", [
    dict(times=[0.0, 1.0], q=[0.5], tau=0.25),          # lengths don't match
    dict(times=[1.0, 0.0], q=[0.5, 0.5], tau=0.25),     # not sorted
    dict(times=[0.0], q=[-0.5], tau=0.25),              # negative dose
    dict(times=[0.0], q=[0.5], tau=0.0),                # tau has to be > 0
    dict(times=[np.nan], q=[0.5], tau=0.25),            # NaN would pass "sorted" and ">= 0" checks
    dict(times=[0.0], q=[np.nan], tau=0.25),
    dict(times=[0.0], q=[0.5], tau=np.inf),
    dict(times=[[0.0, 1.0]], q=[[0.5, 0.5]], tau=0.25),   # same shape, but 2-D
])
def test_bad_dosing_is_rejected(bad):
    with pytest.raises(ValueError):
        Dosing(**bad)


def test_dosing_keeps_its_own_read_only_copy():
    # Changing the array you passed in afterwards shouldn't change the Dosing,
    # and you shouldn't be able to edit d.times / d.q directly either.
    times = np.array([0.0, 1.0, 2.0])
    d = Dosing(times=times, q=np.array([0.5, 0.5, 0.5]), tau=0.25)
    times[0] = 99.0
    assert d.times[0] == 0.0
    with pytest.raises(ValueError):
        d.q[0] = -1.0
    with pytest.raises(ValueError):
        d.times[0] = 5.0


def test_effective_input_is_zero_before_first_dose():
    d = weekly_dosing(1.25, 8, first_dose_week=1.0)
    assert np.all(effective_input([0.0, 0.5, 0.999], d) == 0)


def test_E_with_doses_far_in_the_future_doesnt_overflow():
    # Same idea as the models.py test: exp(+300/tau) would overflow if the future doses
    # weren't zeroed out first. The Week 3 solve_ivp code will call E a lot, so pin it here too.
    late = weekly_dosing(1.25, 8, first_dose_week=300.0)
    with np.errstate(over="raise", invalid="raise"):   # underflow to 0 is fine
        assert np.array_equal(effective_input([0.0, 10.0], late), [0.0, 0.0])


def test_E_counts_the_dose_at_exactly_its_time():
    # Right at t_j the dose is already "on", so E jumps by q/tau = 0.625/0.25 = 2.5 there.
    # (H and G don't care, since they add exactly 0 at s = 0, but E does, and the
    # Week 3 solve_ivp code will rely on this when it starts each piece at t_j.)
    d = weekly_dosing(1.25, 8)
    just_before = effective_input([2.0 - 1e-12], d)[0]
    at_dose = effective_input([2.0], d)[0]
    assert at_dose - just_before == pytest.approx(0.625 / 0.25, rel=1e-6)


def test_area_under_E_is_the_total_dose():
    # Each bump has area q_j, so integrating E over a long time should give the total dose.
    # E jumps up at every dose (right at t_j the new dose is already counted), so I
    # integrate one dose interval at a time with scipy's quad and stop just before the
    # next dose. The last piece runs out to week 40, where E is basically 0 (e^-132).
    d = weekly_dosing(1.25, 8)
    ends = list(d.times[1:]) + [40.0]
    area = sum(quad(lambda x: effective_input(x, d)[0], a, np.nextafter(b, a), limit=200)[0]
               for a, b in zip(d.times, ends))
    assert area == pytest.approx(d.total, rel=1e-9)


def test_visit_offset_only_moves_followups():
    t = visit_times([0, 2, 6, 8, 20], offset_weeks=1 / 7)
    assert t[0] == 0.0
    assert np.allclose(t[1:], np.array([2, 6, 8, 20]) + 1 / 7)
    assert t.dtype == float


def test_default_visit_offset_is_zero():
    assert np.array_equal(visit_times([0, 4, 10, 16, 20]), [0.0, 4.0, 10.0, 16.0, 20.0])


def test_visit_times_doesnt_change_your_array():
    # Without the .copy() inside, calling it twice on a numpy array would shift the visits twice.
    schedule = np.array([0.0, 2.0, 6.0, 8.0, 20.0])
    first = visit_times(schedule, offset_weeks=1 / 7)
    second = visit_times(schedule, offset_weeks=1 / 7)
    assert np.array_equal(schedule, [0.0, 2.0, 6.0, 8.0, 20.0])
    assert np.array_equal(first, second)


def test_baseline_after_first_dose_is_rejected():
    with pytest.raises(ValueError):
        visit_times([1.0, 4.0, 10.0], first_dose_week=0.0)


@pytest.mark.parametrize("schedule, offset", [([0.0, 6.0, 4.0], 0.0), ([0.0, 4.0, 4.0, 10.0, 20.0], 0.0),
                                              ([0.0, np.nan, 4.0], 0.0), ([], 0.0), ([0.0, 4.0, 10.0], np.nan)],
                         ids=["decreasing", "duplicate", "NaN", "empty", "NaN offset"])
def test_bad_visit_schedules_are_rejected(schedule, offset):
    with pytest.raises(ValueError):
        visit_times(schedule, offset_weeks=offset)
