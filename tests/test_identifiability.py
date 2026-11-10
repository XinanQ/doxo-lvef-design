"""Unit tests for identifiability.py: profile_set (the Nov 8 classes, used by fit.py)
and profile_interval (the Week 1 labels, used by W1).

These use made-up profiles where we know the answer, so they don't depend on the
model or the fit at all. The checks on real fits are in test_w1_fit.py and test_fit.py.
"""
import numpy as np
import pytest
from scipy.stats import chi2

from doxo_lvef.identifiability import profile_interval, profile_set

HL = np.geomspace(0.1, 52.0, 60)        # same grid as W1
N = HL.size
THR = chi2.ppf(0.95, 1)


def u_shape(center=3.0, width=10.0):
    """A clean U-shaped profile in log half-life with its bottom near `center`."""
    return width * (np.log(HL) - np.log(center)) ** 2


def check_ends(out, rss, sigma=1.0):
    # The ends should be the first grid points OUTSIDE the set: ruled out themselves,
    # with everything strictly between them inside.
    delta = (rss - rss.min()) / sigma ** 2
    lo, hi = HL.tolist().index(out["low"]), HL.tolist().index(out["high"])
    assert delta[lo] > THR and delta[hi] > THR
    assert np.all(delta[lo + 1:hi] <= THR)


def test_threshold_comes_from_chi_square():
    out = profile_interval(HL, u_shape(), 1.0)
    assert out["threshold"] == pytest.approx(3.841458820694124, rel=1e-12)     # not a hard-coded 3.84
    assert profile_interval(HL, u_shape(), 1.0, level=0.99)["threshold"] == pytest.approx(6.634896601021213)


def test_u_shape_is_identified_with_outer_grid_ends():
    rss = u_shape()
    out = profile_interval(HL, rss, 1.0)
    assert out["status"] == "identified"
    assert out["low"] < 3.0 < out["high"]
    check_ends(out, rss)
    assert not (out["low_set_by_cap"] or out["high_set_by_cap"] or out["disjoint"])


def test_profile_falling_all_the_way_to_52_weeks_gives_a_lower_bound_only():
    # Best fit at the slow end: short half-lives get ruled out, long ones can't be.
    out = profile_interval(HL, u_shape(center=52.0), 1.0)
    assert out["status"] == "lower bound only"
    assert out["low"] is not None and out["high"] is None


def test_profile_rising_from_0_1_weeks_gives_an_upper_bound_only():
    out = profile_interval(HL, u_shape(center=0.1), 1.0)
    assert out["status"] == "upper bound only"
    assert out["low"] is None and out["high"] is not None


def test_flat_profile_is_not_identified():
    out = profile_interval(HL, np.full(N, 5.0), 1.0)
    assert out["status"] == "not identified"
    assert out["low"] is None and out["high"] is None and out["inside"].all()


@pytest.mark.parametrize("side", ["low", "high"])
def test_ruling_out_only_the_grid_end_doesnt_count(side):
    # If the only point ruled out is the grid's own end (0.1 or 52 weeks), the "bound"
    # would just be the edge of the range we searched, so that side stays open.
    end = 0 if side == "low" else -1
    rss = np.zeros(N)
    rss[end] = 10.0
    assert profile_interval(HL, rss, 1.0)["status"] == "not identified"
    # Same data, but the fit at the grid end is on a cap too. It's still the grid end keeping
    # that side open, so no cap flag. (Separate call on purpose: with the cap set in the first
    # call, that check would pass even if the grid-end rule were broken.)
    at_cap = np.zeros(N, dtype=bool)
    at_cap[end] = True
    out = profile_interval(HL, rss, 1.0, at_cap=at_cap)
    assert out["status"] == "not identified"
    assert not (out["low_set_by_cap"] or out["high_set_by_cap"])
    # Rule out one more point and it does count.
    rss[1 if side == "low" else -2] = 10.0
    out = profile_interval(HL, rss, 1.0)
    assert out["status"] == ("lower bound only" if side == "low" else "upper bound only")
    assert out[side] == (HL[1] if side == "low" else HL[-2])


@pytest.mark.parametrize("side", ["low", "high"])
def test_a_capped_first_ruled_out_point_counts_as_open(side):
    rss = u_shape()
    i = HL.tolist().index(profile_interval(HL, rss, 1.0)[side])     # first ruled-out point on that side
    at_cap = np.zeros(N, dtype=bool)
    at_cap[i] = True
    out = profile_interval(HL, rss, 1.0, at_cap=at_cap)
    other = "high" if side == "low" else "low"
    assert out["status"] == ("upper bound only" if side == "low" else "lower bound only")
    assert out[side] is None and out[other] is not None
    assert out[f"{side}_set_by_cap"] and not out[f"{other}_set_by_cap"]


@pytest.mark.parametrize("side", ["low", "high"])
def test_a_cap_somewhere_else_changes_nothing(side):
    rss = u_shape()
    plain = profile_interval(HL, rss, 1.0)
    i = HL.tolist().index(plain[side])
    at_cap = np.zeros(N, dtype=bool)
    at_cap[[i - 1, i + 1, int(np.argmin(rss))]] = True     # one further out, one inside, and the best point
    out = profile_interval(HL, rss, 1.0, at_cap=at_cap)
    assert (out["status"], out["low"], out["high"]) == (plain["status"], plain["low"], plain["high"])
    assert not (out["low_set_by_cap"] or out["high_set_by_cap"])


def test_w_shape_with_a_big_hump_is_disjoint():
    # Two dips at about 1 and 10 weeks; the hump between them is ruled out.
    rss = np.minimum(u_shape(center=1.0, width=40.0), u_shape(center=10.0, width=40.0))
    out = profile_interval(HL, rss, 1.0)
    assert out["disjoint"] and out["status"] == "disjoint"
    # Same shape, but the hump between the dips is squashed under the cutoff: one piece again.
    between = (HL > 1.0) & (HL < 10.0)
    rss[between] = np.minimum(rss[between], THR - 0.1)
    out = profile_interval(HL, rss, 1.0)
    assert not out["disjoint"] and out["status"] == "identified"


def test_a_point_exactly_at_the_cutoff_is_inside():
    rss = np.full(N, 10.0)
    rss[30], rss[29], rss[31] = 0.0, THR, THR
    out = profile_interval(HL, rss, 1.0)
    assert (out["low"], out["high"]) == (HL[28], HL[32])
    rss[29] = THR * (1 + 1e-12)                       # a hair over the cutoff -> ruled out
    assert profile_interval(HL, rss, 1.0)["low"] == HL[29]


def test_shifting_or_rescaling_the_profile_changes_nothing():
    # Only differences in RSS matter, measured in units of sigma^2.
    rss = u_shape()
    base = profile_interval(HL, rss, 1.0)
    for out in (profile_interval(HL, rss + 100.0, 1.0), profile_interval(HL, 4.0 * rss, 2.0)):
        assert (out["status"], out["low"], out["high"]) == (base["status"], base["low"], base["high"])


@pytest.mark.parametrize("kwargs", [
    dict(rss=np.r_[np.nan, np.ones(N - 1)]),
    dict(rss=np.r_[np.inf, np.ones(N - 1)]),
    dict(sigma=0.0), dict(sigma=-1.0), dict(sigma=np.nan),
    dict(level=0.0), dict(level=1.0), dict(level=1.5),
    dict(rss=np.ones(N - 1)),                                  # wrong length
    dict(half_lives=HL[::-1]),                                 # decreasing would swap the ends
    dict(half_lives=np.r_[0.0, HL[1:]]),                       # a half-life of 0 isn't a half-life
    dict(half_lives=np.r_[HL[:-1], np.inf]),
    dict(half_lives=HL[:2], rss=np.ones(2)),                   # too short to have an inside and two sides
    # a column instead of a 1-D grid, with everything else shaped to match so no other check catches it
    dict(half_lives=HL[:, None], rss=u_shape()[:, None], at_cap=np.zeros((N, 1), dtype=bool)),
    dict(at_cap=np.zeros(N - 1, dtype=bool)),
    dict(at_cap=np.zeros(N)),                                  # floats, not bools
], ids=["NaN rss", "inf rss", "sigma 0", "sigma < 0", "sigma NaN", "level 0", "level 1", "level 1.5",
        "length", "decreasing", "zero half-life", "inf half-life", "two points", "2-D grid",
        "at_cap shape", "at_cap floats"])
def test_bad_input_is_rejected(kwargs):
    args = dict(half_lives=HL, rss=u_shape(), sigma=1.0) | kwargs
    with pytest.raises(ValueError):
        profile_interval(**args)


# ---------- profile_set: the four classes from the Nov 8 reply ----------

@pytest.mark.parametrize("center, label", [(3.0, "identified"), (52.0, "open above"), (0.1, "open below")])
def test_profile_set_class_depends_on_which_grid_ends_the_set_reaches(center, label):
    rss = u_shape(center=center)
    out = profile_set(HL, rss, 1.0)
    assert out["class"] == label and not out["has_gap"]
    # low and high are the lowest and highest grid points IN the set, everything between is in too.
    delta = (rss - rss.min()) / 1.0
    lo, hi = HL.tolist().index(out["low"]), HL.tolist().index(out["high"])
    assert np.all(delta[lo:hi + 1] <= THR) and np.array_equal(out["inside"], delta <= THR)
    assert lo == 0 or delta[lo - 1] > THR
    assert hi == N - 1 or delta[hi + 1] > THR


def test_flat_profile_is_uninformative():
    # b_R = 0 gives this: the same RSS at every half-life, so the set is the whole grid.
    out = profile_set(HL, np.full(N, 5.0), 1.0)
    assert out["class"] == "uninformative" and (out["low"], out["high"]) == (HL[0], HL[-1])


def test_profile_set_keeps_a_gap():
    # Two dips at about 1 and 10 weeks with a hump between them that's ruled out: the set has a gap.
    # The class only looks at the grid ends, so it's still identified, and the gap gets its flag.
    rss = np.minimum(u_shape(center=1.0, width=40.0), u_shape(center=10.0, width=40.0))
    out = profile_set(HL, rss, 1.0)
    assert out["class"] == "identified" and out["has_gap"]
    between = (HL > out["low"]) & (HL < out["high"])
    assert not out["inside"][between].all()


def test_the_cutoff_and_the_grid_end_rules_are_the_only_differences():
    # The Week 1 rule calls a side open when its first ruled-out point is the grid end, even though the
    # grid end itself is ruled out. profile_set only asks whether the set reaches the end.
    rss = np.full(N, 10.0)
    rss[1:-1] = 0.0                                   # everything except the two grid ends is in the set
    assert profile_set(HL, rss, 1.0)["class"] == "identified"
    assert profile_interval(HL, rss, 1.0)["status"] == "not identified"


@pytest.mark.parametrize("kwargs", [dict(rss=np.r_[np.nan, np.ones(N - 1)]), dict(sigma=0.0), dict(level=1.0),
                                    dict(half_lives=HL[::-1]), dict(rss=np.ones(N - 1))],
                         ids=["NaN rss", "sigma 0", "level 1", "decreasing", "length"])
def test_profile_set_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        profile_set(**(dict(half_lives=HL, rss=u_shape(), sigma=1.0) | kwargs))
