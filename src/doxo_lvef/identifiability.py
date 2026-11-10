"""Can the data actually pin down the recovery half-life? (profile likelihood on the k grid)

This is the identifiability.py from the proposal's module list.

The idea (Raue et al. 2009): the grid fit already gives us, for every half-life h on
the grid, the best RSS you can get with h held fixed. That's the "profile". With a
known noise SD sigma, RSS/sigma^2 is -2 log-likelihood up to a constant, so

    delta(h) = (RSS(h) - RSS_min) / sigma^2

is the profile likelihood-ratio statistic, approximately chi-square with 1 df at the
true h (Wilks' theorem). The 95% profile-likelihood set is every h with
delta <= chi2.ppf(0.95, 1) = 3.84. The 3.84 is asymptotic, so it's only approximate
here: 5 points for 4 parameters, and parameters sitting on bounds (b_P = 0, the b_R
cap). The coverage in the Monte Carlo runs will tell us how good it is.

There are two functions:

profile_set() is the rule from the Nov 8 reply, and fit.py uses it. Every fit keeps
its best point, and the set is classified by whether it reaches the ends of the grid:

    identified      both ends inside the grid
    open above      reaches the top of the grid (52 weeks): the data can't tell the
                    recovery apart from persistence (G_k -> H as k -> 0)
    open below      reaches the bottom of the grid (0.1 weeks)
    uninformative   reaches both ends

The set can have a gap in it when the profile has a shoulder, so it's returned as it
is, with its lowest and highest member and a has_gap flag.

profile_interval() is the Week 1 rule, kept because scripts/W1.py uses it, so the
Week 1 numbers stay a fixed reference. It has two extra rules of mine on top of the
95% set. A side only counts as closed if the first grid point ruled out on that side
is not the grid's own end, and the fit there is not on a cap (b_R or b_P at 6, or L0
at 50 or 95). Its labels are identified, disjoint, lower bound only, upper bound only
and not identified.

Either way, a label is about one dataset. "identified" doesn't mean the estimate is
accurate: the set can still be very wide.
"""
import numpy as np
from scipy.stats import chi2


def _check_profile(half_lives, rss, sigma, level):
    """Turn the inputs into float arrays, or raise a ValueError. Nothing gets sorted or fixed up
    quietly: decreasing half-lives would swap the two ends, and a NaN in rss would make every
    check meaningless."""
    half_lives = np.asarray(half_lives, dtype=float)
    rss = np.asarray(rss, dtype=float)
    if half_lives.ndim != 1 or half_lives.size < 3:
        raise ValueError("half_lives should be a 1-D grid with at least 3 points")
    if not (np.all(np.isfinite(half_lives)) and np.all(half_lives > 0) and np.all(np.diff(half_lives) > 0)):
        raise ValueError("half_lives have to be finite, positive and strictly increasing")
    if rss.shape != half_lives.shape or not np.all(np.isfinite(rss)):
        raise ValueError("rss has to be finite and the same shape as half_lives")
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma has to be a finite positive number, got {sigma}")
    if not 0 < level < 1:
        raise ValueError(f"level has to be between 0 and 1, got {level}")
    return half_lives, rss


def profile_set(half_lives, rss, sigma, level=0.95):
    """The nominal profile-likelihood set for the half-life on a grid, and its class.

    half_lives : 1-D array (n,), strictly increasing, > 0, at least 3 points (weeks)
    rss        : 1-D array (n,), the best RSS with the half-life fixed at each point
    sigma      : the known noise SD (> 0), same units as the data
    level      : confidence level, 0 < level < 1 (0.95 gives the cutoff 3.8415)

    returns a dict:
        class        identified, open above, open below or uninformative (module docstring)
        low, high    the lowest and highest half-life IN the set (weeks). The set's real ends
                     are somewhere between these and the next grid point out.
        has_gap      True if the set isn't one piece
        threshold    the chi-square cutoff that was used
        delta        (rss - rss.min()) / sigma^2 at every point
        inside       bool array, delta <= threshold: the set itself
    """
    half_lives, rss = _check_profile(half_lives, rss, sigma, level)
    threshold = chi2.ppf(level, 1)                  # 3.841458820694124 at 0.95, don't hard-code 3.84
    delta = (rss - rss.min()) / sigma ** 2
    inside = delta <= threshold
    first, last = np.flatnonzero(inside)[[0, -1]]   # the best point is always inside, so this exists
    reaches_bottom, reaches_top = bool(inside[0]), bool(inside[-1])
    label = {(False, False): "identified", (False, True): "open above",
             (True, False): "open below", (True, True): "uninformative"}[reaches_bottom, reaches_top]
    return {"class": label, "low": float(half_lives[first]), "high": float(half_lives[last]),
            "has_gap": bool(not inside[first:last + 1].all()), "threshold": threshold,
            "delta": delta, "inside": inside}


def profile_interval(half_lives, rss, sigma, level=0.95, at_cap=None):
    """The Week 1 version (W1 uses it): nominal profile-likelihood interval on a grid.

    half_lives, rss, sigma, level : as in profile_set
    at_cap     : optional bool array (n,), True where the fit at that grid point has a
                 coefficient stuck on a cap (see the module docstring). Built by the caller.

    returns a dict:
        status       identified, disjoint, lower bound only, upper bound only or not identified
        low, high    interval ends in weeks, or None if that side is open. Each one is the
                     first grid point OUTSIDE the set, so the interval comes out a bit wide,
                     never too narrow.
        low_set_by_cap, high_set_by_cap
                     True if the first ruled-out point on that side (not the grid end) has
                     its fit on a cap, so that side counts as open
        disjoint     True if the set has a gap in it
        threshold, delta, inside   as in profile_set
    """
    half_lives, rss = _check_profile(half_lives, rss, sigma, level)
    n = half_lives.size
    at_cap = np.zeros(n, dtype=bool) if at_cap is None else np.asarray(at_cap)
    if at_cap.shape != half_lives.shape or at_cap.dtype != bool:
        raise ValueError("at_cap has to be a bool array with the same shape as half_lives")

    threshold = chi2.ppf(level, 1)                  # 3.841458820694124 at 0.95, don't hard-code 3.84
    delta = (rss - rss.min()) / sigma ** 2          # the grid min is >= the true min, so this never overstates
    inside = delta <= threshold
    first, last = np.flatnonzero(inside)[[0, -1]]   # the best point is always inside, so this exists
    disjoint = not inside[first:last + 1].all()

    # i and j are the first points ruled out just below and just above the set.
    # A side counts as closed only if that point isn't the grid's own end
    # (index 0 or n - 1) and the fit there isn't stuck on a cap.
    i, j = first - 1, last + 1
    low_set_by_cap = bool(i >= 1 and at_cap[i])
    high_set_by_cap = bool(j <= n - 2 and at_cap[j])
    low = float(half_lives[i]) if i >= 1 and not at_cap[i] else None
    high = float(half_lives[j]) if j <= n - 2 and not at_cap[j] else None

    if low is not None and high is not None:
        status = "disjoint" if disjoint else "identified"
    elif low is not None:
        status = "lower bound only"
    elif high is not None:
        status = "upper bound only"
    else:
        status = "not identified"
    return {"status": status, "low": low, "high": high, "low_set_by_cap": low_set_by_cap,
            "high_set_by_cap": high_set_by_cap, "disjoint": bool(disjoint), "threshold": threshold,
            "delta": delta, "inside": inside}
