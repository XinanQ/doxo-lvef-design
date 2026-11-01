"""Can the data actually pin down the recovery half-life? (profile likelihood on the k grid)

This is the identifiability.py from the proposal's module list. For now it has one
function, profile_interval(). W1 uses it, and the Week 3 fitting code will too.

The idea (Raue et al. 2009): the grid fit already gives us, for every half-life h on
the grid, the best RSS you can get with h held fixed. That's the "profile". With a
known noise SD sigma,

    delta(h) = (RSS(h) - RSS_min) / sigma^2

is a likelihood-ratio statistic, roughly chi-square with 1 df at the true h. So a
grid point with delta <= chi2.ppf(0.95, 1) = 3.84 stays in the (nominal) 95% set,
and anything above that is "ruled out".

What we report depends on whether the data close that set on each side:

    identified        both sides closed, and the set is one piece
    disjoint          both sides closed, but the set has a gap in it
    lower bound only  only the short side closed  (h >= low)
    upper bound only  only the long side closed   (h <= high)
    not identified    neither side closed

A side counts as closed when the first grid point ruled out on that side is
neither of these:
  - The grid's own end (0.1 or 52 weeks). Then the "bound" is just the edge of the
    range we searched.
  - A point whose fit is on a cap (b_R or b_P at 6, or L0 at 50 or 95). Then we
    can't tell whether the data or our search box rules it out, so the side stays
    open. b_R = 0 or b_P = 0 is NOT a cap: amplitudes >= 0 is part of the model.

"identified" is a per-dataset label. It doesn't mean the estimate is accurate, and
the 95% is only nominal (the model is nonlinear in k, there are just 5 points for
4 parameters, and it uses the true simulation sigma).
"""
import numpy as np
from scipy.stats import chi2


def profile_interval(half_lives, rss, sigma, level=0.95, at_cap=None):
    """Nominal profile-likelihood interval for the half-life, on a grid.

    half_lives : 1-D array (n,), strictly increasing, > 0, at least 3 points (weeks)
    rss        : 1-D array (n,), the best RSS with the half-life fixed at each grid point
    sigma      : the known noise SD (> 0), same units as the data
    level      : confidence level, 0 < level < 1 (0.95 gives the cutoff 3.8415)
    at_cap     : optional bool array (n,), True where the fit at that grid point has a
                 coefficient stuck on a cap (see the module docstring). Built by the caller.

    returns a dict:
        status       one of the five labels in the module docstring
        low, high    interval ends in weeks, or None if that side is open. Each one is the
                     first grid point OUTSIDE the set, so the interval comes out a bit wide,
                     never too narrow. Refining the ends is Week 3.
        low_set_by_cap, high_set_by_cap
                     True if the first ruled-out point on that side (not the grid end) has
                     its fit on a cap, so that side counts as open
        disjoint     True if the set has a gap in it
        threshold    the chi-square cutoff that was used
        delta        (rss - rss.min()) / sigma^2 at every grid point
        inside       bool array, delta <= threshold

    Bad input raises a ValueError. Nothing gets sorted or fixed up quietly: decreasing
    half-lives would swap the two ends, and a NaN in rss would make every check meaningless.
    """
    half_lives = np.asarray(half_lives, dtype=float)
    rss = np.asarray(rss, dtype=float)
    n = half_lives.size
    if half_lives.ndim != 1 or n < 3:
        raise ValueError("half_lives should be a 1-D grid with at least 3 points")
    if not (np.all(np.isfinite(half_lives)) and np.all(half_lives > 0) and np.all(np.diff(half_lives) > 0)):
        raise ValueError("half_lives have to be finite, positive and strictly increasing")
    if rss.shape != half_lives.shape or not np.all(np.isfinite(rss)):
        raise ValueError("rss has to be finite and the same shape as half_lives")
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma has to be a finite positive number, got {sigma}")
    if not 0 < level < 1:
        raise ValueError(f"level has to be between 0 and 1, got {level}")
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
