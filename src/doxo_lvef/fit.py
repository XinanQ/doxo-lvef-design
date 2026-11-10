"""The bounded fitting interface (Week 3): fit_lvef().

With k fixed, L = L0 - b_R G_k - b_P H is linear in (L0, b_R, b_P). So a fit goes
over the half-life grid, solves a bounded linear least-squares problem at each k
(lsq_linear with method "bvls") and keeps the RSS at every k (variable projection).
RSS against half-life is the profile. That part is the same grid fit as
scripts/W1.py. On top of it (Oct 29 and Nov 8 replies):

  - refine=True refines an interior best point in log k between its two grid
    neighbours (minimize_scalar with method="bounded", lsq_linear inside) and keeps
    whichever RSS is lower. Fits at the bound (the grid end, or a coefficient on a
    cap) or with b_R near zero aren't refined. The plain grid (refine=False) is the
    default, since the pilot comparison, the schedule search and the Monte Carlo
    summaries use it.
  - Any other local minimum within 2 units of RSS/sigma^2 of the best one is kept
    too (near_optimal), and refined with refine=True.
  - The best point always comes with its 95% profile-likelihood set and the set's
    class (identifiability.profile_set), the bound hits and the fitted nadir depth.
  - Failed fits get their own status, since the proposal keeps them in the
    denominator (loss 100).
"""
import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar

from doxo_lvef.identifiability import profile_set
from doxo_lvef.inputs import check_times
from doxo_lvef.models import response_basis, simulate_lvef

NAMES = ("L0", "b_R", "b_P")
# Oct 29 reply: a second local minimum within about 2 units of RSS/sigma^2 gets refined and kept too.
NEAR_OPTIMAL_DELTA = 2.0
# Neighbouring profile values closer than this (in units of sigma^2) count as tied (see local_minima).
TIE_TOL = 1e-9
# A coefficient within 1e-9 of its bound's range counts as on the bound (the same tolerance as W1).
BOUND_TOL = 1e-9
# The nadir is looked for on this many evenly spaced times between the first and last visit.
NADIR_POINTS = 4001


def fit_settings(cfg):
    """The half-life grid, the bounds and the profile level from the [fit] section of the config.

    returns a dict to pass straight on:  fit_lvef(t, y, dosing, sigma, **fit_settings(cfg))
    """
    f = cfg["fit"]
    bounds = np.array([f["L0_bounds"], f["b_R_bounds"], f["b_P_bounds"]], dtype=float)
    return {"half_lives": np.geomspace(f["half_life_min_weeks"], f["half_life_max_weeks"], f["n_grid"]),
            "lower": bounds[:, 0], "upper": bounds[:, 1], "level": float(f["profile_ci_level"])}


def solve_at_k(t, y, k, dosing, lower, upper):
    """Best (L0, b_R, b_P) with k held fixed: one bounded linear least-squares problem.

    returns (coefs, rss, ok). coefs are clipped into the bounds, and rss is worked out
    from the clipped coefs. ok is False if lsq_linear failed (status <= 0) or the RSS
    isn't finite.
    """
    H, G = response_basis(t, k, dosing)
    X = np.column_stack([np.ones(t.size), -G, -H])          # columns: L0, b_R, b_P
    # bvls's default max_iter is the number of unknowns (3 here), and it can stop right at the
    # optimum and still report success=False. 50 is plenty (same as W1).
    sol = lsq_linear(X, y, bounds=(lower, upper), method="bvls", max_iter=50)
    coefs = np.clip(sol.x, lower, upper)
    rss = float(np.sum((y - X @ coefs) ** 2))
    return coefs, rss, bool(sol.status > 0 and np.isfinite(rss))


def local_minima(delta, tie_tol=TIE_TOL):
    """Grid indices of the local minima of a profile, in grid order.

    A run of tied neighbours (within tie_tol) counts as one stretch, and a stretch is a
    minimum if the values on both sides of it are higher; it's reported by its lowest
    point. So a completely flat profile (b_R = 0) has exactly one minimum. A grid end
    counts as a minimum if its one neighbour is higher.
    """
    delta = np.asarray(delta, dtype=float)
    runs = np.split(np.arange(delta.size), np.flatnonzero(np.abs(np.diff(delta)) > tie_tol) + 1)
    minima = []
    for r, run in enumerate(runs):
        lowest = delta[run].min()
        left = delta[runs[r - 1][-1]] if r > 0 else np.inf
        right = delta[runs[r + 1][0]] if r + 1 < len(runs) else np.inf
        if left > lowest and right > lowest:
            minima.append(int(run[np.argmin(delta[run])]))
    return minima


def nadir_depth(L0, b_R, b_P, k, dosing, t_from, t_to):
    """How far the curve gets below L0 between t_from and t_to (Nov 8 reply): L0 minus the
    lowest point of the curve, looked for on NADIR_POINTS evenly spaced times."""
    t = np.linspace(t_from, t_to, NADIR_POINTS)
    return float(L0 - simulate_lvef(t, L0, b_R, b_P, k, dosing).min())


def fit_lvef(t, y, dosing, sigma, *, half_lives, lower, upper, level=0.95, refine=False):
    """Fit L0, b_R, b_P and k to LVEF data y at times t.

    t, y       : 1-D arrays, same length (weeks, pp)
    dosing     : inputs.Dosing
    sigma      : the noise SD (> 0). The fit itself doesn't use it; it's only for the
                 95% set and the "within 2 units of RSS/sigma^2" rule.
    half_lives, lower, upper, level : the grid and bounds, usually **fit_settings(cfg).
                 lower and upper are (L0, b_R, b_P).
    refine     : False = plain grid (the default, for the pilot). True = refine the
                 minimum (and any near-optimal one), for reported numbers.

    returns a dict:
        status          "ok" or "failed"
        L0, b_R, b_P, k, rss
                        the best fit (refined if refine=True and it was allowed). NaN if failed.
        half_life       ln 2 / k at the best fit: the point estimate. Never report it without
                        the class and set below, and never average it. None if b_R = 0 (then k
                        isn't in the model; k is still filled in, so the curve can be drawn).
        half_life_grid  the best grid point, before any refinement (None if b_R = 0)
        refined         True if the best fit came from the refinement
        half_life_class identified, open above, open below or uninformative ("failed" if failed)
        set_low, set_high
                        the lowest and highest half-life in the 95% set (weeks)
        set_has_gap     True if the 95% set isn't one piece
        nadir_depth     L0 minus the lowest point of the fitted curve between the first and
                        last visit
        at_grid_end     True if the best half-life is at the grid's end (a k boundary hit)
        on_cap          True if the best fit has b_R or b_P on its upper bound or L0 on either
                        bound (a bound hit)
        coef_at_bound   names of the coefficients on any bound, e.g. "b_P" (b_P = 0 counts here)
        notes           the bound hits and anything else worth knowing, "; "-separated
        near_optimal    list of the local minima within 2 units of RSS/sigma^2 of the best,
                        best first: dicts with grid_index, half_life, k, L0, b_R, b_P, rss,
                        delta = (rss - best rss) / sigma^2 and refined
        profile         dict of arrays over the grid: half_lives, rss, coefs (n, 3), at_cap,
                        solver_ok, delta and in_set (True where the grid point is in the 95% set)
    """
    t = check_times(t)
    y = np.asarray(y, dtype=float)
    if y.shape != t.shape or not np.all(np.isfinite(y)):
        raise ValueError("y has to be 1-D, the same length as t, and free of NaN/inf")
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma has to be a finite positive number, got {sigma}")
    half_lives = np.asarray(half_lives, dtype=float)
    if (half_lives.ndim != 1 or half_lives.size < 3 or not np.all(np.isfinite(half_lives))
            or not np.all(half_lives > 0) or not np.all(np.diff(half_lives) > 0)):
        raise ValueError("half_lives should be a strictly increasing 1-D grid of at least 3 positive numbers")
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(lower < upper):
        raise ValueError("lower and upper should be 3 numbers each (L0, b_R, b_P) with lower < upper")

    # ---- the plain grid ----
    last = half_lives.size - 1
    k_grid = np.log(2) / half_lives                  # fast -> slow, so k goes DOWN along the grid
    solved = [solve_at_k(t, y, k, dosing, lower, upper) for k in k_grid]
    coefs = np.array([c for c, _, _ in solved])
    rss = np.array([r for _, r, _ in solved])
    solver_ok = np.array([ok for _, _, ok in solved])
    tol = BOUND_TOL * (upper - lower)

    def capped(c):
        # Names of the coefficients on a cap: b_R or b_P on its upper bound, or L0 on either bound.
        # b_R = 0 or b_P = 0 isn't a cap: amplitudes >= 0 is part of the model, while 6 and [50, 95]
        # are just our search box.
        return [name for name, value, lo, hi, tol_ in zip(NAMES, c, lower, upper, tol)
                if value >= hi - tol_ or (name == "L0" and value <= lo + tol_)]

    profile = {"half_lives": half_lives, "rss": rss, "coefs": coefs,
               "at_cap": np.array([bool(capped(c)) for c in coefs]), "solver_ok": solver_ok}
    if not solver_ok.all():
        nan = float("nan")
        return {"status": "failed", "L0": nan, "b_R": nan, "b_P": nan, "k": nan, "rss": nan,
                "half_life": nan, "half_life_grid": nan, "refined": False, "half_life_class": "failed",
                "set_low": None, "set_high": None, "set_has_gap": False, "nadir_depth": nan,
                "at_grid_end": False, "on_cap": False, "coef_at_bound": "",
                "notes": f"lsq_linear failed at {np.sum(~solver_ok)} grid point(s)",
                "near_optimal": [], "profile": profile}

    # ---- the best point and any near-optimal ones ----
    delta = (rss - rss.min()) / sigma ** 2
    grid_best = int(np.argmin(rss))
    candidates = [i for i in local_minima(delta) if delta[i] <= NEAR_OPTIMAL_DELTA]

    def refine_between(i):
        # Bounded 1-D search in log k between the two grid neighbours of point i.
        def objective(log_k):
            _, r, ok = solve_at_k(t, y, np.exp(log_k), dosing, lower, upper)
            return r if ok else np.inf
        # xatol is in log k, so it's about 1e-8 relative in the half-life.
        opt = minimize_scalar(objective, bounds=(np.log(k_grid[i + 1]), np.log(k_grid[i - 1])),
                              method="bounded", options={"xatol": 1e-8})
        k = float(np.exp(opt.x))
        c, r, ok = solve_at_k(t, y, k, dosing, lower, upper)
        return k, c, r, ok

    near_optimal = []
    for i in candidates:
        entry = {"grid_index": i, "half_life": float(half_lives[i]), "k": float(k_grid[i]),
                 **dict(zip(NAMES, coefs[i].tolist())), "rss": float(rss[i]), "refined": False}
        # Not at the bound (grid end or a cap) and not b_R near zero: only then refine.
        if refine and 0 < i < last and abs(coefs[i, 1]) > tol[1] and not profile["at_cap"][i]:
            k, c, r, ok = refine_between(i)
            if ok and r < rss[i]:                     # keep whichever is lower
                entry.update(half_life=float(np.log(2) / k), k=k, **dict(zip(NAMES, c.tolist())),
                             rss=r, refined=True)
        near_optimal.append(entry)
    near_optimal.sort(key=lambda e: e["rss"])
    for entry in near_optimal:
        entry["delta"] = (entry["rss"] - near_optimal[0]["rss"]) / sigma ** 2
    best = near_optimal[0]

    # ---- the 95% set and its class ----
    # With refinement, the refined points go into the profile too, so the rise is measured from
    # the refined minimum (and a very narrow set still has its best point in it).
    extra = [e for e in near_optimal if e["refined"]]
    hl_all = np.append(half_lives, [e["half_life"] for e in extra])
    rss_all = np.append(rss, [e["rss"] for e in extra])
    order = np.argsort(hl_all)
    pset = profile_set(hl_all[order], rss_all[order], sigma, level)
    profile["delta"] = (rss - rss_all.min()) / sigma ** 2
    profile["in_set"] = profile["delta"] <= pset["threshold"]

    # ---- bound hits, nadir depth, notes ----
    best_coefs = np.array([best[name] for name in NAMES])
    b_R_zero = bool(abs(best["b_R"]) <= tol[1])
    # With b_R = 0 the profile is flat by construction and the argmin is just a tie-break,
    # so it's not a real grid edge.
    at_grid_end = best["grid_index"] in (0, last) and not b_R_zero
    caps = capped(best_coefs)
    depth = nadir_depth(*best_coefs, best["k"], dosing, t.min(), t.max())
    notes = []
    if b_R_zero:
        notes.append("b_R = 0, so k drops out of the model")
    if at_grid_end:
        notes.append("bound hit: half-life at the grid end")
    notes += [f"bound hit: {name} at {best[name]:g}" for name in caps]
    if pset["has_gap"]:
        notes.append(f"{level:.0%} set has a gap")
    hit = (best_coefs <= lower + tol) | (best_coefs >= upper - tol)
    return {"status": "ok", **{name: best[name] for name in NAMES}, "k": best["k"], "rss": best["rss"],
            "half_life": None if b_R_zero else best["half_life"],
            "half_life_grid": None if b_R_zero else float(half_lives[grid_best]),
            "refined": best["refined"], "half_life_class": pset["class"],
            "set_low": pset["low"], "set_high": pset["high"], "set_has_gap": pset["has_gap"],
            "nadir_depth": depth, "at_grid_end": at_grid_end, "on_cap": bool(caps),
            "coef_at_bound": " ".join(name for name, h in zip(NAMES, hit) if h),
            "notes": "; ".join(notes), "near_optimal": near_optimal, "profile": profile}
