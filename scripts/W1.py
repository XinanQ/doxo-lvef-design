"""Week 1: minimal end-to-end script (simulate -> add noise -> bounded grid fit).

Model (recovery plus persistence):
    dR/dt = b_R * E(t) - k * R,   dP/dt = b_P * E(t),   L(t) = L0 - R(t) - P(t)
Time is in weeks and LVEF in percent. Only the three pilot cases and the three
fixed schedules are used. Run `python scripts/W1.py` from the repo root; results go to outputs/.

Since Week 2 it borrows one tested helper, profile_interval, from the package (src/doxo_lvef).
The model and the grid fit are still its own, so the Week 1 numbers stay a fixed reference.
"""
import csv
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # save figures to file without opening a window
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import lsq_linear

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))   # so we can import the package without installing it

from doxo_lvef.identifiability import profile_interval   # noqa: E402

# ---------------- settings ----------------
# Development seed. We've already looked at results from it, so it must never be reused for the
# final evaluation; those seeds get picked and frozen separately before the final runs.
SEED = 20261019
# All the noise comes from ONE random stream: case c, rep r gets the (c * N_REPS + r)-th block of
# draws on the sorted union of visit weeks. So changing N_REPS, the case list or order, or adding
# a visit week changes the other cases' noise too (editing a case's numbers in place doesn't).
# Fine for this demo; from Week 4 each (case, rep) gets its own stream.
N_REPS = 3          # quick check only; the Week 4 pilot will use 20
TAU = 0.25          # effective input timescale in weeks (assumption, not a PK half-life)
L0_TRUE = 75.0
# Eight weekly doses at weeks 0-7; 1.25 mg/kg normalized by 2 mg/kg gives 0.625.
DOSE_TIMES = np.arange(8.0)
DOSES = np.full(8, 1.25 / 2.0)
# Pilot cases: (a, f_P, recovery half-life in weeks, noise SD)
CASES = [(2, 0.3, 2, 1), (4, 0.7, 8, 1), (1, 0.3, 8, 3)]
# Baseline at week 0 plus four follow-up visits (stored as floats, in weeks).
SCHEDULES = {"uniform": [0.0, 4.0, 10.0, 16.0, 20.0], "treatment": [0.0, 2.0, 6.0, 8.0, 20.0],
             "late": [0.0, 6.0, 12.0, 16.0, 20.0]}
# Visit convention (confirmed in the Oct 29 reply): a visit in a dose week is taken just BEFORE
# that week's dose, i.e. at the trough. H and G_k are continuous at a dose time, so "just
# before" and "at t_j" give the same value. VISIT_OFFSET (weeks) moves follow-up visits later;
# it matters a lot, since with tau = 0.25 one day after a dose already adds ~44% of its H term.
VISIT_OFFSET = 0.0
# Bounds for (L0, b_R, b_P). They're just a search box for the synthetic experiment, not biology.
NAMES = ("L0", "b_R", "b_P")
LOWER = np.array([50.0, 0.0, 0.0])
UPPER = np.array([95.0, 6.0, 6.0])
# 60 log-spaced half-lives from 0.1 to 52 weeks, and the matching k values (fast -> slow).
HALF_LIVES = np.geomspace(0.1, 52.0, 60)
K_GRID = np.log(2) / HALF_LIVES
# A half-life is only reported when the data rule out both faster and slower recovery, i.e. the
# nominal 95% profile interval is closed on both sides and is one piece (chi-square 1 df; see
# identifiability.py).
PROFILE_CI_LEVEL = 0.95
OUT_DIR = REPO / "outputs"


# ---------------- model ----------------
def basis(t, k):
    """Return H(t) and G_k(t), so that L(t) = L0 - b_R * G_k(t) - b_P * H(t).

    Each dose q_j at t_j adds, for s = t - t_j >= 0 (nothing before the dose):
        H += q_j (1 - exp(-s/tau)),   G_k += q_j (exp(-k s) - exp(-s/tau)) / (1 - k tau).
    G_k is 0/0 at k tau = 1. With m = min(k, 1/tau) and d = |1/tau - k| it equals
    q_j exp(-m s) (1 - exp(-d s)) / (tau d), and (1 - exp(-d s))/d -> s as d -> 0, which is the
    proposal's k tau = 1 formula. Pulling out the slower rate m keeps every exponent <= 0, so
    nothing overflows. It's the same overflow-safe form as models.response_basis (since Week 2).
    """
    k = float(k)
    # Rows are times, columns are doses; broadcasting handles all doses at once.
    s = np.asarray(t, dtype=float)[:, None] - DOSE_TIMES[None, :]
    given = s >= 0                          # a dose only acts after it is given (s = 0 adds exactly 0)
    s = np.where(given, s, 0.0)
    m, d = min(k, 1.0 / TAU), abs(1.0 / TAU - k)
    ratio = s if d < 1e-10 else -np.expm1(-d * s) / d     # expm1 stays accurate for small d
    H = (DOSES * given * -np.expm1(-s / TAU)).sum(axis=1)
    G = (DOSES * given * np.exp(-m * s) * ratio / TAU).sum(axis=1)
    return H, G


def lvef(t, L0, b_R, b_P, k):
    H, G = basis(t, k)
    return L0 - b_R * G - b_P * H


def true_params(a, f_P, half_life):
    """b_R = a(1 - f_P) and b_P = a f_P, as in the proposal."""
    return dict(L0=L0_TRUE, b_R=a * (1 - f_P), b_P=a * f_P, k=np.log(2) / half_life)


# ---------------- fitting ----------------
def fit_grid(t, y, sigma):
    """For fixed k the model is linear in (L0, b_R, b_P): solve a bounded linear least-squares
    problem at every k on the grid and keep the smallest RSS. The whole RSS profile is returned.
    sigma is the noise SD, which we know in simulations. The fit itself doesn't use it; it's only
    used to decide what we can say about the half-life."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError(f"sigma has to be a finite positive number, got {sigma}")
    if t.ndim != 1 or t.shape != y.shape or not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("t and y have to be 1-D, the same length, and free of NaN/inf")

    def solve(k):
        H, G = basis(t, k)
        X = np.column_stack([np.ones(len(t)), -G, -H])      # columns: L0, b_R, b_P
        # bvls's default max_iter is the number of unknowns (3 here), and it can stop right at
        # the optimum and still report success=False. 50 is plenty and changes no numbers.
        return lsq_linear(X, y, bounds=(LOWER, UPPER), method="bvls", max_iter=50)

    sols = [solve(k) for k in K_GRID]
    # There's no "failed" label until the Week 3 fitting code, so for now a failed solve stops
    # the run instead of quietly getting a normal-looking label.
    if any(sol.status <= 0 or not np.isfinite(sol.cost) for sol in sols):
        raise RuntimeError("lsq_linear failed at some k on the grid, stopping instead of guessing")
    rss = np.array([2.0 * sol.cost for sol in sols])        # sol.cost is RSS / 2
    coefs = np.array([sol.x for sol in sols])               # (60, 3): best L0, b_R, b_P at every k
    best = int(np.argmin(rss))
    coef = coefs[best]

    # Report problems instead of hiding them.
    tol = 1e-9 * (UPPER - LOWER)
    hit = (coef <= LOWER + tol) | (coef >= UPPER - tol)
    ci = profile_interval(HALF_LIVES, rss, sigma, PROFILE_CI_LEVEL, stuck_on_a_cap(coefs))
    status, low, high = ci["status"], ci["low"], ci["high"]
    grid_raw, at_edge = HALF_LIVES[best], best in (0, len(K_GRID) - 1)
    # We don't refit without the cap, so all the note can say is that the point is on a cap,
    # not that the cap is the ONLY thing ruling it out (see identifiability.py).
    capped = "side counted open (first ruled-out point is on a cap)"
    notes = [text for text, on in [(f"short {capped}", ci["low_set_by_cap"]),
                                   (f"long {capped}", ci["high_set_by_cap"]),
                                   ("95% set has a gap", ci["disjoint"])] if on]
    if coef[1] <= tol[1]:
        # b_R = 0 takes k out of the model: (L0, 0, b_P) gets the same RSS at every k, so the profile
        # is flat by construction and argmin's index 0 is just a tie-break, not a real grid edge.
        status, low, high, grid_raw, at_edge = "not identified", None, None, None, False
        notes = ["b_R = 0, so k drops out of the model"]
    return {**dict(zip(NAMES, coef)), "k": K_GRID[best], "rss": rss[best],
            "half_life_status": status,
            "half_life_hat": HALF_LIVES[best] if status == "identified" else None,
            "half_life_ci_low": low, "half_life_ci_high": high, "half_life_grid_raw": grid_raw,
            "profile_range": np.ptp(rss) / sigma ** 2,       # diagnostic only, the labels come from ci
            "k_at_grid_edge": at_edge, "coef_at_bound": " ".join(n for n, h in zip(NAMES, hit) if h),
            "notes": "; ".join(notes), "profile": rss}


def stuck_on_a_cap(coefs):
    """For each row of coefs (L0, b_R, b_P), True if b_R or b_P sits on its upper bound or L0 on
    either bound. b_R = 0 / b_P = 0 don't count: amplitudes >= 0 is part of the model, while the
    6 and [50, 95] are just our search box (proposal: numerical bounds, not biology)."""
    tol = 1e-9 * (UPPER - LOWER)
    return ((coefs[:, 1:] >= UPPER[1:] - tol[1:]).any(axis=1)
            | (coefs[:, 0] <= LOWER[0] + tol[0]) | (coefs[:, 0] >= UPPER[0] - tol[0]))


def half_life_text(r):
    """How a fitted half-life gets written in the table and the plot titles."""
    if r["half_life_status"] == "identified":
        return f"{r['half_life_hat']:.2f} [{r['half_life_ci_low']:.3g}, {r['half_life_ci_high']:.3g}]"
    if r["half_life_status"] == "lower bound only":
        return f">= {r['half_life_ci_low']:.3g}"
    if r["half_life_status"] == "upper bound only":
        return f"<= {r['half_life_ci_high']:.3g}"
    return "--"


# ---------------- main ----------------
def main():
    rng = np.random.default_rng(SEED)
    # Shift follow-up visits by VISIT_OFFSET; the baseline stays before the first dose.
    visits = {name: np.array(ts) + np.r_[0.0, np.full(len(ts) - 1, VISIT_OFFSET)]
              for name, ts in SCHEDULES.items()}
    # Noise is drawn once on all visit times and shared by the three schedules,
    # so the schedules are compared on the same noise (paired comparison).
    union = np.unique(np.concatenate(list(visits.values())))     # sorted, no repeats

    rows, examples = [], {}
    for case_id, (a, f_P, half_life, sigma) in enumerate(CASES):
        p = true_params(a, f_P, half_life)
        clean = lvef(union, **p)
        for rep in range(N_REPS):
            noisy = clean + rng.normal(0.0, sigma, union.size)
            for name, times in visits.items():
                idx = np.searchsorted(union, times)
                t, y = union[idx], noisy[idx]
                fit = fit_grid(t, y, sigma)
                # Normalized squared amplitude error (the proposal's ranking objective). It's kept for
                # every fit, identified or not, because the ranking averages over all replicates.
                loss = np.mean([((fit[b] - p[b]) / a) ** 2 for b in ("b_R", "b_P")])
                rows.append(dict(case=case_id, a=a, f_P=f_P, half_life_true=half_life, sigma=sigma, rep=rep,
                                 schedule=name, L0_hat=fit["L0"],
                                 b_R_true=p["b_R"], b_R_hat=fit["b_R"], b_P_true=p["b_P"], b_P_hat=fit["b_P"],
                                 # half_life_hat is only filled in when the data pin the half-life down
                                 # from both sides. half_life_grid_raw is the grid's best point, kept so
                                 # nothing is hidden, but it is NOT an estimate, so don't average it.
                                 **{key: fit[key] for key in ("half_life_status", "half_life_hat",
                                                              "half_life_ci_low", "half_life_ci_high",
                                                              "half_life_grid_raw")},
                                 rss=fit["rss"], norm_amp_loss=loss,
                                 **{key: fit[key] for key in ("profile_range", "k_at_grid_edge",
                                                              "coef_at_bound", "notes")}))
                if rep == 0:
                    examples[case_id, name] = (p, t, y, fit)

    save_csv(rows)
    print_summary(rows)
    plot(examples)
    print(f"Saved {len(rows)} fits and the figure in {OUT_DIR}")


def save_csv(rows):
    OUT_DIR.mkdir(exist_ok=True)
    # Plain "\n" line endings and utf-8, so the file comes out the same on Windows and Mac.
    with open(OUT_DIR / "W1_fits.csv", "w", newline="", encoding="utf-8") as f:
        f.write(f"# seed={SEED} n_reps={N_REPS} tau={TAU} visit_offset={VISIT_OFFSET} "
                f"profile_ci_level={PROFILE_CI_LEVEL} ci_ends=outer_grid_bracket\n")
        # Versions get their own line: the regression test skips it, so a different patch version
        # doesn't fail it for no reason, but we can still see what made the file.
        f.write(f"# env: python={platform.python_version()} numpy={np.__version__} "
                f"scipy={scipy.__version__}\n")
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: f"{v:.6g}" if isinstance(v, float) else v for key, v in row.items()} for row in rows)


def print_summary(rows):
    """Print replicate 0 only."""
    print("case  schedule   b_R true/fit   b_P true/fit   half-life true / fit [95% CI]   loss   flags")
    for r in filter(lambda r: r["rep"] == 0, rows):
        flags = [label for label, on in [(r["half_life_status"], half_life_text(r) == "--"),
                                         ("best k at grid edge", r["k_at_grid_edge"]),
                                         (f"at bound: {r['coef_at_bound']}", r["coef_at_bound"]),
                                         (r["notes"], r["notes"])] if on]
        print(f"{r['case']:>4}  {r['schedule']:<9}  {r['b_R_true']:5.2f} / {r['b_R_hat']:<5.2f}"
              f"  {r['b_P_true']:5.2f} / {r['b_P_hat']:<5.2f}  {r['half_life_true']:6.2f} / "
              f"{half_life_text(r):<20}  {r['norm_amp_loss']:6.3f}  {', '.join(flags)}")


def plot(examples):
    # Draw the curves out to the last visit even if VISIT_OFFSET pushes it past week 20.
    t_dense = np.linspace(0, max(20.0, max(t[-1] for _, t, _, _ in examples.values())), 401)
    names = list(SCHEDULES)
    fig, axes = plt.subplots(len(CASES), len(names), figsize=(12, 9), sharex=True, sharey="row")
    for (case_id, name), (p, t, y, fit) in examples.items():
        ax = axes[case_id, names.index(name)]
        ax.axvspan(DOSE_TIMES[0], DOSE_TIMES[-1], color="0.92", lw=0)       # dosing period
        ax.plot(t_dense, lvef(t_dense, **p), "k-", lw=1.5, label="truth")
        ax.plot(t_dense, lvef(t_dense, *(fit[n] for n in NAMES), fit["k"]), "C1--", lw=1.5, label="fit")
        ax.plot(t, y, "C0o", label="simulated data")
        hl = half_life_text(fit)
        if hl == "--":
            hl = fit["half_life_status"] + (" (b_R = 0)" if fit["notes"].startswith("b_R = 0") else "")
        if fit["k_at_grid_edge"]:
            hl += ", best k at grid edge"       # the dashed fit curve is drawn at that edge k
        ax.set_title(f"case {case_id} {CASES[case_id]}, {name}\nfitted half-life (wk): {hl}", fontsize=9)
    plt.setp(axes[-1], xlabel="weeks")
    plt.setp(axes[:, 0], ylabel="LVEF (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"W1 fits, replicate 0 (seed {SEED}); shading marks dosing weeks 0-7")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "W1_fits.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
