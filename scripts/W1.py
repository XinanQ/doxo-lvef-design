"""Week 1: minimal end-to-end script (simulate -> add noise -> bounded grid fit).

Model (recovery plus persistence):
    dR/dt = b_R * E(t) - k * R,   dP/dt = b_P * E(t),   L(t) = L0 - R(t) - P(t)
Time is in weeks and LVEF in percent. Only the three pilot cases and the three
fixed schedules are used. Run `python scripts/W1.py` from the repo root; results go to outputs/.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                   # save figures to file without opening a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear

# ---------------- settings ----------------
SEED = 20261019
N_REPS = 3          # quick check only; the Week 4 pilot will use 20
TAU = 0.25          # effective input timescale in weeks (assumption, not a PK half-life)
L0_TRUE = 75.0
# Eight weekly doses at weeks 0-7; 1.25 mg/kg normalized by 2 mg/kg gives 0.625.
DOSE_TIMES = np.arange(8.0)
DOSES = np.full(8, 1.25 / 2.0)
# Pilot cases: (a, f_P, recovery half-life in weeks, noise SD)
CASES = [(2, 0.3, 2, 1), (4, 0.7, 8, 1), (1, 0.3, 8, 3)]
# Baseline at week 0 plus four follow-up visits.
SCHEDULES = {"uniform": [0, 4, 10, 16, 20], "treatment": [0, 2, 6, 8, 20], "late": [0, 6, 12, 16, 20]}
# Bounds for (L0, b_R, b_P), and a log-spaced grid of 60 half-lives (0.1 to 52 weeks) for k.
NAMES = ("L0", "b_R", "b_P")
LOWER = np.array([50.0, 0.0, 0.0])
UPPER = np.array([95.0, 6.0, 6.0])
K_GRID = np.log(2) / np.geomspace(0.1, 52.0, 60)
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"    # repo-level outputs/


# ---------------- model ----------------
def basis(t, k):
    """Return H(t) and G_k(t), so that L(t) = L0 - b_R * G_k(t) - b_P * H(t).

    Each dose q_j at t_j adds, for s = t - t_j >= 0 (nothing before the dose):
        H += q_j (1 - exp(-s/tau)),   G_k += q_j (exp(-k s) - exp(-s/tau)) / (1 - k tau).
    G_k is 0/0 at k tau = 1. With d = 1/tau - k it equals q_j exp(-k s) (1 - exp(-d s)) / (tau d),
    and (1 - exp(-d s))/d -> s as d -> 0, which is the proposal's k tau = 1 formula.
    """
    # Rows are times, columns are doses; broadcasting handles all doses at once.
    s = np.asarray(t, dtype=float)[:, None] - DOSE_TIMES[None, :]
    given = s >= 0                          # a dose only acts after it is given
    s = np.where(given, s, 0.0)
    d = 1.0 / TAU - k
    ratio = s if abs(d) < 1e-10 else -np.expm1(-d * s) / d     # expm1 stays accurate for small d
    H = (DOSES * given * -np.expm1(-s / TAU)).sum(axis=1)
    G = (DOSES * given * np.exp(-k * s) * ratio / TAU).sum(axis=1)
    return H, G


def lvef(t, L0, b_R, b_P, k):
    H, G = basis(t, k)
    return L0 - b_R * G - b_P * H


def true_params(a, f_P, half_life):
    """b_R = a(1 - f_P) and b_P = a f_P, as in the proposal."""
    return dict(L0=L0_TRUE, b_R=a * (1 - f_P), b_P=a * f_P, k=np.log(2) / half_life)


# ---------------- fitting ----------------
def fit_grid(t, y):
    """For fixed k the model is linear in (L0, b_R, b_P): solve a bounded linear least-squares
    problem at every k on the grid and keep the smallest RSS. The whole RSS profile is returned."""
    def solve(k):
        H, G = basis(t, k)
        X = np.column_stack([np.ones(len(t)), -G, -H])      # columns: L0, b_R, b_P
        return lsq_linear(X, y, bounds=(LOWER, UPPER), method="bvls")

    sols = [solve(k) for k in K_GRID]
    rss = np.array([2.0 * sol.cost for sol in sols])        # sol.cost is RSS / 2
    best = int(np.argmin(rss))
    coef = sols[best].x

    # Report problems instead of hiding them.
    tol = 1e-9 * (UPPER - LOWER)
    hit = (coef <= LOWER + tol) | (coef >= UPPER - tol)
    return {**dict(zip(NAMES, coef)), "k": K_GRID[best], "rss": rss[best],
            "k_unidentifiable": coef[1] <= tol[1],          # data do not depend on k when b_R = 0
            "k_at_grid_edge": best in (0, len(K_GRID) - 1),
            "coef_at_bound": " ".join(n for n, h in zip(NAMES, hit) if h), "profile": rss}


# ---------------- main ----------------
def main():
    rng = np.random.default_rng(SEED)
    # Noise is drawn once on all visit times and shared by the three schedules,
    # so the schedules are compared on the same noise (paired comparison).
    union = np.unique(np.concatenate(list(SCHEDULES.values()))).astype(float)   # sorted, no repeats

    rows, examples = [], {}
    for case_id, (a, f_P, half_life, sigma) in enumerate(CASES):
        p = true_params(a, f_P, half_life)
        clean = lvef(union, **p)
        for rep in range(N_REPS):
            noisy = clean + rng.normal(0.0, sigma, union.size)
            for name, times in SCHEDULES.items():
                idx = np.searchsorted(union, times)
                t, y = union[idx], noisy[idx]
                fit = fit_grid(t, y)
                # Normalized squared amplitude error (the proposal's ranking objective).
                loss = np.mean([((fit[b] - p[b]) / a) ** 2 for b in ("b_R", "b_P")])
                rows.append(dict(case=case_id, a=a, f_P=f_P, half_life=half_life, sigma=sigma, rep=rep,
                                 schedule=name, L0_hat=fit["L0"],
                                 b_R_true=p["b_R"], b_R_hat=fit["b_R"], b_P_true=p["b_P"], b_P_hat=fit["b_P"],
                                 half_life_hat=np.log(2) / fit["k"], rss=fit["rss"], norm_amp_loss=loss,
                                 **{key: fit[key] for key in ("k_unidentifiable", "k_at_grid_edge", "coef_at_bound")}))
                if rep == 0:
                    examples[case_id, name] = (p, t, y, fit)

    save_csv(rows)
    print_summary(rows)
    plot(examples)
    print(f"Saved {len(rows)} fits and the figure in {OUT_DIR}")


def save_csv(rows):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "W1_fits.csv", "w", newline="") as f:
        f.write(f"# seed={SEED} n_reps={N_REPS} tau={TAU} numpy={np.__version__}\n")  # for reproducibility
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: f"{v:.6g}" if isinstance(v, float) else v for key, v in row.items()} for row in rows)


def print_summary(rows):
    """Print replicate 0 only."""
    print("case  schedule   b_R true/fit   b_P true/fit   half-life true/fit   loss   flags")
    for r in filter(lambda r: r["rep"] == 0, rows):
        flags = [label for label, on in [("k-unidentifiable", r["k_unidentifiable"]),
                                         ("k-at-grid-edge", r["k_at_grid_edge"]),
                                         (f"at-bound: {r['coef_at_bound']}", r["coef_at_bound"])] if on]
        print(f"{r['case']:>4}  {r['schedule']:<9}  {r['b_R_true']:5.2f} / {r['b_R_hat']:<5.2f}"
              f"  {r['b_P_true']:5.2f} / {r['b_P_hat']:<5.2f}  {r['half_life']:6.2f} / {r['half_life_hat']:<6.2f}"
              f"       {r['norm_amp_loss']:6.3f}  {', '.join(flags)}")


def plot(examples):
    t_dense = np.linspace(0, 20, 401)
    names = list(SCHEDULES)
    fig, axes = plt.subplots(len(CASES), len(names), figsize=(12, 9), sharex=True, sharey="row")
    for (case_id, name), (p, t, y, fit) in examples.items():
        ax = axes[case_id, names.index(name)]
        ax.axvspan(DOSE_TIMES[0], DOSE_TIMES[-1], color="0.92", lw=0)       # dosing period
        ax.plot(t_dense, lvef(t_dense, **p), "k-", lw=1.5, label="truth")
        ax.plot(t_dense, lvef(t_dense, *(fit[n] for n in NAMES), fit["k"]), "C1--", lw=1.5, label="fit")
        ax.plot(t, y, "C0o", label="observed")
        ax.set_title(f"case {case_id} {CASES[case_id]}, {name}", fontsize=9)
    plt.setp(axes[-1], xlabel="weeks")
    plt.setp(axes[:, 0], ylabel="LVEF (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"W1 fits, replicate 0 (seed {SEED}); shading marks dosing weeks 0-7")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "W1_fits.png", dpi=120)


if __name__ == "__main__":
    main()
