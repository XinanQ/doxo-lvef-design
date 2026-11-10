"""Week 3: check the simulator two ways, and reproduce one known-truth fit.

1. Analytical vs numerical. The closed form (models.simulate_states) against the
   solve_ivp path (models.simulate_states_ode) for the pilot dosing, at every
   half-life on the fit grid plus k*tau = 1 and k = 0. The times are a fine grid
   from 0 to 40 weeks plus every dose time and every candidate visit week, so the
   visits in dose weeks (the s = 0 case) are in there. They should agree to TOL.
2. One known-truth fit. Noiseless data from a = 4, f_P = 0.3, half-life 2 weeks
   (the proposal's Figure 2 numbers) on the treatment schedule, fitted with
   fit.fit_lvef on the plain grid and with refinement. We know the truth, so we
   can see whether the fit gets it back, nadir depth included. sigma = 0.25 only
   goes into the 95% set (this case is identified at that sigma on all three
   schedules).

Run from the repo root:  python scripts/W3.py   (prints the numbers, saves outputs/W3_check.png)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # just save the figure, don't open a window
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))   # so we can import the package without installing it

from doxo_lvef.fit import fit_lvef, fit_settings, nadir_depth, solve_at_k   # noqa: E402
from doxo_lvef.inputs import dosing_from_config, load_config, visit_times   # noqa: E402
from doxo_lvef.models import params_from_case, simulate_lvef, simulate_states, simulate_states_ode   # noqa: E402

OUT_DIR = REPO / "outputs"
# Agreement tolerance for the two paths, in LVEF percentage points. Way below the noise SDs we
# use (1 and 3 pp in the pilot), so anything that passes can't change a fit.
TOL = 1e-6
A, F_P, HALF_LIFE = 4.0, 0.3, 2.0     # the known truth
SCHEDULE = "treatment"                # visits 0, 2, 6, 8, 20; weeks 2 and 6 are dose weeks
SIGMA = 0.25


def check_paths(cfg, dosing, settings, p):
    """Largest |closed form - solve_ivp| in LVEF, over all the k values we check."""
    t = np.unique(np.concatenate([np.linspace(0.0, 40.0, 801), dosing.times,
                                  cfg["visits"]["candidate_weeks"]]))
    ks = [*(np.log(2) / settings["half_lives"]), 1.0 / dosing.tau, 0.0]
    worst, worst_k = 0.0, None
    for k in ks:
        closed = simulate_states(t, p["L0"], p["b_R"], p["b_P"], k, dosing)["L"]
        numerical = simulate_states_ode(t, p["L0"], p["b_R"], p["b_P"], k, dosing)["L"]
        diff = np.max(np.abs(closed - numerical))
        if diff > worst:
            worst, worst_k = diff, k
    # The visits that fall exactly on a dose time (the s = 0 case).
    dose_weeks = np.intersect1d(cfg["visits"]["candidate_weeks"], dosing.times)
    at_doses = max(np.max(np.abs(simulate_states(dose_weeks, p["L0"], p["b_R"], p["b_P"], k, dosing)["L"]
                                 - simulate_states_ode(dose_weeks, p["L0"], p["b_R"], p["b_P"], k, dosing)["L"]))
                   for k in ks)
    print(f"Closed form vs solve_ivp (pilot dosing, b_R = {p['b_R']:g}, b_P = {p['b_P']:g}, "
          f"{len(ks)} values of k, {t.size} times):")
    half_life = "k = 0" if worst_k == 0 else f"half-life {np.log(2) / worst_k:.3g} wk"
    verdict = "they agree" if worst < TOL else "they DON'T agree"
    print(f"  largest |difference| in LVEF: {worst:.1e} pp (at {half_life}); tolerance {TOL:g} pp, so {verdict}")
    print(f"  at the visits on dose times (weeks {', '.join(f'{w:g}' for w in dose_weeks)}): {at_doses:.1e} pp")
    return worst, at_doses


def main():
    cfg = load_config()
    dosing = dosing_from_config(cfg)
    settings = fit_settings(cfg)
    p = params_from_case(A, F_P, HALF_LIFE, cfg["truth"]["L0"])

    # 1. the two paths
    check_paths(cfg, dosing, settings, p)

    # 2. the known-truth fit
    t = visit_times(cfg["schedules"][SCHEDULE], cfg["visits"]["offset_weeks"], dosing.times[0])
    y = simulate_lvef(t, dosing=dosing, **p)
    grid = fit_lvef(t, y, dosing, SIGMA, **settings)
    refined = fit_lvef(t, y, dosing, SIGMA, refine=True, **settings)
    step = np.log(settings["half_lives"][1] / settings["half_lives"][0])
    true_depth = nadir_depth(p["L0"], p["b_R"], p["b_P"], p["k"], dosing, t.min(), t.max())
    print(f"\nKnown-truth fit: noiseless, a = {A:g}, f_P = {F_P:g}, half-life {HALF_LIFE:g} wk, "
          f"{SCHEDULE} schedule (visits {', '.join(f'{v:g}' for v in t)})")
    print("            L0       b_R      b_P      half-life   nadir depth   RSS")
    print(f"  truth     {p['L0']:<8.4f} {p['b_R']:<8.4f} {p['b_P']:<8.4f} {HALF_LIFE:<11.4f} {true_depth:<13.4f} -")
    for name, r in (("grid", grid), ("refined", refined)):
        print(f"  {name:<9} {r['L0']:<8.4f} {r['b_R']:<8.4f} {r['b_P']:<8.4f} {r['half_life']:<11.4f} "
              f"{r['nadir_depth']:<13.4f} {r['rss']:.1e}")
    print(f"  refinement moved the half-life {np.log(refined['half_life'] / grid['half_life']) / step:.2f} "
          f"grid steps; other local minima within 2 units of RSS/sigma^2: {len(refined['near_optimal']) - 1}")
    gap = "has a gap" if refined["set_has_gap"] else "one piece"
    print(f"  {settings['level']:.0%} set (sigma = {SIGMA:g}): {refined['set_low']:.3g} to {refined['set_high']:.3g} wk "
          f"(grid points), {gap}, so the class is {refined['half_life_class']}; "
          f"bound hits: {'yes' if refined['at_grid_end'] or refined['on_cap'] else 'none'}")

    plot(dosing, settings, p, t, y, grid, refined)
    print(f"Saved {OUT_DIR / 'W3_check.png'}")


def plot(dosing, settings, p, t, y, grid, refined):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: how far apart the two paths are over time, for a few recovery speeds.
    tt = np.linspace(0.0, 40.0, 801)
    for half_life, label in [(0.1, "half-life 0.1 wk (grid end)"),
                             (np.log(2) * dosing.tau, "half-life 0.173 wk (k*tau = 1)"),
                             (2.0, "half-life 2 wk"), (52.0, "half-life 52 wk (grid end)")]:
        k = np.log(2) / half_life
        diff = np.abs(simulate_states(tt, p["L0"], p["b_R"], p["b_P"], k, dosing)["L"]
                      - simulate_states_ode(tt, p["L0"], p["b_R"], p["b_P"], k, dosing)["L"])
        axes[0].semilogy(tt, diff, lw=1.2, label=label)
    axes[0].axhline(TOL, color="k", ls="--", lw=1, label=f"tolerance ({TOL:g} pp)")
    axes[0].set_ylim(1e-15, 1e-2)       # leaves room for the legend above the tolerance line
    axes[0].set_title("|closed form - solve_ivp| in LVEF", fontsize=10)
    axes[0].set_xlabel("weeks")
    axes[0].set_ylabel("pp")
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper center")

    # Middle: the known-truth fit.
    tt = np.linspace(0.0, 20.0, 401)
    axes[1].axvspan(dosing.times[0], dosing.times[-1], color="0.92", lw=0)       # dosing period
    axes[1].plot(tt, simulate_lvef(tt, dosing=dosing, **p), "k-", lw=1.5, label="truth")
    axes[1].plot(tt, simulate_lvef(tt, refined["L0"], refined["b_R"], refined["b_P"], refined["k"], dosing),
                 "C1--", lw=1.5, label="refined fit")
    axes[1].plot(t, y, "C0o", label="noiseless data")
    axes[1].set_title(f"known-truth fit, {SCHEDULE} schedule", fontsize=10)
    axes[1].set_xlabel("weeks")
    axes[1].set_ylabel("LVEF (%)")
    axes[1].legend(fontsize=8)

    # Right: the RSS profile near the minimum: grid points vs the refined minimum.
    half_lives = settings["half_lives"]
    near = (half_lives > 1.4) & (half_lives < 2.9)
    fine = np.geomspace(1.4, 2.9, 300)
    fine_rss = [solve_at_k(t, y, np.log(2) / h, dosing, settings["lower"], settings["upper"])[1] for h in fine]
    axes[2].plot(fine, (np.array(fine_rss) - refined["rss"]) / SIGMA ** 2, "0.6", lw=1.2, label="profile")
    axes[2].plot(half_lives[near], (grid["profile"]["rss"][near] - refined["rss"]) / SIGMA ** 2, "C0o",
                 label="grid points")
    axes[2].plot(refined["half_life"], 0.0, "C1*", ms=12, label=f"refined minimum ({refined['half_life']:.3f} wk)")
    axes[2].set_xscale("log")
    axes[2].set_xticks([1.5, 2, 2.5], ["1.5", "2", "2.5"])
    axes[2].minorticks_off()
    axes[2].set_title("RSS profile near the minimum", fontsize=10)
    axes[2].set_xlabel("half-life (weeks, log scale)")
    axes[2].set_ylabel("(RSS - refined RSS) / sigma^2")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"W3 checks: the two simulator paths, and a noiseless known-truth fit "
                 f"(a = {A:g}, f_P = {F_P:g}, half-life {HALF_LIFE:g} wk)", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "W3_check.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
