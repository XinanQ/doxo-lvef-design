"""Week 2: baseline trajectories from the tested simulator (src/doxo_lvef).

This is the "tested baseline trajectories" output for Week 2. It doesn't fit
anything. It just draws what the model does, so we can all look at the same
picture and sanity-check it before the Week 3 fitting code starts using it.

Left panel : persistent-only, recovery-only and mixed responses, using the same
             numbers as the proposal's Figure 2 (a = 4, half-life 2 weeks).
Middle     : for the mixed curve, what R and P are doing underneath L.
Right      : the mixed response at the three recovery half-lives in our scenario
             grid (0.5, 2, 8 weeks), with the pilot visit weeks marked.

Run from the repo root:  python scripts/W2.py   (saves outputs/W2_trajectories.png)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # just save the figure, don't open a window
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))   # so we can import the package without installing it

from doxo_lvef.inputs import dosing_from_config, load_config, visit_times   # noqa: E402
from doxo_lvef.models import params_from_case, simulate_lvef, simulate_states   # noqa: E402

OUT_DIR = REPO / "outputs"


def main():
    cfg = load_config()
    dosing = dosing_from_config(cfg)
    L0 = cfg["truth"]["L0"]
    t = np.linspace(0.0, 40.0, 1601)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax in axes:
        ax.axvspan(dosing.times[0], dosing.times[-1], color="0.92", lw=0)   # dosing period

    # Left: three hypotheses with the proposal's Figure 2 numbers.
    print("Plateaus (long after dosing), should be L0 - b_P * total dose:")
    for label, f_P in [("persistent only (f_P = 1)", 1.0), ("recovery only (f_P = 0)", 0.0),
                       ("mixed (f_P = 0.3)", 0.3)]:
        p = params_from_case(4.0, f_P, 2.0, L0)
        L = simulate_lvef(t, dosing=dosing, **p)
        axes[0].plot(t, L, lw=1.5, label=label)
        expected = L0 - p["b_P"] * dosing.total
        print(f"  {label:<26} L(40) = {L[-1]:6.2f}   expected {expected:6.2f}")
    axes[0].set_title("a = 4, half-life 2 wk (proposal Fig. 2 numbers)", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")

    # Middle: what's going on underneath the mixed curve.
    p = params_from_case(4.0, 0.3, 2.0, L0)
    s = simulate_states(t, dosing=dosing, **p)
    axes[1].plot(t, s["L"], "k-", lw=1.5, label="L = L0 - R - P")
    axes[1].plot(t, L0 - s["R"], "C0--", lw=1.2, label="L0 - R (recovering part only)")
    axes[1].plot(t, L0 - s["P"], "C3--", lw=1.2, label="L0 - P (persistent part only)")
    axes[1].set_title("mixed case split into R and P", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")

    # Right: half-lives from the scenario grid, with the pilot visit weeks.
    for half_life in (0.5, 2.0, 8.0):
        p = params_from_case(2.0, 0.3, half_life, L0)
        axes[2].plot(t, simulate_lvef(t, dosing=dosing, **p), lw=1.5, label=f"half-life {half_life:g} wk")
    # One row of markers per schedule (along the bottom, out of the way of the curves).
    for row, (name, marker) in enumerate(zip(cfg["schedules"], ["o", "s", "^"])):
        weeks = visit_times(cfg["schedules"][name], cfg["visits"]["offset_weeks"], dosing.times[0])
        axes[2].plot(weeks, np.full(weeks.shape, 57.0 - 1.2 * row), marker, ms=5, ls="none",
                     label=f"{name} visits")
    axes[2].set_title("a = 2, f_P = 0.3 at the three scenario half-lives", fontsize=10)
    axes[2].legend(fontsize=7, loc="lower right")

    for ax in axes:
        ax.set_xlabel("weeks")
    axes[0].set_ylabel("LVEF (%)")
    fig.suptitle("W2 baseline trajectories (noiseless; shading = dosing weeks 0-7, "
                 "visits in dose weeks are just before the dose)", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "W2_trajectories.png", dpi=120)
    plt.close(fig)                    # otherwise figures pile up if main() gets called in a loop
    print(f"Saved {OUT_DIR / 'W2_trajectories.png'}")


if __name__ == "__main__":
    main()
