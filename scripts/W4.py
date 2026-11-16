"""Week 4: run the pilot (3 cases x 3 fixed schedules x 20 replicates) and summarize it.

Everything comes from configs/pilot.toml. Each fit is done on the plain grid with the
declared cap of 6 on b_R. A fit that ends on the cap is fitted once more with a cap of 12
and both results are kept (see doxo_lvef.evaluate).

Run from the repo root:  python scripts/W4.py   It writes four files in outputs/:
    W4_fits.csv       one row per fit, with the cap it was fitted with
    W4_failures.csv   the failed fits (just the header if there aren't any)
    W4_summary.csv    the summaries for each case and schedule
    W4_bands.png      replicate 0: the fitted curves for every half-life in the 95% set
"""
import csv
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # just save the figure, don't open a window
import matplotlib.pyplot as plt
import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))   # so we can import the package without installing it

from doxo_lvef.evaluate import pick_visits, pilot_data, run_pilot, summarize   # noqa: E402
from doxo_lvef.fit import fit_lvef, fit_settings   # noqa: E402
from doxo_lvef.inputs import dosing_from_config, load_config   # noqa: E402
from doxo_lvef.models import simulate_lvef   # noqa: E402

OUT_DIR = REPO / "outputs"


def main():
    cfg = load_config()
    declared_cap = cfg["fit"]["b_R_bounds"][1]
    rows = run_pilot(cfg)
    summary = summarize(rows, declared_cap)
    failed = [r for r in rows if r["status"] != "ok"]

    OUT_DIR.mkdir(exist_ok=True)
    header = (f"# seed={cfg['seeds']['development']} n_reps={cfg['pilot']['n_reps']} "
              f"declared_b_R_cap={declared_cap:g} refit_b_R_cap={cfg['fit']['b_R_refit_cap']:g}")
    save_csv(OUT_DIR / "W4_fits.csv", rows, list(rows[0]), header)
    save_csv(OUT_DIR / "W4_failures.csv", failed, list(rows[0]), header)
    save_csv(OUT_DIR / "W4_summary.csv", summary, list(summary[0]), header)
    print_summary(cfg, summary, len(failed))
    plot_bands(cfg)
    print(f"Saved {len(rows)} rows and the figure in {OUT_DIR}")


def save_csv(path, rows, fieldnames, header):
    # Plain "\n" line endings and utf-8, so the file comes out the same on Windows and Mac.
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(header + "\n")
        # Versions get their own line, which the regression test skips.
        f.write(f"# env: python={platform.python_version()} numpy={np.__version__} scipy={scipy.__version__}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: text(value) for key, value in row.items()} for row in rows)


def text(value):
    if value is None:
        return ""
    return f"{value:.6g}" if isinstance(value, float) else value


def print_summary(cfg, summary, n_failed):
    reps = cfg["pilot"]["n_reps"]
    print(f"Pilot: {len(cfg['cases'])} cases x {len(cfg['schedules'])} schedules x {reps} replicates, "
          f"declared cap b_R <= {cfg['fit']['b_R_bounds'][1]:g}. Failed fits: {n_failed}.")
    print("Counts are out of the replicates. 'identified' means identified given the declared cap.\n")
    print("case  schedule   identified  open above  open below  uninformative  covered  "
          "width (log2)  half-life error (log2)  on cap (cap-dep.)  mean loss")
    for s in summary:
        error = "-" if s["median_abs_log2_error"] is None else f"{s['median_abs_log2_error']:.2f} (n={s['n_identified']})"
        on_cap = f"{s['b_R_on_cap']} ({s['cap_dependent']})"
        print(f"{s['case']:>4}  {s['schedule']:<9}  {s['identified']:>10}  {s['open_above']:>10}  "
              f"{s['open_below']:>10}  {s['uninformative']:>13}  {s['covered']:>3}/{s['covered_of']:<3}  "
              f"{s['median_width_log2']:>12.2f}  {error:>22}  {on_cap:>17}  {s['mean_norm_amp_loss']:>9.2f}")
    print()
    print("width = median width of the 95% set. half-life error = median |log2(estimate / truth)| "
          "within the identified fits only.")


def plot_bands(cfg):
    """Replicate 0 of every case and schedule: instead of one fitted line, the fitted curve for
    every grid half-life in the 95% set."""
    dosing = dosing_from_config(cfg)
    settings = fit_settings(cfg)
    tt = np.linspace(0.0, 20.0, 401)
    names = list(cfg["schedules"])
    fig, axes = plt.subplots(len(cfg["cases"]), len(names), figsize=(12, 9), sharex=True, sharey="row")
    for case_id, case in enumerate(cfg["cases"]):
        t_all, y_all, truth = pilot_data(cfg, dosing, case_id, 0)
        for col, name in enumerate(names):
            t, y = pick_visits(cfg, dosing, t_all, y_all, name)
            fit = fit_lvef(t, y, dosing, case["noise_sd"], **settings)
            ax = axes[case_id, col]
            ax.axvspan(dosing.times[0], dosing.times[-1], color="0.92", lw=0)       # dosing period
            in_set = fit["profile"]["in_set"]
            for j, (half_life, coefs) in enumerate(zip(fit["profile"]["half_lives"][in_set],
                                                       fit["profile"]["coefs"][in_set])):
                ax.plot(tt, simulate_lvef(tt, *coefs, np.log(2) / half_life, dosing), "C1-", lw=0.8,
                        alpha=0.35, label="fits in the 95% set" if j == 0 else None)
            ax.plot(tt, simulate_lvef(tt, dosing=dosing, **truth), "k-", lw=1.5, label="truth")
            ax.plot(t, y, "C0o", label="simulated data")
            ax.set_title(f"case {case_id}, {name}: {fit['half_life_class']}\n"
                         f"95% set {fit['set_low']:.3g} to {fit['set_high']:.3g} wk", fontsize=9)
    plt.setp(axes[-1], xlabel="weeks")
    plt.setp(axes[:, 0], ylabel="LVEF (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"W4 pilot, replicate 0 (seed {cfg['seeds']['development']}): every fit in the 95% set, "
                 f"given b_R <= {cfg['fit']['b_R_bounds'][1]:g}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "W4_bands.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
