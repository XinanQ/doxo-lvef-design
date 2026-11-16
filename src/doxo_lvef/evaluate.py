"""Running the pilot (Week 4): simulate, fit and summarize the three fixed schedules.

    pilot_data()   one simulated dataset for a case and replicate
    run_pilot()    every case x replicate x schedule, one row per fit
    summarize()    the Monte Carlo summaries for each case and schedule

Seeds: every (case, replicate) gets its own random stream, made from the development
seed, the case number and the replicate number. So changing the number of replicates,
or editing or adding a case, doesn't change anybody else's noise. The noise is drawn
once on the baseline plus all nine candidate weeks and shared by the schedules, so
they're compared on the same noise (paired), and a new schedule on the candidate grid
doesn't change it either.

The cap on b_R (Nov 10 reply): 6 is the declared bound for all pilot results. A fit
that ends with b_R on the cap gets fitted once more with a cap of 12, both results are
kept as separate rows, and the fit is "cap-dependent" if its class or the low end of
its 95% set changes. Every row says which cap it was fitted with.
"""
import numpy as np

from doxo_lvef.fit import BOUND_TOL, covers_truth, fit_lvef, fit_settings, nadir_depth
from doxo_lvef.inputs import dosing_from_config, load_config, visit_times
from doxo_lvef.models import params_from_case, simulate_lvef

# Proposal: failed fits get normalized loss 100 and stay in the denominator.
FAILED_LOSS = 100.0
CLASSES = ("identified", "open above", "open below", "uninformative")


def pilot_times(cfg, dosing):
    """The baseline plus every candidate week: the times the noise is drawn on."""
    v = cfg["visits"]
    return visit_times([v["baseline_week"], *v["candidate_weeks"]], v["offset_weeks"], dosing.times[0])


def noise_stream(seed, case_id, rep):
    """The random stream for one (case, replicate). Nothing else uses it."""
    return np.random.default_rng([seed, case_id, rep])


def pilot_data(cfg, dosing, case_id, rep):
    """One simulated dataset: the truth for this case plus this replicate's noise.

    returns (t, y, truth): the times from pilot_times, noisy LVEF at those times, and the
    true parameters as a dict (L0, b_R, b_P, k). A schedule picks its visits out of t.
    """
    case = cfg["cases"][case_id]
    truth = params_from_case(case["a"], case["f_P"], case["half_life_weeks"], cfg["truth"]["L0"])
    t = pilot_times(cfg, dosing)
    noise = noise_stream(cfg["seeds"]["development"], case_id, rep).normal(0.0, case["noise_sd"], t.size)
    return t, simulate_lvef(t, dosing=dosing, **truth) + noise, truth


def pick_visits(cfg, dosing, t_all, y_all, schedule):
    """The (t, y) of one schedule, taken out of a dataset from pilot_data."""
    t = visit_times(cfg["schedules"][schedule], cfg["visits"]["offset_weeks"], dosing.times[0])
    idx = np.searchsorted(t_all, t)
    if not np.array_equal(t_all[idx], t):
        raise ValueError(f"schedule {schedule!r} has a visit that isn't the baseline or a candidate week")
    return t, y_all[idx]


def _row(case_id, case, rep, schedule, truth, t, y, dosing, result, settings):
    """One output row for one fit. settings holds the bounds this fit actually used."""
    ok = result["status"] == "ok"
    a, half_life = case["a"], case["half_life_weeks"]
    loss = FAILED_LOSS
    if ok:
        # Normalized squared amplitude error, the proposal's ranking objective.
        loss = float(np.mean([((result[b] - truth[b]) / a) ** 2 for b in ("b_R", "b_P")]))
    covered = covers_truth(t, y, dosing, case["noise_sd"], result, half_life, lower=settings["lower"],
                           upper=settings["upper"], level=settings["level"])
    return {"case": case_id, "a": a, "f_P": case["f_P"], "half_life_true": half_life,
            "sigma": case["noise_sd"], "rep": rep, "schedule": schedule,
            "b_R_cap": float(settings["upper"][1]), "status": result["status"],
            "L0_hat": result["L0"], "b_R_true": truth["b_R"], "b_R_hat": result["b_R"],
            "b_P_true": truth["b_P"], "b_P_hat": result["b_P"],
            "half_life_hat": result["half_life"] if ok else None,
            "half_life_class": result["half_life_class"],
            "set_low": result["set_low"], "set_high": result["set_high"],
            "set_has_gap": result["set_has_gap"], "covered": covered,
            "nadir_depth_true": nadir_depth(truth["L0"], truth["b_R"], truth["b_P"], truth["k"], dosing,
                                            t.min(), t.max()),
            "nadir_depth_hat": result["nadir_depth"], "norm_amp_loss": loss,
            "at_grid_end": result["at_grid_end"], "on_cap": result["on_cap"],
            "cap_dependent": None, "notes": result["notes"]}


def run_pilot(cfg=None, n_reps=None):
    """Fit every case x replicate x schedule on the plain grid. Returns a list of rows (dicts).

    Each fit gives one row with the declared cap on b_R. A fit with b_R on that cap gives a
    second row, fitted with the refit cap, and both rows get cap_dependent (True or False).
    Failed fits stay in as rows with status "failed" and loss 100.
    """
    cfg = load_config() if cfg is None else cfg
    n_reps = cfg["pilot"]["n_reps"] if n_reps is None else n_reps
    dosing = dosing_from_config(cfg)
    settings = fit_settings(cfg)
    wider = dict(settings, upper=settings["upper"].copy())
    wider["upper"][1] = cfg["fit"]["b_R_refit_cap"]
    cap, floor = settings["upper"][1], settings["lower"][1]

    rows = []
    for case_id, case in enumerate(cfg["cases"]):
        for rep in range(n_reps):
            t_all, y_all, truth = pilot_data(cfg, dosing, case_id, rep)
            for schedule in cfg["schedules"]:
                t, y = pick_visits(cfg, dosing, t_all, y_all, schedule)
                fit = fit_lvef(t, y, dosing, case["noise_sd"], **settings)
                row = _row(case_id, case, rep, schedule, truth, t, y, dosing, fit, settings)
                rows.append(row)
                if fit["status"] == "ok" and fit["b_R"] >= cap - BOUND_TOL * (cap - floor):
                    refit = fit_lvef(t, y, dosing, case["noise_sd"], **wider)
                    again = _row(case_id, case, rep, schedule, truth, t, y, dosing, refit, wider)
                    changed = (refit["half_life_class"] != fit["half_life_class"]
                               or refit["set_low"] != fit["set_low"])
                    row["cap_dependent"] = again["cap_dependent"] = bool(changed)
                    rows.append(again)
    return rows


def summarize(rows, declared_cap):
    """Monte Carlo summaries for each case and schedule, from the rows at the declared cap.

    For each (case, schedule):
        n, failed                 replicates, and how many fits failed
        one count per class       identified, open above, open below, uninformative
        covered, covered_of       fits whose 95% set contains the true half-life, out of the
                                  fits that didn't fail. Counts, not percentages: with 20
                                  replicates this is only a smoke test.
        median_width_log2         median width of the 95% set, in log2(half-life)
        n_identified, median_abs_log2_error
                                  half-life error |log2(estimate / truth)|, within the
                                  identified fits only (None if there aren't any)
        mean_norm_amp_loss        mean normalized amplitude loss, failed fits counted as 100
        b_R_on_cap, cap_dependent how many fits ended with b_R on the declared cap, and how
                                  many of those changed class or low end with the refit cap
    """
    groups = {}
    for row in rows:
        if row["b_R_cap"] == declared_cap:
            groups.setdefault((row["case"], row["schedule"]), []).append(row)
    out = []
    for (case_id, schedule), group in groups.items():
        ok = [r for r in group if r["status"] == "ok"]
        identified = [r for r in ok if r["half_life_class"] == "identified" and r["half_life_hat"] is not None]
        errors = [abs(np.log2(r["half_life_hat"] / r["half_life_true"])) for r in identified]
        widths = [np.log2(r["set_high"] / r["set_low"]) for r in ok]
        out.append({"case": case_id, "schedule": schedule, "n": len(group), "failed": len(group) - len(ok),
                    **{c.replace(" ", "_"): sum(r["half_life_class"] == c for r in ok) for c in CLASSES},
                    "covered": sum(r["covered"] is True for r in ok),
                    "covered_of": sum(r["covered"] is not None for r in ok),
                    "median_width_log2": float(np.median(widths)) if widths else None,
                    "n_identified": len(identified),
                    "median_abs_log2_error": float(np.median(errors)) if errors else None,
                    "mean_norm_amp_loss": float(np.mean([r["norm_amp_loss"] for r in group])),
                    "b_R_on_cap": sum(r["cap_dependent"] is not None for r in group),
                    "cap_dependent": sum(r["cap_dependent"] is True for r in group)})
    return out
