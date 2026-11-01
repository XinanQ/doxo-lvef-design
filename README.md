# doxo_lvef_design

This is the code for our undergrad research project with Professor Wei. We're
looking at how doxorubicin affects heart function (LVEF) in rats, and mainly at
*when* it should be measured. If you only get a few measurements, can you
actually tell slow recovery, a lasting drop and delayed effects apart? We try to
answer that with small ODE models, simulations where we know the true answer,
and a few published rat studies (Timm 2020, Cove-Smith 2014, Sang 2021).

Who does what:
- Xinan: simulation and fitting code (fall 2026, until the handoff in Week 6)
- Hamda: estimation and uncertainty
- Diya: comparing measurement schedules / design

## Getting it running

You need Python 3.11 to 3.13 (we use 3.13). From the repo folder:

```
python -m venv .venv            # on Mac/Linux: python3 -m venv .venv
.venv\Scripts\activate          # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python scripts/W1.py
python scripts/W2.py
```

If the tests pass and the two scripts save their files in `outputs/`, you're
good. `requirements.txt` pins the versions we test with, so everyone gets the
same numbers. The scripts overwrite `outputs/`, so run the tests first (one of
them compares W1's output with the committed CSV) and check `git diff outputs/`
afterwards.

## What's where

```
scripts/        one script per milestone (W1.py, W2.py)
src/doxo_lvef/  the package: inputs.py (dosing, E(t), visits, config), models.py (the model),
                identifiability.py (what the data can say about the half-life)
tests/          unit tests, run with `python -m pytest`
configs/        pilot.toml: the pilot settings in one place
data/           data from the papers, see data/README.md
docs/weekly/    a note for each week: what got done, what we're unsure about
outputs/        results from running the scripts (committed so you can look without running anything)
```

The proposal suggests splitting the package into `inputs.py`, `models.py`,
`fit.py`, `identifiability.py`, `design.py` and `evaluate.py`. I'm only adding
each one when we get to that part, so the folder doesn't fill up with empty
files.

## Where we are

| week | dates | what | status |
|---|---|---|---|
| 1 | Oct 19-23 | minimal end-to-end script | done, plus the follow-ups from Prof. Wei's reply (`docs/weekly/W1.md`) |
| 2 | Oct 26-30 | simulator + unit tests | done (`docs/weekly/W2.md`); joint-session model notes still to write |
| 3 | Nov 2-6 | check analytical vs numerical solution, fitting interface | |
| 4 | Nov 9-13 | compare the three fixed schedules (my part: config and seed tests, failure logging, a recorded walkthrough, reviewing Diya's comparison) | |
| 5 | Nov 16-20 | freeze the pilot and hand off | |

`scripts/W1.py` should keep working the whole time, even after the package
grows, so nobody gets stuck waiting on me. A test runs it as a command to make
sure.

## Conventions

- Visits in dose weeks are taken just *before* that week's dose, at the trough
  (confirmed by Professor Wei). Visit times are floats, and any shift goes in
  `offset_weeks` in `configs/pilot.toml` (default 0).
- Fits at the grid edge or with b_R = 0 are labeled, not reported as estimates.
- The plain 60-point k grid is used for the pilot and the schedule search.
  Refinement comes in Week 3, only for reported numbers or decisions near a
  threshold.
- An interval that's open above at 52 weeks means slower than we can resolve
  over 20 weeks, which includes persistence. The proposal reruns this with a
  104-week bound in spring.
- Noise SD 1 is meant as a group-mean SEM regime and SD 3 as an individual-animal
  regime. Both are illustrative.

One rule is still waiting on Professor Wei: a half-life is only reported when
the nominal 95% profile interval is closed on both sides by the data and is one
piece (details in `docs/weekly/W1.md`). Otherwise the fit gets
`lower bound only`, `upper bound only`, `disjoint` or `not identified`. For the
pilot this is just a labeling rule; proper profile-likelihood intervals and
coverage checks are spring work.

## Why we did it this way

- The model has a closed form, L = L0 - b_R G_k - b_P H, so we use it instead of
  an ODE solver. For a fixed k it's linear in L0, b_R and b_P, so we fit on a grid
  over k: at each k it's a bounded linear least-squares problem, with no starting
  values to pick, and we get the whole RSS profile for free.
- The grid and bounds are the proposal's (half-life 0.1 to 52 weeks, L0 in
  [50, 95], b_R and b_P in [0, 6]). They're numerical bounds, not biology.
- The textbook G_k is 0/0 at k*tau = 1, which is inside our grid, so it's
  written with m = min(k, 1/tau) and d = |1/tau - k|. That also keeps every
  exponent <= 0 (with a signed d = 1/tau - k it overflows to NaN for very fast
  recovery far out in time).
- A side of the 95% interval only counts as closed if the data close it. If the
  only thing ruling it out is the grid's end, or a point where the fit is stuck
  on a cap (like b_R at 6), we can't tell whether it's the data or our search
  box, so that side stays open. If b_R = 0, k drops out of the model, so those
  fits are "not identified".
- b_R, b_P and `norm_amp_loss` stay in for every fit. The ranking averages over
  all replicates, and dropping the hard fits would favor schedules that fail
  more often. `half_life_hat` is never averaged, because the fits that have one
  are a selected subset.
- Noise is drawn once on the union of visit times and shared by the three
  schedules, so they're compared on the same noise.
- The seed in `pilot.toml` is a development seed. The final evaluation gets its
  own seeds, picked and frozen before the final runs.
- W1 keeps its own copy of the settings, model and grid fit, so the Week 1 numbers
  stay a fixed reference. Tests check it still agrees with `pilot.toml` and the
  package.
- W1 stops with an error on bad input or a failed solve, instead of quietly
  labeling the fit. Proper "failed" fits (loss 100) come with the Week 3/4 code.

## A few rules from the proposal we're sticking to

- Don't mix up raw data, numbers we digitized from figures, fitted curves and
  simulation truth. Label everything.
- Settle on the models, schedules, thresholds and seeds *before* looking at the
  final results. Failed fits still count.
- If a profile is flat, an estimate hits a bound, or two models can't be told
  apart, say so. That's a real result, not something to hide.
- R and P are just quantities in the model. They aren't measurements of how many
  cells are damaged, and "persistent" means it stays over our follow-up.
