# doxo_lvef_design

This is my simulation and fitting code for our group project on doxorubicin and
LVEF in rats. The question is when to measure LVEF so that a few measurements
can still tell recovery apart from a lasting drop. I check the fits on
simulated data, where I know the true answer.

## Getting it running

You need Python 3.11 to 3.13 (I use 3.13). From the repo folder:

```
python -m venv .venv            # on Mac/Linux: python3 -m venv .venv
.venv\Scripts\activate          # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python scripts/W1.py
python scripts/W2.py
python scripts/W3.py
```

If the tests pass and the three scripts save their files in `outputs/`, you're
good. `requirements.txt` pins the versions I test with, so everyone gets the
same numbers. The scripts overwrite `outputs/`, so run the tests first (one of
them compares W1's output with the committed CSV) and check `git diff outputs/`
afterwards.

## What's where

```
scripts/        one script per milestone (W1.py, W2.py, W3.py)
src/doxo_lvef/  the package: inputs.py (dosing, E(t), visits, config), models.py (the model),
                identifiability.py (what the data can say about the half-life),
                fit.py (the bounded fit)
tests/          unit tests, run with `python -m pytest`
configs/        pilot.toml: the pilot settings in one place
data/           data from published rat studies (none yet), see data/README.md
docs/weekly/    a note for each week: what I did, what I'm unsure about
outputs/        results from running the scripts (committed so you can look without running anything)
```

The proposal suggests splitting the package into `inputs.py`, `models.py`,
`fit.py`, `identifiability.py`, `design.py` and `evaluate.py`. I'm only adding
each one when we get to that part, so the folder doesn't fill up with empty
files.

## My part of the fall pilot

| week | dates | what | status |
|---|---|---|---|
| 1 | Oct 19-23 | minimal end-to-end script | done, plus the Week 1 follow-ups from the Oct 29 reply (`docs/weekly/W1.md`) |
| 2 | Oct 26-30 | simulator + unit tests | done (`docs/weekly/W2.md`); joint-session model notes still to write |
| 3 | Nov 2-6 | check analytical vs numerical solution, fitting interface (`fit.py`) | done (`docs/weekly/W3.md`) |
| 4 | Nov 9-13 | config and seed tests, failure logging, a recorded walkthrough, reviewing the fixed-schedule comparison | |
| 5 | Nov 16-20 | freeze the pilot and hand it off | |

As the proposal asks, `scripts/W1.py` keeps working the whole time, even now that
the package has its own fit, so nobody has to wait on new code.
`tests/test_scripts.py` runs it as a command to make sure.

## Conventions

The first three come from the replies to my emails (Oct 29 and Nov 8).

- Visits in dose weeks are taken just *before* that week's dose, at the trough.
  Visit times are floats, and any shift goes in `offset_weeks` in
  `configs/pilot.toml` (default 0).
- The plain 60-point k grid is used for the pilot, the schedule search and
  Monte Carlo summaries. Refinement between grid points
  (`fit_lvef(..., refine=True)`) is only for reported numbers or decisions near
  a threshold.
- How a fitted half-life gets reported:
  - The best grid point is the point estimate, but it's never reported alone
    and never averaged.
  - Every fit comes with its 95% profile-likelihood set (the half-lives with
    (RSS - min RSS)/sigma^2 <= 3.84) and the class of that set: `identified`
    (both ends inside the grid), `open above` (reaches 52 weeks), `open below`
    (reaches 0.1 weeks) or `uninformative` (both).
  - Bound hits (the half-life at the grid end, b_R at 6 and so on) are flagged,
    and the fitted nadir depth is reported too.
- `open above` means slower than we can resolve over 20 weeks, which includes
  persistence (G_k turns into H as k goes to 0). The proposal reruns this with a
  104-week bound in spring.
- Noise SD 1 is meant as a group-mean SEM regime and SD 3 as an
  individual-animal regime. Both are illustrative.

`scripts/W1.py` still has its Week 1 labels (`identified`, `lower bound only`,
`upper bound only`, `disjoint`, `not identified`). They come from the stricter
rule I used in Week 1 (details in `docs/weekly/W1.md`). W1 keeps them, so its
numbers don't change; `fit.py` uses the classes above.

## Why I did it this way

- The model has a closed form, L = L0 - b_R G_k - b_P H, so I use it instead of
  an ODE solver (the `solve_ivp` path in `models.py` is only there to check it).
  For a fixed k it's linear in L0, b_R and b_P, so the proposal's grid fit over
  k is just a bounded linear least-squares problem at each k (variable
  projection), with no starting values to pick, and I get the whole RSS profile
  for free.
- The grid and bounds are the proposal's (half-life 0.1 to 52 weeks, L0 in
  [50, 95], b_R and b_P in [0, 6]). They're numerical bounds, not biology.
- The textbook G_k is 0/0 at k*tau = 1, which is inside our grid, so I wrote
  it with m = min(k, 1/tau) and d = |1/tau - k|. That also keeps every
  exponent <= 0 (with a signed d = 1/tau - k it overflows to NaN for very fast
  recovery far out in time).
- With sigma known, (RSS - min RSS)/sigma^2 is the profile likelihood-ratio
  statistic, so the 95% set comes straight from the RSS profile the grid fit
  already has. 3.84 is only approximate with one residual degree of freedom; the
  coverage in the Monte Carlo runs will show how good it is.
- b_R and k trade off along a ridge: for half-lives up to about 2 weeks the depth
  of the recovery dip goes roughly like b_R/k. So the cap of 6 on b_R also acts as
  a lower bound on the half-life. That's why a fit on a cap is flagged like a
  half-life at the grid end, and why the nadir depth is reported: the data pin it
  down better than b_R or k.
- If b_R = 0, k drops out of the model, so the profile is flat and the 95% set
  is the whole grid (`uninformative`).
- b_R, b_P and `norm_amp_loss` stay in for every fit. The ranking averages over
  all replicates, and dropping the hard fits would favor schedules that fail
  more often. The half-life is never averaged; the Week 4 summaries look at it
  only within the `identified` class, next to how often each class shows up.
- As the proposal says, noise is drawn once on the union of visit times and
  shared by the three schedules, so they're compared on the same noise.
- W1 keeps its own copy of the settings, model and grid fit, so the Week 1
  numbers stay a fixed reference. `tests/test_config.py` checks that it still
  agrees with `pilot.toml` and the package.
- W1 stops with an error on bad input or a failed solve, instead of quietly
  labeling the fit. `fit.py` has a "failed" status instead (the estimates are
  NaN), and the proposal still counts failed fits in the ranking (loss 100).

## A few rules from the proposal we're sticking to

- Raw data, digitized numbers, fitted curves and simulation truth stay apart
  (`data/raw/` vs `data/extracted/`; `b_R_true` vs `b_R_hat` in the W1 CSV).
- Models, schedules, thresholds and seeds get frozen *before* we see the final
  results. `pilot.toml` has a development seed; the final runs get their own.
- If a profile is flat, an estimate hits a bound, or two models can't be told
  apart, we say so. For the W1 fits that's `half_life_status`, `coef_at_bound`
  and `notes` in the CSV. In `fit.py` it's `half_life_class`, the bound-hit flags
  (`at_grid_end`, `on_cap`) and `notes`.
- R and P are just quantities in the model. They aren't measurements of how many
  cells are damaged, and "persistent" means it stays over our follow-up.
