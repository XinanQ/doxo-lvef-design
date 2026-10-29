# doxo_lvef_design

This is the code for our undergrad research project with Professor Wei. We're
looking at how doxorubicin affects heart function (LVEF) in rats, and mainly at
*when* it should be measured. If you only get a few measurements, can you
actually tell slow recovery, permanent damage and delayed effects apart? We try
to answer that with small ODE models, simulations where we know the true
answer, and a few published rat studies (Timm 2020, Cove-Smith 2014, Sang 2021).

Who does what:
- Xinan: simulation and fitting code (fall 2026, until the handoff in Week 6)
- Hamda: estimation and uncertainty
- Diya: comparing measurement schedules / design

## Getting it running

You need Python 3.13. From the repo folder:

```
python -m venv .venv
.venv\Scripts\activate          # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/W1.py
```

If that last line prints a table and saves two files in `outputs/`, you're good.
The package versions are pinned so we all get the same numbers.

## What's where

```
scripts/        one script per milestone (W1.py for now)
src/doxo_lvef/  the actual package, still basically empty
tests/          unit tests, starting Week 2
configs/        scenario/schedule settings and seeds (we freeze these before the final runs)
data/           data from the papers, see data/README.md
docs/weekly/    a short note for each week: what got done, what we're unsure about
outputs/        results from running the scripts (committed so you can look without running anything)
```

The proposal suggests splitting the package into `inputs.py`, `models.py`,
`fit.py`, `identifiability.py`, `design.py` and `evaluate.py`. I'm only going to
add each one when we actually get to that part, so the folder doesn't fill up
with empty files.

## Where we are

| week | dates | what | status |
|---|---|---|---|
| 1 | Oct 19-23 | minimal end-to-end script | done (`docs/weekly/W1.md`) |
| 2 | Oct 26-30 | simulator + unit tests | working on it |
| 3 | Nov 2-6 | check analytical vs numerical solution, fitting interface | |
| 4 | Nov 9-13 | compare the three fixed schedules | |
| 5 | Nov 16-20 | freeze the pilot and hand off | |

`scripts/W1.py` should keep working the whole time, even after the package
grows, so nobody gets stuck waiting on me.

## A few rules from the proposal we're sticking to

- Don't mix up raw data, numbers we digitized from figures, fitted curves and
  simulation truth. Label everything.
- Settle on the models, schedules, thresholds and seeds *before* looking at the
  final results. Failed fits still count.
- If a profile is flat, an estimate hits a bound, or two models can't be told
  apart, say so. That's a real result, not something to hide.
- R and P are just quantities in the model. They aren't measurements of how many
  cells are damaged.
