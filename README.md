# doxo_lvef_design

Code for our group project on doxorubicin-related LVEF changes in rats. The question
is when cardiac function should be measured, and whether a small number of
measurements is enough to tell slow recovery, persistent dysfunction and delayed
responses apart. We use small ODE models, simulations with known parameters and
published rat studies (Timm 2020, Cove-Smith 2014, Sang 2021).

Team: Xinan Qin (simulation and fitting infrastructure, fall 2026),
Hamda Anjum (estimation and uncertainty), Diya Pidatala (schedule comparison and
design). Mentor: Professor Ning Wei.

## Setup

Python 3.13. From the repo root:

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Versions are pinned in `requirements.txt`.

## Layout

```
scripts/        runnable scripts, one per milestone (W1.py, ...)
src/doxo_lvef/  the package; modules get added as the plan reaches them
tests/          unit tests (start in Week 2)
configs/        scenario/schedule configs and seeds, frozen before evaluation
data/           source data with provenance (see data/README.md)
docs/weekly/    short notes for each week: what was done, open questions
outputs/        generated results, not tracked in git (rerun the scripts)
```

Planned modules in `src/doxo_lvef/`, following the proposal: `inputs.py`,
`models.py`, `fit.py`, `identifiability.py`, `design.py`, `evaluate.py`. They
will be added when each part is actually built, not before.

## Status

| week | dates | what | notes |
|---|---|---|---|
| 1 | Oct 19-23 | minimal end-to-end script (`scripts/W1.py`) | done, see `docs/weekly/W1.md` |
| 2 | Oct 26-30 | baseline simulator + unit tests | in progress |
| 3 | Nov 2-6 | analytical vs numerical check, bounded fitting interface | |
| 4 | Nov 9-13 | three fixed-schedule comparison, config/seed tests | |
| 5 | Nov 16-20 | freeze pilot package, handoff | |

`scripts/W1.py` stays runnable as the package grows.

## Ground rules we're following

These come from the proposal. Most of them are about keeping results honest.

- Raw observations, digitized summaries, fitted curves and simulation truth are
  kept separate and labeled.
- Models, schedules, thresholds and seeds are frozen before looking at final
  evaluation results. Failed fits stay in the denominator.
- Flat profiles, estimates at bounds and unresolved comparisons are reported,
  not hidden.
- R, P (and Z later) are functional model quantities, not measurements of cell
  injury.
