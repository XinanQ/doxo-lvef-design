# Code walkthrough (outline for the recording)

About ten minutes, from the repo root with the venv active. Every command here
is also run by `tests/test_scripts.py`.

1. The layout: `configs/pilot.toml` (all the settings), `src/doxo_lvef/`
   (the package), `scripts/` (one script per week), `tests/`, `outputs/`.
2. `python -m pytest`: everything passes.
3. The model, `src/doxo_lvef/models.py`: L = L0 - b_R G_k - b_P H, the closed
   form in `response_basis`, and the k*tau = 1 limit.
   `python scripts/W2.py` draws the baseline trajectories.
4. The check, `python scripts/W3.py`: the closed form against `solve_ivp`, and
   the noiseless known-truth fit (grid 1.945 weeks, refined 2.000).
5. The fit, `src/doxo_lvef/fit.py`: bounded least squares at each of the 60
   half-lives, the RSS profile, the 95% set and its four classes, bound hits.
6. The pilot, `python scripts/W4.py`: 3 cases x 3 schedules x 20 replicates.
   Open `outputs/W4_fits.csv` (one row per fit, with its cap) and
   `outputs/W4_summary.csv`, then `outputs/W4_bands.png`.
7. Seeds and failures, `src/doxo_lvef/evaluate.py`: one stream per
   (case, replicate), paired noise, failed fits kept with loss 100.
8. `python scripts/W1.py` still runs and gives the Week 1 numbers.
