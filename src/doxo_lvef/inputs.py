"""Inputs to the model: the dosing history, the drug input E(t), visit times and the config.

Everything here is "stuff we know before we fit anything". The model itself
(how the heart responds) lives in models.py.

Units used everywhere in the package:
    time    -> weeks (floats, never integer weeks; see the visit convention below)
    doses   -> normalized, i.e. mg/kg divided by the 2 mg/kg reference
    LVEF    -> percentage points (pp), so 75 means 75%
"""
from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

# configs/ sits two folders up from this file (src/doxo_lvef/inputs.py -> repo root).
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pilot.toml"


@dataclass(frozen=True)
class Dosing:
    """The known treatment history.

    times : array, shape (n_doses,)   when each dose is given (weeks), sorted
    q     : array, shape (n_doses,)   normalized dose amounts (>= 0)
    tau   : float                     effective input timescale (weeks), > 0

    n_doses can be 0 (no treatment at all). That's handy for the "zero dose" test.
    frozen=True just means you can't change it by accident after making it.
    """
    times: np.ndarray
    q: np.ndarray
    tau: float

    def __post_init__(self):
        # Turn whatever we got (lists, ints...) into float arrays, then sanity-check.
        # np.array(..., copy=True) so changing the caller's array later can't sneak past the checks.
        times = np.atleast_1d(np.array(self.times, dtype=float, copy=True))
        q = np.atleast_1d(np.array(self.q, dtype=float, copy=True))
        if times.shape != q.shape or times.ndim != 1:
            raise ValueError(f"times and q need the same 1-D shape, got {times.shape} and {q.shape}")
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(q))):
            raise ValueError("dose times and amounts have to be finite numbers (no NaN/inf)")
        if np.any(np.diff(times) < 0):
            raise ValueError("dose times have to be sorted")
        if np.any(q < 0):
            raise ValueError("dose amounts can't be negative")
        if not (np.isfinite(self.tau) and self.tau > 0):
            raise ValueError(f"tau has to be a finite positive number, got {self.tau}")
        # Make the arrays read-only too, otherwise "frozen" only protects the attribute, not the numbers.
        times.setflags(write=False)
        q.setflags(write=False)
        # A frozen dataclass doesn't let you assign normally, so this is the usual workaround.
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "tau", float(self.tau))

    @property
    def total(self):
        """Sum of the normalized doses (5.0 for the pilot: 8 x 0.625)."""
        return float(self.q.sum())


def weekly_dosing(dose_mg_per_kg, n_doses, first_dose_week=0.0, interval_weeks=1.0,
                  reference_mg_per_kg=2.0, tau_weeks=0.25):
    """Build a regular dosing schedule, e.g. the pilot's 8 weekly doses of 1.25 mg/kg."""
    times = first_dose_week + interval_weeks * np.arange(n_doses, dtype=float)
    q = np.full(n_doses, dose_mg_per_kg / reference_mg_per_kg)
    return Dosing(times=times, q=q, tau=tau_weeks)


def no_dosing(tau_weeks=0.25):
    """No treatment at all. LVEF should just stay at L0."""
    return Dosing(times=np.array([]), q=np.array([]), tau=tau_weeks)


def check_times(t):
    """Make t a 1-D float array, or complain.

    A float becomes shape (1,). 2-D input would quietly broadcast the wrong way
    against the dose times, and NaN would quietly look like "before every dose",
    so both get a ValueError instead.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if t.ndim != 1:
        raise ValueError(f"t should be 1-D (or a single number), got shape {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("t has NaN or inf in it")
    return t


def effective_input(t, dosing):
    """E(t): the drug input, in normalized dose per week.

    Each dose j adds q_j * exp(-(t - t_j)/tau) / tau once it's given and nothing
    before that. Each of these bumps has area q_j, so E is just the effective drug
    input spread out over time (it's a model input, not a measured concentration).

    Heads-up about exactly t = t_j: E counts the dose as already given there
    (s >= 0), so E jumps up by q_j/tau right at the dose time. That's fine for
    the visit convention, because H, G_k, R, P and L don't jump (they're integrals
    of E and add exactly 0 at s = 0). It matters if you integrate E numerically
    (Week 3 solve_ivp): start each piece at t_j and stop just before t_{j+1}.

    t : float or array-like, shape (n_t,)  ->  returns array, shape (n_t,)
    """
    t = check_times(t)
    s = t[:, None] - dosing.times[None, :]          # time since each dose, shape (n_t, n_doses)
    given = s >= 0
    s = np.where(given, s, 0.0)                     # avoids exp overflow for doses in the future
    return (dosing.q * given * np.exp(-s / dosing.tau) / dosing.tau).sum(axis=1)


def visit_times(schedule, offset_weeks=0.0, first_dose_week=0.0):
    """Turn a schedule like [0, 2, 6, 8, 20] into the actual visit times.

    Visit convention (agreed on Oct 29): a visit in a dose week is taken just
    before that week's dose, so offset_weeks = 0 by default. The offset only moves
    the follow-ups; the first entry is the pretreatment baseline and stays put.

    schedule : array-like, shape (n_visits,), baseline first  ->  returns float array (n_visits,)
    """
    t = np.asarray(schedule, dtype=float).copy()
    if t.ndim != 1 or t.size == 0:
        raise ValueError("schedule should be a non-empty 1-D list of weeks")
    if not (np.all(np.isfinite(t)) and np.isfinite(offset_weeks)):
        raise ValueError("visit weeks and the offset have to be finite numbers")
    t[1:] += offset_weeks
    if t[0] > first_dose_week:
        raise ValueError("the baseline visit has to be before (or at) the first dose")
    if np.any(np.diff(t) <= 0):
        raise ValueError("visit times have to be strictly increasing")
    return t


def load_config(path=None):
    """Read configs/pilot.toml (or another TOML file) into a plain dict."""
    with open(path or DEFAULT_CONFIG, "rb") as f:     # tomllib wants the file opened in binary mode
        return tomllib.load(f)


def dosing_from_config(cfg):
    """Make the Dosing object described in the [dosing] section of the config."""
    d = cfg["dosing"]
    return weekly_dosing(d["dose_mg_per_kg"], d["n_doses"], d["first_dose_week"],
                         d["interval_weeks"], d["reference_mg_per_kg"], d["tau_weeks"])
