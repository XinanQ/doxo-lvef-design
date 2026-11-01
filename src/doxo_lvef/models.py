"""The baseline model: recovery plus persistence.

    dR/dt = b_R * E(t) - k * R       R = deficit that recovers (leaky bucket)
    dP/dt = b_P * E(t)               P = deficit that stays over the follow-up (no leak)
    L(t)  = L0 - R(t) - P(t)         L = LVEF, the only thing we actually measure

R = P = 0 before the first dose. R and P are functional model quantities, NOT
measurements of cell death or fibrosis (proposal ground rule 4).

Solving the two equations gives a closed form (this is what everything uses):

    L(t) = L0 - b_R * G_k(t) - b_P * H(t)

where H is "how much dose has come in so far" and G_k is "how much of it hasn't
recovered yet". See response_basis() for the formulas.

Shapes: every function that takes t accepts a float or a 1-D array-like of length
n_t and returns 1-D arrays of length n_t. A plain float t comes back as an array of
length 1. 2-D t or NaN/inf times get a ValueError instead of a silently weird answer.

(Adding noise isn't in here on purpose. W1.py does its own, and a proper noise
helper will come with the Week 4 seed work.)
"""
import numpy as np

from doxo_lvef.inputs import check_times

# Below this, we treat k*tau as exactly 1 and use the limit formula (see response_basis).
_REPEATED_ROOT_TOL = 1e-10


def response_basis(t, k, dosing):
    """Return (H, G) at times t, so that L = L0 - b_R * G - b_P * H.

    For one dose q_j given at t_j, with s = t - t_j:
        s < 0 :  nothing (the dose hasn't happened yet)
        s >= 0:  H += q_j * (1 - exp(-s/tau))
                 G += q_j * (exp(-k*s) - exp(-s/tau)) / (1 - k*tau)

    The G formula breaks (0/0) when k*tau = 1. Let m = min(k, 1/tau) (the slower of the
    two decay rates) and d = |1/tau - k|. Then it's the same thing as
        q_j * exp(-m*s) * (1 - exp(-d*s)) / (tau*d)
    and (1 - exp(-d*s)) / d -> s as d -> 0, which is the proposal's k*tau = 1 formula
    q_j * (s/tau) * exp(-s/tau). np.expm1 keeps it accurate when d is tiny, and pulling
    out the slower rate means no exponent is ever positive, so nothing overflows even
    for a huge k or a huge t (an earlier version gave NaN there).

    Note that at s = 0 both terms are exactly 0. That's what makes the visit
    convention work: a visit at a dose time sees the value from just before the dose.

    t      : float or array-like, shape (n_t,)
    k      : float >= 0, recovery rate per week (half-life = ln 2 / k; k = 0 gives G = H)
    dosing : inputs.Dosing
    returns: H, G, both arrays of shape (n_t,)
    """
    # One k at a time, as a plain float. (A float32 k would quietly make the 1/tau - k
    # below float32 too, and an array k would just crash with a confusing message.)
    if np.ndim(k) != 0:
        raise ValueError(f"k should be one number, got shape {np.shape(k)}")
    k = float(k)
    if not (np.isfinite(k) and k >= 0):
        raise ValueError(f"k has to be a finite number >= 0, got {k}")
    t = check_times(t)
    tau = dosing.tau

    # Rows = times, columns = doses. Broadcasting does all the doses at once.
    s = t[:, None] - dosing.times[None, :]
    given = s >= 0
    s = np.where(given, s, 0.0)        # zero out the future doses so exp() doesn't blow up

    # H part: plain integral of E, no recovery.
    H = (dosing.q * given * -np.expm1(-s / tau)).sum(axis=1)

    # G part: same thing but it leaks away at rate k.
    # Pull out the slower decay rate m so every exponent is <= 0 (no overflow when k is huge).
    m = min(k, 1.0 / tau)
    d = abs(1.0 / tau - k)
    if d < _REPEATED_ROOT_TOL:
        ratio = s                                        # the k*tau = 1 limit
    else:
        ratio = -np.expm1(-d * s) / d
    G = (dosing.q * given * np.exp(-m * s) * ratio / tau).sum(axis=1)
    return H, G


def simulate_states(t, L0, b_R, b_P, k, dosing):
    """Noiseless R(t), P(t) and L(t) from the closed form.

    returns: dict with "R", "P", "L", each an array of shape (n_t,)
    """
    H, G = response_basis(t, k, dosing)
    R = b_R * G
    P = b_P * H
    return {"R": R, "P": P, "L": L0 - R - P}


def simulate_lvef(t, L0, b_R, b_P, k, dosing):
    """Noiseless LVEF at times t. Same as simulate_states(...)["L"], just shorter to call."""
    H, G = response_basis(t, k, dosing)
    return L0 - b_R * G - b_P * H


def params_from_case(a, f_P, half_life_weeks, L0=75.0):
    """Turn a pilot case (a, f_P, half-life) into model parameters.

    b_R = a * (1 - f_P), b_P = a * f_P, k = ln 2 / half-life (proposal, simulation table).
    Every simulated truth in the package goes through here, so a typo in a scenario table
    should fail right here instead of turning into a negative amplitude or an infinite k.
    (W1 keeps its own true_params on purpose; test_config.py checks the two agree on the
    pilot cases.)
    """
    if not (np.isfinite(a) and a >= 0):
        raise ValueError(f"a is an amplitude, so it has to be a finite number >= 0, got {a}")
    if not 0 <= f_P <= 1:
        raise ValueError("f_P is a fraction, so it has to be between 0 and 1")
    if not (np.isfinite(half_life_weeks) and half_life_weeks > 0):
        raise ValueError(f"half-life has to be a positive number of weeks, got {half_life_weeks}")
    return {"L0": L0, "b_R": a * (1 - f_P), "b_P": a * f_P, "k": np.log(2) / half_life_weeks}
