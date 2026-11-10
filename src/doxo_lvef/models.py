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

simulate_states_ode() solves the same two equations numerically with solve_ivp
instead. It's only there to check the closed form (Week 3); nothing else uses it.

Shapes: every function that takes t accepts a float or a 1-D array-like of length
n_t and returns 1-D arrays of length n_t. A plain float t comes back as an array of
length 1. 2-D t or NaN/inf times get a ValueError instead of a silently weird answer.

(Adding noise isn't in here on purpose. W1.py does its own, and a proper noise
helper will come with the Week 4 seed work.)
"""
import numpy as np
from scipy.integrate import solve_ivp

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

    At s = 0 both terms are exactly 0. That's what makes the visit
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


def simulate_states_ode(t, L0, b_R, b_P, k, dosing, rtol=1e-10, atol=1e-12):
    """Same output as simulate_states, but from solving the two ODEs with solve_ivp.

    This is the numerical path for checking the closed form. It's a lot slower, so
    nothing else uses it.

    It integrates piecewise between doses: one solve_ivp call from t_j to t_{j+1},
    starting from wherever the previous piece ended (R = P = 0 before the first
    dose). Inside a piece E(t) is smooth and only has the doses given so far in it,
    so the solver never has to step across the jump E makes at a dose time.

    The s = 0 term: a time exactly at a dose t_j gets the state at the start of
    piece j, which is where piece j - 1 ended, so dose j hasn't added anything yet.
    That's the same "just before the dose" value the closed form gives. This is
    also why each piece builds its own E from doses 0..j instead of calling
    inputs.effective_input: that one already counts dose j + 1 at t = t_{j+1}, and
    the solver does evaluate the right-hand side at the end of the piece.

    DOP853 (an explicit 8th-order method) with rtol 1e-10 and atol 1e-12 agrees with
    the closed form to better than 1e-8 over our half-life grid (scripts/W3.py prints
    the number). It isn't meant for huge k, where the equations get stiff and it
    slows down.

    returns: dict with "R", "P", "L", each an array of shape (n_t,)
    """
    if np.ndim(k) != 0:
        raise ValueError(f"k should be one number, got shape {np.shape(k)}")
    k = float(k)
    if not (np.isfinite(k) and k >= 0):
        raise ValueError(f"k has to be a finite number >= 0, got {k}")
    t = check_times(t)
    times, q, tau = dosing.times, dosing.q, dosing.tau
    R, P = np.zeros(t.size), np.zeros(t.size)          # stays 0 before the first dose
    if times.size:
        # Piece j runs from dose j to dose j + 1. The last one runs to the last time we need.
        ends = np.append(times[1:], max(t.max(), times[-1]))
        state = np.zeros(2)
        for j, (start, end) in enumerate(zip(times, ends)):
            last = j == times.size - 1
            here = (t >= start) & ((t <= end) if last else (t < end))
            if end == start:
                # Two doses at the same time: nothing to integrate, the state just carries on.
                R[here], P[here] = state
                continue

            def rhs(s, y, t_on=times[:j + 1], q_on=q[:j + 1]):
                E = np.sum(q_on * np.exp(-(s - t_on) / tau)) / tau     # doses 0..j only
                return [b_R * E - k * y[0], b_P * E]

            sol = solve_ivp(rhs, (start, end), state, method="DOP853", rtol=rtol, atol=atol,
                            dense_output=True)
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed between t = {start} and {end}: {sol.message}")
            if here.any():
                R[here], P[here] = sol.sol(t[here])
            state = sol.y[:, -1]
    return {"R": R, "P": P, "L": L0 - R - P}


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
