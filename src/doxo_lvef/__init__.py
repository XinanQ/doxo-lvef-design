"""Doxorubicin LVEF monitoring-design project.

Modules get added week by week as we actually build them:
    inputs.py          (Week 2)  dosing, E(t), visit times, config loading
    models.py          (Week 2)  the recovery + persistence model and its closed form
                       (Week 3)  plus a solve_ivp path that's only there to check the closed form
    identifiability.py (Week 2)  what the data can say about the half-life: profile_interval (W1's labels)
                       (Week 3)  plus profile_set: the 95% set and its class (Nov 8 reply)
    fit.py             (Week 3)  fit_lvef: the bounded grid fit, refinement, near-optimal minima
"""
