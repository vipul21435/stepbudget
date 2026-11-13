"""Seeded simulation shared by the tests: exponential times to success and to giving up."""

import numpy as np

from stepbudget.data import Run
from stepbudget.estimators import CENSORED, GAVE_UP, SUCCESS


def true_curve(budget, a, g):
    """Share solved within ``budget`` when nothing is stopped by a limit."""
    return a / (a + g) * (1.0 - np.exp(-(a + g) * np.asarray(budget, dtype=float)))


def simulate(n, a, g, limits, seed):
    """Return (budget used, event, hidden time to success, hidden time to giving up, limit)."""
    rng = np.random.default_rng(seed)
    need = rng.exponential(1.0 / a, n)
    quit_at = rng.exponential(1.0 / g, n)
    limit = rng.choice(np.asarray(limits, dtype=float), n)
    used = np.minimum(np.minimum(need, quit_at), limit)
    event = np.where(
        need <= np.minimum(quit_at, limit), SUCCESS, np.where(quit_at <= limit, GAVE_UP, CENSORED)
    )
    return used, event, need, quit_at, limit


def as_runs(used, event, system="s", prefix="item"):
    pairs = zip(used, event, strict=True)
    return [Run(f"{prefix}-{i}", system, float(b), int(e)) for i, (b, e) in enumerate(pairs)]


def rerun_under(need, quit_at, limit):
    """Share solved when every run is repeated under one common limit."""
    return float(np.mean(need <= np.minimum(quit_at, limit)))
