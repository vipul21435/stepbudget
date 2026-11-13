"""Kaplan-Meier and Aalen-Johansen estimators on budget-to-event data.

A run ends in one of three ways, at the budget it had used by then:

* ``SUCCESS``  - it solved the item;
* ``GAVE_UP``  - it stopped on its own without solving the item;
* ``CENSORED`` - the limit stopped it, so all that is known is that solving
  the item would have needed more than the budget it used.

The success-by-budget curve is the cumulative incidence of ``SUCCESS`` with
``GAVE_UP`` as a competing event (Aalen and Johansen 1978).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

CENSORED = 0
SUCCESS = 1
GAVE_UP = 2

ArrayLike = Sequence[float] | np.ndarray


def _z(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def _step(times: np.ndarray, values: np.ndarray, budget: ArrayLike | float, before: float):
    """Evaluate a right-continuous step function at ``budget``."""
    b = np.asarray(budget, dtype=float)
    idx = np.searchsorted(times, b, side="right") - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], before)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class EventTable:
    """Counts (or weights) at each distinct budget, in ascending order."""

    times: np.ndarray
    at_risk: np.ndarray
    success: np.ndarray
    gave_up: np.ndarray
    censored: np.ndarray
    total: float


def event_table(
    budgets: ArrayLike, events: Sequence[int] | np.ndarray, weights: ArrayLike | None = None
) -> EventTable:
    """Tabulate runs by the budget at which they ended.

    ``at_risk`` at a budget counts the runs that had not ended before it; runs
    censored at a budget are still at risk at that budget, so an event and a
    censoring at the same budget are ordered event first.
    """
    t = np.asarray(budgets, dtype=float)
    e = np.asarray(events, dtype=int)
    if t.ndim != 1 or t.shape != e.shape:
        raise ValueError("budgets and events must be one-dimensional and of equal length")
    if t.size == 0:
        raise ValueError("no runs")
    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError("budgets must be finite and non-negative")
    if np.any((e < 0) | (e > 2)):
        raise ValueError("events must be 0 (censored), 1 (success) or 2 (gave up)")
    if weights is None:
        w = np.ones_like(t)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != t.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite, non-negative and match budgets")
        if w.sum() <= 0:
            raise ValueError("weights sum to zero")
    times, inv = np.unique(t, return_inverse=True)
    k = len(times)
    d1 = np.bincount(inv, weights=w * (e == SUCCESS), minlength=k)
    d2 = np.bincount(inv, weights=w * (e == GAVE_UP), minlength=k)
    c = np.bincount(inv, weights=w * (e == CENSORED), minlength=k)
    leaving = d1 + d2 + c
    total = float(leaving.sum())
    at_risk = total - np.concatenate(([0.0], np.cumsum(leaving)[:-1]))
    at_risk = np.maximum(at_risk, leaving)  # guards rounding with fractional weights
    return EventTable(times, at_risk, d1, d2, c, total)


@dataclass(frozen=True)
class KaplanMeier:
    """Product-limit estimate of the share of runs with no event by each budget."""

    times: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    survival: np.ndarray
    variance: np.ndarray

    def at(self, budget: ArrayLike | float):
        """Survival at ``budget`` (1 before the first observed budget)."""
        return _step(self.times, self.survival, budget, 1.0)

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise interval from Greenwood's variance on the log(-log) scale."""
        return _loglog_interval(self.survival, self.variance, level)


def kaplan_meier(
    budgets: ArrayLike, observed: Sequence[int] | np.ndarray, weights: ArrayLike | None = None
) -> KaplanMeier:
    """Kaplan-Meier estimate; ``observed`` is 1 for an event and 0 for censoring.

    The variance is Greenwood's formula,
    ``S(t)^2 * sum_{t_i <= t} d_i / (n_i * (n_i - d_i))``.
    """
    obs = np.asarray(observed, dtype=int)
    if np.any((obs < 0) | (obs > 1)):
        raise ValueError("observed must be 0 or 1")
    tab = event_table(budgets, obs, weights)
    d, n = tab.success, tab.at_risk
    surv = np.cumprod(1.0 - d / n)
    left = n - d
    term = np.divide(d, n * left, out=np.zeros_like(d), where=left > 0)
    var = surv**2 * np.cumsum(term)
    return KaplanMeier(tab.times, n, d, surv, var)


def _loglog_interval(
    value: np.ndarray, variance: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Interval for a probability from its variance, on the log(-log) scale.

    The transform keeps both ends inside [0, 1]. Where the estimate is exactly
    0 or 1 the variance estimate is 0 and the interval collapses to the point.
    """
    z = _z(level)
    p = np.asarray(value, dtype=float)
    lo = p.copy()
    hi = p.copy()
    ok = (p > 0.0) & (p < 1.0) & (variance > 0.0)
    if np.any(ok):
        q = p[ok]
        se = np.sqrt(variance[ok]) / (q * np.abs(np.log(q)))
        centre = np.log(-np.log(q))
        lo[ok] = np.exp(-np.exp(centre + z * se))
        hi[ok] = np.exp(-np.exp(centre - z * se))
    return lo, hi


@dataclass(frozen=True)
class SuccessCurve:
    """Success as a function of budget for one system.

    All arrays are aligned with ``times``, the distinct budgets at which a run
    ended. ``success`` is the cumulative incidence of success, ``gave_up`` the
    cumulative incidence of giving up, ``running`` the share still running
    after each budget, and ``empirical`` the plain share of all runs that
    succeeded within the budget (limit-reached runs counted as not solved).
    """

    times: np.ndarray
    at_risk: np.ndarray
    n_success: np.ndarray
    n_gave_up: np.ndarray
    n_censored: np.ndarray
    running: np.ndarray
    success: np.ndarray
    gave_up: np.ndarray
    variance: np.ndarray
    empirical: np.ndarray
    total: float

    def at(self, budget: ArrayLike | float):
        """Cumulative incidence of success at ``budget``."""
        return _step(self.times, self.success, budget, 0.0)

    def empirical_at(self, budget: ArrayLike | float):
        """Plain share of runs that succeeded within ``budget``."""
        return _step(self.times, self.empirical, budget, 0.0)

    def running_at(self, budget: ArrayLike | float):
        """Estimated share of runs still running after ``budget``."""
        return _step(self.times, self.running, budget, 1.0)

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise interval for ``success`` at each entry of ``times``."""
        return _loglog_interval(self.success, self.variance, level)

    def interval_at(self, budget: float, level: float = 0.95) -> tuple[float, float]:
        """Pointwise interval for the cumulative incidence of success at ``budget``."""
        lo, hi = self.interval(level)
        return float(_step(self.times, lo, budget, 0.0)), float(_step(self.times, hi, budget, 0.0))

    @property
    def first_censoring(self) -> float | None:
        """Smallest budget at which a run was stopped by the limit, if any."""
        hit = self.times[self.n_censored > 0]
        return float(hit[0]) if len(hit) else None

    @property
    def max_budget(self) -> float:
        """Largest budget any run used."""
        return float(self.times[-1])


def success_curve(
    budgets: ArrayLike, events: Sequence[int] | np.ndarray, weights: ArrayLike | None = None
) -> SuccessCurve:
    """Aalen-Johansen cumulative incidence of success by budget.

    With ``n_i`` runs still going just before budget ``t_i``, of which ``s_i``
    succeed and ``g_i`` give up at ``t_i``::

        running(t) = prod_{t_i <= t} (1 - (s_i + g_i) / n_i)
        success(t) = sum_{t_i <= t} running(t_{i-1}) * s_i / n_i

    Runs stopped by the limit leave the risk set without an event. The
    variance is the delta-method estimate for a cumulative incidence in its
    usual three-sum form, computed with running sums so the cost is linear in
    the number of distinct budgets. It assumes independent runs; with repeated runs of an
    item use the bootstrap in :mod:`stepbudget.analysis` instead.
    """
    tab = event_table(budgets, events, weights)
    n, s, g = tab.at_risk, tab.success, tab.gave_up
    d = s + g
    running = np.cumprod(1.0 - d / n)
    lag = np.concatenate(([1.0], running[:-1]))
    f1 = np.cumsum(lag * s / n)
    f2 = np.cumsum(lag * g / n)
    empirical = np.cumsum(s) / tab.total

    left = n - d
    a = np.divide(d, n * left, out=np.zeros_like(d), where=left > 0)
    first = f1**2 * np.cumsum(a) - 2.0 * f1 * np.cumsum(a * f1) + np.cumsum(a * f1**2)
    second = np.cumsum(lag**2 * s * (n - s) / n**3)
    h = lag * s / n**2
    third = f1 * np.cumsum(h) - np.cumsum(h * f1)
    var = np.maximum(first + second - 2.0 * third, 0.0)
    return SuccessCurve(
        tab.times, n, s, g, tab.censored, running, f1, f2, var, empirical, tab.total
    )
