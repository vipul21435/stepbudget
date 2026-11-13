"""Marginal return of a larger limit and the cost of a success."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .analysis import Prepared, bootstrap_items, curve_for, item_weights, percentile_interval
from .data import Run
from .estimators import SUCCESS


def marginal_return(
    runs: Sequence[Run], edges: Sequence[float], threshold: float | None = None
) -> dict:
    """Solved share gained per extra unit of limit, between consecutive budgets.

    For each pair of consecutive ``edges`` the row holds ``per_unit_limit``,
    the rise of the success curve divided by the width of the step, and
    ``per_unit_spent``, the same rise divided by the budget runs actually
    spend there (only runs still going spend it). With ``threshold`` the
    result names the smallest edge from which ``per_unit_limit`` stays below
    it up to the largest edge; ``None`` when the last step is still at or
    above the threshold.
    """
    grid = np.asarray(sorted(set(float(e) for e in edges)), dtype=float)
    if len(grid) < 2:
        raise ValueError("need at least two budgets")
    curve = curve_for(runs)
    if grid[-1] > curve.max_budget:
        raise ValueError(f"budgets must not exceed {curve.max_budget:g}, the largest one used")
    solved = np.asarray(curve.at(grid))
    spent = np.asarray(curve.spend_at(grid))
    width = np.diff(grid)
    rise = np.diff(solved)
    burn = np.diff(spent)
    per_spent = np.divide(rise, burn, out=np.full(len(rise), np.nan), where=burn > 0)
    rows = [
        {
            "from": float(grid[k]),
            "to": float(grid[k + 1]),
            "gain": float(rise[k]),
            "per_unit_limit": float(rise[k] / width[k]),
            "per_unit_spent": float(per_spent[k]),
        }
        for k in range(len(width))
    ]
    flat_from = None
    if threshold is not None:
        below = np.array([r["per_unit_limit"] < threshold for r in rows])
        k = len(below)
        while k > 0 and below[k - 1]:
            k -= 1
        flat_from = float(grid[k]) if k < len(below) else None
    return {"threshold": threshold, "below_threshold_from": flat_from, "rows": rows}


def cost_of_success(
    runs: Sequence[Run],
    limits: Sequence[float],
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Expected budget spent per solved item under each limit.

    ``spend`` is the expected budget one run uses under the limit, successful
    or not (the area under the still-running curve), ``solved`` the success
    curve at the limit, and ``cost`` their ratio: the budget burned by runs
    that gave up or were stopped is charged to the successes. The interval is
    a percentile interval from a bootstrap over items. ``as_run`` is the plain
    total of the budgets used divided by the number of successes, under
    whatever limits the runs actually had; when the runs carry a second budget
    (``Run.second``) its total per item and per success is reported there too.
    """
    grid = np.asarray(list(limits), dtype=float)
    curve = curve_for(runs)
    if np.any(grid > curve.max_budget):
        raise ValueError(f"limits must not exceed {curve.max_budget:g}, the largest budget used")
    prep = Prepared.from_runs(runs)
    samples = bootstrap_items(
        len(prep.items), lambda c: prep.cost_at(grid, c), n_boot=n_boot, seed=seed
    )
    lo, hi = percentile_interval(samples, level)
    solved = np.asarray(curve.at(grid))
    spent = np.asarray(curve.spend_at(grid))
    rows = [
        {
            "limit": float(b),
            "spend": float(spent[k]),
            "solved": float(solved[k]),
            "cost": float(spent[k] / solved[k]) if solved[k] > 0 else float("nan"),
            "low": float(lo[k]),
            "high": float(hi[k]),
        }
        for k, b in enumerate(grid)
    ]
    w = item_weights(runs)
    used = float(sum(wi * r.budget for wi, r in zip(w, runs, strict=True)))
    wins = float(sum(wi for wi, r in zip(w, runs, strict=True) if r.event == SUCCESS))
    as_run = {
        "spend": used / float(w.sum()),
        "solved": wins / float(w.sum()),
        "cost": used / wins if wins > 0 else float("nan"),
    }
    if all(r.second is not None for r in runs):
        other = float(sum(wi * r.second for wi, r in zip(w, runs, strict=True)))
        as_run["second_spend"] = other / float(w.sum())
        as_run["second_cost"] = other / wins if wins > 0 else float("nan")
    return {"level": level, "n_boot": n_boot, "seed": seed, "as_run": as_run, "rows": rows}
