"""Two systems at equal budgets."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .analysis import Prepared, bootstrap_items, curve_for, percentile_interval
from .data import Run
from .estimators import SUCCESS, SuccessCurve


def logrank(runs_a: Sequence[Run], runs_b: Sequence[Run]) -> dict:
    """Log-rank test (Mantel 1966) on the budget to success.

    Giving up and reaching the limit both remove a run from the risk set
    without an event, so the test compares the rate of success among runs
    still going (the cause-specific hazard), not the curves directly. It
    treats runs as independent: it ignores that the two systems ran the same
    items and that an item may have several runs.
    """
    ta = np.array([r.budget for r in runs_a], dtype=float)
    tb = np.array([r.budget for r in runs_b], dtype=float)
    sa = np.array([r.event == SUCCESS for r in runs_a])
    sb = np.array([r.event == SUCCESS for r in runs_b])
    times = np.unique(np.concatenate((ta[sa], tb[sb])))
    if len(times) == 0:
        return {"chi2": 0.0, "p": 1.0, "observed_a": 0.0, "expected_a": 0.0}
    na = len(ta) - np.searchsorted(np.sort(ta), times, side="left")
    nb = len(tb) - np.searchsorted(np.sort(tb), times, side="left")
    da = np.array([np.sum(ta[sa] == t) for t in times], dtype=float)
    db = np.array([np.sum(tb[sb] == t) for t in times], dtype=float)
    n, d = na + nb, da + db
    expected = d * na / n
    spread = np.where(n > 1, d * (na / n) * (nb / n) * (n - d) / np.maximum(n - 1, 1), 0.0)
    obs, exp, var = float(da.sum()), float(expected.sum()), float(spread.sum())
    chi2 = (obs - exp) ** 2 / var if var > 0 else 0.0
    return {
        "chi2": chi2,
        "p": math.erfc(math.sqrt(chi2 / 2.0)),
        "observed_a": obs,
        "expected_a": exp,
    }


def crossovers(a: SuccessCurve, b: SuccessCurve) -> dict:
    """Budgets at which the lead changes, over the range both curves cover."""
    top = min(a.max_budget, b.max_budget)
    grid = np.union1d(a.times, b.times)
    grid = grid[grid <= top]
    diff = np.asarray(a.at(grid)) - np.asarray(b.at(grid))
    lead = 0
    first = None
    changes: list[dict] = []
    for budget, value in zip(grid, diff, strict=True):
        sign = 0 if abs(value) < 1e-12 else (1 if value > 0 else -1)
        if sign == 0:
            continue
        if lead == 0:
            first = "a" if sign > 0 else "b"
        elif sign != lead:
            changes.append({"budget": float(budget), "leader": "a" if sign > 0 else "b"})
        lead = sign
    final = None if lead == 0 else ("a" if lead > 0 else "b")
    return {"first_leader": first, "changes": changes, "final_leader": final, "up_to": float(top)}


def compare(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    budgets: Sequence[float],
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Difference in solved share (a minus b) at equal budgets.

    Only items that both systems ran are used. The interval is a percentile
    interval from a paired bootstrap: items are drawn with replacement and
    each drawn item brings all its runs of both systems, so the pairing and
    any repeated runs are kept. The result is the same for the same ``seed``.
    """
    common = [i for i in dict.fromkeys(r.item for r in runs_a) if i in {r.item for r in runs_b}]
    if not common:
        raise ValueError("the two systems share no items")
    keep = set(common)
    kept_a = [r for r in runs_a if r.item in keep]
    kept_b = [r for r in runs_b if r.item in keep]
    curve_a, curve_b = curve_for(kept_a), curve_for(kept_b)
    top = min(curve_a.max_budget, curve_b.max_budget)
    grid = np.asarray(list(budgets), dtype=float)
    if np.any(grid > top):
        raise ValueError(
            f"budgets must not exceed {top:g}, the largest budget both systems reached"
        )
    grid_all = np.append(grid, top)
    pa, pb = Prepared.from_runs(kept_a, common), Prepared.from_runs(kept_b, common)
    samples = bootstrap_items(
        len(common),
        lambda c: pa.success_at(grid_all, c) - pb.success_at(grid_all, c),
        n_boot=n_boot,
        seed=seed,
    )
    lo, hi = percentile_interval(samples, level)
    rows = []
    for k, b in enumerate(grid):
        fa, fb = float(curve_a.at(b)), float(curve_b.at(b))
        rows.append(
            {
                "budget": float(b),
                "a": fa,
                "b": fb,
                "difference": fa - fb,
                "low": float(lo[k]),
                "high": float(hi[k]),
                "share_a_ahead": float(np.mean(samples[:, k] > 0)),
            }
        )
    cross = crossovers(curve_a, curve_b)
    end = samples[:, -1]
    if cross["final_leader"] == "a":
        cross["final_leader_share"] = float(np.mean(end > 0))
    elif cross["final_leader"] == "b":
        cross["final_leader_share"] = float(np.mean(end < 0))
    else:
        cross["final_leader_share"] = None
    return {
        "items": len(common),
        "dropped_a": len({r.item for r in runs_a}) - len(common),
        "dropped_b": len({r.item for r in runs_b}) - len(common),
        "level": level,
        "n_boot": n_boot,
        "seed": seed,
        "rows": rows,
        "logrank": logrank(kept_a, kept_b),
        "crossovers": cross,
    }
