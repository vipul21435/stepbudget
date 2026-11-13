"""Curves per system, item clustering and the bootstrap over items."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .data import Run
from .estimators import GAVE_UP, SUCCESS, SuccessCurve, spend, success_curve


def systems(runs: Sequence[Run]) -> list[str]:
    """System names in order of first appearance."""
    return list(dict.fromkeys(r.system for r in runs))


def select(runs: Sequence[Run], system: str) -> list[Run]:
    """Runs of one system."""
    out = [r for r in runs if r.system == system]
    if not out:
        known = ", ".join(systems(runs))
        raise ValueError(f"no runs for system {system!r} (systems in the file: {known})")
    return out


def item_weights(runs: Sequence[Run]) -> np.ndarray:
    """Weight of each run so that every item counts once.

    A run of an item that was run ``k`` times gets weight ``1 / k``; the curve
    is then a share of items, not of runs.
    """
    counts = Counter(r.item for r in runs)
    return np.array([1.0 / counts[r.item] for r in runs])


def has_repeats(runs: Sequence[Run]) -> bool:
    """True when some item has more than one run."""
    return len({r.item for r in runs}) < len(runs)


def curve_for(runs: Sequence[Run]) -> SuccessCurve:
    """Success-by-budget curve of one system's runs, each item weighted once."""
    budgets = [r.budget for r in runs]
    events = [r.event for r in runs]
    weights = item_weights(runs) if has_repeats(runs) else None
    return success_curve(budgets, events, weights)


@dataclass(frozen=True)
class Prepared:
    """One system's runs laid out for repeated re-weighting by item."""

    items: tuple[str, ...]
    times: np.ndarray
    inverse: np.ndarray
    item_index: np.ndarray
    base_weight: np.ndarray
    is_success: np.ndarray
    is_gave_up: np.ndarray

    @classmethod
    def from_runs(cls, runs: Sequence[Run], items: Sequence[str] | None = None) -> Prepared:
        if items is None:
            items = list(dict.fromkeys(r.item for r in runs))
        index = {name: i for i, name in enumerate(items)}
        budgets = np.array([r.budget for r in runs], dtype=float)
        events = np.array([r.event for r in runs], dtype=int)
        times, inverse = np.unique(budgets, return_inverse=True)
        return cls(
            tuple(items),
            times,
            inverse,
            np.array([index[r.item] for r in runs], dtype=int),
            item_weights(runs),
            events == SUCCESS,
            events == GAVE_UP,
        )

    def curves(self, item_counts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Success and still-running shares at ``times`` with items re-weighted."""
        w = self.base_weight
        if item_counts is not None:
            w = w * item_counts[self.item_index]
        k = len(self.times)
        s = np.bincount(self.inverse, weights=w * self.is_success, minlength=k)
        g = np.bincount(self.inverse, weights=w * self.is_gave_up, minlength=k)
        leaving = np.bincount(self.inverse, weights=w, minlength=k)
        n = leaving.sum() - np.concatenate(([0.0], np.cumsum(leaving)[:-1]))
        n = np.maximum(n, leaving)
        safe = np.where(n > 0, n, 1.0)
        running = np.cumprod(1.0 - (s + g) / safe)
        lag = np.concatenate(([1.0], running[:-1]))
        return np.cumsum(lag * s / safe), running

    def success_at(self, budgets: np.ndarray, item_counts: np.ndarray | None = None) -> np.ndarray:
        """Cumulative incidence of success at chosen budgets."""
        f1, _ = self.curves(item_counts)
        idx = np.searchsorted(self.times, budgets, side="right") - 1
        return np.where(idx >= 0, f1[np.clip(idx, 0, None)], 0.0)

    def cost_at(self, budgets: np.ndarray, item_counts: np.ndarray | None = None) -> np.ndarray:
        """Expected budget spent per solved item at chosen limits (nan with no success)."""
        f1, running = self.curves(item_counts)
        idx = np.searchsorted(self.times, budgets, side="right") - 1
        solved = np.where(idx >= 0, f1[np.clip(idx, 0, None)], 0.0)
        spent = spend(self.times, running, budgets)
        return np.divide(spent, solved, out=np.full(len(solved), np.nan), where=solved > 0)


def bootstrap_items(
    n_items: int,
    statistic: Callable[[np.ndarray], np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Draw items with replacement and collect ``statistic(item_counts)``.

    ``item_counts[i]`` is how many times item ``i`` was drawn. All runs of a
    drawn item come along, so repeated runs of an item stay together. The
    result has one row per replicate and is the same for the same ``seed``.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_items, size=n_items), minlength=n_items)
        rows.append(np.asarray(statistic(counts), dtype=float))
    return np.vstack(rows)


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval over bootstrap rows, ignoring undefined replicates."""
    tail = (1.0 - level) / 2.0
    lo = np.nanquantile(samples, tail, axis=0)
    hi = np.nanquantile(samples, 1.0 - tail, axis=0)
    return lo, hi


def curve_interval(
    runs: Sequence[Run],
    budgets: Sequence[float],
    level: float = 0.95,
    method: str = "auto",
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Pointwise interval for the success curve at ``budgets``.

    ``method`` is ``"analytic"`` (delta-method variance, independent runs),
    ``"bootstrap"`` (percentile interval, items resampled with their runs) or
    ``"auto"``, which uses the bootstrap only when some item has several runs.
    Returns the lower ends, the upper ends and the method used.
    """
    if method not in {"auto", "analytic", "bootstrap"}:
        raise ValueError("method must be auto, analytic or bootstrap")
    if method == "auto":
        method = "bootstrap" if has_repeats(runs) else "analytic"
    grid = np.asarray(budgets, dtype=float)
    if method == "analytic":
        curve = curve_for(runs)
        pairs = [curve.interval_at(float(b), level) for b in grid]
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), method
    prep = Prepared.from_runs(runs)
    samples = bootstrap_items(
        len(prep.items), lambda c: prep.success_at(grid, c), n_boot=n_boot, seed=seed
    )
    lo, hi = percentile_interval(samples, level)
    return lo, hi, method
