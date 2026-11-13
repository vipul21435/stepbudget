"""What a different limit would have scored."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .analysis import curve_for, item_weights
from .data import Run
from .estimators import CENSORED, GAVE_UP, SUCCESS, SuccessCurve


@dataclass(frozen=True)
class LimitScore:
    """The share of items solved under a limit other than the one used.

    ``kind`` is ``"exact"`` when no run was stopped before the limit (the score
    follows by truncation), ``"bounded"`` when some were (the score lies
    between ``none_solve`` and ``all_solve`` and ``estimate`` is the
    censoring-aware curve), and ``"extrapolated"`` when the limit is larger
    than any budget observed.
    """

    budget: float
    kind: str
    none_solve: float
    all_solve: float
    estimate: float
    stopped_before: float

    def as_dict(self) -> dict:
        return asdict(self)


def tail_rates(runs: Sequence[Run], start: float) -> tuple[float, float]:
    """Successes and give-ups per unit of budget spent beyond ``start``."""
    w = item_weights(runs)
    spent = sum(wi * (r.budget - start) for wi, r in zip(w, runs, strict=True) if r.budget > start)
    if spent <= 0:
        return 0.0, 0.0
    late = [(wi, r) for wi, r in zip(w, runs, strict=True) if r.budget > start]
    wins = sum(wi for wi, r in late if r.event == SUCCESS)
    quits = sum(wi for wi, r in late if r.event == GAVE_UP)
    return wins / spent, quits / spent


def extrapolate(
    curve: SuccessCurve, runs: Sequence[Run], budget: float, tail: float = 0.5
) -> float:
    """Success share at a limit beyond the data, under constant tail rates.

    The per-unit rates of success and of giving up are measured over the last
    ``tail`` fraction of the observed budget range and assumed to stay the same
    beyond it. With rates ``a`` (success) and ``g`` (giving up) and a share
    ``R`` still running at the largest observed budget ``L``::

        success(b) = success(L) + R * a / (a + g) * (1 - exp(-(a + g) * (b - L)))

    This is an assumption, not an estimate the data can check.
    """
    if not 0.0 < tail <= 1.0:
        raise ValueError("tail must be in (0, 1]")
    top = curve.max_budget
    base = float(curve.success[-1])
    if budget <= top:
        return float(curve.at(budget))
    a, g = tail_rates(runs, top * (1.0 - tail))
    if a <= 0:
        return base
    remaining = float(curve.running[-1])
    return base + remaining * a / (a + g) * (1.0 - math.exp(-(a + g) * (budget - top)))


def score_at_limit(runs: Sequence[Run], budget: float, tail: float = 0.5) -> LimitScore:
    """Share of items one system would have solved under ``budget``.

    ``none_solve`` counts the runs that succeeded within ``budget``;
    ``all_solve`` adds every run the limit stopped before ``budget``, as if each
    would have gone on to succeed within it. Both are shares of items.
    """
    if not budget >= 0:
        raise ValueError("budget must be non-negative")
    w = item_weights(runs)
    total = float(w.sum())
    used = np.array([r.budget for r in runs])
    event = np.array([r.event for r in runs])
    lower = float(w[(event == SUCCESS) & (used <= budget)].sum() / total)
    stopped = float(w[(event == CENSORED) & (used < budget)].sum() / total)
    curve = curve_for(runs)
    if budget > curve.max_budget:
        kind, estimate = "extrapolated", extrapolate(curve, runs, budget, tail)
    elif stopped > 0:
        kind, estimate = "bounded", float(curve.at(budget))
    else:
        kind, estimate = "exact", lower
    return LimitScore(float(budget), kind, lower, lower + stopped, estimate, stopped)
