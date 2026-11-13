"""Success rate of an agent as a function of its step or cost limit."""

from .analysis import curve_for, curve_interval, item_weights, select, systems
from .compare import compare, crossovers, logrank
from .data import Run, classify, read_runs
from .economics import cost_of_success, marginal_return
from .estimators import (
    CENSORED,
    GAVE_UP,
    SUCCESS,
    KaplanMeier,
    SuccessCurve,
    event_table,
    kaplan_meier,
    spend,
    success_curve,
)
from .limits import LimitScore, extrapolate, score_at_limit

__version__ = "0.1.0"

__all__ = [
    "CENSORED",
    "GAVE_UP",
    "SUCCESS",
    "KaplanMeier",
    "LimitScore",
    "Run",
    "SuccessCurve",
    "classify",
    "compare",
    "cost_of_success",
    "crossovers",
    "curve_for",
    "curve_interval",
    "event_table",
    "extrapolate",
    "item_weights",
    "kaplan_meier",
    "logrank",
    "marginal_return",
    "read_runs",
    "score_at_limit",
    "select",
    "spend",
    "success_curve",
    "systems",
]
