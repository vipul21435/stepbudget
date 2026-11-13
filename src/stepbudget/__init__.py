"""Success rate of an agent as a function of its step or cost limit."""

from .data import Run, classify, read_runs
from .estimators import (
    CENSORED,
    GAVE_UP,
    SUCCESS,
    KaplanMeier,
    SuccessCurve,
    event_table,
    kaplan_meier,
    success_curve,
)

__version__ = "0.1.0"

__all__ = [
    "CENSORED",
    "GAVE_UP",
    "SUCCESS",
    "KaplanMeier",
    "Run",
    "SuccessCurve",
    "classify",
    "event_table",
    "kaplan_meier",
    "read_runs",
    "success_curve",
]
