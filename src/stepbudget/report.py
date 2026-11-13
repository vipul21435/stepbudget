"""Plain-text rendering helpers shared by the commands."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence


def fmt_budget(value: float) -> str:
    """Budgets print as integers when they are whole numbers."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def pct(value: float | None) -> str:
    """A share as a percentage with one decimal."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100.0 * value:.1f} %"


def pct_range(lo: float | None, hi: float | None) -> str:
    """An interval of shares as percentages."""
    if lo is None or hi is None:
        return "-"
    return f"{100.0 * lo:.1f} to {100.0 * hi:.1f} %"


def num(value: float | None, digits: int = 4) -> str:
    """A plain number with a fixed count of significant digits."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}g}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _clean(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def to_json(obj) -> str:
    """JSON with numpy values converted and non-finite numbers written as null."""
    return json.dumps(_clean(obj), indent=2)
