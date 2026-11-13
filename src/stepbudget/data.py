"""Reading runs from CSV or JSON lines and deciding how each one ended."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .estimators import CENSORED, GAVE_UP, SUCCESS

TRUE_WORDS = frozenset(
    {"1", "true", "t", "yes", "y", "success", "solved", "resolved", "pass", "passed"}
)
FALSE_WORDS = frozenset(
    {"0", "false", "f", "no", "n", "failure", "fail", "failed", "unsolved", "unresolved", ""}
)
DEFAULT_LIMIT_REASONS = ("limit reached", "limit", "limit_reached", "max_steps", "timeout")


@dataclass(frozen=True)
class Run:
    """One run of one system on one item."""

    item: str
    system: str
    budget: float
    event: int
    exit_reason: str | None = None


def parse_outcome(value: object) -> bool:
    """Read a success flag from a boolean, a number or a common word."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value != value:
            raise ValueError("cannot read outcome nan as success or failure")
        return value != 0
    word = str(value).strip().lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"cannot read outcome {value!r} as success or failure")


def classify(
    success: bool,
    budget: float,
    exit_reason: str | None = None,
    limit_reasons: Iterable[str] = DEFAULT_LIMIT_REASONS,
    limit: float | None = None,
) -> int:
    """Decide the event type of a run.

    A successful run is a success at the budget it used, whatever its exit
    reason. A failed run is censored when its exit reason is one of
    ``limit_reasons`` (compared case-insensitively) or, when ``limit`` is
    given, when it used at least ``limit``. Any other failed run gave up.
    """
    if success:
        return SUCCESS
    if exit_reason is not None:
        wanted = {r.strip().lower() for r in limit_reasons}
        if exit_reason.strip().lower() in wanted:
            return CENSORED
    if limit is not None and budget >= limit:
        return CENSORED
    return GAVE_UP


def _rows(path: Path) -> Iterable[Mapping[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = path.read_text(encoding="utf-8-sig")
        if text.lstrip().startswith("["):
            try:
                rows = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: not valid JSON ({exc})") from exc
            for n, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: entry {n} is not a JSON object")
                yield row
            return
    if suffix in {".jsonl", ".ndjson", ".json"}:
        with path.open(encoding="utf-8-sig") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: not valid JSON ({exc})") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected a JSON object")
                yield row
    else:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh)


def read_runs(
    path: str | Path,
    *,
    item_col: str = "item",
    system_col: str = "system",
    outcome_col: str = "outcome",
    budget_col: str = "budget",
    exit_col: str | None = "exit_reason",
    limit_reasons: Iterable[str] = DEFAULT_LIMIT_REASONS,
    limit: float | None = None,
    default_system: str = "system",
) -> list[Run]:
    """Read one run per row from a CSV file, a JSON-lines file or a JSON array of objects.

    ``budget_col`` names the column to use as the budget axis, so a file with
    both steps and cost can be analysed on either. The system column and the
    exit-reason column may be absent.
    """
    path = Path(path)
    limit_reasons = tuple(limit_reasons)
    runs: list[Run] = []
    for n, row in enumerate(_rows(path), 1):
        for col in (item_col, outcome_col, budget_col):
            if col not in row or row[col] is None:
                raise ValueError(f"{path}: row {n} has no value for column {col!r}")
        try:
            budget = float(row[budget_col])  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            bad = row[budget_col]
            raise ValueError(f"{path}: row {n} has a non-numeric budget {bad!r}") from exc
        if not budget >= 0 or budget == float("inf"):
            raise ValueError(f"{path}: row {n} has budget {budget}, expected a finite value >= 0")
        reason = row.get(exit_col) if exit_col else None
        reason = None if reason is None or str(reason).strip() == "" else str(reason)
        system = row.get(system_col)
        system = default_system if system is None or str(system) == "" else str(system)
        ok = parse_outcome(row[outcome_col])
        event = classify(ok, budget, reason, limit_reasons, limit)
        runs.append(Run(str(row[item_col]), system, budget, event, reason))
    if not runs:
        raise ValueError(f"{path}: no runs")
    return runs
