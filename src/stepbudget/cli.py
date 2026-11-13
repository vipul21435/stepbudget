"""Command line: ``stepbudget curve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .analysis import curve_for, curve_interval, has_repeats, select, systems
from .data import DEFAULT_LIMIT_REASONS, Run, read_runs
from .estimators import CENSORED, GAVE_UP, SUCCESS
from .report import fmt_budget, pct, pct_range, table, to_json


def default_grid(runs: Sequence[Run], points: int = 10) -> list[float]:
    """Evenly spaced budgets up to the largest one used, whole when the data is."""
    values = sorted({r.budget for r in runs})
    if len(values) <= points:
        return values
    top = values[-1]
    grid = np.linspace(top / points, top, points)
    if all(float(v).is_integer() for v in values):
        grid = np.unique(np.maximum(np.round(grid), 1.0))
    return [float(g) for g in grid]


def counts(runs: Sequence[Run]) -> dict[str, int]:
    """How many runs ended each way."""
    return {
        "runs": len(runs),
        "items": len({r.item for r in runs}),
        "success": sum(r.event == SUCCESS for r in runs),
        "gave_up": sum(r.event == GAVE_UP for r in runs),
        "limit_reached": sum(r.event == CENSORED for r in runs),
    }


def curve_report(
    runs: Sequence[Run],
    budgets: Sequence[float] | None = None,
    level: float = 0.95,
    method: str = "auto",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """The success-by-budget curve of one system at chosen budgets."""
    grid = list(budgets) if budgets else default_grid(runs)
    curve = curve_for(runs)
    lo, hi, used = curve_interval(runs, grid, level, method, n_boot, seed)
    rows = []
    for b, low, high in zip(grid, lo, hi, strict=True):
        idx = int(np.searchsorted(curve.times, b, side="right"))
        at_risk = float(curve.at_risk[idx]) if idx < len(curve.times) else 0.0
        rows.append(
            {
                "budget": b,
                "success": curve.at(b),
                "low": float(low),
                "high": float(high),
                "plain": curve.empirical_at(b),
                "running": curve.running_at(b),
                "at_risk_after": at_risk,
            }
        )
    differ = np.flatnonzero(np.abs(curve.success - curve.empirical) > 1e-12)
    return {
        "counts": counts(runs),
        "level": level,
        "interval": used,
        "clustered": has_repeats(runs),
        "first_limit_budget": curve.first_censoring,
        "curves_differ_from": float(curve.times[differ[0]]) if len(differ) else None,
        "max_budget": curve.max_budget,
        "rows": rows,
    }


def difference_note(rep: dict) -> str:
    """When the censoring-aware curve and the plain share part ways."""
    first, differ = rep["first_limit_budget"], rep["curves_differ_from"]
    if first is None:
        return "No run was stopped by the limit, so the two curves are the same everywhere."
    if differ is None:
        return (
            f"The first run stopped by the limit had used {fmt_budget(first)}, and no run "
            "succeeded at a larger budget, so the two curves are the same everywhere."
        )
    return (
        f"The first run stopped by the limit had used {fmt_budget(first)}. The two curves are "
        f"the same below {fmt_budget(differ)} and differ from there on: the plain share counts "
        "every run stopped earlier as unsolved at every larger budget, while the "
        "censoring-aware curve lets the runs still going stand in for it."
    )


def curve_markdown(name: str, rep: dict) -> str:
    c = rep["counts"]
    level = f"{100 * rep['level']:g} %"
    head = [
        f"## {name}",
        "",
        f"{c['runs']} runs on {c['items']} items: {c['success']} succeeded, "
        f"{c['gave_up']} gave up, {c['limit_reached']} were stopped by the limit.",
        "",
    ]
    rows = [
        [
            fmt_budget(r["budget"]),
            pct(r["success"]),
            pct_range(r["low"], r["high"]),
            pct(r["plain"]),
            pct(r["running"]),
        ]
        for r in rep["rows"]
    ]
    header = ["budget", "solved within budget", f"{level} interval", "plain share", "still running"]
    how = "delta-method variance" if rep["interval"] == "analytic" else "bootstrap over items"
    tail = ["", f"Interval: {how}. {difference_note(rep)}"]
    return "\n".join(head + [table(header, rows)] + tail)


def _reader_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="CSV or JSON-lines file with one run per row")
    p.add_argument("--item-col", default="item")
    p.add_argument("--system-col", default="system")
    p.add_argument("--outcome-col", default="outcome")
    p.add_argument("--budget-col", default="budget", help="column used as the budget axis")
    p.add_argument("--exit-col", default="exit_reason")
    p.add_argument(
        "--limit-reasons",
        default=",".join(DEFAULT_LIMIT_REASONS),
        help="comma-separated exit reasons that mean the limit stopped the run",
    )
    p.add_argument(
        "--limit",
        type=float,
        default=None,
        help="treat failed runs that used at least this much as stopped by the limit",
    )
    p.add_argument("--format", choices=["md", "json"], default="md")
    p.add_argument("--level", type=float, default=0.95, help="interval level")
    p.add_argument("--interval", choices=["auto", "analytic", "bootstrap"], default="auto")
    p.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=0)


def _load(args: argparse.Namespace) -> list[Run]:
    return read_runs(
        args.path,
        item_col=args.item_col,
        system_col=args.system_col,
        outcome_col=args.outcome_col,
        budget_col=args.budget_col,
        exit_col=args.exit_col,
        limit_reasons=[r for r in args.limit_reasons.split(",") if r.strip()],
        limit=args.limit,
    )


def _cmd_curve(args: argparse.Namespace) -> str:
    runs = _load(args)
    names = args.system or systems(runs)
    reports = {
        name: curve_report(
            select(runs, name), args.budgets, args.level, args.interval, args.boot, args.seed
        )
        for name in names
    }
    if args.format == "json":
        return to_json(reports)
    return "\n\n".join(curve_markdown(name, rep) for name, rep in reports.items())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stepbudget",
        description="Success rate as a function of the step or cost limit.",
    )
    parser.add_argument("--version", action="version", version=f"stepbudget {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("curve", help="success by budget for each system")
    _reader_options(p)
    p.add_argument("--system", action="append", help="system to report (repeatable; default all)")
    p.add_argument("--budgets", type=float, nargs="+", help="budgets to report")
    p.set_defaults(func=_cmd_curve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        out = args.func(args)
    except (ValueError, OSError) as exc:
        print(f"stepbudget: {exc}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
