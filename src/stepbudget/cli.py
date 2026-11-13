"""Command line: ``stepbudget curve | at | compare | marginal | cost``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .analysis import curve_for, curve_interval, has_repeats, item_weights, select, systems
from .compare import compare
from .data import DEFAULT_LIMIT_REASONS, Run, read_runs
from .economics import cost_of_success, marginal_return
from .estimators import CENSORED, GAVE_UP, SUCCESS
from .limits import score_at_limit
from .report import fmt_budget, num, pct, pct_range, table, to_json


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
    ended = np.sort([r.budget for r in runs])
    for b, low, high in zip(grid, lo, hi, strict=True):
        rows.append(
            {
                "budget": b,
                "success": curve.at(b),
                "low": float(low),
                "high": float(high),
                "plain": curve.empirical_at(b),
                "running": curve.running_at(b),
                "runs_past": int(len(ended) - np.searchsorted(ended, b, side="right")),
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
        f"{c['gave_up']} ended unsolved before the limit, "
        f"{c['limit_reached']} ({pct(c['limit_reached'] / c['runs'])} of runs) "
        "were stopped by the limit.",
        "",
    ]
    rows = [
        [
            fmt_budget(r["budget"]),
            pct(r["success"]),
            pct_range(r["low"], r["high"]),
            pct(r["plain"]),
            pct(r["running"]),
            str(r["runs_past"]),
        ]
        for r in rep["rows"]
    ]
    header = [
        "budget",
        "solved within budget",
        f"{level} interval",
        "plain share",
        "still running",
        "runs past budget",
    ]
    how = "delta-method variance" if rep["interval"] == "analytic" else "bootstrap over items"
    tail = ["", f"Interval: {how}. {difference_note(rep)}"]
    return "\n".join(head + [table(header, rows)] + tail)


def at_report(
    runs: Sequence[Run],
    budgets: Sequence[float] = (),
    fractions: Sequence[float] = (),
    level: float = 0.95,
    method: str = "auto",
    n_boot: int = 1000,
    seed: int = 0,
    tail: float = 0.5,
) -> dict:
    """What other limits would have scored for one system."""
    curve = curve_for(runs)
    top = curve.max_budget
    wanted = [float(b) for b in budgets] + [f * top for f in fractions]
    if not wanted:
        wanted = [0.25 * top, 0.5 * top]
    wanted = sorted(set(wanted))
    inside = [b for b in wanted if b <= top]
    lo, hi, used = curve_interval(runs, inside, level, method, n_boot, seed)
    bands = {b: (float(low), float(high)) for b, low, high in zip(inside, lo, hi, strict=True)}
    rows = []
    for b in wanted:
        row = score_at_limit(runs, b, tail).as_dict()
        row["low"], row["high"] = bands.get(b, (None, None))
        rows.append(row)
    w = item_weights(runs)
    total = float(w.sum())
    as_run = {
        "solved": float(sum(wi for wi, r in zip(w, runs, strict=True) if r.event == SUCCESS))
        / total,
        "stopped_by_limit": float(
            sum(wi for wi, r in zip(w, runs, strict=True) if r.event == CENSORED)
        )
        / total,
        "largest_budget": top,
    }
    return {
        "counts": counts(runs),
        "level": level,
        "interval": used,
        "as_run": as_run,
        "rows": rows,
    }


KIND_NOTE = {
    "exact": "by truncation",
    "bounded": "censoring-aware estimate",
    "extrapolated": "extrapolation",
}


def at_markdown(name: str, rep: dict) -> str:
    level = f"{100 * rep['level']:g} %"
    run = rep["as_run"]
    head = [
        f"## {name}",
        "",
        f"As run: {pct(run['solved'])} of items solved, {pct(run['stopped_by_limit'])} stopped "
        f"by the limit, largest budget used {fmt_budget(run['largest_budget'])}.",
        "",
    ]
    rows = [
        [
            fmt_budget(r["budget"]),
            pct(r["estimate"]),
            KIND_NOTE[r["kind"]],
            pct(r["none_solve"]),
            pct(r["all_solve"]),
            pct_range(r["low"], r["high"]),
        ]
        for r in rep["rows"]
    ]
    header = [
        "limit",
        "solved",
        "how",
        "if no stopped run solves",
        "if every stopped run solves",
        f"{level} interval",
    ]
    notes = []
    kinds = {r["kind"] for r in rep["rows"]}
    if "exact" in kinds:
        notes.append(
            "By truncation: no run was stopped before this limit, so the score is a count, "
            "given that a run behaves the same up to the smaller limit."
        )
    if "bounded" in kinds:
        notes.append(
            "Censoring-aware estimate: some runs were stopped before this limit; the two "
            "bound columns hold without assumptions."
        )
    if "extrapolated" in kinds:
        notes.append(
            "Extrapolation: the limit is beyond every observed budget. The figure assumes the "
            "success and stopping rates of the last part of the observed range continue; "
            "only the two bound columns are supported by the data."
        )
    return "\n".join(head + [table(header, rows), ""] + notes)


def compare_markdown(a: str, b: str, rep: dict) -> str:
    level = f"{100 * rep['level']:g} %"
    lines = [
        f"## {a} against {b}",
        "",
        f"{rep['items']} items run by both systems; difference is {a} minus {b}; "
        f"{level} interval from a paired bootstrap over items "
        f"({rep['n_boot']} replicates, seed {rep['seed']}).",
        "",
    ]
    rows = [
        [
            fmt_budget(r["budget"]),
            pct(r["a"]),
            pct(r["b"]),
            f"{100 * r['difference']:+.1f} points",
            f"{100 * r['low']:+.1f} to {100 * r['high']:+.1f}",
            pct(r["share_a_ahead"]),
        ]
        for r in rep["rows"]
    ]
    header = ["budget", a, b, "difference", f"{level} interval", f"replicates with {a} ahead"]
    lines += [table(header, rows), ""]
    lr = rep["logrank"]
    lines.append(
        f"Log-rank test on the budget to success: chi-square {num(lr['chi2'])} on 1 degree of "
        f"freedom, p = {num(lr['p'], 3)} ({a} had {num(lr['observed_a'], 6)} successes against "
        f"{num(lr['expected_a'], 6)} expected if both succeeded at the same rate among runs "
        "still going). The test treats runs as independent."
    )
    cross = rep["crossovers"]
    name = {"a": a, "b": b, None: "neither"}
    if not cross["changes"]:
        if cross["final_leader"] is None:
            lines.append("The curves never separate.")
        else:
            lines.append(
                f"No crossover: {name[cross['final_leader']]} is ahead wherever the curves "
                f"differ, up to budget {fmt_budget(cross['up_to'])}."
            )
    else:
        steps = ", ".join(
            f"{name[c['leader']]} from {fmt_budget(c['budget'])}" for c in cross["changes"]
        )
        lines.append(
            f"Lead changes: {name[cross['first_leader']]} is ahead first, then {steps} "
            f"(curves compared up to budget {fmt_budget(cross['up_to'])})."
        )
    if cross["final_leader_share"] is not None:
        lines.append(
            f"{name[cross['final_leader']]} is ahead at budget {fmt_budget(cross['up_to'])} in "
            f"{pct(cross['final_leader_share'])} of the bootstrap replicates."
        )
    return "\n".join(lines)


def marginal_markdown(name: str, rep: dict) -> str:
    rows = [
        [
            f"{fmt_budget(r['from'])} to {fmt_budget(r['to'])}",
            f"{100 * r['gain']:+.2f} points",
            num(100 * r["per_unit_limit"], 3),
            num(100 * r["per_unit_spent"], 3),
        ]
        for r in rep["rows"]
    ]
    header = [
        "limit raised from",
        "solved share gained",
        "successes per 100 items per unit of limit",
        "successes per 100 units actually spent",
    ]
    lines = [f"## {name}", "", table(header, rows)]
    if rep["threshold"] is not None:
        thr = num(100 * rep["threshold"], 3)
        if rep["below_threshold_from"] is None:
            lines += ["", f"The gain is still at or above {thr} per 100 items per unit at the end."]
        else:
            lines += [
                "",
                f"From budget {fmt_budget(rep['below_threshold_from'])} on, each further unit "
                f"of limit adds fewer than {thr} successes per 100 items.",
            ]
    return "\n".join(lines)


def cost_markdown(name: str, rep: dict) -> str:
    level = f"{100 * rep['level']:g} %"
    rows = [
        [
            fmt_budget(r["limit"]),
            pct(r["solved"]),
            num(r["spend"]),
            num(r["cost"]),
            f"{num(r['low'])} to {num(r['high'])}",
        ]
        for r in rep["rows"]
    ]
    header = ["limit", "solved", "spent per item", "spent per solved item", f"{level} interval"]
    run = rep["as_run"]
    tail = (
        f"As run: {num(run['spend'])} spent per item, {pct(run['solved'])} solved, "
        f"{num(run['cost'])} per solved item. Interval: bootstrap over items "
        f"({rep['n_boot']} replicates, seed {rep['seed']})."
    )
    return "\n".join([f"## {name}", "", table(header, rows), "", tail])


def step_edges(runs: Sequence[Run], step: float | None) -> list[float]:
    """Budgets 0, step, 2 * step, ... up to the largest budget used."""
    top = max(r.budget for r in runs)
    if step is None:
        step = top / 10.0
        if all(float(r.budget).is_integer() for r in runs):
            step = max(1.0, round(step))
    if step <= 0:
        raise ValueError("step must be positive")
    edges = list(np.arange(0.0, top + 1e-9, step))
    if edges[-1] < top:
        edges.append(top)
    return [float(e) for e in edges]


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
    if args.plot:
        from .plot import plot_curves

        plot_curves(
            {name: select(runs, name) for name in names},
            args.plot,
            level=args.level,
            seed=args.seed,
            xlabel=args.xlabel or args.budget_col,
            xmax=args.xmax,
        )
    if args.format == "json":
        return to_json(reports)
    return "\n\n".join(curve_markdown(name, rep) for name, rep in reports.items())


def _per_system(args: argparse.Namespace, build, render) -> str:
    runs = _load(args)
    names = args.system or systems(runs)
    reports = {name: build(select(runs, name)) for name in names}
    if args.format == "json":
        return to_json(reports)
    return "\n\n".join(render(name, rep) for name, rep in reports.items())


def _cmd_at(args: argparse.Namespace) -> str:
    return _per_system(
        args,
        lambda runs: at_report(
            runs,
            args.budgets or (),
            args.fractions or (),
            args.level,
            args.interval,
            args.boot,
            args.seed,
            args.tail,
        ),
        at_markdown,
    )


def _cmd_marginal(args: argparse.Namespace) -> str:
    return _per_system(
        args,
        lambda runs: marginal_return(runs, step_edges(runs, args.step), args.threshold),
        marginal_markdown,
    )


def _cmd_cost(args: argparse.Namespace) -> str:
    return _per_system(
        args,
        lambda runs: cost_of_success(
            runs, args.limits or default_grid(runs, 5), args.level, args.boot, args.seed
        ),
        cost_markdown,
    )


def _cmd_compare(args: argparse.Namespace) -> str:
    runs = _load(args)
    names = systems(runs)
    if args.a is None or args.b is None:
        if len(names) != 2:
            raise ValueError(
                "name the two systems with --a and --b (the file has: " + ", ".join(names) + ")"
            )
        a, b = names
    else:
        a, b = args.a, args.b
    runs_a, runs_b = select(runs, a), select(runs, b)
    budgets = args.budgets
    if not budgets:
        top = min(max(r.budget for r in runs_a), max(r.budget for r in runs_b))
        budgets = [g for g in default_grid(runs_a + runs_b, 5) if g <= top]
    rep = compare(runs_a, runs_b, budgets, args.level, args.boot, args.seed)
    rep["a"], rep["b"] = a, b
    if args.format == "json":
        return to_json(rep)
    return compare_markdown(a, b, rep)


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
    p.add_argument("--plot", help="also write a chart of the curves here (needs matplotlib)")
    p.add_argument("--xlabel", help="label of the chart's budget axis")
    p.add_argument("--xmax", type=float, help="right end of the chart's budget axis")
    p.set_defaults(func=_cmd_curve)

    p = sub.add_parser("at", help="what a smaller or larger limit would have scored")
    _reader_options(p)
    p.add_argument("--system", action="append", help="system to report (repeatable; default all)")
    p.add_argument("--budgets", type=float, nargs="+", help="limits to score")
    p.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        help="limits as fractions of the largest budget used (default 0.25 0.5)",
    )
    p.add_argument(
        "--tail",
        type=float,
        default=0.5,
        help="fraction of the observed range whose rates feed the extrapolation",
    )
    p.set_defaults(func=_cmd_at)

    p = sub.add_parser("compare", help="two systems at equal budgets")
    _reader_options(p)
    p.add_argument("--a", help="first system")
    p.add_argument("--b", help="second system")
    p.add_argument("--budgets", type=float, nargs="+", help="budgets to compare at")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("marginal", help="successes gained per additional unit of limit")
    _reader_options(p)
    p.add_argument("--system", action="append", help="system to report (repeatable; default all)")
    p.add_argument("--step", type=float, help="width of each step of the limit")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="gain in solved share per unit of limit below which a larger limit is called flat",
    )
    p.set_defaults(func=_cmd_marginal)

    p = sub.add_parser("cost", help="budget spent per solved item, failures included")
    _reader_options(p)
    p.add_argument("--system", action="append", help="system to report (repeatable; default all)")
    p.add_argument("--limits", type=float, nargs="+", help="limits to cost")
    p.set_defaults(func=_cmd_cost)
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
