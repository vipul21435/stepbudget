"""Use stepbudget from Python on the synthetic example runs.

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from stepbudget import (
    compare,
    cost_of_success,
    curve_for,
    marginal_return,
    read_runs,
    score_at_limit,
    select,
)


def main() -> None:
    runs = read_runs(Path(__file__).with_name("runs.csv"))
    alpha, beta = select(runs, "alpha"), select(runs, "beta")

    curve = curve_for(alpha)
    low, high = curve.interval_at(60, level=0.95)
    print(f"alpha solved within 60 steps: {curve.at(60):.3f} ({low:.3f} to {high:.3f})")
    print(f"plain share at 60 steps:      {curve.empirical_at(60):.3f}")

    for limit in (20, 60, 120):
        s = score_at_limit(alpha, limit)
        print(
            f"limit {limit:>3}: {s.estimate:.3f} [{s.kind}], "
            f"between {s.none_solve:.3f} and {s.all_solve:.3f}"
        )

    rep = compare(alpha, beta, [20, 80], n_boot=500, seed=0)
    for row in rep["rows"]:
        print(
            f"alpha - beta at {row['budget']:.0f}: {row['difference']:+.3f} "
            f"({row['low']:+.3f} to {row['high']:+.3f})"
        )
    print("lead changes:", rep["crossovers"]["changes"])

    gain = marginal_return(alpha, [0, 20, 40, 60, 80], threshold=0.001)
    print("gain per step:", [round(r["per_unit_limit"], 4) for r in gain["rows"]])

    cost = cost_of_success(beta, [40, 80], n_boot=500, seed=0)
    for row in cost["rows"]:
        print(f"beta, limit {row['limit']:.0f}: {row['cost']:.1f} steps per solved item")


if __name__ == "__main__":
    main()
