"""Write examples/runs.csv: seeded synthetic runs of two made-up systems.

Each item has a hidden number of steps a system needs to solve it and a hidden
number of steps after which the system gives up; the run ends at whichever
comes first, or at the step limit. "alpha" was run under a limit of 40 steps on
half the items and 80 on the rest; "beta" under 80 everywhere.

    python examples/make_runs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

N_ITEMS = 120


def main() -> None:
    rng = np.random.default_rng(2026)
    difficulty = rng.normal(0.0, 0.6, size=N_ITEMS)
    rows = []
    for system, median, give_up_mean in (("alpha", 30.0, 90.0), ("beta", 45.0, 300.0)):
        need = np.ceil(np.exp(np.log(median) + difficulty + rng.normal(0, 0.5, N_ITEMS)))
        quit_at = np.ceil(rng.exponential(give_up_mean, N_ITEMS))
        for i in range(N_ITEMS):
            limit = 40 if system == "alpha" and i % 2 == 0 else 80
            if need[i] <= min(quit_at[i], limit):
                steps, outcome, reason = need[i], "success", "submitted"
            elif quit_at[i] <= limit:
                steps, outcome, reason = quit_at[i], "failure", "gave up"
            else:
                steps, outcome, reason = limit, "failure", "limit reached"
            rows.append((f"item-{i + 1:03d}", system, outcome, int(steps), reason))
    out = Path(__file__).with_name("runs.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "system", "outcome", "budget", "exit_reason"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
