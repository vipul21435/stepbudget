"""Chart of the success-by-budget curves (needs the ``plot`` extra)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .analysis import Prepared, bootstrap_items, curve_for, has_repeats, percentile_interval
from .data import Run


def plot_curves(
    by_system: Mapping[str, Sequence[Run]],
    path: str | Path,
    level: float = 0.95,
    n_boot: int = 300,
    seed: int = 0,
    xlabel: str = "budget",
    title: str | None = None,
) -> Path:
    """Draw each system's censoring-aware curve with its band and the plain share.

    Solid line: cumulative incidence of success. Shaded: pointwise interval
    (bootstrap over items when items have repeated runs). Dashed: plain share
    of runs solved within the budget.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ValueError("the chart needs matplotlib: pip install 'stepbudget[plot]'") from exc

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=120)
    for name, runs in by_system.items():
        curve = curve_for(runs)
        x = np.concatenate(([0.0], curve.times))
        y = np.concatenate(([0.0], curve.success))
        (line,) = ax.step(x, 100 * y, where="post", label=name, linewidth=1.6)
        if has_repeats(runs):
            prep = Prepared.from_runs(runs)
            samples = bootstrap_items(
                len(prep.items), lambda c, p=prep: p.curves(c)[0], n_boot, seed
            )
            lo, hi = percentile_interval(samples, level)
        else:
            lo, hi = curve.interval(level)
        ax.fill_between(
            x,
            100 * np.concatenate(([0.0], lo)),
            100 * np.concatenate(([0.0], hi)),
            step="post",
            alpha=0.18,
            color=line.get_color(),
            linewidth=0,
        )
        ax.step(
            x,
            100 * np.concatenate(([0.0], curve.empirical)),
            where="post",
            linestyle="--",
            linewidth=1.0,
            color=line.get_color(),
        )
    ax.plot([], [], color="0.3", linestyle="--", linewidth=1.0, label="plain share (dashed)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("items solved within budget (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out)
    plt.close(fig)
    return out
