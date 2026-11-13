import numpy as np
import pytest

from sim import as_runs, simulate
from stepbudget.analysis import (
    Prepared,
    bootstrap_items,
    curve_for,
    curve_interval,
    has_repeats,
    item_weights,
    select,
    systems,
)
from stepbudget.data import Run
from stepbudget.estimators import success_curve


def test_item_weights_make_every_item_count_once():
    runs = [Run("a", "s", 1, 1), Run("a", "s", 2, 2), Run("a", "s", 3, 0), Run("b", "s", 4, 1)]
    assert item_weights(runs) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1])
    assert has_repeats(runs)
    curve = curve_for(runs)
    assert curve.total == pytest.approx(2.0)
    # item a contributes a third of a success, item b a whole one, out of two items
    assert curve.empirical[-1] == pytest.approx((1 / 3 + 1) / 2)


def test_repeating_every_run_leaves_the_curve_unchanged():
    used, event, *_ = simulate(200, 0.03, 0.01, [20, 60], seed=7)
    once = as_runs(used, event)
    thrice = once * 3
    a, b = curve_for(once), curve_for(thrice)
    assert a.times == pytest.approx(b.times)
    assert a.success == pytest.approx(b.success)


def test_prepared_matches_success_curve():
    used, event, *_ = simulate(300, 0.03, 0.01, [20, 60], seed=8)
    runs = as_runs(used, event)
    prep = Prepared.from_runs(runs)
    curve = success_curve(used, event)
    f1, running = prep.curves()
    assert f1 == pytest.approx(curve.success)
    assert running == pytest.approx(curve.running)
    grid = np.array([0.0, 5.0, 30.0, 1000.0])
    assert prep.success_at(grid) == pytest.approx(np.asarray(curve.at(grid)))
    counts = np.ones(len(prep.items), dtype=int)
    assert prep.success_at(grid, counts) == pytest.approx(np.asarray(curve.at(grid)))


def test_bootstrap_is_deterministic_for_a_seed():
    used, event, *_ = simulate(150, 0.03, 0.01, [20, 60], seed=9)
    runs = as_runs(used, event)
    one = curve_interval(runs, [10, 40], method="bootstrap", n_boot=200, seed=5)
    two = curve_interval(runs, [10, 40], method="bootstrap", n_boot=200, seed=5)
    other = curve_interval(runs, [10, 40], method="bootstrap", n_boot=200, seed=6)
    assert one[0].tolist() == two[0].tolist() and one[1].tolist() == two[1].tolist()
    assert one[0].tolist() != other[0].tolist()
    assert one[2] == "bootstrap"


def test_bootstrap_and_analytic_intervals_agree_for_independent_runs():
    used, event, *_ = simulate(600, 0.03, 0.01, [20, 60, 120], seed=10)
    runs = as_runs(used, event)
    lo_a, hi_a, how = curve_interval(runs, [30, 90])
    lo_b, hi_b, _ = curve_interval(runs, [30, 90], method="bootstrap", n_boot=600, seed=1)
    assert how == "analytic"
    assert lo_a == pytest.approx(lo_b, abs=0.02)
    assert hi_a == pytest.approx(hi_b, abs=0.02)


def test_clustered_runs_get_a_wider_interval_than_independence_would_give():
    # every item is run 10 times with the same result: 10x the rows, no extra information
    used, event, *_ = simulate(80, 0.03, 0.01, [200], seed=11)
    runs = [
        Run(f"item-{i}", "s", float(b), int(e))
        for i, (b, e) in enumerate(zip(used, event, strict=True))
        for _ in range(10)
    ]
    lo, hi, how = curve_interval(runs, [40], n_boot=500, seed=2)
    assert how == "bootstrap"
    naive = success_curve([r.budget for r in runs], [r.event for r in runs])
    n_lo, n_hi = naive.interval_at(40.0)
    assert (hi[0] - lo[0]) > 2.0 * (n_hi - n_lo)
    single = as_runs(used, event)
    s_lo, s_hi, _ = curve_interval(single, [40], method="analytic")
    assert (hi[0] - lo[0]) == pytest.approx(s_hi[0] - s_lo[0], rel=0.25)


def test_systems_select_and_bad_arguments():
    runs = [Run("a", "x", 1, 1), Run("a", "y", 2, 1), Run("b", "x", 3, 2)]
    assert systems(runs) == ["x", "y"]
    assert len(select(runs, "x")) == 2
    with pytest.raises(ValueError, match="x, y"):
        select(runs, "z")
    with pytest.raises(ValueError):
        curve_interval(runs, [1], method="exact")
    with pytest.raises(ValueError):
        bootstrap_items(3, lambda c: c, n_boot=0)
