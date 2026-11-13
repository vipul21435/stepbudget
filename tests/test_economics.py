import numpy as np
import pytest

from sim import as_runs, simulate
from stepbudget.analysis import curve_for
from stepbudget.economics import cost_of_success, marginal_return


def test_spend_equals_mean_truncated_budget_when_nothing_is_stopped_earlier():
    used, event, *_ = simulate(2000, 0.03, 0.01, [80], seed=30)
    curve = curve_for(as_runs(used, event))
    for limit in (0, 7.5, 20, 55, 80):
        assert curve.spend_at(limit) == pytest.approx(float(np.mean(np.minimum(used, limit))))
    grid = np.array([10.0, 30.0, 80.0])
    assert np.asarray(curve.spend_at(grid)) == pytest.approx(
        [float(np.mean(np.minimum(used, b))) for b in grid]
    )


def test_cost_equals_total_spent_over_successes_by_truncation():
    used, event, *_ = simulate(2000, 0.03, 0.01, [80], seed=31)
    runs = as_runs(used, event)
    rep = cost_of_success(runs, [20, 40, 80], n_boot=100, seed=0)
    for row in rep["rows"]:
        b = row["limit"]
        solved = np.sum((event == 1) & (used <= b))
        assert row["cost"] == pytest.approx(np.sum(np.minimum(used, b)) / solved)
        assert row["low"] <= row["cost"] <= row["high"]
    # at the limit the runs really had, the estimate is the plain ratio
    assert rep["rows"][-1]["cost"] == pytest.approx(rep["as_run"]["cost"])
    assert rep["as_run"]["cost"] == pytest.approx(used.sum() / np.sum(event == 1))
    assert rep["as_run"]["spend"] == pytest.approx(used.mean())


def test_cost_is_deterministic_and_rejects_limits_beyond_the_data():
    used, event, *_ = simulate(300, 0.03, 0.01, [40, 80], seed=32)
    runs = as_runs(used, event)
    assert cost_of_success(runs, [30], n_boot=80, seed=3) == cost_of_success(
        runs, [30], n_boot=80, seed=3
    )
    with pytest.raises(ValueError, match="must not exceed"):
        cost_of_success(runs, [500], n_boot=10)


def test_cost_with_no_success_is_undefined_not_infinite():
    runs = as_runs([5, 6, 7], [2, 0, 2])
    rep = cost_of_success(runs, [6], n_boot=20)
    assert np.isnan(rep["rows"][0]["cost"])
    assert np.isnan(rep["as_run"]["cost"])


def test_marginal_gains_add_up_to_the_curve():
    used, event, *_ = simulate(1500, 0.03, 0.01, [40, 80], seed=33)
    runs = as_runs(used, event)
    edges = [0, 10, 20, 40, 60, 80]
    rep = marginal_return(runs, edges)
    curve = curve_for(runs)
    assert sum(r["gain"] for r in rep["rows"]) == pytest.approx(float(curve.at(80)))
    for row in rep["rows"]:
        width = row["to"] - row["from"]
        assert row["per_unit_limit"] == pytest.approx(row["gain"] / width)
        # only runs still going spend budget, so the gain per unit spent is the larger one
        assert row["per_unit_spent"] >= row["per_unit_limit"] - 1e-12
    assert rep["below_threshold_from"] is None and rep["threshold"] is None


def test_marginal_per_unit_spent_recovers_the_success_rate():
    # with constant rates the gain per unit actually spent is the success rate itself
    used, event, *_ = simulate(60000, 0.03, 0.01, [100], seed=34)
    rep = marginal_return(as_runs(used, event), [0, 20, 40, 60])
    for row in rep["rows"]:
        assert row["per_unit_spent"] == pytest.approx(0.03, rel=0.08)


def test_threshold_names_the_budget_from_which_the_gain_stays_low():
    # successes at 1..4, then nothing up to 10
    runs = as_runs([1, 2, 3, 4, 10, 10, 10, 10], [1, 1, 1, 1, 2, 2, 0, 0])
    rep = marginal_return(runs, [0, 2, 4, 6, 8, 10], threshold=0.01)
    assert rep["below_threshold_from"] == pytest.approx(4.0)
    rising = marginal_return(runs, [0, 2, 4], threshold=0.01)
    assert rising["below_threshold_from"] is None
    flat = marginal_return(runs, [6, 8, 10], threshold=0.01)
    assert flat["below_threshold_from"] == pytest.approx(6.0)
    with pytest.raises(ValueError):
        marginal_return(runs, [3])
    with pytest.raises(ValueError, match="must not exceed"):
        marginal_return(runs, [0, 50])
