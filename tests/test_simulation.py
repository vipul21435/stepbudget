import numpy as np
import pytest

from sim import as_runs, rerun_under, simulate, true_curve
from stepbudget.analysis import curve_for
from stepbudget.estimators import success_curve
from stepbudget.limits import extrapolate, score_at_limit

A, G = 0.03, 0.01


def test_estimator_recovers_the_curve_and_the_plain_share_does_not():
    used, event, *_ = simulate(40000, A, G, [10, 20, 40, 80, 160], seed=1)
    curve = success_curve(used, event)
    grid = np.array([5, 10, 20, 40, 80, 150])
    truth = true_curve(grid, A, G)
    assert np.asarray(curve.at(grid)) == pytest.approx(truth, abs=0.015)
    plain = np.asarray(curve.empirical_at(grid))
    # up to the smallest limit nothing is censored and both agree with the truth
    assert plain[:2] == pytest.approx(truth[:2], abs=0.015)
    # beyond it the plain share falls short by far more than the estimator's error
    assert truth[4] - plain[4] > 0.15
    assert truth[5] - plain[5] > 0.2


def test_dropping_limit_runs_is_biased_too():
    used, event, *_ = simulate(40000, A, G, [10, 20, 40, 80, 160], seed=2)
    kept = event != 0
    dropped = success_curve(used[kept], event[kept])
    full = success_curve(used, event)
    for budget in (10, 20, 40):
        truth = true_curve(budget, A, G)
        # runs that ended early are over-represented once the stopped ones are dropped
        assert dropped.at(budget) - truth > 0.05
        assert full.at(budget) == pytest.approx(truth, abs=0.015)


def test_interval_covers_the_truth_at_the_nominal_rate():
    hits = 0
    reps = 200
    for seed in range(reps):
        used, event, *_ = simulate(300, A, G, [20, 60, 120], seed=100 + seed)
        lo, hi = success_curve(used, event).interval_at(60.0, 0.95)
        hits += lo <= true_curve(60, A, G) <= hi
    assert 0.90 <= hits / reps <= 0.99


def test_truncation_equals_rerunning_under_the_smaller_limit():
    used, event, need, quit_at, limit = simulate(5000, A, G, [80], seed=3)
    runs = as_runs(used, event)
    for smaller in (10, 25, 40, 80):
        score = score_at_limit(runs, smaller)
        assert score.kind == "exact"
        assert score.none_solve == pytest.approx(score.all_solve)
        assert score.estimate == pytest.approx(rerun_under(need, quit_at, smaller))


def test_bounds_hold_the_rerun_score_when_runs_were_stopped_earlier():
    used, event, need, quit_at, limit = simulate(5000, A, G, [20, 40, 80], seed=4)
    runs = as_runs(used, event)
    for b in (30, 60, 80):
        score = score_at_limit(runs, b)
        assert score.kind == "bounded"
        truth = rerun_under(need, quit_at, b)
        assert score.none_solve <= truth <= score.all_solve
        assert score.none_solve <= score.estimate <= score.all_solve
        assert score.estimate == pytest.approx(truth, abs=0.03)
        assert score.stopped_before == pytest.approx(score.all_solve - score.none_solve)


def test_extrapolation_is_labelled_bounded_and_close_under_constant_rates():
    used, event, need, quit_at, limit = simulate(20000, A, G, [60], seed=5)
    runs = as_runs(used, event)
    score = score_at_limit(runs, 120)
    assert score.kind == "extrapolated"
    assert score.none_solve <= score.estimate <= score.all_solve
    # the simulated rates really are constant, so here the assumption holds
    assert score.estimate == pytest.approx(rerun_under(need, quit_at, 120), abs=0.02)
    assert score.none_solve == pytest.approx(float(np.mean(event == 1)))
    assert score.all_solve == pytest.approx(float(np.mean(event != 2)))
    far = score_at_limit(runs, 1e6)
    assert far.estimate <= far.all_solve + 1e-12


def test_extrapolation_without_late_successes_stays_flat():
    runs = as_runs([1, 2, 3, 10, 10], [1, 1, 2, 0, 0])
    curve = curve_for(runs)
    assert extrapolate(curve, runs, 50) == pytest.approx(float(curve.success[-1]))
    assert extrapolate(curve, runs, 2) == pytest.approx(float(curve.at(2)))
    with pytest.raises(ValueError):
        extrapolate(curve, runs, 50, tail=0)
    with pytest.raises(ValueError):
        score_at_limit(runs, -1)
