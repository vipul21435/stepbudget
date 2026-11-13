"""Comparison against the lifelines package (test-only dependency)."""

import warnings

import numpy as np
import pytest

from stepbudget.estimators import kaplan_meier, success_curve

lifelines = pytest.importorskip("lifelines")


def test_kaplan_meier_matches_lifelines_with_ties():
    rng = np.random.default_rng(21)
    t = rng.integers(1, 30, size=250).astype(float)
    obs = rng.random(250) < 0.7
    ours = kaplan_meier(t, obs)
    ref = lifelines.KaplanMeierFitter().fit(t, obs)
    theirs = ref.survival_function_at_times(ours.times).to_numpy()
    assert ours.survival == pytest.approx(theirs, abs=1e-10)


def test_cumulative_incidence_matches_lifelines_without_ties():
    rng = np.random.default_rng(22)
    n = 200
    t = np.sort(rng.exponential(10.0, size=n)) + np.arange(n) * 1e-6
    events = rng.choice([0, 1, 2], size=n, p=[0.25, 0.45, 0.30])
    ours = success_curve(t, events)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter = lifelines.AalenJohansenFitter(calculate_variance=True)
        ref = fitter.fit(t, events, event_of_interest=1)
    index = np.asarray(ref.cumulative_density_.index, dtype=float)
    where = np.searchsorted(index, ours.times, side="right") - 1
    theirs = np.asarray(ref.cumulative_density_).reshape(len(index), -1)[where, 0]
    assert ours.success == pytest.approx(theirs, abs=1e-10)
    their_var = np.asarray(ref.variance_).reshape(len(index), -1)[where, 0]
    # lifelines divides by zero at the last budget when no run is left after it
    finite = np.isfinite(their_var)
    assert finite.sum() >= len(their_var) - 1
    assert ours.variance[finite] == pytest.approx(their_var[finite], rel=1e-6, abs=1e-12)
