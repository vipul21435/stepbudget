import numpy as np
import pytest

from stepbudget.estimators import (
    CENSORED,
    GAVE_UP,
    SUCCESS,
    event_table,
    kaplan_meier,
    success_curve,
)


def test_kaplan_meier_hand_worked_table():
    # times 1, 2+, 3, 3, 4+, 5 (+ marks censoring)
    km = kaplan_meier([1, 2, 3, 3, 4, 5], [1, 0, 1, 1, 0, 1])
    assert km.times.tolist() == [1, 2, 3, 4, 5]
    assert km.at_risk.tolist() == [6, 5, 4, 2, 1]
    expected = [5 / 6, 5 / 6, 5 / 6 * 2 / 4, 5 / 6 * 2 / 4, 0.0]
    assert km.survival == pytest.approx(expected)
    # Greenwood at t=3: S^2 * (1/(6*5) + 2/(4*2))
    s3 = 5 / 6 * 2 / 4
    assert km.variance[2] == pytest.approx(s3**2 * (1 / 30 + 2 / 8))
    assert km.at(0.5) == pytest.approx(1.0)
    assert km.at(3.5) == pytest.approx(s3)


def test_success_curve_hand_worked_table():
    # budget: 1 success, 2 gave up, 2 success, 3 limit, 4 success, 5 limit
    budgets = [1, 2, 2, 3, 4, 5]
    events = [SUCCESS, GAVE_UP, SUCCESS, CENSORED, SUCCESS, CENSORED]
    c = success_curve(budgets, events)
    assert c.times.tolist() == [1, 2, 3, 4, 5]
    assert c.at_risk.tolist() == [6, 5, 3, 2, 1]
    # running: 5/6, 5/6*3/5 = 1/2, 1/2, 1/2*1/2 = 1/4, 1/4
    assert c.running == pytest.approx([5 / 6, 0.5, 0.5, 0.25, 0.25])
    # success: 1/6, +5/6*1/5 = 1/3, 1/3, +1/2*1/2 = 7/12, 7/12
    assert c.success == pytest.approx([1 / 6, 1 / 3, 1 / 3, 7 / 12, 7 / 12])
    assert c.gave_up == pytest.approx([0, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
    assert c.empirical == pytest.approx([1 / 6, 2 / 6, 2 / 6, 3 / 6, 3 / 6])
    assert c.first_censoring == pytest.approx(3.0)
    assert c.at(0) == pytest.approx(0.0)
    assert c.at(4.5) == pytest.approx(7 / 12)
    assert c.running_at(0.5) == pytest.approx(1.0)


def test_shares_add_to_one():
    rng = np.random.default_rng(3)
    budgets = rng.integers(1, 40, size=300)
    events = rng.integers(0, 3, size=300)
    c = success_curve(budgets, events)
    assert c.success + c.gave_up + c.running == pytest.approx(np.ones(len(c.times)))
    assert np.all(np.diff(c.success) >= -1e-12)


def test_without_censoring_curve_is_empirical_and_variance_binomial():
    rng = np.random.default_rng(5)
    budgets = rng.integers(1, 30, size=200)
    events = rng.choice([SUCCESS, GAVE_UP], size=200)
    c = success_curve(budgets, events)
    assert c.success == pytest.approx(c.empirical)
    assert c.first_censoring is None
    p = c.success
    assert c.variance == pytest.approx(p * (1 - p) / 200)


def test_censoring_only_at_the_largest_budget_keeps_curve_empirical():
    budgets = [3, 5, 8, 10, 10, 10]
    events = [SUCCESS, GAVE_UP, SUCCESS, CENSORED, CENSORED, SUCCESS]
    c = success_curve(budgets, events)
    assert c.success == pytest.approx(c.empirical)


def test_without_giving_up_curve_is_one_minus_kaplan_meier():
    rng = np.random.default_rng(7)
    budgets = rng.integers(1, 25, size=150)
    events = rng.choice([SUCCESS, CENSORED], size=150, p=[0.6, 0.4])
    c = success_curve(budgets, events)
    km = kaplan_meier(budgets, events == SUCCESS)
    assert c.success == pytest.approx(1 - km.survival)
    # with a single event type the variance reduces to Greenwood's formula
    assert c.variance == pytest.approx(km.variance)


def test_integer_weights_equal_repeated_rows():
    budgets = np.array([1, 2, 2, 3, 4, 5, 6])
    events = np.array([1, 2, 1, 0, 1, 0, 2])
    w = np.array([1, 3, 2, 1, 2, 1, 2])
    weighted = success_curve(budgets, events, w)
    repeated = success_curve(np.repeat(budgets, w), np.repeat(events, w))
    assert weighted.success == pytest.approx(repeated.success)
    assert weighted.running == pytest.approx(repeated.running)


def test_interval_contains_estimate_and_stays_in_unit_range():
    rng = np.random.default_rng(11)
    c = success_curve(rng.integers(1, 50, size=120), rng.integers(0, 3, size=120))
    lo, hi = c.interval(0.95)
    assert np.all(lo <= c.success + 1e-12) and np.all(c.success <= hi + 1e-12)
    assert np.all(lo >= 0) and np.all(hi <= 1)
    lo90, hi90 = c.interval(0.90)
    assert np.all(hi90 - lo90 <= hi - lo + 1e-12)


@pytest.mark.parametrize(
    "budgets, events",
    [([], []), ([1, -2], [1, 1]), ([1, 2], [1, 3]), ([1, float("nan")], [1, 1]), ([1, 2], [1])],
)
def test_bad_input_is_rejected(budgets, events):
    with pytest.raises(ValueError):
        event_table(budgets, events)


def test_bad_weights_and_level_are_rejected():
    with pytest.raises(ValueError):
        success_curve([1, 2], [1, 1], [0, 0])
    with pytest.raises(ValueError):
        success_curve([1, 2], [1, 1], [1, -1])
    with pytest.raises(ValueError):
        success_curve([1, 2], [1, 0]).interval(1.5)
    with pytest.raises(ValueError):
        kaplan_meier([1, 2], [1, 2])
