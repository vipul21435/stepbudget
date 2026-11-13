import numpy as np
import pytest

from sim import as_runs, simulate
from stepbudget.analysis import curve_for
from stepbudget.compare import compare, crossovers, logrank
from stepbudget.data import Run


def two_systems(n=200, seed=0):
    ua, ea, *_ = simulate(n, 0.05, 0.02, [30, 90], seed=seed)
    ub, eb, *_ = simulate(n, 0.02, 0.002, [90], seed=seed + 1)
    return as_runs(ua, ea, "fast"), as_runs(ub, eb, "slow")


def test_paired_bootstrap_is_deterministic_for_a_seed():
    a, b = two_systems()
    one = compare(a, b, [10, 30, 60], n_boot=300, seed=4)
    two = compare(a, b, [10, 30, 60], n_boot=300, seed=4)
    other = compare(a, b, [10, 30, 60], n_boot=300, seed=5)
    assert one == two
    assert [r["low"] for r in one["rows"]] != [r["low"] for r in other["rows"]]
    # the point estimates do not depend on the seed
    assert [r["difference"] for r in one["rows"]] == [r["difference"] for r in other["rows"]]


def test_difference_matches_the_two_curves_and_sits_in_its_interval():
    a, b = two_systems()
    rep = compare(a, b, [10, 30, 60], n_boot=400, seed=1)
    ca, cb = curve_for(a), curve_for(b)
    for row in rep["rows"]:
        assert row["a"] == pytest.approx(float(ca.at(row["budget"])))
        assert row["b"] == pytest.approx(float(cb.at(row["budget"])))
        assert row["difference"] == pytest.approx(row["a"] - row["b"])
        assert row["low"] <= row["difference"] <= row["high"]
        assert 0.0 <= row["share_a_ahead"] <= 1.0
    assert rep["items"] == 200


def test_a_system_against_itself_has_zero_difference_in_every_replicate():
    a, _ = two_systems()
    same = [Run(r.item, "copy", r.budget, r.event) for r in a]
    rep = compare(a, same, [10, 30], n_boot=100, seed=0)
    for row in rep["rows"]:
        assert row["difference"] == pytest.approx(0.0, abs=1e-12)
        assert row["low"] == pytest.approx(0.0, abs=1e-12)
        assert row["high"] == pytest.approx(0.0, abs=1e-12)
    assert rep["logrank"]["chi2"] == pytest.approx(0.0, abs=1e-12)
    assert rep["logrank"]["p"] == pytest.approx(1.0)
    assert rep["crossovers"]["final_leader"] is None


def test_pairing_narrows_the_interval_when_items_drive_both_systems():
    rng = np.random.default_rng(12)
    n = 300
    hard = rng.random(n) < 0.5
    budget_a = np.where(hard, 50.0, 10.0) + rng.integers(0, 3, n)
    budget_b = np.where(hard, 52.0, 12.0) + rng.integers(0, 3, n)
    a = [Run(f"i{k}", "a", float(budget_a[k]), 1) for k in range(n)]
    b = [Run(f"i{k}", "b", float(budget_b[k]), 1) for k in range(n)]
    paired = compare(a, b, [30], n_boot=400, seed=3)["rows"][0]
    shuffled = [Run(f"i{(k * 7 + 1) % n}", "b", r.budget, r.event) for k, r in enumerate(b)]
    unpaired = compare(a, shuffled, [30], n_boot=400, seed=3)["rows"][0]
    assert paired["high"] - paired["low"] < 0.5 * (unpaired["high"] - unpaired["low"])


def test_only_common_items_are_compared():
    a = [Run("x", "a", 1, 1), Run("y", "a", 2, 1), Run("z", "a", 3, 2)]
    b = [Run("y", "b", 2, 1), Run("z", "b", 4, 1), Run("w", "b", 5, 0)]
    rep = compare(a, b, [2], n_boot=50)
    assert rep["items"] == 2 and rep["dropped_a"] == 1 and rep["dropped_b"] == 1
    with pytest.raises(ValueError, match="share no items"):
        compare(a[:1], b[2:], [1], n_boot=10)
    with pytest.raises(ValueError, match="must not exceed"):
        compare(a, b, [100], n_boot=10)


def test_crossover_is_found_where_the_lead_changes():
    # a solves early then stalls; b starts late and ends ahead
    a = as_runs([1, 2, 3, 20, 20, 20], [1, 1, 1, 0, 0, 0])
    b = as_runs([5, 6, 7, 8, 9, 20], [1, 1, 1, 1, 1, 0])
    cross = crossovers(curve_for(a), curve_for(b))
    assert cross["first_leader"] == "a"
    assert cross["final_leader"] == "b"
    assert cross["changes"] == [{"budget": 8.0, "leader": "b"}]
    assert cross["up_to"] == pytest.approx(20.0)
    none = crossovers(curve_for(b), curve_for(b))
    assert none["changes"] == [] and none["first_leader"] is None


def test_logrank_hand_worked():
    # group a: success at 1 and 3; group b: success at 2, censored at 4
    a = as_runs([1, 3], [1, 1])
    b = as_runs([2, 4], [1, 0])
    out = logrank(a, b)
    # t=1: n=(2,2) d=1 -> E=1/2, V=1/4; t=2: n=(1,2) d=1 -> E=1/3, V=2/9; t=3: n=(1,1) -> 1/2, 1/4
    assert out["observed_a"] == pytest.approx(2.0)
    assert out["expected_a"] == pytest.approx(1 / 2 + 1 / 3 + 1 / 2)
    var = 1 / 4 + 2 / 9 + 1 / 4
    assert out["chi2"] == pytest.approx((2 - 4 / 3) ** 2 / var)
    assert logrank(as_runs([1], [0]), as_runs([2], [2]))["p"] == pytest.approx(1.0)


def test_logrank_matches_lifelines():
    stats = pytest.importorskip("lifelines.statistics")
    a, b = two_systems(n=250, seed=20)
    for runs in (a, b):
        for k, r in enumerate(runs):
            runs[k] = Run(r.item, r.system, float(np.ceil(r.budget)), r.event)
    ours = logrank(a, b)
    ref = stats.logrank_test(
        [r.budget for r in a],
        [r.budget for r in b],
        [r.event == 1 for r in a],
        [r.event == 1 for r in b],
    )
    assert ours["chi2"] == pytest.approx(ref.test_statistic, rel=1e-8)
    assert ours["p"] == pytest.approx(ref.p_value, rel=1e-6)
