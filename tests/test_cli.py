import json
from pathlib import Path

import pytest

from stepbudget.cli import main

EXAMPLE = str(Path(__file__).resolve().parents[1] / "examples" / "runs.csv")


def write_csv(path, rows):
    lines = ["item,system,outcome,budget,exit_reason"]
    lines += [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_curve_markdown_on_example(capsys):
    assert main(["curve", EXAMPLE, "--budgets", "20", "40", "80"]) == 0
    out = capsys.readouterr().out
    assert "## alpha" in out and "## beta" in out
    assert "| 40 | 54.2 % | 44.8 to 62.6 % | 54.2 % | 25.8 % | 19 |" in out
    assert "stopped by the limit" in out


def test_curve_json_matches_hand_worked_table(tmp_path, capsys):
    path = write_csv(
        tmp_path / "r.csv",
        [
            ("a", "s", "success", 1, "submitted"),
            ("b", "s", "failure", 2, "gave up"),
            ("c", "s", "success", 2, "submitted"),
            ("d", "s", "failure", 3, "limit reached"),
            ("e", "s", "success", 4, "submitted"),
            ("f", "s", "failure", 5, "limit reached"),
        ],
    )
    assert main(["curve", path, "--format", "json", "--budgets", "2", "4"]) == 0
    rep = json.loads(capsys.readouterr().out)["s"]
    assert rep["counts"] == {"runs": 6, "items": 6, "success": 3, "gave_up": 1, "limit_reached": 2}
    assert rep["rows"][0]["success"] == pytest.approx(1 / 3)
    assert rep["rows"][1]["success"] == pytest.approx(7 / 12)
    assert rep["rows"][1]["plain"] == pytest.approx(0.5)
    assert [r["runs_past"] for r in rep["rows"]] == [3, 1]
    assert rep["first_limit_budget"] == pytest.approx(3.0)
    assert rep["curves_differ_from"] == pytest.approx(4.0)


def test_missing_file_and_unknown_system_exit_with_2(tmp_path, capsys):
    assert main(["curve", str(tmp_path / "nope.csv")]) == 2
    assert main(["curve", EXAMPLE, "--system", "gamma"]) == 2
    assert "gamma" in capsys.readouterr().err


def test_at_reports_truncation_bounds_and_extrapolation(capsys):
    assert main(["at", EXAMPLE, "--system", "alpha", "--budgets", "40", "60", "120"]) == 0
    out = capsys.readouterr().out
    assert "| 40 | 54.2 % | by truncation | 54.2 % | 54.2 % |" in out
    assert "| 60 | 66.4 % | censoring-aware estimate | 61.7 % | 71.7 % |" in out
    assert "| 120 |" in out and "| extrapolation | 62.5 % | 78.3 % | - |" in out
    assert "As run: 62.5 % of items solved, 15.8 % stopped by the limit" in out


def test_at_json_default_fractions(capsys):
    assert main(["at", EXAMPLE, "--format", "json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert [r["budget"] for r in rep["beta"]["rows"]] == pytest.approx([20.0, 40.0])
    args = ["at", EXAMPLE, "--format", "json", "--budgets", "70", "30", "70", "--fractions", "0.5"]
    assert main(args) == 0
    again = json.loads(capsys.readouterr().out)
    assert [r["budget"] for r in again["alpha"]["rows"]] == pytest.approx([30.0, 40.0, 70.0])
    assert all(r["kind"] == "exact" for r in rep["beta"]["rows"])
    assert rep["beta"]["as_run"]["solved"] == pytest.approx(88 / 120)


def test_compare_markdown_and_json(capsys):
    args = ["compare", EXAMPLE, "--budgets", "20", "80", "--boot", "200"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "## alpha against beta" in out
    assert "Lead changes: alpha is ahead first, then beta from 63" in out
    assert "Log-rank test" in out
    assert main(args + ["--format", "json", "--a", "beta", "--b", "alpha"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["a"] == "beta" and rep["items"] == 120
    assert rep["rows"][0]["difference"] == pytest.approx(0.15 - 28 / 120)
    assert rep["crossovers"]["changes"] == [{"budget": 63.0, "leader": "a"}]


def test_compare_needs_names_when_the_file_has_other_than_two_systems(tmp_path, capsys):
    path = write_csv(
        tmp_path / "three.csv",
        [("a", s, "success", 1, "submitted") for s in ("x", "y", "z")],
    )
    assert main(["compare", path]) == 2
    assert "--a and --b" in capsys.readouterr().err


def test_marginal_and_cost(capsys):
    assert main(["marginal", EXAMPLE, "--system", "alpha", "--step", "10"]) == 0
    out = capsys.readouterr().out
    assert "| 20 to 30 | +22.50 points | 2.25 |" in out
    assert "From budget 70 on" in out
    assert main(["marginal", EXAMPLE, "--format", "json", "--threshold", "0.0"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["alpha"]["below_threshold_from"] is None
    assert main(["cost", EXAMPLE, "--limits", "40", "80", "--boot", "100"]) == 0
    out = capsys.readouterr().out
    assert "| 40 | 54.2 % | 25.44 | 46.97 |" in out
    assert main(["cost", EXAMPLE, "--format", "json", "--boot", "50"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["beta"]["as_run"]["cost"] == pytest.approx(rep["beta"]["rows"][-1]["cost"])


def test_curve_writes_a_chart(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "curves.png"
    assert main(["curve", EXAMPLE, "--plot", str(out), "--xlabel", "steps", "--xmax", "60"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_with_repeated_runs_uses_the_bootstrap_band(tmp_path):
    pytest.importorskip("matplotlib")
    from stepbudget.data import Run
    from stepbudget.plot import plot_curves

    runs = [Run(f"i{k % 20}", "s", float(1 + (k * 7) % 30), k % 3) for k in range(60)]
    out = plot_curves({"s": runs}, tmp_path / "c.png", n_boot=30, title="t", xlabel="steps")
    assert out.stat().st_size > 1000


def test_bad_step_and_default_grids(capsys):
    assert main(["marginal", EXAMPLE, "--step", "-1"]) == 2
    assert main(["curve", EXAMPLE, "--format", "json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert len(rep["alpha"]["rows"]) == 10
    assert rep["alpha"]["rows"][-1]["budget"] == pytest.approx(80.0)
