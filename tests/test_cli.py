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
    assert "| 40 | 54.2 % |" in out
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
    assert rep["first_limit_budget"] == pytest.approx(3.0)
    assert rep["curves_differ_from"] == pytest.approx(4.0)


def test_missing_file_and_unknown_system_exit_with_2(tmp_path, capsys):
    assert main(["curve", str(tmp_path / "nope.csv")]) == 2
    assert main(["curve", EXAMPLE, "--system", "gamma"]) == 2
    assert "gamma" in capsys.readouterr().err
