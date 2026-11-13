import json

import pytest

from stepbudget.data import classify, parse_outcome, read_runs
from stepbudget.estimators import CENSORED, GAVE_UP, SUCCESS


def test_classify_rules():
    assert classify(True, 10, "limit reached") == SUCCESS
    assert classify(False, 10, "Limit Reached ") == CENSORED
    assert classify(False, 10, "error") == GAVE_UP
    assert classify(False, 10, None) == GAVE_UP
    assert classify(False, 50, None, limit=50) == CENSORED
    assert classify(False, 49, None, limit=50) == GAVE_UP
    assert classify(False, 10, "exit_context", limit_reasons=["exit_context"]) == CENSORED
    assert classify(False, 10, "limit", limit_reasons=["exit_context"]) == GAVE_UP


@pytest.mark.parametrize("value", [True, 1, 1.0, "1", "true", "Success", " resolved ", "PASS"])
def test_outcome_true(value):
    assert parse_outcome(value) is True


@pytest.mark.parametrize("value", [False, 0, "0", "false", "Failure", "", "unresolved"])
def test_outcome_false(value):
    assert parse_outcome(value) is False


def test_outcome_unknown_word_is_an_error():
    with pytest.raises(ValueError, match="maybe"):
        parse_outcome("maybe")


def test_read_csv_with_custom_columns_and_limit(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text(
        "task_name,agent,resolved,steps,cost\n"
        "t1,a,true,12,0.4\n"
        "t2,a,false,50,2.0\n"
        "t3,a,false,20,0.9\n",
        encoding="utf-8",
    )
    kw = dict(item_col="task_name", system_col="agent", outcome_col="resolved", exit_col=None)
    runs = read_runs(path, budget_col="steps", limit=50, **kw)
    assert [r.event for r in runs] == [SUCCESS, CENSORED, GAVE_UP]
    assert [r.budget for r in runs] == pytest.approx([12, 50, 20])
    assert runs[0].system == "a" and runs[0].item == "t1"
    by_cost = read_runs(path, budget_col="cost", **kw)
    assert [r.budget for r in by_cost] == pytest.approx([0.4, 2.0, 0.9])
    assert [r.event for r in by_cost] == [SUCCESS, GAVE_UP, GAVE_UP]


def test_read_json_lines_without_system_or_exit_reason(tmp_path):
    path = tmp_path / "runs.jsonl"
    rows = [
        {"item": "a", "outcome": True, "budget": 3},
        {"item": "b", "outcome": False, "budget": 9, "exit_reason": "limit reached"},
        {"item": "c", "outcome": 0, "budget": 4.5, "exit_reason": None},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    runs = read_runs(path)
    assert [r.event for r in runs] == [SUCCESS, CENSORED, GAVE_UP]
    assert {r.system for r in runs} == {"system"}
    assert runs[2].exit_reason is None


@pytest.mark.parametrize(
    "text, message",
    [
        ("item,outcome,budget\n", "no runs"),
        ("item,outcome\na,true\n", "budget"),
        ("item,outcome,budget\na,true,many\n", "non-numeric"),
        ("item,outcome,budget\na,true,-3\n", "finite value"),
        ("item,outcome,budget\na,true,inf\n", "finite value"),
        ("item,outcome,budget\na,true,nan\n", "finite value"),
        ("item,outcome,budget\na,perhaps,3\n", "perhaps"),
    ],
)
def test_bad_csv_rows_are_reported(tmp_path, text, message):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_runs(path)


@pytest.mark.parametrize("text", ["{not json}\n", "[1, 2]\n"])
def test_bad_json_lines_are_reported(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        read_runs(path)


def test_json_array_and_byte_order_mark(tmp_path):
    rows = [
        {"item": "a", "outcome": "resolved", "budget": 3},
        {"item": "b", "outcome": "unresolved", "budget": 9, "exit_reason": "limit"},
    ]
    array = tmp_path / "runs.json"
    array.write_text(json.dumps(rows), encoding="utf-8")
    assert [r.event for r in read_runs(array)] == [SUCCESS, CENSORED]
    lines = tmp_path / "lines.json"
    lines.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert [r.event for r in read_runs(lines)] == [SUCCESS, CENSORED]
    with_mark = tmp_path / "bom.csv"
    with_mark.write_text("item,outcome,budget\na,true,2\n", encoding="utf-8-sig")
    assert read_runs(with_mark)[0].item == "a"
    broken = tmp_path / "broken.json"
    broken.write_text("[{\"item\": 1, \"outcome\": 1, \"budget\": 1}, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="entry 2"):
        read_runs(broken)
    broken.write_text("[{\"item\": 1},", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_runs(broken)


def test_nan_outcome_is_an_error():
    with pytest.raises(ValueError, match="nan"):
        parse_outcome(float("nan"))
