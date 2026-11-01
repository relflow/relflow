"""Execution records are immutable and never append to published YAML."""

import json

import pytest
from reporting import save


def test_independent_json_records_preserve_each_attempt(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    save({"seed": 1, "checks": {"signal": True}}, first)
    save({"seed": 2, "checks": {"signal": False}}, second)
    assert json.loads(first.read_text())["seed"] == 1
    assert json.loads(second.read_text())["checks"] == {"signal": False}
    with pytest.raises(FileExistsError):
        save({"seed": 3}, first)
    assert json.loads(first.read_text())["seed"] == 1


def test_nonfinite_metrics_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        save({"metric": float("nan")}, tmp_path / "result.json")
