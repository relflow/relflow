"""Execution records are immutable and never append to published YAML."""

import json
import sys
from types import SimpleNamespace

import pytest
import reporting
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


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metrics_are_rejected(tmp_path, metric):
    with pytest.raises(ValueError):
        save({"metrics": {"nested": [metric]}}, tmp_path / "result.json")


def test_recorded_source_identity_binds_shared_policy_bytes_without_importing_them(tmp_path, monkeypatch):
    root = tmp_path / "suite"
    helpers = root / "experiments"
    script = helpers / "P001.py"
    script.parent.mkdir(parents=True)
    script.write_text("def run(seed, steps, accelerator):\n    return {}, {'signal': True}\n")
    for name in ("reporting.py", "source.py", "run.py"):
        (helpers / name).write_text("# shared runtime fixture\n")
    policy = helpers / "policies.py"
    policy.write_text("raise RuntimeError('policy must not execute while hashing')\n")
    package = tmp_path / "installed/relflow"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# independently installed package fixture\n")
    monkeypatch.setattr(reporting, "ROOT", root)
    monkeypatch.setattr(reporting, "__file__", str(helpers / "reporting.py"))
    monkeypatch.setitem(sys.modules, "relflow", SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(cuda=None), cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(reporting.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setenv("EVIDENCE_REVISION", "fixture-source")
    before = reporting.provenance(script)
    policy.write_text("raise RuntimeError('policy changed; still must not execute')\n")
    after = reporting.provenance(script)
    assert before["source_sha256"] != after["source_sha256"]
    assert before["script_sha256"] == after["script_sha256"]
    assert before["code_sha256"] == after["code_sha256"]
    assert before["relflow_source_sha256"] == after["relflow_source_sha256"]
