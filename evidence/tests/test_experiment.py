"""Standalone experiment declarations retain ordinary Python and immutable reports."""

import importlib.util
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import reporting
from reporting import experiment, report
from source import Experiment


def test_decorator_retains_original_function_without_execution():
    calls = []

    def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        calls.append((seed, steps, accelerator))
        return {}, {"signal": True}

    code, signature, annotations = run.__code__, inspect.signature(run), run.__annotations__.copy()
    decorated = experiment("P001", seed=123)(run)

    assert decorated is run
    assert decorated.__code__ is code
    assert inspect.signature(decorated) == signature
    assert decorated.__annotations__ == annotations
    assert not hasattr(decorated, "__wrapped__")
    assert calls == []
    assert decorated.experiment == Experiment("P001", 123)
    with pytest.raises(AttributeError):
        decorated.experiment.seed = 999
    with pytest.raises(TypeError):
        decorated.experiment[0] = "P002"
    assert decorated.experiment == Experiment("P001", 123)


@pytest.mark.parametrize("identifier,seed", [("P1", 0), ("P001", 123), ("P999", 2**32 - 1)])
def test_decorator_accepts_stable_identifiers_and_seed_boundaries(identifier, seed):
    def run(seed, steps, accelerator):
        return {}, {"signal": True}

    assert experiment(identifier, seed=seed)(run).experiment == Experiment(identifier, seed)


@pytest.mark.parametrize("identifier", [None, 1, "", "P", "p001", "P-1", "P001suffix"])
def test_decorator_rejects_invalid_identifiers(identifier):
    with pytest.raises(ValueError, match="stable P identifier"):
        experiment(identifier, seed=123)


@pytest.mark.parametrize("seed", [None, True, False, -1, 2**32, 1.0, "123"])
def test_decorator_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        experiment("P001", seed=seed)


@pytest.mark.parametrize(
    "signature",
    [
        "seed, steps",
        "seed, steps, accelerator, extra",
        "seed, steps, device",
        "seed, steps=None, accelerator='cpu'",
        "seed, steps, *, accelerator",
        "seed, steps, accelerator, /",
        "seed, steps, accelerator, *args",
        "seed, steps, accelerator, **kwargs",
    ],
)
def test_decorator_rejects_incompatible_call_signatures(signature):
    namespace = {}
    exec(f"def run({signature}):\n    return {{}}, {{'signal': True}}\n", namespace)
    with pytest.raises(ValueError, match="signature"):
        experiment("P001", seed=123)(namespace["run"])


def test_decorator_rejects_async_functions_and_callable_objects():
    async def run(seed, steps, accelerator):
        return {}, {"signal": True}

    class Callable:
        def __call__(self, seed, steps, accelerator):
            return {}, {"signal": True}

    for function in (run, Callable(), len):
        with pytest.raises(TypeError, match="synchronous function"):
            experiment("P001", seed=123)(function)


def test_duplicate_decoration_cannot_replace_original_metadata():
    @experiment("P001", seed=123)
    def run(seed, steps, accelerator):
        return {}, {"signal": True}

    declared = run.experiment
    for identifier, seed in (("P001", 123), ("P002", 456)):
        with pytest.raises(ValueError, match="already registered"):
            experiment(identifier, seed=seed)(run)
        assert run.experiment is declared


def test_repeated_imports_have_no_process_registry_collisions(tmp_path):
    path = tmp_path / "toy.py"
    path.write_text(
        "from reporting import experiment\n"
        "executions = 0\n"
        "@experiment('P001', seed=123)\n"
        "def run(seed, steps, accelerator):\n"
        "    global executions\n"
        "    executions += 1\n"
        "    return {}, {'signal': True}\n"
    )
    copies = []
    for name in ("experiment_first_import", "experiment_second_import"):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        copies.append(module)

    first, second = copies
    assert first.run is not second.run
    assert first.run.experiment == second.run.experiment == Experiment("P001", 123)
    assert first.executions == second.executions == 0
    assert first.run(123, None, "cpu") == ({}, {"signal": True})
    assert first.executions == 1
    assert second.executions == 0


@pytest.fixture
def cli(monkeypatch, tmp_path):
    threads, sources = [], []
    provenance = {"source_sha256": "a" * 64, "wheel_sha256": "b" * 64}

    def inspect_source(path):
        sources.append(path)
        return provenance.copy()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=threads.append))
    monkeypatch.setattr(reporting, "provenance", inspect_source)
    monkeypatch.delenv("EVIDENCE_JSON_OUTPUT", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    output = tmp_path / "record.json"
    monkeypatch.setattr(sys, "argv", ["toy.py", "--output", str(output)])
    return SimpleNamespace(output=output, threads=threads, sources=sources, provenance=provenance)


@pytest.mark.parametrize("passed,outcome", [(True, "met"), (False, "not_met")])
def test_report_uses_declared_defaults_and_retains_unmet_checks(cli, capsys, passed, outcome):
    calls = []

    @experiment("P007", seed=123)
    def run(seed, steps, accelerator):
        calls.append((seed, steps, accelerator))
        return {"loss": 0.25, "nested": {"samples": (1, 2)}}, {"signal": passed}

    report(run)

    recorded = json.loads(cli.output.read_text())
    assert calls == [(123, None, "cpu")]
    assert cli.threads == [1]
    assert cli.sources == [Path(run.__code__.co_filename).resolve()]
    assert recorded["provenance"] == cli.provenance
    assert recorded["metrics"] == {"loss": 0.25, "nested": {"samples": [1, 2]}}
    assert recorded["checks"] == {"signal": passed}
    assert recorded["outcome"] == outcome
    assert recorded["mode"] == "full"
    assert recorded["seed"] == 123
    assert recorded["steps_override"] is None
    assert recorded["accelerator"] == "cpu"
    assert recorded["threads"] == 1
    assert recorded["duration_seconds"] >= 0
    assert datetime.fromisoformat(recorded["started_at"]).utcoffset().total_seconds() == 0
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    stdout = capsys.readouterr().out
    printed, end = json.JSONDecoder().raw_decode(stdout)
    assert printed == {"evidence": "P007", **recorded}
    assert stdout[end:].strip() == f"Recorded P007 in {cli.output}"


def test_cli_overrides_keep_environment_output_and_existing_cuda_configuration(cli, monkeypatch, tmp_path):
    calls = []

    @experiment("P001", seed=123)
    def run(seed, steps, accelerator):
        calls.append((seed, steps, accelerator))
        return {"loss": 0.5}, {"signal": True}

    selected = tmp_path / "selected" / "result.json"
    monkeypatch.setenv("EVIDENCE_JSON_OUTPUT", str(selected))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toy.py",
            "--seed",
            "456",
            "--steps",
            "3",
            "--accelerator",
            "mps",
            "--threads",
            "2",
            "--output",
            str(cli.output),
        ],
    )

    report(run)

    recorded = json.loads(selected.read_text())
    assert calls == [(456, 3, "mps")]
    assert cli.threads == [2]
    assert recorded["mode"] == "smoke"
    assert recorded["seed"] == 456
    assert recorded["steps_override"] == 3
    assert recorded["accelerator"] == "mps"
    assert recorded["threads"] == 2
    assert not cli.output.exists()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_default_output_uses_declared_identifier(cli, monkeypatch, tmp_path):
    @experiment("P042", seed=123)
    def run(seed, steps, accelerator):
        return {}, {"signal": True}

    monkeypatch.setattr(reporting, "ROOT", tmp_path)
    monkeypatch.setattr(reporting.time, "time_ns", lambda: 314)
    monkeypatch.setattr(sys, "argv", ["toy.py"])
    report(run)
    assert json.loads((tmp_path / "runs/local/P042-314.json").read_text())["outcome"] == "met"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "0"],
        ["--steps", "-1"],
        ["--threads", "0"],
        ["--threads", "-1"],
        ["--accelerator", "unsupported"],
        ["--seed", "not-an-integer"],
    ],
)
def test_invalid_cli_options_fail_before_execution_or_recording(cli, monkeypatch, arguments):
    calls = []

    @experiment("P001", seed=123)
    def run(seed, steps, accelerator):
        calls.append((seed, steps, accelerator))
        return {}, {"signal": True}

    monkeypatch.setattr(sys, "argv", ["toy.py", "--output", str(cli.output), *arguments])
    with pytest.raises(SystemExit) as error:
        report(run)
    assert error.value.code == 2
    assert calls == cli.threads == cli.sources == []
    assert not cli.output.exists()


@pytest.mark.parametrize("metadata", [None, {"id": "P001", "seed": 123}, SimpleNamespace(id="P001", seed=123)])
def test_report_requires_an_experiment_declaration(cli, metadata):
    def run(seed, steps, accelerator):
        pytest.fail("An undeclared function must not execute")

    if metadata is not None:
        run.experiment = metadata
    with pytest.raises(ValueError, match="Decorate the reporting function"):
        report(run)
    assert cli.threads == cli.sources == []
    assert not cli.output.exists()


@pytest.mark.parametrize("failure", ["function", "provenance", "missing-checks"])
def test_report_records_errors_before_exiting(cli, monkeypatch, failure):
    calls = []

    @experiment("P001", seed=123)
    def run(seed, steps, accelerator):
        calls.append((seed, steps, accelerator))
        if failure == "function":
            raise RuntimeError("toy execution failed")
        return {"loss": 0.5}, {}

    if failure == "provenance":

        def unavailable(path):
            raise RuntimeError("toy provenance unavailable")

        monkeypatch.setattr(reporting, "provenance", unavailable)

    with pytest.raises(SystemExit) as error:
        report(run)
    assert error.value.code == 1
    recorded = json.loads(cli.output.read_text())
    assert recorded["outcome"] == "error"
    assert recorded["seed"] == 123
    assert "Traceback (most recent call last)" in recorded["error"]
    if failure == "provenance":
        assert calls == []
        assert recorded["provenance"] == {}
        assert "toy provenance unavailable" in recorded["error"]
    else:
        assert calls == [(123, None, "cpu")]
        assert recorded["provenance"] == cli.provenance
        if failure == "function":
            assert "toy execution failed" in recorded["error"]
        else:
            assert recorded["metrics"] == {"loss": 0.5}
            assert recorded["checks"] == {}
            assert "must return its behavioral checks" in recorded["error"]


def test_cli_cannot_overwrite_a_previous_record(cli):
    calls = []

    @experiment("P001", seed=123)
    def run(seed, steps, accelerator):
        calls.append((seed, steps, accelerator))
        return {"attempt": len(calls)}, {"signal": len(calls) == 1}

    report(run)
    original = cli.output.read_bytes()
    with pytest.raises(FileExistsError):
        report(run)
    assert cli.output.read_bytes() == original
    assert json.loads(original)["checks"] == {"signal": True}
    assert json.loads(original)["metrics"] == {"attempt": 1}
