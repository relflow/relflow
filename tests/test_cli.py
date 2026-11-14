"""CLI contracts and executable workflows over real nested file inputs."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from textwrap import dedent

import polars as pl
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import yaml

import relflow as rf
import relflow.cli.commands as commands
from relflow.cli import main
from relflow.cli.config import Run, read
from relflow.cli.jobs import Job


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "model.yaml"
    model.write_text(
        dedent("""\
        preset: small
        d_model: 8
        n_layers: 1
        n_heads: 2
        fields:
          events:
            type: branch
            length: 3
            fields:
              amount:
                type: number
              label:
                type: boolean
                mask: true
        """)
    )
    frame = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "events": [[{"amount": 1.0, "label": True}], [{"amount": 2.0, "label": False}], []],
        }
    )
    data = tmp_path / "data.parquet"
    frame.write_parquet(data)
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "seed": 17,
                "data": {"predict": "data.parquet"},
                "predict": {"batch_size": 1, "accelerator": "cpu", "precision": "32-true"},
            }
        )
    )
    return model, data, config


def test_inspect_resolves_the_python_schema_without_a_graph(inputs, monkeypatch):
    model, _, _ = inputs
    expected = rf.Model.from_yaml(model)

    def forbidden(*args, **kwargs):
        pytest.fail("inspection constructed a neural graph")

    monkeypatch.setattr(rf.Model, "__init__", forbidden)
    report = rf.inspect(model)
    assert report["schema"] == expected.schema.model_dump(mode="json")
    assert report["objectives"] == ["/events/label"]
    assert report["stored_tensors"] == 0
    assert {node["address"] for node in report["nodes"]} == {"/", "/events", "/events/amount", "/events/label"}


def test_checkpoint_inspection_does_not_construct_a_model(inputs, tmp_path, monkeypatch):
    model, _, _ = inputs
    instance = rf.Model.from_yaml(model)
    path = tmp_path / "model.rf"
    instance.save(path)

    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint inspection constructed a model")

    monkeypatch.setattr(rf.Model, "__init__", forbidden)
    report = rf.inspect(path)
    assert report["schema"] == instance.schema.model_dump(mode="json")
    assert report["stored_tensors"] > 0
    assert report["stored_tensor_bytes"] > 0
    assert not report["resumable"]


def test_structural_check_reports_zero_values_read(inputs, monkeypatch):
    model, data, _ = inputs

    def forbidden(*args, **kwargs):
        pytest.fail("default preflight constructed a model")

    monkeypatch.setattr(rf.Model, "__init__", forbidden)
    report = rf.check(model, data={"train": str(data)})
    assert report["phases"][-1]["source_rows"] == 0
    assert report["phases"][-1]["status"] == "passed"
    assert Path(report["sources"]["train"]["files"][0]["path"]) == data


def test_sampled_check_uses_processors_and_obeys_source_row_cap(inputs):
    model, data, _ = inputs
    seen = []

    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        seen.append(len(frame))
        return frame

    report = rf.check(model, data={"train": str(data)}, execute=True, sample=2, preprocessor=prepare)
    assert report["phases"][-1]["source_rows"] == 2
    assert report["phases"][-1]["processed_rows"] == 2
    assert sum(seen) == 2


def test_dataset_preflight_reports_skipped_scope(inputs):
    model, data, _ = inputs

    @rf.preprocess(scope="dataset")
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        pytest.fail("dataset preprocessor ran on a truncated source")

    report = rf.check(model, data={"train": str(data)}, execute=True, preprocessor=prepare)
    assert report["phases"][-1]["status"] == "skipped"


@pytest.mark.parametrize(
    "content", ["version: 2", "version: 1\nversion: 1", "version: 1\nfit:\n  typo: 3", "version: true"]
)
def test_run_file_rejects_unknown_versions_duplicates_and_options(tmp_path, content):
    path = tmp_path / "run.yaml"
    path.write_text(content)
    with pytest.raises((ValueError, yaml.YAMLError)):
        read(path)


def test_run_paths_resolve_against_configuration(inputs, monkeypatch, tmp_path):
    _, data, config = inputs
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert read(config).data["predict"] == str(data)


def test_inspection_does_not_resolve_unused_processors(inputs, tmp_path, capsys):
    model, _, config = inputs
    payload = yaml.safe_load(config.read_text())
    payload.update(preprocessor="unavailable_application:prepare", postprocessor="unavailable_application:write")
    config.write_text(yaml.safe_dump(payload))
    assert main(["inspect", str(model), "--config", str(config), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["objectives"] == ["/events/label"]


def test_serving_passes_the_checkpoint_to_the_runtime_without_preloading(inputs, tmp_path, monkeypatch, capsys):
    model, _, _ = inputs
    checkpoint = tmp_path / "model.rf"
    rf.Model.from_yaml(model).save(checkpoint)
    served = []

    def forbidden(*args, **kwargs):
        pytest.fail("CLI loaded a checkpoint that belongs to the serving runtime")

    def serve(deployment):
        served.append((deployment.checkpoint, deployment.host))

    monkeypatch.setattr(commands.Artifact, "read", forbidden)
    monkeypatch.setattr(rf.Deployment, "serve", serve)
    monkeypatch.delenv("RELFLOW_HOST", raising=False)
    assert main(["serve", str(checkpoint), "--json"]) == 0
    assert served == [(checkpoint, "127.0.0.1")]
    assert json.loads(capsys.readouterr().out)["status"] == "success"


def test_explicit_training_input_can_be_checked_without_a_validation_split(inputs, capsys):
    model, data, _ = inputs
    assert main(["check", str(model), "--input", str(data), "--stage", "train", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["phases"][-1]["split"] == "train"


def test_duplicate_run_yaml_is_an_input_error(inputs, tmp_path, capsys):
    model, _, _ = inputs
    config = tmp_path / "duplicate.yaml"
    config.write_text("version: 1\nversion: 1\n")
    assert main(["check", str(model), "--config", str(config), "--json"]) == 2
    assert "duplicate key" in json.loads(capsys.readouterr().out)["error"]["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {"fit": {"devices": 2}},
        {"loader": {"num_workers": True}},
        {"predict": {"batch_size": 0}},
        {"serve": {"batch_timeout": float("nan")}},
    ],
)
def test_run_settings_enforce_declared_boundaries(payload):
    with pytest.raises(ValueError):
        Run.model_validate(payload)


def test_json_stdout_has_only_the_report_on_success_and_failure(inputs, capsys):
    model, _, _ = inputs
    assert main(["inspect", str(model), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["command"] == "inspect"
    assert main(["check", str(model), "--sample", "2", "--json"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "failed"
    assert "requires --execute" in captured.err


def test_human_reports_show_the_model_and_check_results(inputs, capsys, monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    model, data, _ = inputs
    assert main(["inspect", str(model)]) == 0
    captured = capsys.readouterr()
    assert "/events/amount" in captured.out
    assert "/events/label" in captured.out
    assert "target" in captured.out
    assert "Stored tensors" in captured.out
    assert main(["check", str(model), "--input", str(data)]) == 0
    captured = capsys.readouterr()
    assert "Preflight" in captured.out
    assert "passed" in captured.out
    assert "tensorized" in captured.out
    assert "--execute" in captured.out
    assert "\x1b[" not in captured.out


def test_rich_output_treats_model_paths_as_literal_text(inputs, capsys, monkeypatch):
    model, _, _ = inputs
    monkeypatch.setenv("COLUMNS", "240")
    named = model.with_name("[bold]model.yaml")
    model.rename(named)
    assert main(["inspect", str(named)]) == 0
    assert "[bold]model.yaml" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["inspect", "check"])
def test_json_bypasses_rich_even_when_terminal_colors_are_forced(inputs, capsys, monkeypatch, command):
    model, _, _ = inputs
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "30")
    assert main([command, str(model), "--json"]) == 0
    output = capsys.readouterr().out
    assert "\x1b[" not in output
    assert len(output.splitlines()) == 1
    assert json.loads(output)["command"] == command


def test_existing_job_is_a_contract_error_and_keeps_its_receipt(inputs, tmp_path, capsys):
    model, data, _ = inputs
    checkpoint = tmp_path / "saved.rf"
    rf.Model.from_yaml(model).save(checkpoint)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "_metadata").mkdir()
    report = output / "_metadata/report.json"
    report.write_text('{"status": "success"}')
    assert main(["predict", str(checkpoint), "--input", str(data), "--output", str(output), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert report.read_text() == '{"status": "success"}'


def test_job_failure_preserves_receipt_and_has_no_success_marker(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="failure"):
        with Job(output, {"run": {}}):
            raise RuntimeError("failure")
    assert json.loads((output / "_metadata/report.json").read_text())["status"] == "failed"
    assert not (output / "_SUCCESS").exists()
    assert not list(output.rglob("*.partial"))
    with pytest.raises(FileExistsError):
        with Job(output, {}):
            pass


def test_interrupted_prediction_retains_its_failed_report(inputs, tmp_path, monkeypatch, capsys):
    model, data, _ = inputs
    checkpoint = tmp_path / "model.rf"
    rf.Model.from_yaml(model).save(checkpoint)
    output = tmp_path / "interrupted"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(rf.Model, "predict", interrupt)
    assert main(["predict", str(checkpoint), "--input", str(data), "--output", str(output), "--json"]) == 130
    receipt = json.loads((output / "_metadata/report.json").read_text())
    assert receipt["status"] == "failed"
    assert json.loads(capsys.readouterr().out) == receipt
    assert not (output / "_SUCCESS").exists()
    assert not list(output.rglob("*.partial"))


def invoke(*arguments: str, cwd: Path, env=None):
    command = shutil.which("relflow")
    assert command is not None, "the installed wheel must expose its console entry point"
    result = subprocess.run([command, *arguments], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.integration
def test_executable_streamed_prediction_matches_python(inputs, tmp_path):
    model, data, config = inputs
    checkpoint = tmp_path / "model.rf"
    restored = rf.Model.from_yaml(model)
    restored.save(checkpoint)
    expected = restored.predict(pl.read_parquet(data), retain=("id",), batch_size=2)
    payload = yaml.safe_load(config.read_text())
    payload["data"]["predict"] = "unavailable.parquet"
    config.write_text(yaml.safe_dump(payload))
    for format in ("parquet", "ipc"):
        directory = tmp_path / format
        report = invoke(
            "predict",
            str(checkpoint),
            "--config",
            str(config),
            "--batch-size",
            "2",
            "--input",
            str(data),
            "--output",
            str(directory),
            "--retain",
            "id",
            "--format",
            format,
            "--accelerator",
            "cpu",
            "--precision",
            "32-true",
            "--json",
            cwd=tmp_path,
        )
        shard = directory / ("rank-0.parquet" if format == "parquet" else "rank-0.arrow")
        actual = pq.read_table(shard) if format == "parquet" else ipc.open_file(shard).read_all()
        assert pl.from_arrow(actual).equals(expected)
        assert report["written_rows"] == len(expected)
        resolved = yaml.safe_load((directory / "_metadata/resolved.yaml").read_text())
        assert resolved["run"]["predict"]["batch_size"] == 2
        assert resolved["run"]["data"]["predict"] == str(data)
        assert (directory / "_SUCCESS").exists()
        assert rf.source(str(directory)).discover().schema.equals(actual.schema)


@pytest.mark.integration
def test_module_entry_point_and_installed_extension_imports(inputs, tmp_path):
    model, data, _ = inputs
    app = tmp_path / "application.py"
    app.write_text(
        dedent("""\
        import polars as pl
        import relflow as rf
        from typing import Literal

        extension = rf.Extension(name="application_number", types=(int, float))
        for role, component in rf.TENSORFIELDS["number"].components.items():
            if role != rf.Component.Request:
                extension.register(role, component)

        @extension.register(rf.Component.Request)
        class Amount(rf.Number):
            type: Literal["application_number"] = "application_number"

        @rf.preprocess
        def prepare(frame: pl.DataFrame) -> pl.DataFrame:
            print("application diagnostic")
            return frame
        """)
    )
    config = tmp_path / "application.yaml"
    model.write_text(model.read_text().replace("type: number", "type: application_number"))
    config.write_text(yaml.safe_dump({"version": 1, "imports": ["application"], "preprocessor": "application:prepare"}))
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    report = invoke(
        "check", str(model), "--config", str(config), "--input", str(data), "--execute", "--json", cwd=tmp_path, env=env
    )
    assert report["phases"][-1]["processed_rows"] == 3
    result = subprocess.run(
        [sys.executable, "-m", "relflow", "inspect", str(model), "--import", "application", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["objectives"] == ["/events/label"]


@pytest.mark.integration
def test_empty_inputs_and_failed_prediction_publish_distinct_receipts(inputs, tmp_path):
    model, data, _ = inputs
    checkpoint = tmp_path / "saved.rf"
    rf.Model.from_yaml(model).save(checkpoint)
    empty = tmp_path / "empty.parquet"
    pl.read_parquet(data).head(0).write_parquet(empty)
    output = tmp_path / "empty-output"
    report = invoke(
        "predict",
        str(checkpoint),
        "--input",
        str(empty),
        "--output",
        str(output),
        "--format",
        "ipc",
        "--json",
        cwd=tmp_path,
    )
    assert report["written_rows"] == 0
    assert ipc.open_file(output / "rank-0.arrow").read_all().num_rows == 0
    assert (output / "_SUCCESS").exists()

    broken = tmp_path / "broken.parquet"
    pl.DataFrame({"events": [[{"amount": [1.0], "label": True}]]}).write_parquet(broken)
    failed = tmp_path / "failed-output"
    result = subprocess.run(
        [
            shutil.which("relflow"),
            "predict",
            str(checkpoint),
            "--input",
            str(broken),
            "--output",
            str(failed),
            "--json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads((failed / "_metadata/report.json").read_text())
    assert receipt["status"] == "failed"
    assert json.loads(result.stdout) == receipt
    assert not (failed / "_SUCCESS").exists()
    assert not list(failed.rglob("*.partial"))


@pytest.mark.integration
def test_executable_serves_the_saved_model(inputs, tmp_path):
    model, data, _ = inputs
    instance = rf.Model.from_yaml(model)
    checkpoint = tmp_path / "serve.rf"
    instance.save(checkpoint)
    frame = pl.read_parquet(data)
    expected = instance.predict(frame, retain=("id",)).to_dicts()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [
                shutil.which("relflow"),
                "serve",
                str(checkpoint),
                "--port",
                str(port),
                "--accelerator",
                "cpu",
                "--retain",
                "id",
            ],
            cwd=tmp_path,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert process.poll() is None, (tmp_path / "server.log").read_text()
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.05)
            else:
                pytest.fail("CLI server did not become ready: " + (tmp_path / "server.log").read_text())
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/predict",
                data=json.dumps(frame.to_dicts()).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                assert json.loads(response.read()) == expected
                assert response.headers["X-RelFlow-Model-Version"] == instance.version
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
