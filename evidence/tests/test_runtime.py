"""Shared Docker images preserve exact dependencies and offline wheel installation."""

from pathlib import Path

import pytest
import runtime
from evidence_ci import identity, plan, runtime_inputs


def source(root):
    script = root / "experiments/family/signal.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "# %% [markdown]\n# ---\n# title: Signal\n# ---\n# %%\n"
        "from reporting import experiment, report\n"
        "@experiment('P901', seed=123)\n"
        "def run(seed, steps, accelerator):\n    return {}, {'signal': True}\n"
        "if __name__ == '__main__':\n    report(run)\n"
    )
    (root / "experiments/reporting.py").write_text("# trusted reporting fixture\n")
    return script


def test_wheel_installation_is_in_an_offline_cached_layer(tmp_path, monkeypatch):
    wheel = tmp_path / "relflow-0.4.3-py3-none-any.whl"
    wheel.write_bytes(b"verified-wheel")
    recipes = []

    def build(command, **kwargs):
        context = Path(command[-1])
        recipes.append((context / "Dockerfile").read_text())
        assert (context / wheel.name).read_bytes() == wheel.read_bytes()
        assert command[command.index("--network") + 1] == "none"

    monkeypatch.setattr(runtime.subprocess, "run", build)
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: "sha256:installed-wheel\n")
    assert runtime.wheel_image({"image": "verified-base", "image_id": "sha256:base"}, wheel) == "sha256:installed-wheel"
    assert "FROM verified-base\n" in recipes[0]
    assert "pip install --no-deps --no-index" in recipes[0]


def test_loaded_dependency_image_is_verified_and_tagged_for_offline_wheel_builds(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.12.0+cpu\n")
    wheel = tmp_path / "relflow-0.4.5-py3-none-any.whl"
    wheel.write_bytes(b"verified-wheel")
    loaded = "sha256:" + "a" * 64
    installed = "sha256:" + "b" * 64
    commands, recipes = [], []

    def inspect(command, **kwargs):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return (loaded if command[3] == loaded else installed) + "\n"
        assert command[:2] == ["docker", "version"]
        return "27.5.1\n"

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] is True
        if command[:3] == ["docker", "image", "tag"]:
            return
        assert command[:2] == ["docker", "build"]
        assert command[command.index("--network") + 1] == "none"
        recipes.append((Path(command[-1]) / "Dockerfile").read_text())

    monkeypatch.setattr(runtime.subprocess, "check_output", inspect)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime.platform, "processor", lambda: "fixture CPU")
    execution = runtime.environment(requirements, workers=2, image_id=loaded)
    expected = (
        "relflow-evidence:"
        + runtime.identity({"lock": runtime.digest(requirements), "recipe": runtime.runtime_recipe(requirements)})[:20]
    )

    assert commands[0] == ["docker", "image", "inspect", loaded, "--format", "{{.Id}}"]
    assert commands[1] == ["docker", "image", "tag", loaded, expected]
    assert execution["image"] == expected
    assert execution["image_id"] == execution["identity"]["image_id"] == loaded
    assert runtime.wheel_image(execution, wheel) == installed
    assert recipes[0].splitlines()[0] == f"FROM {expected}"
    assert f"FROM {loaded}\n" not in recipes[0]


def test_loaded_dependency_image_mismatch_is_rejected_before_tagging(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.12.0+cpu\n")
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **kwargs: "sha256:" + "b" * 64 + "\n")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unverified image was tagged"))

    with pytest.raises(ValueError, match="Loaded dependency image differs"):
        runtime.environment(requirements, workers=2, image_id="sha256:" + "a" * 64)


def test_environment_records_actual_image_and_cpu_runtime(tmp_path, monkeypatch):
    lock = tmp_path / "requirements.txt"
    lock.write_text("torch==2.5.0\n")
    builds = []

    def docker(command, **kwargs):
        assert command[:2] == ["docker", "build"]
        context = Path(command[-1])
        builds.append((context / "Dockerfile").read_text())
        assert (context / "requirements.txt").read_bytes() == lock.read_bytes()

    monkeypatch.setattr(runtime.subprocess, "run", docker)
    monkeypatch.setattr(
        runtime.subprocess,
        "check_output",
        lambda command, **kwargs: "sha256:actual-image\n" if command[1] == "image" else "actual-docker-version\n",
    )
    measured = runtime.environment(lock, 2)
    assert measured["identity"]["image_id"] == "sha256:actual-image"
    assert measured["identity"]["docker"] == "actual-docker-version"
    assert measured["identity"]["host"]["cpu_count"]
    assert measured["identity"]["resources"]["workers"] == 2
    assert measured["identity"]["resources"]["network"] == "none"
    assert builds and "python:3.12-slim@sha256:b18b" in builds[0]
    assert "pip install --no-cache-dir torch==2.5.0+cpu" in builds[0]


def test_worker_inputs_exclude_unbound_files_and_preserve_shared_data(tmp_path):
    suite = tmp_path / "suite"
    script = source(suite)
    other = script.with_name("other.py")
    other.write_text(script.read_text().replace("P901", "P902"))
    data = suite / "experiments/data.bin"
    data.write_bytes(b"\x00training input")
    data.chmod(0o640)
    (suite / "results.json").write_text("unbound publication input")
    manifest = plan(
        run_id="isolated", candidate={}, baseline={}, evidence_sha="source", executor_sha="trusted", root=suite
    )
    task = manifest["tasks"][0]
    inputs = runtime_inputs(suite)
    scripts = {script.relative_to(suite).as_posix(), other.relative_to(suite).as_posix()}
    view = runtime.isolate(suite, tmp_path / "inputs", task, inputs, scripts)
    assert identity(runtime_inputs(view)) == task["inputs_sha256"]
    assert (view / task["script"]).read_bytes() == script.read_bytes()
    assert (view / "experiments/reporting.py").is_file()
    assert (view / "experiments/data.bin").read_bytes() == data.read_bytes()
    assert (view / "experiments/data.bin").stat().st_mode & 0o777 == 0o640
    assert not (view / other.relative_to(suite)).exists()
    assert not (view / "results.json").exists()
    # A separate copy prevents writes through any accidental alias to the source.
    (view / "experiments/data.bin").write_bytes(b"changed copy")
    assert data.read_bytes() == b"\x00training input"


@pytest.mark.parametrize("change", ["bytes", "mode", "symlink"])
def test_worker_rejects_input_changes_after_planning(tmp_path, change):
    suite = tmp_path / "suite"
    script = source(suite)
    manifest = plan(
        run_id="isolated", candidate={}, baseline={}, evidence_sha="source", executor_sha="trusted", root=suite
    )
    task = manifest["tasks"][0]
    inputs = runtime_inputs(suite)
    if change == "bytes":
        script.write_text(script.read_text() + "changed = True\n")
    elif change == "mode":
        script.chmod(0o755)
    else:
        script.unlink()
        script.symlink_to(suite / "experiments/reporting.py")
    with pytest.raises(ValueError, match="Execution input"):
        runtime.isolate(suite, tmp_path / "inputs", task, inputs, {task["script"]})
