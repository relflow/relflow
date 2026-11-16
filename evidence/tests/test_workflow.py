"""Hosted Evidence keeps exact event inputs and a separately pinned executor."""

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import publication
import pytest
import releases
import yaml
from hosted import PROFILE

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/evidence-hosted.yml"
CONFIG = yaml.safe_load(WORKFLOW.read_text())
STEPS = CONFIG["jobs"]["prepare"]["steps"]


@pytest.fixture
def history(tmp_path, monkeypatch):
    for name in subprocess.check_output(["git", "rev-parse", "--local-env-vars"], text=True).splitlines():
        monkeypatch.delenv(name, raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    revisions = []
    for version in ("0.4.7", "0.5.0", "0.5.1"):
        (tmp_path / "pyproject.toml").write_text(f'[project]\nversion="{version}"\n')
        subprocess.run(["git", "add", "pyproject.toml"], cwd=tmp_path, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Evidence test",
                "-c",
                "user.email=evidence@example.invalid",
                "commit",
                "-qm",
                version,
            ],
            cwd=tmp_path,
            check=True,
        )
        revisions.append(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip())
    monkeypatch.setattr(releases, "BASELINE", revisions[1])
    return tmp_path, revisions


def test_supported_baseline_preserves_the_actual_event_base(history):
    root, (_, baseline, candidate) = history
    result = releases.comparison(root, candidate, baseline)
    assert result["baseline"] == result["event_base"] == baseline
    assert result["reason"] == "event-base"


def test_pre_05_event_base_uses_the_fixed_supported_anchor(history):
    root, (base, anchor, candidate) = history
    result = releases.comparison(root, candidate, base)
    assert result == {
        "schema": 1,
        "event_base": base,
        "baseline": anchor,
        "reason": "pre-0.5",
        "candidate_version": "0.5.1",
        "baseline_version": "0.5.0",
    }
    with pytest.raises(ValueError, match="candidate.*requires relflow >=0.5.0"):
        releases.comparison(root, base, base)


@pytest.mark.parametrize("available", [False, True])
def test_unrelated_base_requires_an_authenticated_forced_push(history, available):
    root, (_, anchor, candidate) = history
    unavailable = "f" * 40
    if available:
        unavailable = subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Evidence test",
                "-c",
                "user.email=evidence@example.invalid",
                "commit-tree",
                "HEAD^{tree}",
                "-m",
                "unrelated history",
            ],
            cwd=root,
            text=True,
        ).strip()
    with pytest.raises(ValueError, match="available ancestor"):
        releases.comparison(root, candidate, unavailable)
    selected = releases.comparison(root, candidate, unavailable, forced=True)
    assert selected["event_base"] == unavailable
    assert selected["baseline"] == anchor
    assert selected["reason"] == "rewritten-push"


def test_anchor_must_be_supported_and_an_ancestor(history, monkeypatch):
    root, (old, anchor, candidate) = history
    monkeypatch.setattr(releases, "BASELINE", old)
    with pytest.raises(ValueError, match="baseline.*requires relflow >=0.5.0"):
        releases.comparison(root, candidate, old)
    monkeypatch.setattr(releases, "BASELINE", candidate)
    with pytest.raises(ValueError, match="baseline.*available ancestor"):
        releases.comparison(root, anchor, old)


def test_suite_revision_cannot_replace_the_trusted_host_executor():
    checkouts = {step["with"]["path"]: step["with"] for step in STEPS if step.get("uses") == "actions/checkout@v4"}
    executor, suite = checkouts["executor"], checkouts["suite"]
    assert executor["repository"] == suite["repository"] == "relflow/relflow"
    assert re.fullmatch(r"[0-9a-f]{40}", executor["ref"])
    assert suite["ref"] == "${{ inputs.suite_ref }}"
    assert executor["persist-credentials"] is suite["persist-credentials"] is False


@pytest.mark.parametrize(
    "reference,accepted",
    [
        ("a" * 40, True),
        ("main", False),
        ("a" * 39, False),
        ("A" * 40, False),
        ("a" * 40 + "\n", False),
        ("$(echo injected)", False),
    ],
)
def test_suite_reference_guard_requires_an_exact_commit(reference, accepted):
    validation = next(step["run"] for step in STEPS if step.get("name") == "Validate the immutable suite reference")
    result = subprocess.run(
        ["bash", "-c", validation], env=dict(os.environ, SUITE_REF=reference, BASELINE_SHA="b" * 40), check=False
    )
    assert (result.returncode == 0) == accepted


@pytest.mark.parametrize("name", ["pull_request", "merge_group", "push", "workflow_dispatch"])
@pytest.mark.parametrize("supported", [False, True])
def test_comparison_base_is_bound_to_the_actual_event(tmp_path, monkeypatch, name, supported):
    candidate, baseline, suite = "c" * 40, "b" * 40, "a" * 40
    measured = baseline if supported else "d" * 40
    event = {
        "pull_request": {"base": {"sha": baseline}},
        "merge_group": {"base_sha": baseline},
        "before": baseline,
        "forced": True,
    }
    selection = {"event_base": baseline, "baseline": measured, "reason": "event-base" if supported else "pre-0.5"}
    request = {
        "candidate": candidate,
        "baseline": measured,
        "suite": suite,
        "run_id": 123,
        "run_attempt": 2,
        "event": name,
    }
    monkeypatch.chdir(tmp_path)
    (tmp_path / "inputs").mkdir()
    request_path = tmp_path / "inputs/request.json"
    request_path.write_text(json.dumps(request))
    comparison_path = tmp_path / "inputs/comparison.json"
    comparison_path.write_text(json.dumps(selection))
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    for key, value in {
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_EVENT_NAME": name,
        "CANDIDATE_SHA": candidate,
        "BASELINE_SHA": measured,
        "SUITE_REF": suite,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: baseline + "\n")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    def select(project, revision, base, *, forced):
        assert project == Path("executor") and revision == candidate and base == baseline
        assert forced is (name == "push")
        return selection

    monkeypatch.setattr(releases, "comparison", select)
    step = next(step["run"] for step in STEPS if step.get("name") == "Bind the requested inputs to the GitHub event")
    source = step.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    exec(compile(source, "evidence-hosted.yml:request", "exec"), {})
    comparison_path.write_text(json.dumps(dict(selection, event_base="e" * 40)))
    with pytest.raises(ValueError, match="triggering GitHub event"):
        exec(compile(source, "evidence-hosted.yml:request", "exec"), {})
    comparison_path.write_text(json.dumps(selection))
    request_path.write_text(json.dumps(dict(request, baseline="e" * 40)))
    with pytest.raises(ValueError, match="triggering GitHub event"):
        exec(compile(source, "evidence-hosted.yml:request", "exec"), {})
    request_path.write_text(json.dumps(dict(request, run_attempt=1)))
    with pytest.raises(ValueError, match="re-run all jobs"):
        exec(compile(source, "evidence-hosted.yml:request", "exec"), {})


def test_standard_cpu_matrix_matches_the_execution_profile_and_collects_failures():
    jobs = CONFIG["jobs"]
    assert set(jobs) == {"prepare", "shard", "collect", "plots"}
    assert all(job["runs-on"] == "ubuntu-24.04" for job in jobs.values())
    assert jobs["shard"]["strategy"] == {
        "fail-fast": False,
        "max-parallel": PROFILE["shards"],
        "matrix": {"shard": list(range(PROFILE["shards"]))},
    }
    assert jobs["collect"]["needs"] == ["prepare", "shard"]
    assert jobs["collect"]["if"].startswith("always()")
    assert jobs["collect"]["name"] == "Collect full paired suite"
    assert CONFIG["permissions"] == {"contents": "read", "actions": "read"}
    assert not any("tailscale" in str(step).lower() for job in jobs.values() for step in job["steps"])


@pytest.mark.parametrize("job", ["prepare", "shard", "collect", "plots"])
def test_every_job_checks_out_the_same_immutable_executor(job):
    checkouts = [
        step["with"]
        for step in CONFIG["jobs"][job]["steps"]
        if step.get("uses") == "actions/checkout@v4" and step["with"]["path"] == "executor"
    ]
    assert len(checkouts) == 1
    assert re.fullmatch(r"[0-9a-f]{40}", checkouts[0]["ref"])
    assert checkouts[0]["persist-credentials"] is False
    assert checkouts[0]["ref"] == next(
        step["with"]["ref"]
        for step in STEPS
        if step.get("uses") == "actions/checkout@v4" and step["with"]["path"] == "executor"
    )
    assert checkouts[0]["ref"] == publication.EXECUTOR


def test_wheel_preparation_records_the_supported_baseline_before_building(history, monkeypatch):
    root, (base, anchor, candidate) = history
    workspace = root / "workspace"
    workspace.mkdir()
    (workspace / "candidate").symlink_to(root, target_is_directory=True)
    event = workspace / "event.json"
    event.write_text(json.dumps({"pull_request": {"base": {"sha": base}}}))
    for key, value in {
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_EVENT_NAME": "pull_request",
        "CANDIDATE_SHA": candidate,
        "SUITE_REF": candidate,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
    }.items():
        monkeypatch.setenv(key, value)
    consumer = yaml.safe_load((WORKFLOW.parent / "evidence.yml").read_text())
    steps = consumer["jobs"]["wheels"]["steps"]
    resolver = next(step["with"]["ref"] for step in steps if step.get("with", {}).get("path") == "executor")
    assert resolver == publication.EXECUTOR
    build = next(step["run"] for step in steps if step.get("id") == "inputs")
    assert build.index("comparison(") < build.index("worktree add") < build.index("uv build")
    assert build.index("scripts/release.py licensing") < build.index("uv build")
    source = build.split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    monkeypatch.chdir(workspace)
    exec(compile(source, "evidence.yml:request", "exec"), {})
    selection = json.loads((workspace / "inputs/comparison.json").read_text())
    request = json.loads((workspace / "inputs/request.json").read_text())
    assert selection == releases.comparison(root, candidate, base)
    assert request["baseline"] == anchor and selection["event_base"] == base


@pytest.mark.parametrize("job", ["shard", "collect", "plots"])
@pytest.mark.parametrize("prepared,current,accepted", [("1", "1", True), ("1", "2", False), ("", "2", False)])
def test_partial_retry_cannot_mix_previous_attempts(job, prepared, current, accepted):
    step = next(
        step["run"]
        for step in CONFIG["jobs"][job]["steps"]
        if step.get("name") == "Require the current workflow attempt"
    )
    result = subprocess.run(
        ["bash", "-c", step],
        env=dict(os.environ, PREPARED_ATTEMPT=prepared, GITHUB_RUN_ATTEMPT=current),
        check=False,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is accepted
    if not accepted:
        assert "re-run all jobs" in result.stdout


def test_only_runtime_dependencies_are_cached_and_learning_has_no_retry_loop():
    caches = [
        (job, step)
        for job, value in CONFIG["jobs"].items()
        for step in value["steps"]
        if step.get("uses", "").startswith("actions/cache/")
    ]
    assert len(caches) == 2 and all(job == "prepare" for job, _ in caches)
    assert all(step["with"]["path"] == "${{ runner.temp }}/evidence-build-cache" for _, step in caches)
    learning = next(
        step["run"]
        for step in CONFIG["jobs"]["shard"]["steps"]
        if step.get("name") == "Execute each planned measurement once in isolated containers"
    )
    assert "hosted.py run" in learning
    assert "--runtime-sha256" in learning and "--runtime-id" in learning
    assert "--steps" not in learning and "for " not in learning
    for job in ("shard", "collect"):
        upload = CONFIG["jobs"][job]["steps"][-1]
        assert upload["if"] == "always()"
        assert "github.run_attempt" in upload["with"]["name"]


def test_runtime_is_attested_before_the_plan_is_created():
    names = [step.get("name") for step in STEPS]
    assert names.index("Build the shared CPU dependency image") < names.index(
        "Plan all canonical baseline and candidate tasks"
    )
    plan = next(step["run"] for step in STEPS if step.get("name") == "Plan all canonical baseline and candidate tasks")
    assert "--runtime-manifest runtime/runtime.json" in plan


def test_active_execution_has_no_lab_coordinator_or_ssh_modules():
    root = WORKFLOW.parents[2]
    retired = {"lab_run", "lab_stop", "lab_worker", "ssh_run"}
    for name in retired:
        assert not (root / "evidence" / f"{name}.py").exists()
    for path in (root / "evidence").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported = {part for alias in node.names for part in alias.name.split(".")}
            elif isinstance(node, ast.ImportFrom):
                imported = set((node.module or "").split(".")) | {
                    part for alias in node.names for part in alias.name.split(".")
                }
            else:
                continue
            assert imported.isdisjoint(retired), path
    for path in WORKFLOW.parent.glob("*.yml"):
        assert not any(f"{name}.py" in path.read_text() for name in retired), path
    assert (root / "evidence/runtime.py").is_file()
    assert (root / "evidence/worker.py").is_file()
