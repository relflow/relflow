"""Hosted Evidence keeps exact event inputs and a separately pinned executor."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from hosted import PROFILE

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/evidence-hosted.yml"
CONFIG = yaml.safe_load(WORKFLOW.read_text())
STEPS = CONFIG["jobs"]["prepare"]["steps"]


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
def test_comparison_base_is_bound_to_the_actual_event(tmp_path, monkeypatch, name):
    candidate, baseline, suite = "c" * 40, "b" * 40, "a" * 40
    event = {
        "pull_request": {"base": {"sha": baseline}},
        "merge_group": {"base_sha": baseline},
        "before": baseline,
    }
    request = {
        "candidate": candidate,
        "baseline": baseline,
        "suite": suite,
        "run_id": 123,
        "run_attempt": 2,
        "event": name,
    }
    monkeypatch.chdir(tmp_path)
    (tmp_path / "inputs").mkdir()
    request_path = tmp_path / "inputs/request.json"
    request_path.write_text(json.dumps(request))
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    for key, value in {
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_EVENT_NAME": name,
        "CANDIDATE_SHA": candidate,
        "BASELINE_SHA": baseline,
        "SUITE_REF": suite,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: baseline + "\n")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    step = next(step["run"] for step in STEPS if step.get("name") == "Bind the requested inputs to the GitHub event")
    source = step.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    exec(compile(source, "evidence-hosted.yml:request", "exec"), {})
    request_path.write_text(json.dumps(dict(request, baseline="d" * 40)))
    with pytest.raises(ValueError, match="triggering GitHub event"):
        exec(compile(source, "evidence-hosted.yml:request", "exec"), {})
    request_path.write_text(json.dumps(dict(request, run_attempt=1)))
    with pytest.raises(ValueError, match="re-run all jobs"):
        exec(compile(source, "evidence-hosted.yml:request", "exec"), {})


def test_standard_cpu_matrix_matches_the_execution_profile_and_collects_failures():
    jobs = CONFIG["jobs"]
    assert set(jobs) == {"prepare", "shard", "collect"}
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


@pytest.mark.parametrize("job", ["prepare", "shard", "collect"])
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


@pytest.mark.parametrize("job", ["shard", "collect"])
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
