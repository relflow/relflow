"""The privileged lab workflow accepts suite data without trusting its host code."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/evidence-lab.yml"
STEPS = yaml.safe_load(WORKFLOW.read_text())["jobs"]["run"]["steps"]


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
    exec(compile(source, "evidence-lab.yml:request", "exec"), {})
    request_path.write_text(json.dumps(dict(request, baseline="d" * 40)))
    with pytest.raises(ValueError, match="triggering GitHub event"):
        exec(compile(source, "evidence-lab.yml:request", "exec"), {})
