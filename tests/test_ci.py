"""The merge gate rejects incomplete jobs, and releases consume exact successful main CI."""

import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"
GATE = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]["required"]["steps"][0]["run"]
RELEASE = yaml.safe_load((WORKFLOWS / "release.yml").read_text())["jobs"]["detect_version"]["steps"][0]["with"][
    "script"
]
EVENTS = ["pull_request", "merge_group", "push", "workflow_dispatch"]


@pytest.fixture
def gate(monkeypatch):
    def run(event, results, head="relflow/relflow"):
        monkeypatch.setenv("EVENT_NAME", event)
        monkeypatch.setenv("REPOSITORY", "relflow/relflow")
        monkeypatch.setenv("PR_REPOSITORY", head)
        monkeypatch.setenv("RESULTS", json.dumps({name: {"result": result} for name, result in results.items()}))
        exec(compile(GATE, "ci.yml:required", "exec"), {})

    return run


@pytest.mark.parametrize("event", EVENTS)
def test_complete_jobs_satisfy_the_gate(gate, event):
    gate(
        event,
        {
            "tests": "success",
            "evidence": "skipped" if event == "push" else "success",
            "package": "success" if event == "push" else "skipped",
        },
    )


@pytest.mark.parametrize("event,status", itertools.product(EVENTS, ["failure", "cancelled", "skipped"]))
def test_normal_tests_are_always_required(gate, event, status):
    results = {
        "tests": status,
        "evidence": "skipped" if event == "push" else "success",
        "package": "success" if event == "push" else "skipped",
    }
    with pytest.raises(AssertionError):
        gate(event, results)


@pytest.mark.parametrize(
    "event,status",
    itertools.product(["pull_request", "merge_group", "workflow_dispatch"], ["failure", "cancelled", "skipped"]),
)
def test_evidence_errors_or_skips_never_satisfy_premerge_validation(gate, event, status):
    with pytest.raises(AssertionError):
        gate(event, {"tests": "success", "evidence": status, "package": "skipped"})


def test_forks_and_missing_jobs_cannot_report_a_passing_gate(gate):
    with pytest.raises(AssertionError):
        gate("pull_request", {"tests": "success", "evidence": "success", "package": "skipped"}, head="fork/relflow")
    with pytest.raises(AssertionError):
        gate("pull_request", {"tests": "success", "evidence": "success"})


@pytest.fixture
def release():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the GitHub-script release guard")

    def run(change=None, manual=False, available=True):
        candidate = {
            "id": 123,
            "conclusion": "success",
            "event": "push",
            "head_branch": "main",
            "head_sha": "main-sha",
            "head_repository": {"full_name": "relflow/relflow"},
            "path": ".github/workflows/ci.yml",
        }
        candidate.update(change or {})
        fixture = {"source": RELEASE, "run": candidate, "manual": manual, "available": available}
        harness = """
        const fixture = JSON.parse(process.env.RELEASE_FIXTURE);
        const outputs = {};
        const context = {repo: {owner: 'relflow', repo: 'relflow'}, payload: fixture.manual ? {} : {workflow_run: fixture.run}};
        const github = {rest: {
          git: {getRef: async () => ({data: {object: {sha: 'main-sha'}}})},
          actions: {listWorkflowRuns: async () => ({data: {workflow_runs: fixture.available ? [fixture.run] : []}})},
        }};
        const core = {setOutput: (key, value) => {outputs[key] = value}, info: () => {}};
        const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
        await new AsyncFunction('github', 'context', 'core', fixture.source)(github, context, core);
        process.stdout.write(JSON.stringify(outputs));
        """
        return subprocess.run(
            [node, "--input-type=module", "-e", harness],
            env=dict(os.environ, RELEASE_FIXTURE=json.dumps(fixture)),
            capture_output=True,
            text=True,
        )

    return run


@pytest.mark.parametrize("manual", [False, True])
def test_release_selects_the_successful_run_of_the_exact_main_commit(release, manual):
    result = release(manual=manual)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"run_id": "123", "source_sha": "main-sha", "current": "true"}


@pytest.mark.parametrize(
    "change",
    [
        {"conclusion": "failure"},
        {"event": "pull_request"},
        {"head_branch": "feature"},
        {"head_repository": {"full_name": "fork/relflow"}},
        {"path": ".github/workflows/other.yml"},
    ],
)
def test_release_rejects_runs_outside_the_trusted_main_ci_boundary(release, change):
    assert release(change).returncode != 0


def test_release_skips_superseded_main_and_rejects_missing_validation(release):
    stale = release({"head_sha": "older-main"})
    assert stale.returncode == 0
    assert json.loads(stale.stdout) == {"current": "false"}
    assert release(manual=True, available=False).returncode != 0
