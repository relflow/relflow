"""Privileged publication accepts only complete, current, source-bound CI data."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import publication
import pytest
import reuse
import yaml
from evidence_ci import collect, finish, plan, write_json
from experiments.source import frontmatter, metadata
from publication import (
    EXECUTOR,
    archive,
    check_run,
    compatible,
    integrity,
    publish,
    retain,
    selection_changes,
)

SCRIPT = """# %% [markdown]
# ---
# title: An experiment
# ---
from reporting import experiment, report
@experiment({identifier!r}, seed=7)
def run(seed, steps, accelerator):
    return {{}}, {{'signal': True}}
if __name__ == '__main__':
    report(run)
"""
README = "Introduction\n<!-- evidence-results:start -->\nOld table\n<!-- evidence-results:end -->\nAfterword\n"


def commit(root, message):
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", message],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True, capture_output=True)
    for identifier in ("P021", "P025", "P038"):
        path = root / f"evidence/experiments/example/{identifier.lower()}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SCRIPT.format(identifier=identifier))
    (root / "README.md").write_text(README)
    (root / "pyproject.toml").write_text('[project]\nname="example"\nversion="0.4.2"\n')
    return root, commit(root, "base")


def run(main):
    return {
        "id": 123,
        "run_attempt": 2,
        "status": "completed",
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": main,
        "head_repository": {"full_name": "relflow/relflow"},
        "path": ".github/workflows/ci.yml",
        "workflow_id": 42,
        "html_url": "https://github.com/relflow/relflow/actions/runs/123",
        "referenced_workflows": [
            {
                "path": "relflow/relflow/.github/workflows/evidence-lab.yml@" + "b" * 40,
                "sha": "b" * 40,
            }
        ],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"id": 124},
        {"run_attempt": 0},
        {"status": "in_progress"},
        {"conclusion": "failure"},
        {"event": "pull_request"},
        {"event": "merge_group"},
        {"head_branch": "feature"},
        {"head_sha": "stale"},
        {"head_repository": {"full_name": "fork/relflow"}},
        {"path": ".github/workflows/other.yml"},
    ],
)
def test_run_boundary_rejects_untrusted_incomplete_or_stale_ci(change):
    with pytest.raises(ValueError, match="successful internal CI"):
        check_run(run("main") | change, "main", 123)


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_run_boundary_accepts_only_successful_exact_main(event):
    check_run(run("main") | {"event": event}, "main", 123)


def test_selection_classifier_preserves_program_other_headers_and_readme_boundaries(repository):
    root, _ = repository
    path = root / "evidence/experiments/example/p021.py"
    original = path.read_text()
    updated = frontmatter(
        original,
        metadata(original) | {"evidence-result": {"metrics": {"score": 1}}, "evidence-report": {"run_id": "new"}},
    )
    path.write_text(updated)
    (root / "README.md").write_text(README.replace("Old table", "New table"))
    commit(root, "selected result")
    assert selection_changes(
        {"evidence/experiments/example/p021.py": original, "README.md": README},
        {"evidence/experiments/example/p021.py": updated, "README.md": README.replace("Old table", "New table")},
    )
    for invalid in [
        updated.replace("seed=7", "seed=8"),
        updated.replace("An experiment", "Other question"),
        updated + "# unrelated prose\n",
    ]:
        assert not selection_changes(
            {"evidence/experiments/example/p021.py": original}, {"evidence/experiments/example/p021.py": invalid}
        )
    assert not selection_changes(
        {"README.md": README}, {"README.md": README.replace("Introduction", "Different introduction")}
    )
    assert not selection_changes({"src/relflow/model.py": "x=1"}, {"src/relflow/model.py": "x=2"})
    assert not selection_changes({"README.md": README}, {})


def test_first_parent_merge_classification_and_bootstrap_runtime_guard(repository):
    root, base = repository
    subprocess.run(["git", "checkout", "-b", "publication"], cwd=root, check=True, capture_output=True)
    path = root / "evidence/experiments/example/p021.py"
    path.write_text(frontmatter(path.read_text(), metadata(path.read_text()) | {"evidence-result": {"mode": "full"}}))
    commit(root, "publication")
    subprocess.run(["git", "checkout", "main"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "merge",
            "--no-ff",
            "publication",
            "-m",
            "merge selection",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    compatible(root, base, head)
    (root / "pyproject.toml").write_text("changed runtime dependencies")
    changed = commit(root, "runtime changed")
    with pytest.raises(ValueError, match="runtime input"):
        compatible(root, base, changed)


def test_archive_bytes_ignore_download_times_and_reject_links(tmp_path):
    source = tmp_path / "reports"
    source.mkdir()
    path = source / "manifest.json"
    path.write_text('{"valid":true}')
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    archive(source, first)
    os.utime(path, (100, 100))
    archive(source, second)
    assert first.read_bytes() == second.read_bytes()
    (source / "linked").symlink_to(path)
    with pytest.raises(ValueError, match="regular files"):
        archive(source, second)


@pytest.fixture
def artifacts(repository, tmp_path):
    root, main = repository
    inputs, reports = tmp_path / "inputs", tmp_path / "reports"
    identities = {}
    for role in ("baseline", "candidate"):
        path = inputs / role / "tested.whl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not imported or installed")
        identities[role] = {"revision": main, "wheel_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (inputs / "requirements.txt").write_text("locked==1\n")
    write_json(
        inputs / "request.json",
        {
            "candidate": main,
            "baseline": main,
            "suite": main,
            "run_id": 123,
            "run_attempt": 2,
            "event": "workflow_dispatch",
        },
    )
    manifest = plan(
        run_id="gh-123-2",
        candidate=identities["candidate"],
        baseline=identities["baseline"],
        evidence_sha=main,
        executor_sha=EXECUTOR,
        root=root / "evidence",
    )
    manifest["requirements_sha256"] = hashlib.sha256((inputs / "requirements.txt").read_bytes()).hexdigest()
    write_json(reports / "manifest.json", manifest)
    for task in manifest["tasks"]:
        metrics = {
            "hash_auc": 0.9,
            "category_oov_auc": 0.5,
            "median": {"intact_nrmse": 0.1, "broken_nrmse": 1.2},
            "state_accuracy": 1.0,
            "prefilled_accuracy": 0.5,
        }
        result = {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": task["accelerator"],
            "outcome": "met",
            "checks": {"signal": True},
            "metrics": metrics,
            "provenance": {
                "evidence_revision": main,
                "relflow_revision": main,
                "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        }
        collect(reports, {"task": task, "error": None, "result": result, "log": "one bounded execution log"})
    finish(reports, root=root / "evidence")
    return reports, inputs


@pytest.mark.parametrize("fault", ["corrupt", "extra", "missing-log", "incomplete-inventory"])
def test_report_inventory_and_every_task_log_are_required(artifacts, fault):
    reports, _ = artifacts
    integrity(reports)
    if fault == "corrupt":
        (reports / "report.html").write_text("corrupted report")
    elif fault == "extra":
        (reports / "unexpected.txt").write_text("not covered")
    elif fault == "missing-log":
        next((reports / "logs").rglob("*.txt")).unlink()
        checksums = json.loads((reports / "checksums.json").read_text())
        checksums = {name: value for name, value in checksums.items() if (reports / name).is_file()}
        (reports / "checksums.json").write_text(json.dumps(checksums))
    else:
        checksums = json.loads((reports / "checksums.json").read_text())
        checksums.pop("report.html")
        (reports / "checksums.json").write_text(json.dumps(checksums))
    with pytest.raises(ValueError, match="checksum|task log"):
        integrity(reports)


@pytest.mark.parametrize("failure", [None, "render", "stale-main"])
@pytest.mark.parametrize("retry", [False, True])
def test_release_publication_creates_validated_assets_without_changing_sources(
    repository, artifacts, tmp_path, monkeypatch, failure, retry
):
    root, main = repository
    reports, inputs = artifacts
    (root / "guide.md").write_text("Trusted new static publisher tooling/prose.\n")
    checkout = commit(root, "bootstrap")
    calls, retained = [], []

    def api(path, **kwargs):
        if path.endswith("/git/ref/heads/main"):
            calls.append("main")
            return {"object": {"sha": "advanced" if failure == "stale-main" and len(calls) > 1 else main}}
        if path.endswith("jobs?per_page=100"):
            return {
                "jobs": [
                    {
                        "id": 91,
                        "name": "Evidence / Run full paired suite on lab / Run full paired suite on lab",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ]
            }
        if "/attempts/2" in path:
            return dict(run(main), conclusion="failure")
        if "/actions/runs/" in path:
            return dict(run(main), run_attempt=3 if retry else 2)
        return {"id": 42}

    monkeypatch.setattr(reuse, "api", api)
    real_run = subprocess.run

    def execute(command, *args, **kwargs):
        if command[:3] == ["gh", "run", "download"]:
            source = reports if command[command.index("--name") + 1] == "relflow-evidence" else inputs
            shutil.copytree(source, command[command.index("--dir") + 1])
            return subprocess.CompletedProcess(command, 0)
        if command[0] == "make":
            if failure == "render":
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(publication, "api", api)
    monkeypatch.setattr(
        publication,
        "verified_artifact",
        lambda run_id, name, target: shutil.copytree(reports if name == "relflow-evidence" else inputs, target),
    )
    monkeypatch.setattr(publication.subprocess, "run", execute)
    monkeypatch.setattr(publication, "retain", lambda *args: retained.append(args))
    output = tmp_path / "publication"
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            publish(root, output, 123)
        assert not retained
    else:
        publish(root, output, 123)
        assert retained[0][1] == main
        assert (output / "docs-base.txt").read_text().strip() == checkout
        selected = json.loads((output / "publication.json").read_text())
        assert selected["version"] == "0.4.2"
        assert selected["report"]["github_run"]["run_attempt"] == 2
        assert selected["revision"] == main
        assert selected["report"]["release"].endswith("/evidence-v0.4.2")
        assert all(result["provenance"]["relflow_revision"] == main for result in selected["results"].values())
        assert not subprocess.check_output(["git", "diff", "--name-only"], cwd=root).strip()
        assert "publication.json" in retained[0][3]
        assert "measurements.md" in retained[0][3]
        assert not (output / "evidence-update.patch").exists()


@pytest.mark.parametrize("corrupt", [False, True])
def test_existing_immutable_release_is_reused_or_rejected_without_overwrite(tmp_path, monkeypatch, corrupt):
    (tmp_path / "one.json").write_text("correct")
    writes = []

    def api(path, **kwargs):
        if "/git/ref/tags/" in path:
            return {"object": {"type": "commit", "sha": "tested"}}
        release = {
            "tag_name": "evidence-test",
            "target_commitish": "tested",
            "draft": False,
            "assets": [{"name": "one.json"}],
        }
        return [[release]] if kwargs.get("paginate") else release

    def execute(command, **kwargs):
        if command[:3] == ["gh", "release", "download"]:
            (Path(command[command.index("--dir") + 1]) / "one.json").write_text("different" if corrupt else "correct")
        else:
            writes.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(publication, "api", api)
    monkeypatch.setattr(publication.subprocess, "run", execute)
    if corrupt:
        with pytest.raises(ValueError, match="Immutable report asset differs"):
            retain("evidence-test", "tested", tmp_path, ["one.json"])
    else:
        retain("evidence-test", "tested", tmp_path, ["one.json"])
    assert writes == []


def test_workflow_never_promotes_pr_runs_or_installs_model_runtime():
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github/workflows/publish-evidence.yml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    job = workflow["jobs"]["publish"]
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_call", "workflow_dispatch", "workflow_run"}
    assert "!inputs.preview" in job["if"] and "inputs.library_run_id != ''" in job["if"]
    assert "pull-requests" not in job["permissions"]
    checkout = next(step["with"] for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["persist-credentials"] is False and checkout["fetch-depth"] == 0
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "uv sync --locked --project docs" in commands
    assert "--version" in commands and "--automatic" not in commands
    assert "uv pip install" not in commands and "--with" not in commands
    release = yaml.safe_load((Path(__file__).resolve().parents[2] / ".github/workflows/release.yml").read_text())
    assert release["jobs"]["evidence"]["uses"] == "./.github/workflows/publish-evidence.yml"
    assert "evidence" in release["jobs"]["publish"]["needs"]


def test_ci_attempts_keep_all_artifacts_and_checkpoint_consumers_separate():
    workflows = Path(__file__).resolve().parents[2] / ".github/workflows"
    for name in ("tests.yml", "docs.yml", "publish-evidence.yml", "evidence-hosted.yml", "evidence.yml"):
        value = yaml.safe_load((workflows / name).read_text())
        for job in value["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses") == "actions/upload-artifact@v4":
                    assert "github.run_attempt" in step["with"]["name"], (name, step)
    tests = yaml.safe_load((workflows / "tests.yml").read_text())["jobs"]
    checkpoint = next(
        step["with"]["name"]
        for step in tests["checkpoint"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    for name in ("suite", "preview"):
        downloads = [
            step["with"]["name"] for step in tests[name]["steps"] if step.get("uses") == "actions/download-artifact@v4"
        ]
        assert checkpoint in downloads
    coverage = yaml.safe_load((workflows / "coverage.yml").read_text())["jobs"]
    downloads = [
        step["with"]["name"]
        for job in coverage.values()
        for step in job["steps"]
        if step.get("uses") == "actions/download-artifact@v4"
    ]
    assert downloads == ["coverage-attempt-${{ github.event.workflow_run.run_attempt }}"]


@pytest.mark.parametrize("fault", ["attempt", "candidate", "baseline", "suite", "executor", "regression"])
def test_report_cannot_relabel_other_attempts_revisions_or_failed_gates(fault):
    manifest = {
        "run_id": "gh-123-2",
        "candidate": {"revision": "main"},
        "baseline": {"revision": "main"},
        "evidence_sha": "main",
        "executor_sha": EXECUTOR,
    }
    summary = {"passed": True}
    if fault == "attempt":
        manifest["run_id"] = "gh-123-1"
    elif fault in {"candidate", "baseline"}:
        manifest[fault]["revision"] = "different"
    elif fault == "suite":
        manifest["evidence_sha"] = "different"
    elif fault == "executor":
        manifest["executor_sha"] = "different"
    else:
        summary["passed"] = False
    with pytest.raises(ValueError, match="Report identity"):
        publication.check_report(manifest, summary, run("main"), "main")


def test_published_version_never_reselects_results_or_creates_a_pull_request(repository, tmp_path, monkeypatch):
    root, _ = repository
    monkeypatch.setattr(publication, "api", lambda *args, **kwargs: {"tag_name": "v0.4.2", "draft": False})
    monkeypatch.setattr(publication, "verified_artifact", lambda *args: pytest.fail("A released version is frozen"))
    monkeypatch.setattr(
        publication.subprocess, "run", lambda *args, **kwargs: pytest.fail("No external write is needed")
    )
    publish(root, tmp_path / "publication", 456)
    assert not (tmp_path / "publication").exists()


@pytest.mark.parametrize("interrupted_after_upload", [False, True])
def test_interrupted_draft_without_tag_resumes_without_duplicate_create_or_overwriting_assets(
    tmp_path, monkeypatch, interrupted_after_upload
):
    for name in ("first.json", "second.json"):
        (tmp_path / name).write_text(name)
    present = {"first.json"} if interrupted_after_upload else set()
    state = {"published": False}
    writes = []

    def api(path, **kwargs):
        if kwargs.get("paginate"):
            return [
                [{"tag_name": "unrelated", "target_commitish": "other", "draft": False, "assets": []}],
                [
                    {
                        "tag_name": "evidence-test",
                        "target_commitish": "tested",
                        "draft": True,
                        "assets": [{"name": name} for name in present],
                    }
                ],
            ]
        if "/git/ref/tags/" in path:
            if not state["published"]:
                assert kwargs.get("missing") is True
                return {}
            return {"object": {"type": "commit", "sha": "tested"}}
        assert state["published"], "Draft resumption must not use the published-by-tag lookup"
        return {"draft": False, "assets": [{"name": name} for name in present]}

    def execute(command, **kwargs):
        assert command[:2] == ["gh", "release"]
        operation = command[2]
        if operation == "download":
            for name in present:
                (Path(command[command.index("--dir") + 1]) / name).write_text(name)
        elif operation == "upload":
            uploaded = {Path(name).name for name in command[6:]}
            assert not uploaded & present
            present.update(uploaded)
            writes.append("upload")
        elif operation == "edit":
            assert present == {"first.json", "second.json"}
            assert "--draft=false" in command and "--latest=false" in command
            state["published"] = True
            writes.append("publish")
        else:
            pytest.fail("An interrupted draft must never create a duplicate release")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(publication, "api", api)
    monkeypatch.setattr(publication.subprocess, "run", execute)
    retain("evidence-test", "tested", tmp_path, ["first.json", "second.json"])
    assert writes == ["upload", "publish"]


def test_tag_without_release_cannot_redirect_new_report_to_untested_code(tmp_path, monkeypatch):
    def api(path, **kwargs):
        if kwargs.get("paginate"):
            return [[]]
        return {"object": {"type": "commit", "sha": "different"}}

    monkeypatch.setattr(publication, "api", api)
    monkeypatch.setattr(
        publication.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Wrong existing tag must fail before all writes"),
    )
    with pytest.raises(ValueError, match="Existing report tag differs"):
        retain("evidence-test", "tested", tmp_path, [])
