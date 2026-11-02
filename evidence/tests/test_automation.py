"""Privileged publication accepts only complete, current, source-bound CI data."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import publication
import pytest
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
    selection_only,
)

SCRIPT = """# %% [markdown]
# ---
# evidence-id: {identifier}
# title: An experiment
# ---
report(seed=7)
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
    root, before = repository
    path = root / "evidence/experiments/example/p021.py"
    original = path.read_text()
    updated = frontmatter(
        original,
        metadata(original) | {"evidence-result": {"metrics": {"score": 1}}, "evidence-report": {"run_id": "new"}},
    )
    path.write_text(updated)
    (root / "README.md").write_text(README.replace("Old table", "New table"))
    selected = commit(root, "selected result")
    assert selection_only(root, before, selected)
    assert not selection_only(root, selected, selected)
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
    assert selection_only(root, head + "^1", head)
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
def test_manual_bootstrap_creates_validated_patch_before_any_external_write(
    repository, artifacts, tmp_path, monkeypatch, failure
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
        if "/actions/runs/" in path:
            return run(main)
        return {"id": 42}

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
    monkeypatch.setattr(publication, "propose", lambda *args: pytest.fail("Manual mode must not push or create a PR"))
    output = tmp_path / "publication"
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            publish(root, output, 123)
        assert not retained
    else:
        publish(root, output, 123)
        assert retained[0][1] == main
        assert (output / "docs-base.txt").read_text().strip() == checkout
        patch = (output / "evidence-update.patch").read_text()
        assert "evidence-result" in patch and "evidence-results:start" in patch
        assert "relflow_revision: " + main in patch
        assert "guide.md" not in patch


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
    assert triggers["workflow_run"] == {"workflows": ["CI"], "branches": ["main"], "types": ["completed"]}
    job = workflow["jobs"]["publish"]
    assert "head_branch == 'main'" in job["if"]
    assert "workflow_run.event == 'push'" in job["if"] and "workflow_run.event == 'workflow_dispatch'" in job["if"]
    checkout = next(step["with"] for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["persist-credentials"] is False and checkout["fetch-depth"] == 0
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "uv sync --locked --project docs" in commands
    assert "--automatic" in commands
    assert "uv pip install" not in commands and "--with" not in commands


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


@pytest.mark.parametrize("changed_branch", [False, True])
def test_rerun_reuses_exact_remote_branch_without_recommit_force_or_overwriting_human_changes(
    repository, tmp_path, monkeypatch, changed_branch
):
    root, main = repository
    (root / "README.md").write_text(README.replace("Old table", "New table"))

    def api(path, **kwargs):
        if "/git/ref/heads/automation/" in path:
            return {"object": {"sha": "existing-commit"}}
        if "/git/commits/" in path:
            tree = subprocess.check_output(["git", "write-tree"], cwd=root, text=True).strip()
            return {"tree": {"sha": "human-changed" if changed_branch else tree}, "parents": [{"sha": main}]}
        return {"object": {"sha": main}}

    real_output, real_run = subprocess.check_output, subprocess.run

    def output(command, *args, **kwargs):
        if command[:3] == ["gh", "pr", "list"]:
            return '[{"number":456}]'
        return real_output(command, *args, **kwargs)

    def execute(command, *args, **kwargs):
        if command[0] == "gh" or any(part in {"push", "commit", "checkout"} for part in command):
            pytest.fail("An existing matching bot branch must never be recommitted, force-pushed, or replaced")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(publication, "api", api)
    monkeypatch.setattr(publication.subprocess, "check_output", output)
    monkeypatch.setattr(publication.subprocess, "run", execute)
    if changed_branch:
        with pytest.raises(ValueError, match="Existing evidence bot branch differs"):
            publication.propose(root, tmp_path, run(main), "evidence-test", main, ["README.md"])
    else:
        publication.propose(root, tmp_path, run(main), "evidence-test", main, ["README.md"])
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == main


def test_automatic_selection_only_main_skips_publication_but_not_ci(repository, artifacts, tmp_path, monkeypatch):
    root, base = repository
    reports, inputs = artifacts
    (root / "README.md").write_text(README.replace("Old table", "New table"))
    main = commit(root, "selection-only merge")
    (inputs / "request.json").write_text(
        json.dumps(
            {
                "candidate": main,
                "baseline": base,
                "suite": main,
                "run_id": 123,
                "run_attempt": 2,
                "event": "workflow_dispatch",
            }
        )
    )

    def api(path, **kwargs):
        if "/git/ref/" in path:
            return {"object": {"sha": main}}
        if "/actions/runs/" in path:
            return run(main)
        return {"id": 42}

    monkeypatch.setattr(publication, "api", api)
    monkeypatch.setattr(
        publication,
        "verified_artifact",
        lambda run_id, name, target: shutil.copytree(reports if name == "relflow-evidence" else inputs, target),
    )
    monkeypatch.setattr(publication, "retain", lambda *args: pytest.fail("No recursive release"))
    monkeypatch.setattr(publication, "propose", lambda *args: pytest.fail("No recursive publication PR"))
    output = tmp_path / "not-created"
    publish(root, output, 123, automatic=True)
    assert not output.exists()


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
