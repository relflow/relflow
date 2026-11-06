"""Equivalent input reuse verifies immutable execution, never fabricated fresh provenance."""

import base64
import copy
import hashlib
import io
import json
import shutil
import subprocess
import zipfile

import publication
import pytest
import reuse
from evidence_ci import collect, digest, finish, plan, write_json
from experiments.source import frontmatter, metadata, registry
from record import select

EXECUTOR = "a" * 40
HELPER = "b" * 40
IDENTITY = {"image": "sha256:" + "c" * 64, "cpu": "verified CPU", "kernel": "test", "env": {"OMP_NUM_THREADS": "1"}}
SCRIPT = "# %% [markdown]\n# ---\n# evidence-id: {identifier}\n# title: A bounded experiment\n# ---\nreport(seed=7)\n"
README = "Intro\n<!-- evidence-results:start -->\nOld measurements\n<!-- evidence-results:end -->\nEnd\n"


def git(root, *args, input=None):
    return subprocess.check_output(["git", *args], cwd=root, input=input).decode().strip()


def commit(root, message):
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


def legacy_selection(root, reports):
    """Model a historical inline selection so retained-report normalization stays covered."""
    document = select(reports, root=root / "evidence", readme_target=root / "README.md")
    for identifier, entry in registry(root / "evidence").items():
        script = root / "evidence" / entry["script"]
        header = dict(
            entry["header"],
            **{"evidence-result": document["results"][identifier], "evidence-report": document["report"]},
        )
        script.write_text(frontmatter(script.read_text(), header))


def wheel(path, readme, code=b"runtime", *, headers=None, mode=0o100644 << 16, corrupt=False, extra=False):
    files = {
        "relflow/model.py": code,
        "relflow-0.4.2.dist-info/METADATA": (
            headers or b"Metadata-Version: 2.4\nName: relflow\nVersion: 0.4.2\nRequires-Dist: locked==1"
        )
        + b"\n\n"
        + readme,
        "relflow-0.4.2.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    record = "".join(
        f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()},{len(content)}\n"
        for name, content in files.items()
    )
    files["relflow-0.4.2.dist-info/RECORD"] = (record + "relflow-0.4.2.dist-info/RECORD,,\n").encode()
    if corrupt:
        files["relflow/model.py"] += b"not covered by the recorded hash"
    if extra:
        files["relflow/unrecorded.py"] = b"unrecorded"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.external_attr = mode
            archive.writestr(entry, content)


def current_inputs(bundle):
    root, reports, inputs, run, requested = bundle
    destination = inputs.parent / "current-inputs"
    destination.mkdir(exist_ok=True)
    shutil.copy2(inputs / "requirements.txt", destination / "requirements.txt")
    for role in ("candidate", "baseline"):
        target = destination / role
        target.mkdir(exist_ok=True)
        readme = git(root, "show", requested[role] + ":README.md").encode() + b"\n"
        code = git(root, "show", requested[role] + ":src/model.py").encode() + b"\n"
        wheel(target / "tested.whl", readme, code)
    request = {name: requested[name] for name in ("candidate", "baseline", "suite", "run_id", "run_attempt", "event")}
    (destination / "request.json").write_text(json.dumps(request))
    return destination


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    for identifier in ("P021", "P025", "P038"):
        path = root / "evidence/experiments/example" / (identifier.lower() + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SCRIPT.format(identifier=identifier))
    (root / "README.md").write_text(README)
    (root / "uv.lock").write_text("locked-runtime\n")
    (root / "pyproject.toml").write_text('[project]\nversion="0.4.2"\n')
    (root / "src").mkdir()
    (root / "src/model.py").write_text("prediction = 1\n")
    base = commit(root, "actual comparison base")
    (root / "src/model.py").write_text("prediction = 2\n")
    candidate = commit(root, "candidate implementation")
    inputs, reports = tmp_path / "inputs", tmp_path / "reports"
    identities = {}
    for role, revision in (("baseline", base), ("candidate", candidate)):
        path = inputs / role / "tested.whl"
        path.parent.mkdir(parents=True)
        wheel(path, README.encode(), git(root, "show", revision + ":src/model.py").encode() + b"\n")
        identities[role] = {"revision": revision, "wheel_sha256": digest(path)}
    (inputs / "requirements.txt").write_text("locked==1\n")
    write_json(
        inputs / "request.json",
        {
            "candidate": candidate,
            "baseline": base,
            "suite": candidate,
            "run_id": 123,
            "run_attempt": 2,
            "event": "push",
        },
    )
    manifest = plan(
        run_id="gh-123-2",
        candidate=identities["candidate"],
        baseline=identities["baseline"],
        evidence_sha=candidate,
        executor_sha=EXECUTOR,
        root=root / "evidence",
    )
    manifest["requirements_sha256"] = digest(inputs / "requirements.txt")
    write_json(reports / "manifest.json", manifest)
    write_json(reports / "execution.json", {"identity": IDENTITY})
    for task in manifest["tasks"]:
        result = {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": "cpu",
            "outcome": "not_met",
            "checks": {"bounded passing check": True, "honest unmet check": False},
            "metrics": {
                "hash_auc": 0.9,
                "category_oov_auc": 0.5,
                "median": {"intact_nrmse": 0.1, "broken_nrmse": 1.2},
                "state_accuracy": 1.0,
                "prefilled_accuracy": 0.5,
            },
            "provenance": {
                "evidence_revision": candidate,
                "relflow_revision": manifest[task["role"]]["revision"],
                "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        }
        collect(reports, {"task": task, "error": None, "result": result, "log": "complete immutable execution log"})
    finish(reports, root=root / "evidence")
    run = {
        "id": 123,
        "run_attempt": 2,
        "status": "completed",
        "conclusion": "success",
        "event": "push",
        "head_branch": "main",
        "head_sha": candidate,
        "head_repository": {"full_name": reuse.REPOSITORY},
        "workflow_id": 42,
        "path": ".github/workflows/ci.yml",
        "html_url": "https://github.com/relflow/relflow/actions/runs/123",
        "referenced_workflows": [
            {"path": f"{reuse.REPOSITORY}/.github/workflows/evidence-lab.yml@{HELPER}", "sha": HELPER}
        ],
    }
    requested = {
        "candidate": candidate,
        "baseline": base,
        "suite": candidate,
        "executor": EXECUTOR,
        "execution_identity": IDENTITY,
        "run_id": 456,
        "run_attempt": 1,
        "event": "push",
    }
    return root, reports, inputs, run, requested


def verify(bundle):
    root, reports, inputs, run, requested = bundle
    return reuse.verify(
        root,
        reports,
        inputs,
        run,
        requested,
        current_inputs=current_inputs(bundle),
        helper=HELPER,
        source={"artifacts": []},
    )


def test_exact_input_reuse_preserves_records_revisions_and_unmet_checks(bundle):
    root, reports, _, _, requested = bundle
    original = reuse.delivered(reports)
    receipt = verify(bundle)
    assert receipt["passed"] is True
    assert receipt["requested"] == requested
    assert receipt["source"]["candidate"]["revision"] == requested["candidate"]
    assert receipt["source"]["baseline"]["revision"] == requested["baseline"]
    assert reuse.delivered(reports) == original
    assert git(root, "status", "--porcelain") == ""


@pytest.mark.parametrize(
    "fault",
    [
        "base",
        "candidate",
        "suite",
        "lock",
        "mode",
        "path",
        "gate",
        "comment",
        "readme-forgery",
        "header-forgery",
        "unrelated-header",
    ],
)
def test_input_equivalence_fails_closed_except_exact_measured_selection(bundle, fault):
    root, reports, _, _, requested = bundle
    if fault == "base":
        (root / "src/model.py").write_text("prediction = 999\n")
        requested["baseline"] = commit(root, "a different actual baseline")
    else:
        script = root / "evidence/experiments/example/p021.py"
        if fault == "candidate":
            (root / "src/model.py").write_text("prediction = 999\n")
        elif fault == "suite":
            script.write_text(script.read_text().replace("seed=7", "seed=8"))
        elif fault == "lock":
            (root / "uv.lock").write_text("different lock\n")
        elif fault == "mode":
            script.chmod(0o755)
        elif fault == "path":
            (root / "evidence/experiments/extra-policy.txt").write_text("new tracked runtime input")
        elif fault == "gate":
            script.write_text(script.read_text() + "check_threshold = 0.9\n")
        elif fault == "comment":
            script.write_text(script.read_text() + "# changed nonselection prose\n")
        elif fault.startswith("readme"):
            text = (
                README.replace("Intro", "Other intro")
                if fault == "readme-prose"
                else README.replace("Old measurements", "fabricated 100% score")
            )
            (root / "README.md").write_text(text)
        else:
            header = metadata(script.read_text())
            if fault == "header-forgery":
                header["evidence-result"] = {"metrics": {"fabricated": 1}}
            else:
                header["title"] = "A different experiment"
            script.write_text(frontmatter(script.read_text(), header))
        changed = commit(root, fault)
        requested["candidate"] = requested["suite"] = changed
    with pytest.raises(ValueError):
        verify(bundle)


def test_selection_only_pr_must_match_archived_records_and_bounded_readme(bundle):
    root, reports, _, _, requested = bundle
    original_candidate = requested["candidate"]
    legacy_selection(root, reports)
    selected = commit(root, "select actual complete measurements")
    requested["candidate"] = requested["suite"] = selected
    requested["baseline"] = original_candidate
    receipt = verify(bundle)
    assert receipt["source"]["candidate"]["revision"] == original_candidate
    assert receipt["requested"]["candidate"] == selected
    assert set(receipt["input_sha256"]) == {"candidate", "baseline", "suite"}
    assert receipt["role_mapping"] == {"baseline": "candidate", "candidate": "candidate"}
    assert receipt["source"]["baseline"]["revision"] != receipt["requested"]["baseline"]


@pytest.mark.parametrize("fault", ["check-type", "passed-type", "release", "manifest-hash"])
def test_selection_normalization_requires_exact_values_types_and_retained_report_identity(bundle, fault):
    root, reports, _, _, requested = bundle
    original = requested["candidate"]
    legacy_selection(root, reports)
    script = root / "evidence/experiments/example/p021.py"
    header = metadata(script.read_text())
    if fault == "check-type":
        header["evidence-result"]["checks"]["bounded passing check"] = 1
    elif fault == "passed-type":
        header["evidence-report"]["passed"] = 1
    elif fault == "release":
        header["evidence-report"]["release"] = (
            "https://github.com/relflow/relflow/releases/tag/evidence-ci-999-attempt-1"
        )
    else:
        header["evidence-report"]["manifest_sha256"] = "0" * 64
    script.write_text(frontmatter(script.read_text(), header))
    requested["candidate"] = requested["suite"] = commit(root, "invalid selection value")
    requested["baseline"] = original
    with pytest.raises(ValueError):
        verify(bundle)


@pytest.mark.parametrize("fault", ["failed", "partial", "fork", "attempt", "helper", "executor", "environment"])
def test_untrusted_or_incomplete_execution_is_never_a_reuse_hit(bundle, fault):
    _, _, _, run, requested = bundle
    if fault == "failed":
        run["conclusion"] = "failure"
    elif fault == "partial":
        run["status"] = "in_progress"
    elif fault == "fork":
        run["head_repository"]["full_name"] = "fork/relflow"
    elif fault == "attempt":
        run["run_attempt"] = 3
    elif fault == "helper":
        run["referenced_workflows"][0]["sha"] = "d" * 40
    elif fault == "executor":
        requested["executor"] = "d" * 40
    else:
        requested["execution_identity"] = dict(IDENTITY, kernel="changed actual kernel")
    with pytest.raises(ValueError):
        verify(bundle)


def test_pr_synthetic_merge_binds_head_and_actual_base_without_relabelling(bundle):
    root, reports, inputs, run, requested = bundle
    original = json.loads((reports / "manifest.json").read_text())
    candidate = original["candidate"]["revision"]
    base = original["baseline"]["revision"]
    tree = git(root, "rev-parse", candidate + "^{tree}")
    synthetic = git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit-tree",
        tree,
        "-p",
        base,
        "-p",
        candidate,
        input=b"synthetic PR merge\n",
    )
    manifest = dict(original, candidate=dict(original["candidate"], revision=synthetic))
    run.update(event="pull_request", pull_requests=[{"base": {"sha": base}, "head": {"sha": candidate}}])
    request = dict(json.loads((inputs / "request.json").read_text()), candidate=synthetic, event="pull_request")
    reuse.trusted(run, manifest, helper=HELPER, executor=EXECUTOR, root=root, request=request)
    with pytest.raises(ValueError, match="actual comparison base|Original input request"):
        reuse.trusted(
            run,
            dict(manifest, baseline=dict(manifest["baseline"], revision=candidate)),
            helper=HELPER,
            executor=EXECUTOR,
            root=root,
            request=request,
        )


def test_original_pr_merge_parent_proof_survives_real_depth_one_fetch(bundle, tmp_path, monkeypatch):
    root, reports, inputs, run, _ = bundle
    manifest = json.loads((reports / "manifest.json").read_text())
    base = manifest["baseline"]["revision"]
    head = manifest["candidate"]["revision"]
    tree = git(root, "rev-parse", head + "^{tree}")
    merge = git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit-tree",
        tree,
        "-p",
        base,
        "-p",
        head,
        input=("PR merge\n\nparent " + "d" * 40 + "\n").encode(),
    )
    runner = tmp_path / "runner-executor"
    runner.mkdir()
    git(runner, "init", "-q")
    git(runner, "fetch", "--quiet", "--depth=1", str(root), base)
    assert (
        subprocess.run(["git", "cat-file", "-e", merge + "^{commit}"], cwd=runner, capture_output=True).returncode != 0
    )
    real_run = subprocess.run
    fetched = []

    def execute(command, *args, **kwargs):
        if command[:2] == ["git", "fetch"] and f"https://github.com/{reuse.REPOSITORY}.git" in command:
            fetched.append(command)
            command = [str(root) if part == f"https://github.com/{reuse.REPOSITORY}.git" else part for part in command]
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(reuse.subprocess, "run", execute)
    reuse.ensure(runner, merge)
    assert len(fetched) == 1 and "--depth=1" in fetched[0]
    assert merge in (runner / ".git/shallow").read_text().splitlines()
    assert git(runner, "show", "-s", "--format=%P", merge) == ""
    run.update(event="pull_request", pull_requests=[{"base": {"sha": base}, "head": {"sha": head}}])
    manifest["candidate"]["revision"] = merge
    request = dict(json.loads((inputs / "request.json").read_text()), candidate=merge, event="pull_request")
    reuse.trusted(run, manifest, helper=HELPER, executor=EXECUTOR, root=runner, request=request)
    manifest["candidate"]["revision"] = head
    reuse.ensure(runner, head)
    run["head_sha"] = "e" * 40
    with pytest.raises(ValueError, match="actual comparison base|Original input request"):
        reuse.trusted(
            run, manifest, helper=HELPER, executor=EXECUTOR, root=runner, request=dict(request, candidate=head)
        )


def test_miss_diagnostics_are_bounded_and_do_not_expose_subprocess_arguments():
    message = reuse.reason(ValueError("Public source mismatch\n" + "x" * 1000))
    assert len(message) <= 256 and "\n" not in message
    assert message.startswith("ValueError: Public source mismatch ")
    assert reuse.reason(subprocess.CalledProcessError(1, ["command", "private-argument"])) == "CalledProcessError"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "same\\windows"])
def test_archive_handoff_rejects_traversal_and_links(tmp_path, name):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(name, "untrusted archive member")
    with pytest.raises(ValueError, match="unsafe"):
        reuse.unpack(data.getvalue(), tmp_path / "artifact", zipped=True)


def test_artifact_bytes_must_match_github_digest_before_extraction(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("manifest.json", "{}")
    row = {"id": 1, "name": "relflow-evidence", "expired": False, "digest": "sha256:" + "0" * 64}
    monkeypatch.setattr(reuse, "api", lambda path: {"artifacts": [row]})
    monkeypatch.setattr(reuse.subprocess, "check_output", lambda *args, **kwargs: payload.getvalue())
    with pytest.raises(ValueError, match="GitHub digest"):
        reuse.artifact(123, "relflow-evidence", tmp_path / "untrusted")
    assert not (tmp_path / "untrusted").exists()


def test_attempt_artifact_selection_checks_every_page_and_rejects_duplicates(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("manifest.json", "{}")
    row = {
        "id": 101,
        "name": "relflow-evidence-attempt-9",
        "expired": False,
        "digest": "sha256:" + hashlib.sha256(payload.getvalue()).hexdigest(),
    }
    monkeypatch.setattr(
        reuse,
        "api",
        lambda path: {"artifacts": [row]} if path.endswith("&page=2") else {"artifacts": [], "total_count": 101},
    )
    monkeypatch.setattr(reuse.subprocess, "check_output", lambda *args, **kwargs: payload.getvalue())
    assert reuse.artifact(123, row["name"], tmp_path / "retained")["id"] == 101
    assert (tmp_path / "retained/manifest.json").read_text() == "{}"
    monkeypatch.setattr(reuse, "api", lambda path: {"artifacts": [row, row]})
    with pytest.raises(ValueError, match="one unexpired immutable artifact"):
        reuse.artifact(123, row["name"], tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_force_fresh_never_performs_network_lookup_or_creates_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, "api", lambda *args: pytest.fail("Force fresh must skip reuse discovery"))
    assert (
        reuse.find(tmp_path, tmp_path / "not-created", {}, current_inputs=tmp_path, helper=HELPER, force=True) is False
    )
    assert not (tmp_path / "not-created").exists()


def test_privileged_publication_rejects_receipt_for_other_current_request(bundle):
    root, reports, inputs, run, requested = bundle
    receipt = verify(bundle)
    current = dict(run, id=requested["run_id"], run_attempt=requested["run_attempt"])
    for change in ({"candidate": requested["baseline"]}, {"baseline": requested["candidate"]}, {"run_attempt": 9}):
        invalid = copy.deepcopy(receipt)
        invalid["requested"].update(change)
        with pytest.raises(ValueError, match="current main CI request"):
            reuse.verify_receipt(
                root,
                reports,
                inputs,
                invalid,
                current,
                requested["baseline"],
                current_inputs=inputs.parent / "current-inputs",
            )


@pytest.mark.parametrize("tamper", [None, "receipt", "delivery"])
def test_privileged_publication_reauthenticates_terminal_source_and_every_delivered_file(bundle, monkeypatch, tamper):
    root, reports, inputs, run, requested = bundle
    identities = [
        {"id": 11, "name": "relflow-evidence", "digest": "sha256:" + "1" * 64},
        {"id": 12, "name": "evidence-inputs", "digest": "sha256:" + "2" * 64},
    ]
    requested_inputs = current_inputs(bundle)
    receipt = reuse.verify(
        root,
        reports,
        inputs,
        run,
        requested,
        current_inputs=requested_inputs,
        helper=HELPER,
        source={"repository": reuse.REPOSITORY, "artifacts": identities},
    )
    delivered_reports = reports.parent / "delivered"
    shutil.copytree(reports, delivered_reports)
    if tamper == "receipt":
        receipt["manifest_sha256"] = "0" * 64
    elif tamper == "delivery":
        (delivered_reports / "extra-unverified-file").write_text("not authenticated by original artifact")
    monkeypatch.setattr(reuse, "api", lambda path: run)

    def artifact(run_id, name, destination):
        assert run_id == 123
        shutil.copytree(reports if name == "relflow-evidence" else inputs, destination)
        return next(row for row in identities if row["name"] == name)

    monkeypatch.setattr(reuse, "artifact", artifact)
    current = dict(run, id=requested["run_id"], run_attempt=requested["run_attempt"])
    if tamper:
        with pytest.raises(ValueError, match="authenticated terminal source"):
            reuse.verify_receipt(
                root,
                delivered_reports,
                inputs,
                receipt,
                current,
                requested["baseline"],
                current_inputs=requested_inputs,
                executor=EXECUTOR,
            )
    else:
        assert (
            reuse.verify_receipt(
                root,
                delivered_reports,
                inputs,
                receipt,
                current,
                requested["baseline"],
                current_inputs=requested_inputs,
                executor=EXECUTOR,
            )
            == run
        )


@pytest.mark.parametrize("fault", [None, "runtime", "headers", "mode", "record", "extra", "body", "outside"])
def test_wheel_payload_keeps_runtime_and_metadata_headers_while_allowing_git_bound_description(tmp_path, fault):
    old = README.encode()
    new = old.replace(b"Old measurements", b"New measurements")
    source, current = tmp_path / "source.whl", tmp_path / "current.whl"
    wheel(source, old)
    kwargs = {}
    body = new
    if fault == "runtime":
        kwargs["code"] = b"changed runtime"
    elif fault == "headers":
        kwargs["headers"] = b"Name: relflow\nVersion: 0.4.2\nRequires-Dist: changed"
    elif fault == "mode":
        kwargs["mode"] = 0o100755 << 16
    elif fault == "record":
        kwargs["corrupt"] = True
    elif fault == "extra":
        kwargs["extra"] = True
    elif fault == "body":
        body += b"untracked README bytes"
    elif fault == "outside":
        body = new.replace(b"Intro", b"Other introduction")
    wheel(current, body, **kwargs)
    if fault not in {None, "outside"}:
        with pytest.raises(ValueError):
            reuse.wheel_equivalence(source, current, old, new)
    else:
        proof = reuse.wheel_equivalence(source, current, old, body)
        assert proof["source_raw_sha256"] != proof["requested_raw_sha256"]
        assert len(proof["payload_sha256"]) == 64


def test_publisher_retains_original_pr_measurements_for_authenticated_multicommit_main_request(
    bundle, monkeypatch, tmp_path
):
    root, reports, inputs, execution_run, requested = bundle
    tested = requested["candidate"]
    # The actual prepush base is C, while main's immediate parent is a later commit.
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "intermediate commit",
    )
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "main merge",
    )
    main = git(root, "rev-parse", "HEAD")
    assert git(root, "rev-parse", "HEAD^1") != tested
    requested.update(candidate=main, baseline=tested, suite=main)
    requested_inputs = current_inputs(bundle)
    identities = [
        {"id": 11, "name": "relflow-evidence", "digest": "sha256:" + "1" * 64},
        {"id": 12, "name": "evidence-inputs", "digest": "sha256:" + "2" * 64},
    ]
    receipt = reuse.verify(
        root,
        reports,
        inputs,
        execution_run,
        requested,
        current_inputs=requested_inputs,
        helper=HELPER,
        source={"repository": reuse.REPOSITORY, "artifacts": identities},
    )
    current = dict(execution_run, id=requested["run_id"], run_attempt=requested["run_attempt"], head_sha=main)
    retained = []
    original = reuse.delivered(reports)

    def api(path, **kwargs):
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": main}}
        if "/actions/runs/" in path:
            return current
        return {"id": 42}

    def original_artifact(run_id, name, destination):
        assert run_id == execution_run["id"]
        shutil.copytree(reports if name == "relflow-evidence" else inputs, destination)
        return next(row for row in identities if row["name"] == name)

    def current_artifact(run_id, name, destination):
        assert run_id == current["id"]
        if name == "evidence-reuse":
            destination.mkdir()
            (destination / "reuse.json").write_text(json.dumps(receipt))
        else:
            source = {
                "relflow-evidence": reports,
                "evidence-inputs": requested_inputs,
                "evidence-reused-inputs": inputs,
            }[name]
            shutil.copytree(source, destination)

    real_run = subprocess.run

    def execute(command, *args, **kwargs):
        if command[0] == "make":
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(publication, "api", api)
    monkeypatch.setattr(publication, "verified_artifact", current_artifact)
    monkeypatch.setattr(publication, "EXECUTOR", EXECUTOR)
    monkeypatch.setattr(publication, "retain", lambda *args: retained.append(args))
    monkeypatch.setattr(publication.subprocess, "run", execute)
    monkeypatch.setattr(reuse, "api", lambda path: execution_run)
    monkeypatch.setattr(reuse, "artifact", original_artifact)
    output = tmp_path / "publication"
    publication.publish(root, output, current["id"])
    assert retained[0][1] == main
    assert "evidence-reuse.json" in retained[0][3]
    assert json.loads((output / "github-run.json").read_text())["run_id"] == execution_run["id"]
    assert json.loads((output / "evidence-reuse.json").read_text())["role_mapping"] == {
        "baseline": "candidate",
        "candidate": "candidate",
    }
    selected = json.loads((output / "publication.json").read_text())
    assert selected["revision"] == main
    assert all(result["provenance"]["relflow_revision"] == tested for result in selected["results"].values())
    assert not git(root, "diff", "--name-only")
    assert reuse.delivered(reports) == original


@pytest.mark.parametrize("change", ["docs", "tests", "publication", "readme"])
def test_presentation_changes_reuse_verified_runtime_without_training(bundle, change):
    root, _, _, _, requested = bundle
    path = {
        "docs": "docs/example.qmd",
        "tests": "tests/test_example.py",
        "publication": "evidence/results.json",
        "readme": "README.md",
    }[change]
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        README.replace("Intro", "New badge and documentation") if change == "readme" else "presentation only\n"
    )
    requested["candidate"] = requested["suite"] = commit(root, change)
    assert verify(bundle)["passed"] is True


def test_declared_input_profile_rejects_changes_to_any_exposed_script_bytes(bundle):
    root, reports, _, _, requested = bundle
    runtime = dict(IDENTITY, resources={"input_layout": "declared-files-v1"})
    for name in ("execution.json", "summary.json", "checksums.json"):
        (reports / name).unlink()
    write_json(reports / "execution.json", {"identity": runtime})
    finish(reports, root=root / "evidence")
    requested["execution_identity"] = runtime
    assert verify(bundle)["passed"] is True
    legacy_selection(root, reports)
    requested["candidate"] = requested["suite"] = commit(root, "inline selection changes exposed script bytes")
    with pytest.raises(ValueError, match="runtime input bytes"):
        verify(bundle)


@pytest.mark.parametrize("association", [[], [{"head": {"sha": "d" * 40}, "base": {"sha": "e" * 40}}]])
def test_pr_provenance_uses_authenticated_request_when_github_associations_change(bundle, association):
    root, reports, inputs, run, _ = bundle
    manifest = json.loads((reports / "manifest.json").read_text())
    head, base = manifest["candidate"]["revision"], manifest["baseline"]["revision"]
    merge = git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit-tree",
        git(root, "rev-parse", head + "^{tree}"),
        "-p",
        base,
        "-p",
        head,
        input=b"synthetic PR merge\n",
    )
    manifest["candidate"]["revision"] = merge
    run.update(event="pull_request", pull_requests=association)
    request = dict(json.loads((inputs / "request.json").read_text()), candidate=merge, event="pull_request")
    reuse.trusted(run, manifest, helper=HELPER, executor=EXECUTOR, root=root, request=request)
    with pytest.raises(ValueError, match="Original input request"):
        reuse.trusted(run, manifest, helper=HELPER, executor=EXECUTOR, root=root, request=dict(request, baseline=head))


@pytest.mark.parametrize("fault", [None, "running", "changed-head", "failed-evidence", "missing-evidence", "future"])
def test_partial_ci_retry_preserves_original_successful_evidence_attempt(bundle, monkeypatch, fault):
    _, _, _, original, _ = bundle
    original = copy.deepcopy(original)
    original["conclusion"] = "failure"
    original["referenced_workflows"].append(
        {"path": "relflow/relflow/.github/workflows/tests.yml@" + "a" * 40, "sha": "a" * 40}
    )
    latest = dict(
        original,
        run_attempt=3,
        conclusion="success",
        referenced_workflows=list(reversed(original["referenced_workflows"])),
    )
    job = {
        "id": 42,
        "name": "Evidence / Run full paired suite on lab / Run full paired suite on lab",
        "status": "completed",
        "conclusion": "success",
    }
    if fault == "running":
        latest["status"] = "in_progress"
    elif fault == "changed-head":
        latest["head_sha"] = "d" * 40
    elif fault == "failed-evidence":
        job["conclusion"] = "failure"
    monkeypatch.setattr(
        reuse,
        "api",
        lambda path: (
            {"jobs": [] if fault == "missing-evidence" else [job]} if path.endswith("jobs?per_page=100") else original
        ),
    )
    if fault is not None:
        with pytest.raises(ValueError):
            reuse.attempt(latest, 4 if fault == "future" else 2)
    else:
        verified = reuse.attempt(latest, 2)
        assert verified["run_attempt"] == 2 and verified["conclusion"] == "failure"
        assert verified["validation"] == {
            "run_attempt": 3,
            "head_sha": original["head_sha"],
            "conclusion": "success",
            "evidence_job_id": 42,
        }
        assert "validation" not in original


@pytest.mark.parametrize("fault", [None, "failed", "skipped", "shard-only", "duplicate", "mixed-helper", "mutable"])
def test_hosted_publication_authenticates_the_complete_collector(bundle, monkeypatch, fault):
    root, reports, inputs, original, _ = bundle
    run = copy.deepcopy(original)
    run["referenced_workflows"] = [
        {"path": f"{reuse.REPOSITORY}/.github/workflows/evidence-hosted.yml@{HELPER}", "sha": HELPER}
    ]
    manifest = json.loads((reports / "manifest.json").read_text()) | {"backend": "github-hosted-docker"}
    request = json.loads((inputs / "request.json").read_text())
    job = {
        "id": 42,
        "name": "Evidence / Run full paired suite on GitHub / Collect full paired suite",
        "status": "completed",
        "conclusion": "success",
    }
    if fault in {"failed", "skipped"}:
        job["conclusion"] = "failure" if fault == "failed" else "skipped"
    elif fault == "shard-only":
        job["name"] = "Evidence / Run paired shard (0)"
    elif fault == "mixed-helper":
        run["referenced_workflows"].extend(original["referenced_workflows"])
    elif fault == "mutable":
        run["referenced_workflows"][0]["path"] = f"{reuse.REPOSITORY}/.github/workflows/evidence-hosted.yml@main"
    monkeypatch.setattr(reuse, "api", lambda path: {"jobs": [job, job] if fault == "duplicate" else [job]})
    if fault:
        with pytest.raises(ValueError):
            reuse.trusted(run, manifest, helper=HELPER, executor=EXECUTOR, root=root, request=request)
    else:
        reuse.trusted(run, manifest, helper=HELPER, executor=EXECUTOR, root=root, request=request)
        assert reuse.artifact_name(run, "relflow-evidence") == "relflow-evidence-attempt-2"
        assert reuse.artifact_name(run, "evidence-inputs") == "evidence-inputs-attempt-2"
        assert reuse.artifact_name(original, "relflow-evidence") == "relflow-evidence"


def test_collector_authentication_reads_all_job_pages(bundle, monkeypatch):
    _, _, _, original, _ = bundle
    run = copy.deepcopy(original)
    run["referenced_workflows"][0]["path"] = f"{reuse.REPOSITORY}/.github/workflows/evidence-hosted.yml@{HELPER}"
    collector = {
        "id": 42,
        "name": "Evidence / Collect full paired suite",
        "status": "completed",
        "conclusion": "success",
    }
    paths = []

    def api(path):
        paths.append(path)
        return {"jobs": [collector]} if path.endswith("&page=2") else {"jobs": [], "total_count": 101}

    monkeypatch.setattr(reuse, "api", api)
    assert reuse.successful_job(run, 2) == collector
    assert len(paths) == 2


def test_versioned_release_assets_are_considered_for_report_reuse(tmp_path, monkeypatch):
    seen = []
    release = {"id": 77, "tag_name": "evidence-v0.4.3", "draft": False}
    monkeypatch.setattr(
        reuse,
        "api",
        lambda path: (
            {"id": 42} if path.endswith("ci.yml") else {"workflow_runs": []} if "/runs?" in path else [release]
        ),
    )

    def retained(candidate, destination):
        seen.append(candidate)
        raise ValueError("invalid test archive remains a miss")

    monkeypatch.setattr(reuse, "release_bundle", retained)
    assert not reuse.find(
        tmp_path, tmp_path / "out", {"execution_identity": IDENTITY}, current_inputs=tmp_path, helper=HELPER
    )
    assert seen == [release]
