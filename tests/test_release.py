"""Release retries preserve the original checked distributions on both hosts."""

import copy
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import release

SOURCE = "a" * 40
LATER = "b" * 40
VERSION = "0.4.2"
WHEEL = f"relflow-{VERSION}-py3-none-any.whl"
SDIST = f"relflow-{VERSION}.tar.gz"
REPOSITORY = "relflow/relflow"


def test_distribution_artifacts_are_immutable_per_attempt():
    assert release.artifact_name(1) == "python-package-distributions-attempt-1"
    assert release.artifact_name(2) == "python-package-distributions-attempt-2"


@pytest.mark.parametrize("attempt", [0, -1, True, "2", None])
def test_distribution_artifact_name_requires_a_real_positive_attempt(attempt):
    with pytest.raises(ValueError, match="positive CI attempt"):
        release.artifact_name(attempt)


def distributions(root, name="package", source=SOURCE, content=b"original", run_id=71):
    directory = root / name
    directory.mkdir()
    project = root / f"project-{name}"
    project.mkdir()
    (project / "pyproject.toml").write_text(f'[project]\nversion = "{VERSION}"\n')
    metadata = f"Metadata-Version: 2.4\nName: relflow\nVersion: {VERSION}\n\n".encode()
    with zipfile.ZipFile(directory / WHEEL, "w") as archive:
        archive.writestr(f"relflow-{VERSION}.dist-info/METADATA", metadata)
        archive.writestr(f"relflow-{VERSION}.dist-info/WHEEL", b"Root-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("relflow/__init__.py", content)
    with tarfile.open(directory / SDIST, "w:gz") as archive:
        member = tarfile.TarInfo(f"relflow-{VERSION}/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))
    release.inventory(directory, source, run_id, 1, root=project)
    return directory


@pytest.fixture
def host(tmp_path, monkeypatch):
    directory = distributions(tmp_path)
    state = {
        "release": None,
        "tag": None,
        "files": {},
        "pypi": {},
        "writes": [],
        "checked": [],
        "original": directory,
    }

    def release_state(repository, tag):
        assert repository == REPOSITORY and tag == f"v{VERSION}"
        if state["release"] is None:
            return None
        return {**state["release"], "assets": [{"id": name, "name": name} for name in state["files"]]}

    def run(command, **kwargs):
        state["writes"].append(command)
        operation = command[1:3]
        if operation == ["release", "create"]:
            state["release"] = {"draft": True, "target_commitish": command[command.index("--target") + 1]}
        elif operation == ["release", "upload"]:
            for value in command[command.index("--repo") + 2 :]:
                path = Path(value)
                assert path.name not in state["files"], "Existing assets must never be overwritten"
                state["files"][path.name] = path.read_bytes()
        elif operation == ["release", "edit"]:
            state["release"]["draft"] = False
            state["tag"] = state["release"]["target_commitish"]
        elif operation == ["run", "download"]:
            destination = Path(command[command.index("--dir") + 1])
            for path in state["original"].iterdir():
                shutil.copyfile(path, destination / path.name)
        else:
            raise AssertionError(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(release, "release_state", release_state)
    monkeypatch.setattr(release, "tag_commit", lambda repository, tag: state["tag"])
    monkeypatch.setattr(release, "asset", lambda repository, item: state["files"][item["id"]])
    monkeypatch.setattr(release, "pypi", lambda version: state["pypi"])
    monkeypatch.setattr(release, "checked_run", lambda repository, receipt: state["checked"].append(receipt))
    monkeypatch.setattr(release.subprocess, "run", run)

    def publish(current=directory, source=SOURCE):
        return release.prepare(
            current,
            tmp_path / "dist",
            tmp_path / "retained",
            repository=REPOSITORY,
            source_sha=source,
            version=VERSION,
        )

    state["publish"] = publish
    return state


def published(host, *, draft=False):
    host["release"] = {"draft": draft, "target_commitish": SOURCE}
    host["tag"] = None if draft else SOURCE
    host["files"] = {path.name: path.read_bytes() for path in host["original"].iterdir()}
    host["pypi"] = release.verify(host["original"])["files"].copy()


def test_new_release_preflights_and_uploads_a_draft_then_stages_only_distributions(host, tmp_path):
    result = host["publish"]()
    assert result == {"should_publish": "true", "source_sha": SOURCE, "version": VERSION}
    assert [command[1:3] for command in host["writes"]] == [
        ["release", "create"],
        ["release", "upload"],
        ["release", "edit"],
    ]
    assert "--draft" in host["writes"][0]
    assert all("--clobber" not in command for command in host["writes"])
    assert host["tag"] == SOURCE and not host["release"]["draft"]
    assert {path.name for path in (tmp_path / "dist").iterdir()} == {WHEEL, SDIST}
    assert release.verify(tmp_path / "retained")["source_sha"] == SOURCE


@pytest.mark.parametrize("with_release", [False, True])
def test_conflicting_tag_is_rejected_before_any_provider_write(host, with_release):
    if with_release:
        published(host)
    host["tag"] = LATER
    with pytest.raises(ValueError, match="existing tag points"):
        host["publish"]()
    assert host["writes"] == []


@pytest.mark.parametrize("provider", ["files", "pypi"])
def test_differing_published_bytes_are_never_clobbered(host, provider):
    published(host)
    host[provider][WHEEL] = b"different bytes" if provider == "files" else "0" * 64
    with pytest.raises(ValueError, match="published bytes are immutable"):
        host["publish"]()
    assert host["writes"] == []


def test_existing_pypi_version_does_not_skip_a_missing_github_release(host):
    host["pypi"] = release.verify(host["original"])["files"].copy()
    result = host["publish"]()
    assert result["should_publish"] == "false"
    assert host["tag"] == SOURCE and set(host["files"]) == {WHEEL, SDIST, "build.json"}


@pytest.mark.parametrize("draft", [False, True])
def test_partial_github_and_pypi_states_resume_without_overwriting(host, tmp_path, draft):
    published(host, draft=draft)
    del host["files"][SDIST]
    del host["pypi"][WHEEL]
    before = host["files"][WHEEL]
    result = host["publish"]()
    upload = next(command for command in host["writes"] if command[1:3] == ["release", "upload"])
    assert [Path(value).name for value in upload[upload.index("--repo") + 2 :]] == [SDIST]
    assert host["files"][WHEEL] == before
    assert {path.name for path in (tmp_path / "dist").iterdir()} == {WHEEL}
    assert result["should_publish"] == "true" and host["checked"]


def test_interrupted_new_draft_without_receipt_or_tag_can_resume(host):
    host["release"] = {"draft": True, "target_commitish": SOURCE}
    host["files"][WHEEL] = (host["original"] / WHEEL).read_bytes()
    host["publish"]()
    assert host["tag"] == SOURCE
    assert not any(command[1:3] == ["release", "create"] for command in host["writes"])


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("attempt", [1, 3])
def test_interrupted_draft_at_older_main_recovers_only_its_original_ci_artifact(
    host, tmp_path, monkeypatch, partial, attempt
):
    host["release"] = {"draft": True, "target_commitish": SOURCE}
    if partial:
        host["files"][WHEEL] = (host["original"] / WHEEL).read_bytes()
    later = distributions(tmp_path, "later", LATER, b"later main", run_id=99)
    monkeypatch.setattr(
        release, "github", lambda *args, **kwargs: {"workflow_runs": [{"id": 71, "run_attempt": attempt}]}
    )
    result = host["publish"](later, LATER)
    assert result["source_sha"] == SOURCE and host["tag"] == SOURCE
    assert all(receipt["ci_run_id"] == 71 for receipt in host["checked"])
    assert all(receipt["ci_run_attempt"] == 1 for receipt in host["checked"])
    assert host["files"][WHEEL] == (host["original"] / WHEEL).read_bytes()
    assert not any(command[1:3] == ["release", "create"] for command in host["writes"])


def test_draft_recovery_rejects_nonmatching_partial_assets_before_release_writes(host, tmp_path, monkeypatch):
    host["release"] = {"draft": True, "target_commitish": SOURCE}
    host["files"][WHEEL] = b"conflicting published bytes"
    later = distributions(tmp_path, "later", LATER, b"later main", run_id=99)
    monkeypatch.setattr(release, "github", lambda *args, **kwargs: {"workflow_runs": [{"id": 71, "run_attempt": 1}]})
    with pytest.raises(ValueError, match="no matching retained"):
        host["publish"](later, LATER)
    assert not any(command[1] == "release" for command in host["writes"])


def test_complete_draft_still_requires_original_ci_before_publishing(host):
    published(host, draft=True)
    host["publish"]()
    assert host["checked"] and host["tag"] == SOURCE
    assert [command[1:3] for command in host["writes"]] == [["release", "edit"]]


def test_missing_legacy_build_receipt_fails_closed(host):
    published(host)
    del host["files"]["build.json"]
    with pytest.raises(ValueError, match="repair it explicitly"):
        host["publish"]()
    assert host["writes"] == []


def test_same_version_on_later_main_preserves_original_source_and_files(host, tmp_path):
    published(host)
    later = distributions(tmp_path, "later", LATER, b"new main", run_id=99)
    result = host["publish"](later, LATER)
    assert result["source_sha"] == SOURCE and result["should_publish"] == "false"
    assert host["writes"] == [] and host["checked"] == []
    assert (tmp_path / "retained" / WHEEL).read_bytes() == (host["original"] / WHEEL).read_bytes()
    assert list((tmp_path / "dist").iterdir()) == []


def test_later_main_repairs_only_from_original_ci_artifact(host, tmp_path):
    published(host)
    del host["files"][WHEEL]
    del host["pypi"][WHEEL]
    later = distributions(tmp_path, "later", LATER, b"new main", run_id=99)
    result = host["publish"](later, LATER)
    assert result["source_sha"] == SOURCE
    assert host["checked"][0]["ci_run_id"] == 71
    assert any(command[1:3] == ["run", "download"] for command in host["writes"])
    assert (tmp_path / "dist" / WHEEL).read_bytes() == (host["original"] / WHEEL).read_bytes()


def test_pypi_conflict_prevents_even_github_creation(host):
    host["pypi"] = {WHEEL: "0" * 64}
    with pytest.raises(ValueError, match="PyPI"):
        host["publish"]()
    assert host["writes"] == []


@pytest.mark.parametrize("unexpected", ["relflow-0.4.2-cp312-linux.whl", "../escape", "other.tar.gz"])
def test_receipt_rejects_unexpected_distribution_inventory(tmp_path, unexpected):
    directory = distributions(tmp_path)
    value = json.loads((directory / "build.json").read_text())
    value["files"][unexpected] = "0" * 64
    with pytest.raises(ValueError, match="exactly"):
        release.manifest(value)


@pytest.mark.parametrize("name", [WHEEL, SDIST])
def test_distribution_metadata_is_checked_even_when_hashes_match(tmp_path, name):
    directory = distributions(tmp_path)
    path = directory / name
    metadata = b"Name: json2vec\nVersion: 0.4.2\n"
    if name == WHEEL:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"relflow-{VERSION}.dist-info/METADATA", metadata)
            archive.writestr(f"relflow-{VERSION}.dist-info/WHEEL", b"Root-Is-Purelib: true\nTag: py3-none-any\n")
    else:
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo(f"relflow-{VERSION}/PKG-INFO")
            member.size = len(metadata)
            archive.addfile(member, io.BytesIO(metadata))
    receipt = json.loads((directory / "build.json").read_text())
    receipt["files"][name] = release.digest(path.read_bytes())
    (directory / "build.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="archive Name/Version"):
        release.verify(directory)


def test_artifact_source_hash_and_extra_file_are_verified(tmp_path):
    directory = distributions(tmp_path)
    with pytest.raises(ValueError, match="checked CI commit"):
        release.verify(directory, source_sha=LATER)
    (directory / WHEEL).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes do not match"):
        release.verify(directory)
    (directory / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="unexpected files"):
        release.verify(directory)


@pytest.fixture
def ci(monkeypatch):
    receipt = {"ci_run_id": 71, "ci_run_attempt": 1, "source_sha": SOURCE}
    run = {
        "id": 71,
        "workflow_id": 81,
        "status": "completed",
        "conclusion": "success",
        "event": "push",
        "head_branch": "main",
        "head_sha": SOURCE,
        "head_repository": {"full_name": REPOSITORY},
        "path": ".github/workflows/ci.yml",
        "run_attempt": 1,
    }
    job = {"run_id": 71, "run_attempt": 1, "head_sha": SOURCE, "status": "completed", "conclusion": "success"}
    state = {
        "receipt": receipt,
        "original": copy.deepcopy(run),
        "latest": copy.deepcopy(run),
        "producer": [{**job, "name": "Package / Build checked distributions"}],
        "gate": [{**job, "name": "CI / required"}],
        "reads": [],
    }

    def github(repository, path, **kwargs):
        assert repository == REPOSITORY
        state["reads"].append(path)
        if path == f"actions/runs/71/attempts/{state['receipt']['ci_run_attempt']}":
            return state["original"]
        if path == "actions/runs/71":
            return state["latest"]
        assert kwargs == {"paginate": True}
        if path == "actions/runs/71/attempts/1/jobs?per_page=100":
            jobs = state["producer"] + (state["gate"] if state["latest"]["run_attempt"] == 1 else [])
            return [{"jobs": jobs[:1]}, {"jobs": jobs[1:]}]
        assert path == f"actions/runs/71/attempts/{state['latest']['run_attempt']}/jobs?per_page=100"
        return [{"jobs": state["gate"]}]

    monkeypatch.setattr(release, "github", github)
    return state


def retry(ci):
    ci["original"]["conclusion"] = "failure"
    ci["latest"]["run_attempt"] = 3
    ci["gate"][0]["run_attempt"] = 3


@pytest.mark.parametrize("targeted", [False, True])
def test_release_authenticates_original_build_and_current_full_gate_without_rewriting_receipt(ci, targeted):
    if targeted:
        retry(ci)
    original = copy.deepcopy(ci["receipt"])
    release.checked_run(REPOSITORY, ci["receipt"])
    assert ci["receipt"] == original
    assert "actions/runs/71/attempts/1/jobs?per_page=100" in ci["reads"]
    assert f"actions/runs/71/attempts/{3 if targeted else 1}/jobs?per_page=100" in ci["reads"]


@pytest.mark.parametrize("attempt", ["original", "latest"])
@pytest.mark.parametrize(
    "change",
    [
        {"id": 72},
        {"status": "in_progress"},
        {"conclusion": None},
        {"event": "pull_request"},
        {"head_branch": "feature"},
        {"head_sha": LATER},
        {"head_repository": {"full_name": "fork/relflow"}},
        {"path": ".github/workflows/other.yml"},
        {"run_attempt": True},
    ],
)
def test_repair_rejects_untrusted_original_or_validation_attempts(ci, attempt, change):
    retry(ci)
    ci[attempt].update(copy.deepcopy(change))
    with pytest.raises(ValueError):
        release.checked_run(REPOSITORY, ci["receipt"])


@pytest.mark.parametrize(
    "change",
    [
        {"conclusion": "failure"},
        {"conclusion": "cancelled"},
        {"workflow_id": 82},
        {"run_attempt": 0},
    ],
)
def test_repair_rejects_failed_or_mismatched_latest_full_gate(ci, change):
    retry(ci)
    ci["latest"].update(change)
    with pytest.raises(ValueError):
        release.checked_run(REPOSITORY, ci["receipt"])


def test_release_validation_cannot_precede_original_build(ci):
    ci["receipt"]["ci_run_attempt"] = 2
    ci["original"]["run_attempt"] = 2
    with pytest.raises(ValueError, match="cannot precede"):
        release.checked_run(REPOSITORY, ci["receipt"])


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_successful_job_retry_accepts_failed_original_run_only_with_later_validation(ci, conclusion):
    retry(ci)
    ci["original"]["conclusion"] = conclusion
    release.checked_run(REPOSITORY, ci["receipt"])
    ci["latest"]["run_attempt"] = 1
    with pytest.raises(ValueError, match="latest successful full CI"):
        release.checked_run(REPOSITORY, ci["receipt"])


@pytest.mark.parametrize("role", ["producer", "gate"])
@pytest.mark.parametrize(
    "change",
    [
        {"conclusion": "failure"},
        {"conclusion": "skipped"},
        {"status": "in_progress"},
        {"run_id": 72},
        {"run_attempt": 2},
        {"head_sha": LATER},
        {"name": "unrelated successful job"},
    ],
)
def test_release_requires_the_exact_successful_original_build_and_current_gate_jobs(ci, role, change):
    retry(ci)
    ci[role][0].update(change)
    with pytest.raises(ValueError, match="requires one successful"):
        release.checked_run(REPOSITORY, ci["receipt"])


@pytest.mark.parametrize("role", ["producer", "gate"])
@pytest.mark.parametrize("count", [0, 2])
def test_release_rejects_missing_or_duplicate_required_jobs(ci, role, count):
    retry(ci)
    ci[role] *= count
    with pytest.raises(ValueError, match="requires one successful"):
        release.checked_run(REPOSITORY, ci["receipt"])


def test_annotated_tag_is_peeled_to_actual_commit(monkeypatch):
    responses = iter([{"object": {"type": "tag", "sha": LATER}}, {"object": {"type": "commit", "sha": SOURCE}}])
    monkeypatch.setattr(release, "github", lambda *args, **kwargs: next(responses))
    assert release.tag_commit(REPOSITORY, "v0.4.2") == SOURCE


def test_pypi_verification_waits_for_exact_complete_inventory(tmp_path, monkeypatch):
    directory = distributions(tmp_path)
    expected = release.verify(directory)["files"]
    responses = iter([{WHEEL: expected[WHEEL]}, expected])
    monkeypatch.setattr(release, "pypi", lambda version: next(responses))
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    release.verify_pypi(directory, attempts=2)
    assert sleeps == [10]
