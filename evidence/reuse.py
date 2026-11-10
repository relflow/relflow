"""Verify equivalent full evidence without relabelling its original execution."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from evidence_ci import digest, identity, resolved, task_key
from experiments.source import describe, metadata, registry
from record import readme_content, validate

REPOSITORY = "relflow/relflow"
RESULTS = {"evidence-result", "evidence-report"}
MARKERS = ("<!-- evidence-results:start -->", "<!-- evidence-results:end -->")
SHA = re.compile(r"[0-9a-f]{40}")
LIMIT = 512 * 1024 * 1024
HELPERS = {
    # Authenticate archived reports; the retired workflow has no execution path.
    "evidence-lab.yml": "Run full paired suite on lab / Run full paired suite on lab",
    "evidence-hosted.yml": "/ Collect full paired suite",
}


def wheel_equivalence(source: Path, current: Path, source_readme: bytes, current_readme: bytes) -> dict:
    """Compare validated runtime payloads; the Git-bound README description is presentation only."""
    start = b"<!-- evidence-results:start -->"
    end = b"<!-- evidence-results:end -->"
    for content in (source_readme, current_readme):
        if start not in content and end not in content:
            continue
        if (
            content.count(start) != 1
            or content.count(end) != 1
            or not re.search(rb"(?m)^" + re.escape(start) + rb"\r?$", content)
            or not re.search(rb"(?m)^" + re.escape(end) + rb"\r?$", content)
            or content.index(start) >= content.index(end)
        ):
            raise ValueError("Wheel README normalization requires one ordered standalone result marker pair")
    identities = []
    raw_hashes = []
    for path, expected_readme in ((source, source_readme), (current, current_readme)):
        raw_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)) or sum(entry.file_size for entry in entries) > 1024**3:
                raise ValueError("Wheel contains duplicate members or exceeds the payload bound")
            payload = {}
            for entry in entries:
                name = PurePosixPath(entry.filename)
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (
                    name.is_absolute()
                    or not name.parts
                    or ".." in name.parts
                    or "\\" in entry.filename
                    or any(ord(char) < 32 or ord(char) == 127 for char in entry.filename)
                    or entry.flag_bits & 1
                    or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or mode not in {0, stat.S_IFREG, stat.S_IFDIR}
                    or (mode == stat.S_IFDIR and not entry.is_dir())
                    or (mode == stat.S_IFREG and entry.is_dir())
                    or name.as_posix().rstrip("/") != entry.filename.rstrip("/")
                ):
                    raise ValueError("Wheel contains unsafe, special or noncanonical members")
                payload[entry.filename] = archive.read(entry)
            metadata = [name for name in names if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)]
            records = [name for name in names if re.fullmatch(r"[^/]+\.dist-info/RECORD", name)]
            if len(metadata) != 1 or records != [metadata[0].removesuffix("METADATA") + "RECORD"]:
                raise ValueError("Wheel requires one unambiguous matching METADATA and RECORD")
            metadata_name, record_name = metadata[0], records[0]
            if any(entry.is_dir() for entry in entries if entry.filename in {metadata_name, record_name}):
                raise ValueError("Wheel metadata must be regular files")
            boundary = re.search(rb"\r?\n\r?\n", payload[metadata_name])
            if boundary is None:
                raise ValueError("Wheel METADATA is missing its description boundary")
            header = payload[metadata_name][: boundary.start()]
            separator = boundary.group()
            body = payload[metadata_name][boundary.end() :]
            if body != expected_readme:
                raise ValueError("Wheel METADATA description must equal the exact Git README")
            normalized_metadata = header + separator
            lines = payload[record_name].splitlines(keepends=True)
            recorded = set()
            normalized_rows = []
            for line in lines:
                try:
                    rows = list(csv.reader([line.decode("utf-8")], strict=True))
                except csv.Error as error:
                    raise ValueError("Wheel RECORD contains malformed CSV") from error
                if len(rows) != 1:
                    raise ValueError("Wheel RECORD contains ambiguous CSV rows")
                row = rows[0]
                if len(row) != 3 or row[0] in recorded or row[0] not in payload:
                    raise ValueError("Wheel RECORD contains missing, duplicate or unknown rows")
                name, checksum, size = row
                recorded.add(name)
                if name == record_name:
                    if checksum or size:
                        raise ValueError("Wheel RECORD self-entry must have empty hash and size")
                else:
                    expected = (
                        "sha256="
                        + base64.urlsafe_b64encode(hashlib.sha256(payload[name]).digest()).rstrip(b"=").decode()
                    )
                    if checksum != expected or size != str(len(payload[name])):
                        raise ValueError("Wheel RECORD hash or size differs from its raw member bytes")
                if name == metadata_name:
                    if not re.fullmatch(
                        re.escape(metadata_name.encode()) + rb",sha256=[A-Za-z0-9_-]+,[0-9]+(?:\r?\n)?", line
                    ):
                        raise ValueError("Wheel METADATA RECORD row must use canonical CSV spelling")
                    newline = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
                    checksum = base64.urlsafe_b64encode(hashlib.sha256(normalized_metadata).digest()).rstrip(b"=")
                    line = (
                        metadata_name.encode()
                        + b",sha256="
                        + checksum
                        + b","
                        + str(len(normalized_metadata)).encode()
                        + newline
                    )
                normalized_rows.append(line)
            expected_names = {entry.filename for entry in entries if not entry.is_dir()}
            if recorded != expected_names:
                raise ValueError("Wheel RECORD must cover every regular archive member exactly once")
            payload[metadata_name] = normalized_metadata
            payload[record_name] = b"".join(normalized_rows)
            identity = {
                entry.filename: {
                    "sha256": hashlib.sha256(payload[entry.filename]).hexdigest(),
                    "external_attr": entry.external_attr,
                    "create_system": entry.create_system,
                    "directory": entry.is_dir(),
                }
                for entry in entries
            }
            identities.append(
                hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            )
    if identities[0] != identities[1]:
        raise ValueError("Wheel payload, metadata headers, RECORD or member modes differ")
    return {"source_raw_sha256": raw_hashes[0], "requested_raw_sha256": raw_hashes[1], "payload_sha256": identities[0]}


def api(path: str) -> dict:
    return json.loads(subprocess.check_output(["gh", "api", path]))


def workflow(run: dict) -> tuple[str, str]:
    """Identify one immutable Evidence helper, including historical lab executions."""
    references = set()
    for row in run.get("referenced_workflows", []):
        for name in HELPERS:
            prefix = REPOSITORY + "/.github/workflows/" + name + "@"
            if not row.get("path", "").startswith(prefix):
                continue
            revision = row.get("sha", "")
            if not SHA.fullmatch(revision) or row["path"] != prefix + revision:
                raise ValueError("CI must identify one immutable trusted evidence helper")
            references.add((name, revision))
    if len(references) != 1:
        raise ValueError("CI must identify one immutable trusted evidence helper")
    return references.pop()


def successful_job(run: dict, number: int, *, review: bool = False) -> dict:
    """Authenticate the collector, rather than treating a successful shard as the gate."""
    name, _ = workflow(run)
    path = f"repos/{REPOSITORY}/actions/runs/{run['id']}/attempts/{number}/jobs?per_page=100"
    listing = api(path)
    jobs = list(listing["jobs"])
    for page in range(2, (listing.get("total_count", len(jobs)) + 99) // 100 + 1):
        jobs.extend(api(path + f"&page={page}")["jobs"])
    evidence = [job for job in jobs if job["name"].endswith(HELPERS[name])]
    conclusions = {"success", "failure"} if review else {"success"}
    if (
        len(evidence) != 1
        or evidence[0].get("conclusion") not in conclusions
        or evidence[0].get("status") != "completed"
    ):
        raise ValueError("The original attempt has no complete successful Evidence job")
    return evidence[0]


def artifact_name(run: dict, name: str) -> str:
    """Keep hosted attempts immutable; old lab artifacts retain their original names."""
    helper, _ = workflow(run)
    return f"{name}-attempt-{run['run_attempt']}" if helper == "evidence-hosted.yml" else name


def attempt(run: dict, number: int) -> dict:
    """Keep original Evidence provenance when a later retry completes the remaining CI jobs."""
    if type(number) is not int or number < 1 or number > run["run_attempt"]:
        raise ValueError("Evidence input attempt is outside the authenticated CI run")
    if number == run["run_attempt"]:
        return run
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError("Earlier Evidence artifacts require the complete CI run to finish successfully")
    original = api(f"repos/{REPOSITORY}/actions/runs/{run['id']}/attempts/{number}")
    fields = (
        "id",
        "head_sha",
        "event",
        "head_branch",
        "workflow_id",
        "path",
        "head_repository",
    )
    original_workflows = sorted(
        original.get("referenced_workflows", []), key=lambda row: (row.get("path", ""), row.get("sha", ""))
    )
    current_workflows = sorted(
        run.get("referenced_workflows", []), key=lambda row: (row.get("path", ""), row.get("sha", ""))
    )
    if (
        original.get("run_attempt") != number
        or original_workflows != current_workflows
        or any(original.get(name) != run.get(name) for name in fields)
    ):
        raise ValueError("Retried CI does not bind the original Evidence source and workflow")
    evidence = successful_job(original, number)
    return dict(
        original,
        validation={
            "run_attempt": run["run_attempt"],
            "head_sha": run["head_sha"],
            "conclusion": "success",
            "evidence_job_id": evidence["id"],
        },
    )


def helper_sha(run: dict) -> str:
    return workflow(run)[1]


def ensure(root: Path, revision: str) -> None:
    if not SHA.fullmatch(revision):
        raise ValueError("Original execution needs a full commit SHA")
    result = subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=root, capture_output=True)
    if result.returncode:
        subprocess.run(
            ["git", "fetch", "--no-tags", "--depth=1", f"https://github.com/{REPOSITORY}.git", revision],
            cwd=root,
            check=True,
            capture_output=True,
        )


def tree(root: Path, revision: str) -> dict[str, tuple[str, bytes]]:
    """Read every tracked path and mode, without export-ignore or working-tree filters."""
    if not SHA.fullmatch(revision):
        raise ValueError("Evidence identity requires a full commit SHA")
    rows = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", revision], cwd=root).split(b"\0")
    entries = []
    for row in filter(None, rows):
        identity, name = row.split(b"\t", 1)
        mode, kind, object_id = identity.decode().split()
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError("Evidence reuse rejects linked or special tracked inputs")
        entries.append((name.decode(), mode, object_id))
    process = subprocess.run(
        ["git", "cat-file", "--batch"],
        input="".join(object_id + "\n" for _, _, object_id in entries).encode(),
        stdout=subprocess.PIPE,
        cwd=root,
        check=True,
    )
    if len(process.stdout) > LIMIT:
        raise ValueError("Tracked evidence inputs exceed the reuse bound")
    stream = io.BytesIO(process.stdout)
    result = {}
    for name, mode, object_id in entries:
        header = stream.readline().decode().strip().split()
        if len(header) != 3 or header[:2] != [object_id, "blob"]:
            raise ValueError("Git returned an incomplete evidence input object")
        content = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("Git returned a truncated evidence input object")
        result[name] = mode, content
    return result


def stripped(source: str) -> bytes:
    """Remove only the two top-level selection values; retain every other byte."""
    metadata(source)
    lines = source.splitlines(keepends=True)
    boundaries = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == "# ---"][:2]
    if len(boundaries) != 2:
        raise ValueError("Evidence selection headers require canonical delimiters")
    start, end = boundaries
    selected = False
    retained = []
    found = set()
    for index, line in enumerate(lines):
        if start < index < end:
            key = re.match(r"^# ([A-Za-z][A-Za-z0-9_-]*):", line)
            if key:
                selected = key[1] in RESULTS
                if selected:
                    if key[1] in found:
                        raise ValueError("Evidence selection fields must not be duplicated")
                    found.add(key[1])
            if selected:
                continue
        retained.append(line)
    if found != metadata(source).keys() & RESULTS:
        raise ValueError("Evidence selection keys must use canonical top-level spelling")
    return "".join(retained).encode()


def archive(source: Path, destination: Path) -> None:
    """Produce reproducible report bytes without following links or special files."""
    with destination.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for path in sorted(source.rglob("*")):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError(f"Evidence artifact must contain regular files: {path}")
                if path.is_file():
                    content = path.read_bytes()
                    entry = tarfile.TarInfo(path.relative_to(source).as_posix())
                    entry.size = len(content)
                    entry.mode = 0o644
                    bundle.addfile(entry, io.BytesIO(content))


def selected(
    header: dict,
    manifest: dict,
    records: list[dict],
    manifest_sha256: str,
    reports_sha256: str,
    run: dict,
    release: str | None,
) -> None:
    identifier = header["evidence-id"]
    expected = next(
        record["result"]
        for record in records
        if record["task"]["id"] == identifier and record["task"]["role"] == "candidate"
    )
    try:
        same_result = json.dumps(header.get("evidence-result"), sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{identifier}: selected measurements must be finite JSON values") from error
    if not same_result:
        raise ValueError(f"{identifier}: normalized selection differs from the archived candidate record")
    report = header.get("evidence-report", {})
    wanted = {
        "run_id": manifest["run_id"],
        "candidate": manifest["candidate"],
        "baseline": manifest["baseline"],
        "evidence_revision": manifest["evidence_sha"],
        "executor_revision": manifest["executor_sha"],
        "manifest_sha256": manifest_sha256,
        "passed": True,
    }
    if json.dumps({key: report.get(key) for key in wanted}, sort_keys=True, allow_nan=False) != json.dumps(
        wanted, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"{identifier}: normalized selection changes original execution provenance")
    if set(report) - (wanted.keys() | {"github_run", "release", "reports_sha256"}):
        raise ValueError(f"{identifier}: normalized selection adds unknown report metadata")
    if "reports_sha256" in report and report["reports_sha256"] != reports_sha256:
        raise ValueError(f"{identifier}: selected bundle hash differs from the archived report")
    if "github_run" in report:
        expected_run = {
            key: run[key]
            for key in ("run_attempt", "head_sha", "head_branch", "event", "html_url", "workflow_id", "path")
        }
        expected_run["run_id"] = run["id"]
        if report["github_run"] != expected_run:
            raise ValueError(f"{identifier}: selected workflow metadata differs from the original attempt")
    allowed = {f"https://github.com/{REPOSITORY}/releases/tag/evidence-ci-{run['id']}-attempt-{run['run_attempt']}"}
    if release:
        allowed.add(release)
    if "release" in report and report["release"] not in allowed:
        raise ValueError(f"{identifier}: selected report must identify an immutable retained evidence release")


def equivalent(
    root: Path,
    original: str,
    requested: str,
    manifest: dict,
    records: list[dict],
    manifest_sha256: str,
    reports_sha256: str,
    run: dict,
    release: str | None,
    *,
    scope: str = "all",
) -> str:
    """Compare complete trees, allowing only measured publication selection changes."""
    before, after = tree(root, original), tree(root, requested)
    if scope == "package":
        before = {
            name: value
            for name, value in before.items()
            if name.startswith("src/") or name in {"pyproject.toml", "uv.lock"}
        }
        after = {
            name: value
            for name, value in after.items()
            if name.startswith("src/") or name in {"pyproject.toml", "uv.lock"}
        }
    elif scope == "suite":
        before = {
            name: value
            for name, value in before.items()
            if name.startswith("evidence/experiments/") or name == "README.md"
        }
        after = {
            name: value
            for name, value in after.items()
            if name.startswith("evidence/experiments/") or name == "README.md"
        }
    elif scope != "all":
        raise ValueError(f"Unknown execution input scope: {scope}")
    if before.keys() != after.keys():
        raise ValueError("Evidence input paths differ")
    headers = {}
    normalized = {}
    for name, (mode, content) in after.items():
        old_mode, old = before[name]
        if mode != old_mode:
            raise ValueError(f"Evidence input mode differs: {name}")
        if re.fullmatch(r"evidence/experiments/[^/]+/[^/]+\.py", name):
            left, right = old.decode(), content.decode()
            header = describe(right)
            headers[header["evidence-id"]] = {
                "header": header,
                "run": header.get("evidence-result"),
                "page": "docs/evidence/" + name.removeprefix("evidence/experiments/").replace("_", "-"),
            }
            if old != content:
                if stripped(left) != stripped(right):
                    raise ValueError(f"Evidence executable source or nonselection bytes differ: {name}")
                if {key: value for key, value in metadata(left).items() if key not in RESULTS} != {
                    key: value for key, value in header.items() if key not in RESULTS
                }:
                    raise ValueError(f"Evidence selection aliases change another header value: {name}")
                selected(header, manifest, records, manifest_sha256, reports_sha256, run, release)
            normalized[name] = mode, stripped(right)
        elif name == "README.md":
            normalized[name] = mode, content
        else:
            if old != content:
                raise ValueError(f"Evidence tracked input differs: {name}")
            normalized[name] = mode, content
    if "README.md" in after:
        old = before["README.md"][1].decode()
        new = after["README.md"][1].decode()
        if old != new:
            if any(old.count(marker) != 1 or new.count(marker) != 1 for marker in MARKERS):
                raise ValueError("README normalization requires one bounded marker pair")
            old_table = old.split(MARKERS[0])[1].split(MARKERS[1])[0]
            new_table = new.split(MARKERS[0])[1].split(MARKERS[1])[0]
            if old_table != new_table:
                if new != readme_content(headers, new):
                    raise ValueError("README metrics differ from the selected archived measurements")
                for identifier in ("P021", "P025", "P038"):
                    selected(
                        headers[identifier]["header"], manifest, records, manifest_sha256, reports_sha256, run, release
                    )
        # The description is verified against Git by wheel_equivalence; it never changes execution.
        normalized.pop("README.md", None)
    checksum = hashlib.sha256()
    for name, (mode, content) in sorted(normalized.items()):
        checksum.update(name.encode() + b"\0" + mode.encode() + b"\0" + hashlib.sha256(content).digest())
    return checksum.hexdigest()


def integrity(reports: Path) -> None:
    files = {}
    for path in reports.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Evidence report contains a linked or special file")
        if path.is_file() and path != reports / "checksums.json":
            files[path.relative_to(reports).as_posix()] = digest(path)
    if files != json.loads((reports / "checksums.json").read_text()):
        raise ValueError("Evidence report checksum inventory differs from the complete retained payload")
    manifest = json.loads((reports / "manifest.json").read_text())
    required = {"manifest.json", "summary.json", "report.html", "execution.json"}
    required.update(f"logs/{task_key(task)}.txt" for task in manifest["tasks"])
    if not required <= files.keys():
        raise ValueError("Evidence report is missing execution identity, summary, report or task logs")


def trusted(
    run: dict, manifest: dict, *, helper: str, executor: str, root: Path, request: dict, review: bool = False
) -> None:
    """Bind actual tested commits to an immutable successful internal workflow attempt."""
    if (
        run.get("status") != "completed"
        or (
            run.get("conclusion") != "success"
            and not (review and run.get("conclusion") == "failure")
            and not (
                run.get("validation", {}).get("conclusion") == "success"
                and run["validation"].get("head_sha") == run.get("head_sha")
                and type(run["validation"].get("run_attempt")) is int
                and run["validation"]["run_attempt"] > run["run_attempt"]
                and type(run["validation"].get("evidence_job_id")) is int
            )
        )
        or run.get("head_repository", {}).get("full_name") != REPOSITORY
        or run.get("path") != ".github/workflows/ci.yml"
        or run.get("event") not in {"pull_request", "push", "workflow_dispatch", "merge_group"}
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] < 1
        or manifest["run_id"] != f"gh-{run['id']}-{run['run_attempt']}"
        or manifest["executor_sha"] != executor
        or helper_sha(run) != helper
    ):
        raise ValueError("Reuse source is not a successful internal immutable evidence workflow attempt")
    if manifest.get("backend") == "github-hosted-docker":
        if workflow(run)[0] != "evidence-hosted.yml":
            raise ValueError("Hosted execution must use its immutable hosted Evidence helper")
        successful_job(run, run["run_attempt"], review=review)
    expected = {
        "candidate": manifest["candidate"]["revision"],
        "baseline": manifest["baseline"]["revision"],
        "suite": manifest["evidence_sha"],
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "event": run["event"],
    }
    if request != expected:
        raise ValueError("Original input request differs from its actual execution and trusted CI attempt")
    candidate = request["candidate"]
    if run["event"] == "pull_request":
        if candidate == run["head_sha"]:
            subprocess.run(["git", "merge-base", "--is-ancestor", request["baseline"], candidate], cwd=root, check=True)
        else:
            commit = subprocess.check_output(["git", "cat-file", "-p", candidate], cwd=root, text=True)
            parents = [
                line.removeprefix("parent ")
                for line in commit.partition("\n\n")[0].splitlines()
                if line.startswith("parent ")
            ]
            expected_parents = [request["baseline"], run["head_sha"]]
            if parents != expected_parents:
                raise ValueError(
                    f"PR source merge parents differ from its actual comparison base and head: expected {expected_parents}; found {parents}"
                )
    elif candidate != run.get("head_sha"):
        raise ValueError("Reuse source candidate differs from the trusted workflow commit")


def comparison(suite: Path, manifest: dict, measured: dict, role_mapping: dict) -> list[dict]:
    """Reapply the requested comparison without changing measured roles or checks."""
    requested_unmet = []
    introduced = set(manifest.get("introduced", []))
    if introduced and role_mapping != {"baseline": "baseline", "candidate": "candidate"}:
        raise ValueError("Introduced proof reuse cannot exchange baseline and candidate roles")
    for identifier in registry(suite):
        if identifier in introduced:
            if not all(measured["candidate", identifier]["checks"].values()):
                raise ValueError("Introduced proof reuse cannot promote unmet candidate checks")
            continue
        left = measured[role_mapping["baseline"], identifier]["checks"]
        right = measured[role_mapping["candidate"], identifier]["checks"]
        if left.keys() != right.keys() or any(passed and not right[name] for name, passed in left.items()):
            raise ValueError("Mapped requested comparison changes checks or regresses a baseline-passing check")
        requested_unmet.extend({"id": identifier, "check": name} for name, passed in left.items() if not passed)
    return requested_unmet


def verify(
    root: Path,
    reports: Path,
    inputs: Path,
    run: dict,
    requested: dict,
    *,
    current_inputs: Path,
    helper: str,
    source: dict,
) -> dict:
    """Always validate the original full bundle, then prove all requested inputs equivalent."""
    integrity(reports)
    manifest = json.loads((reports / "manifest.json").read_text())
    current_request = json.loads((current_inputs / "request.json").read_text())
    if current_request != {
        name: requested[name] for name in ("candidate", "baseline", "suite", "run_id", "run_attempt", "event")
    }:
        raise ValueError("Requested input artifact differs from this CI request")
    if digest(current_inputs / "requirements.txt") != manifest["requirements_sha256"]:
        raise ValueError("Requested runtime requirements differ from the original locked environment")
    for revision in (manifest["evidence_sha"], manifest["candidate"]["revision"], manifest["baseline"]["revision"]):
        ensure(root, revision)
    trusted(
        run,
        manifest,
        helper=helper,
        executor=requested["executor"],
        root=root,
        request=json.loads((inputs / "request.json").read_text()),
    )
    with tempfile.TemporaryDirectory() as temporary:
        suite = Path(temporary) / "evidence"
        for name, (mode, content) in tree(root, manifest["evidence_sha"]).items():
            if name.startswith("evidence/"):
                target = Path(temporary) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(0o755 if mode == "100755" else 0o644)
        measured, records, summary = validate(reports, inputs, root=suite)
        if manifest != measured or summary != json.loads((reports / "summary.json").read_text()):
            raise ValueError("Archived summary differs from its complete measured records")
        request = json.loads((inputs / "request.json").read_text())
        expected_request = {
            "candidate": manifest["candidate"]["revision"],
            "baseline": manifest["baseline"]["revision"],
            "suite": manifest["evidence_sha"],
            "run_id": run["id"],
            "run_attempt": run["run_attempt"],
            "event": run["event"],
        }
        if request != expected_request:
            raise ValueError("Original input request differs from its actual execution and trusted CI attempt")
        if (
            manifest.get("mode") != "full"
            or manifest.get("full_catalog") is not True
            or any(task.get("steps") is not None for task in manifest["tasks"])
        ):
            raise ValueError("Reuse requires declared full budgets and complete catalog coverage")
        execution = json.loads((reports / "execution.json").read_text())
        if not requested.get("execution_identity") or execution.get("identity") != requested["execution_identity"]:
            raise ValueError("Requested image/environment policy is missing or differs from the original execution")
        if execution["identity"].get("resources", {}).get("input_layout") == "declared-files-v1":
            current_runtime = {
                name.removeprefix("evidence/"): {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "mode": int(mode, 8) & 0o777,
                }
                for name, (mode, content) in tree(root, requested["suite"]).items()
                if name.startswith("evidence/experiments/") and "__pycache__" not in PurePosixPath(name).parts
            }
            if identity(current_runtime) != manifest["runtime_sha256"]:
                raise ValueError("Requested runtime input bytes differ from the original execution")
        bundle = Path(temporary) / "original-reports.tar.gz"
        archive(reports, bundle)
        reports_sha256 = digest(bundle)
        identities, role_mapping, wheels = {}, {}, {}
        for role in ("candidate", "baseline"):
            for source_role in ("candidate",) if role == "candidate" else ("baseline", "candidate"):
                try:
                    identities[role] = equivalent(
                        root,
                        manifest[source_role]["revision"],
                        requested[role],
                        manifest,
                        records,
                        digest(reports / "manifest.json"),
                        reports_sha256,
                        run,
                        source.get("release"),
                        scope="package",
                    )
                    original_wheels = list((inputs / source_role).glob("*.whl"))
                    requested_wheels = list((current_inputs / role).glob("*.whl"))
                    if len(original_wheels) != 1 or len(requested_wheels) != 1:
                        raise ValueError("Execution equivalence requires exactly one wheel for each role")
                    wheels[role] = wheel_equivalence(
                        original_wheels[0],
                        requested_wheels[0],
                        tree(root, manifest[source_role]["revision"])["README.md"][1],
                        tree(root, requested[role])["README.md"][1],
                    )
                    role_mapping[role] = source_role
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"No complete measured source role matches requested {role} inputs")
        identities["suite"] = equivalent(
            root,
            manifest["evidence_sha"],
            requested["suite"],
            manifest,
            records,
            digest(reports / "manifest.json"),
            reports_sha256,
            run,
            source.get("release"),
            scope="suite",
        )
        measured = (
            resolved(manifest, records, reports, root=suite)
            if any("reuse" in record for record in records)
            else records
        )
        measured = {(record["task"]["role"], record["task"]["id"]): record["result"] for record in measured}
        requested_unmet = comparison(suite, manifest, measured, role_mapping)
    return {
        "schema": 1,
        "kind": "verified-full-report-reuse",
        "helper": helper,
        "requested": requested,
        "input_sha256": identities,
        "requested_inputs": {
            "files": delivered(current_inputs),
            "requirements_sha256": digest(current_inputs / "requirements.txt"),
            "request_sha256": digest(current_inputs / "request.json"),
            "wheels": wheels,
        },
        "role_mapping": role_mapping,
        "requested_comparison": {
            "complete": True,
            "passed": True,
            "experiments": len(registry(suite)),
            "baseline_unmet_checks": requested_unmet,
        },
        "source": {
            **source,
            "run_id": run["id"],
            "run_attempt": run["run_attempt"],
            "head_sha": run["head_sha"],
            "event": run["event"],
            "candidate": manifest["candidate"],
            "baseline": manifest["baseline"],
            "suite": manifest["evidence_sha"],
            "executor": manifest["executor_sha"],
            "github_run": {
                key: run[key]
                for key in ("id", "run_attempt", "head_sha", "head_branch", "event", "html_url", "workflow_id", "path")
            },
        },
        "manifest_sha256": digest(reports / "manifest.json"),
        "checksums_sha256": digest(reports / "checksums.json"),
        "complete": True,
        "passed": True,
    }


def unpack(payload: bytes, destination: Path, *, zipped: bool) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    archive = zipfile.ZipFile(io.BytesIO(payload)) if zipped else tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz")
    with archive:
        entries = archive.infolist() if zipped else archive.getmembers()
        total = 0
        names = set()
        for entry in entries:
            name = entry.filename if zipped else entry.name
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or name in names
                or path.as_posix() != name.rstrip("/")
            ):
                raise ValueError("Evidence archive contains unsafe or duplicate paths")
            names.add(name)
            directory = entry.is_dir() if zipped else entry.isdir()
            if zipped:
                mode = entry.external_attr >> 16
                special = stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}
                size = entry.file_size
            else:
                special = not (entry.isfile() or entry.isdir())
                size = entry.size
            total += size
            if special or total > LIMIT:
                raise ValueError("Evidence archive contains special files or exceeds its size bound")
            target = destination / name
            if directory:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                content = archive.read(entry) if zipped else archive.extractfile(entry).read()
                target.write_bytes(content)


def artifact(run_id: int, name: str, destination: Path) -> dict:
    path = f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100"
    listing = api(path)
    artifacts = list(listing["artifacts"])
    for page in range(2, (listing.get("total_count", len(artifacts)) + 99) // 100 + 1):
        artifacts.extend(api(path + f"&page={page}")["artifacts"])
    matches = [row for row in artifacts if row["name"] == name and not row["expired"]]
    if len(matches) != 1:
        raise ValueError("Reuse source needs one unexpired immutable artifact per name")
    row = matches[0]
    expected = row.get("digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected) or row.get("size_in_bytes", 0) > LIMIT:
        raise ValueError("Reuse requires an API-authenticated artifact digest")
    payload = subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{row['id']}/zip"])
    if "sha256:" + hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Downloaded artifact differs from its GitHub digest")
    unpack(payload, destination, zipped=True)
    return {key: row[key] for key in ("id", "name", "digest")}


def release_bundle(release: dict, staging: Path) -> tuple[dict, dict]:
    """Recover a terminal execution after transient Actions inputs have expired."""
    required = {"evidence-reports.tar.gz", "evidence-inputs.tar.gz", "github-run.json", "SHA256SUMS"}
    assets = {row["name"]: row for row in release["assets"]}
    if release.get("draft") or not required <= assets.keys():
        raise ValueError("Retained release lacks its complete immutable report/input inventory")
    payloads = {}
    identities = []
    for name in sorted(required):
        asset = assets[name]
        expected = asset.get("digest", "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected) or asset.get("size", 0) > LIMIT:
            raise ValueError("Retained release needs API-authenticated asset digests")
        content = subprocess.check_output(
            ["gh", "api", "-H", "Accept: application/octet-stream", f"repos/{REPOSITORY}/releases/assets/{asset['id']}"]
        )
        if "sha256:" + hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Retained release asset differs from its GitHub digest")
        payloads[name] = content
        identities.append({key: asset[key] for key in ("id", "name", "digest")})
    sums = {}
    for line in payloads["SHA256SUMS"].decode().splitlines():
        checksum, name = line.split("  ", 1)
        if name in sums:
            raise ValueError("Retained release checksum inventory contains duplicate paths")
        sums[name] = checksum
    if not required - {"SHA256SUMS"} <= sums.keys() or any(
        hashlib.sha256(payloads[name]).hexdigest() != sums[name] for name in required - {"SHA256SUMS"}
    ):
        raise ValueError("Retained release checksums differ from report/input assets")
    unpack(payloads["evidence-reports.tar.gz"], staging / "reports", zipped=False)
    unpack(payloads["evidence-inputs.tar.gz"], staging / "inputs", zipped=False)
    manifest = json.loads((staging / "reports/manifest.json").read_text())
    original = re.fullmatch(r"gh-(\d+)-(\d+)", manifest["run_id"])
    if not original:
        raise ValueError("Retained report does not identify its terminal execution")
    run = attempt(api(f"repos/{REPOSITORY}/actions/runs/{original[1]}"), int(original[2]))
    recorded = json.loads(payloads["github-run.json"])
    if (
        recorded.get("run_id", recorded.get("id")) != run["id"]
        or recorded.get("run_attempt") != run["run_attempt"]
        or recorded.get("head_sha") != run["head_sha"]
    ):
        raise ValueError("Retained report workflow metadata differs from the original attempt")
    return run, {
        "repository": REPOSITORY,
        "release_id": release["id"],
        "release": release["html_url"],
        "assets": identities,
    }


def reason(error: Exception) -> str:
    """Bound public-input validation diagnostics without exposing subprocess context."""
    if isinstance(error, (ValueError, KeyError)):
        return type(error).__name__ + ": " + " ".join(str(error).split())[:240]
    return type(error).__name__


def find(root: Path, output: Path, requested: dict, *, current_inputs: Path, helper: str, force: bool = False) -> bool:
    if output.exists():
        raise ValueError("Reuse output must be a fresh directory")
    if force or not requested.get("execution_identity"):
        print("Full evidence reuse miss: forced fresh or no immutable execution policy")
        return False
    workflow = api(f"repos/{REPOSITORY}/actions/workflows/ci.yml")
    runs = api(f"repos/{REPOSITORY}/actions/workflows/ci.yml/runs?status=success&per_page=10")["workflow_runs"]
    releases = [
        row
        for row in api(f"repos/{REPOSITORY}/releases?per_page=20")
        if re.fullmatch(
            r"evidence-(?:ci-\d+-attempt-\d+|v[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?)",
            row["tag_name"],
        )
        and not row["draft"]
    ][:3]
    for kind, candidate in [("actions", run) for run in runs] + [("release", release) for release in releases]:
        run = candidate
        try:
            with tempfile.TemporaryDirectory() as temporary:
                staging = Path(temporary)
                if kind == "actions":
                    if run.get("workflow_id") != workflow["id"]:
                        continue
                    source = {
                        "repository": REPOSITORY,
                        "artifacts": [
                            artifact(run["id"], artifact_name(run, "relflow-evidence"), staging / "reports"),
                            artifact(run["id"], artifact_name(run, "evidence-inputs"), staging / "inputs"),
                        ],
                    }
                else:
                    run, source = release_bundle(candidate, staging)
                    if run.get("workflow_id") != workflow["id"]:
                        continue
                # A re-uploaded reuse bundle must bind to its original run, never create a chain.
                manifest = json.loads((staging / "reports/manifest.json").read_text())
                original = re.fullmatch(r"gh-(\d+)-(\d+)", manifest["run_id"])
                if not original:
                    raise ValueError("Original report run identity is invalid")
                if int(original[1]) != run["id"]:
                    continue
                run = attempt(run, int(original[2]))
                receipt = verify(
                    root,
                    staging / "reports",
                    staging / "inputs",
                    run,
                    requested,
                    current_inputs=current_inputs,
                    helper=helper,
                    source=source,
                )
                shutil.copytree(staging, output)
                (output / "reuse.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
                return True
        except (
            ValueError,
            KeyError,
            OSError,
            subprocess.CalledProcessError,
            tarfile.TarError,
            zipfile.BadZipFile,
        ) as error:
            print(f"Full evidence reuse miss for CI {run.get('id')}: {reason(error)}")
    return False


def delivered(directory: Path) -> dict[str, str]:
    values = {}
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Evidence handoff contains linked or special files")
        if path.is_file():
            values[path.relative_to(directory).as_posix()] = digest(path)
    return values


def verify_receipt(
    root: Path,
    reports: Path,
    inputs: Path,
    receipt: dict,
    current: dict,
    baseline: str,
    *,
    current_inputs: Path,
    executor: str | None = None,
) -> dict:
    """Privileged publication reauthenticates reuse instead of trusting a PR-produced claim."""
    requested = receipt.get("requested", {})
    if (
        receipt.get("schema") != 1
        or receipt.get("kind") != "verified-full-report-reuse"
        or requested.get("candidate") != current["head_sha"]
        or requested.get("suite") != current["head_sha"]
        or requested.get("baseline") != baseline
        or requested.get("run_id") != current["id"]
        or requested.get("run_attempt") != current["run_attempt"]
        or requested.get("event") != current["event"]
        or receipt.get("helper") != helper_sha(current)
        or (executor is not None and requested.get("executor") != executor)
    ):
        raise ValueError("Reuse receipt does not bind the current main CI request and actual baseline")
    source = receipt["source"]
    run = attempt(api(f"repos/{REPOSITORY}/actions/runs/{source['run_id']}"), source["run_attempt"])
    if run.get("workflow_id") != current["workflow_id"]:
        raise ValueError("Reuse source belongs to a different CI workflow")
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        if "release_id" in source:
            release = api(f"repos/{REPOSITORY}/releases/{source['release_id']}")
            original, origin = release_bundle(release, staging)
            if (original["id"], original["run_attempt"]) != (run["id"], run["run_attempt"]):
                raise ValueError("Retained release changed its source execution")
        else:
            origin = {
                "repository": REPOSITORY,
                "artifacts": [
                    artifact(run["id"], artifact_name(run, "relflow-evidence"), staging / "reports"),
                    artifact(run["id"], artifact_name(run, "evidence-inputs"), staging / "inputs"),
                ],
            }
        expected = verify(
            root,
            staging / "reports",
            staging / "inputs",
            run,
            requested,
            current_inputs=current_inputs,
            helper=receipt["helper"],
            source=origin,
        )
        if (
            expected != receipt
            or delivered(reports) != delivered(staging / "reports")
            or delivered(inputs) != delivered(staging / "inputs")
        ):
            raise ValueError("Reuse receipt or delivered payload differs from its authenticated terminal source")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "baseline", "suite", "executor"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--helper")
    parser.add_argument("--execution-identity", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not SHA.fullmatch(getattr(args, name)) for name in ("candidate", "baseline", "suite", "executor")) or (
        args.helper is not None and not SHA.fullmatch(args.helper)
    ):
        parser.error("All execution references must be full commit SHAs")
    requested = {name: getattr(args, name) for name in ("candidate", "baseline", "suite", "executor")}
    probe = json.loads(args.execution_identity.read_text()) if args.execution_identity else {}
    requested["execution_identity"] = probe.get("identity")
    requested["run_id"] = int(os.environ.get("GITHUB_RUN_ID", "0"))
    requested["run_attempt"] = int(os.environ.get("GITHUB_RUN_ATTEMPT", "0"))
    requested["event"] = os.environ.get("GITHUB_EVENT_NAME", "")
    root = Path(__file__).resolve().parents[1]
    try:
        if args.force_fresh or not requested.get("execution_identity"):
            hit = False
        else:
            helper = args.helper or helper_sha(api(f"repos/{REPOSITORY}/actions/runs/{requested['run_id']}"))
            hit = find(root, args.output.resolve(), requested, current_inputs=args.inputs.resolve(), helper=helper)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Full evidence reuse miss: {reason(error)}")
        hit = False
    values = {"hit": str(hit).lower()}
    if hit:
        receipt = json.loads((args.output / "reuse.json").read_text())
        values.update(
            reports=str(args.output / "reports"),
            inputs=str(args.output / "inputs"),
            receipt=str(args.output / "reuse.json"),
            source_run_id=str(receipt["source"]["run_id"]),
            source_run_attempt=str(receipt["source"]["run_attempt"]),
        )
        for role in ("candidate", "baseline"):
            values[role + "_wheel"] = str(next((args.output / "inputs" / role).glob("*.whl")))
    encoded = "".join(f"{key}={value}\n" for key, value in values.items())
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
