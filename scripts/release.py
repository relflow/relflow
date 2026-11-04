"""Verify one CI build and reconcile immutable GitHub and PyPI distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ARTIFACT = "python-package-distributions"
FIELDS = {"schema", "source_sha", "version", "files", "ci_run_id", "ci_run_attempt"}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def manifest(value: dict) -> dict:
    """Accept only the two distributions supported by the shared universal build."""
    if not isinstance(value, dict) or set(value) != FIELDS or type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("Distribution build.json must use the exact schema 1 fields")
    if not isinstance(value["source_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["source_sha"]):
        raise ValueError("Distribution source_sha must be a full lowercase commit SHA")
    version = value["version"]
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?", version
    ):
        raise ValueError("Distribution version must be a normalized release or prerelease version")
    expected = {f"relflow-{version}-py3-none-any.whl", f"relflow-{version}.tar.gz"}
    files = value["files"]
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("Distribution inventory must contain exactly the RelFlow universal wheel and sdist")
    if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in files.values()):
        raise ValueError("Distribution inventory requires lowercase SHA256 hashes")
    if any(type(value[key]) is not int or value[key] < 1 for key in ("ci_run_id", "ci_run_attempt")):
        raise ValueError("Distribution receipt requires a positive CI run ID and attempt")
    return value


def metadata(path: Path, version: str) -> None:
    """Check archive metadata without importing or extracting package code."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            prefix = f"relflow-{version}.dist-info/"
            if names.count(prefix + "METADATA") != 1 or names.count(prefix + "WHEEL") != 1:
                raise ValueError(f"{path.name}: expected one matching wheel metadata directory")
            content = archive.read(prefix + "METADATA")
            wheel = BytesParser().parsebytes(archive.read(prefix + "WHEEL"))
            if wheel.get_all("Tag") != ["py3-none-any"] or wheel.get("Root-Is-Purelib", "").lower() != "true":
                raise ValueError(f"{path.name}: shared wheel must be pure Python and py3-none-any")
    else:
        with tarfile.open(path, "r:gz") as archive:
            name = f"relflow-{version}/PKG-INFO"
            members = [member for member in archive.getmembers() if member.name == name]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(f"{path.name}: expected one matching sdist PKG-INFO")
            stream = archive.extractfile(members[0])
            assert stream is not None
            content = stream.read()
    fields = BytesParser().parsebytes(content)
    if fields.get_all("Name") != ["relflow"] or fields.get_all("Version") != [version]:
        raise ValueError(f"{path.name}: archive Name/Version must be relflow/{version}")


def verify(directory: Path, *, source_sha: str | None = None, version: str | None = None) -> dict:
    receipt = manifest(json.loads((directory / "build.json").read_text()))
    if source_sha is not None and receipt["source_sha"] != source_sha:
        raise ValueError("Distribution source does not match the checked CI commit")
    if version is not None and receipt["version"] != version:
        raise ValueError("Distribution version does not match the checked project version")
    if {path.name for path in directory.iterdir()} != {*receipt["files"], "build.json"}:
        raise ValueError("Distribution artifact contains missing or unexpected files")
    for name, expected in receipt["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"{name}: distribution bytes do not match the CI receipt")
        metadata(path, receipt["version"])
    return receipt


def inventory(directory: Path, source_sha: str, run_id: int, attempt: int, *, root: Path = Path(".")) -> dict:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    receipt = manifest(
        {
            "schema": 1,
            "source_sha": source_sha,
            "version": version,
            "files": {path.name: digest(path.read_bytes()) for path in directory.iterdir() if path.is_file()},
            "ci_run_id": run_id,
            "ci_run_attempt": attempt,
        }
    )
    (directory / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return verify(directory, source_sha=source_sha, version=version)


def github(repository: str, path: str, *, missing: bool = False, paginate: bool = False):
    command = ["gh", "api", f"repos/{repository}/{path}"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        if missing and "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(f"GitHub read failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def tag_commit(repository: str, tag: str) -> str | None:
    reference = github(repository, f"git/ref/tags/{tag}", missing=True)
    if reference is None:
        return None
    value = reference["object"]
    for _ in range(10):
        if value["type"] == "commit":
            return value["sha"]
        if value["type"] != "tag":
            break
        value = github(repository, f"git/tags/{value['sha']}")["object"]
    raise ValueError(f"{tag}: release tag does not resolve to a commit")


def release_state(repository: str, tag: str) -> dict | None:
    pages = github(repository, "releases?per_page=100", paginate=True)
    matches = [release for page in pages for release in page if release["tag_name"] == tag]
    if len(matches) > 1:
        raise ValueError(f"{tag}: multiple releases use the same tag")
    return matches[0] if matches else None


def asset(repository: str, item: dict) -> bytes:
    return subprocess.check_output(
        ["gh", "api", f"repos/{repository}/releases/assets/{item['id']}", "-H", "Accept: application/octet-stream"]
    )


def pypi(version: str) -> dict[str, str]:
    try:
        with urlopen(f"https://pypi.org/pypi/relflow/{version}/json", timeout=30) as response:
            values = json.load(response)["urls"]
    except HTTPError as error:
        if error.code != 404:
            raise
        return {}
    files = {item["filename"]: item["digests"]["sha256"] for item in values}
    if len(files) != len(values):
        raise ValueError("PyPI returned duplicate distribution filenames")
    return files


def compare_files(expected: dict[str, str], present: dict[str, str], host: str) -> list[str]:
    if set(present) - set(expected):
        raise ValueError(f"{host}: published version has unexpected distribution files")
    for name, checksum in present.items():
        if checksum != expected[name]:
            raise ValueError(f"{host}: {name} differs from the original CI artifact; published bytes are immutable")
    return sorted(set(expected) - set(present))


def checked_attempt(repository: str, receipt: dict, run: dict, attempt: int) -> None:
    expected = {
        "id": receipt["ci_run_id"],
        "run_attempt": attempt,
        "status": "completed",
        "event": "push",
        "head_branch": "main",
        "head_sha": receipt["source_sha"],
        "path": ".github/workflows/ci.yml",
    }
    if (
        type(run.get("run_attempt")) is not int
        or any(run.get(key) != value for key, value in expected.items())
        or (run.get("head_repository") or {}).get("full_name") != repository
        or not run.get("conclusion")
    ):
        raise ValueError("Release requires completed main push CI attempts for the original distribution source")


def checked_job(repository: str, receipt: dict, attempt: int, name: str) -> None:
    pages = github(
        repository,
        f"actions/runs/{receipt['ci_run_id']}/attempts/{attempt}/jobs?per_page=100",
        paginate=True,
    )
    jobs = [job for page in pages for job in page["jobs"] if job["name"] == name]
    expected = {
        "run_id": receipt["ci_run_id"],
        "run_attempt": attempt,
        "head_sha": receipt["source_sha"],
        "status": "completed",
        "conclusion": "success",
    }
    if len(jobs) != 1 or any(jobs[0].get(key) != value for key, value in expected.items()):
        raise ValueError(f"Release requires one successful {name!r} job in CI attempt {attempt}")


def checked_run(repository: str, receipt: dict) -> None:
    """Preserve the build's producer while authenticating a later successful full CI gate."""
    path = f"actions/runs/{receipt['ci_run_id']}"
    original = github(repository, f"{path}/attempts/{receipt['ci_run_attempt']}")
    checked_attempt(repository, receipt, original, receipt["ci_run_attempt"])
    latest = github(repository, path)
    attempt = latest.get("run_attempt")
    if type(attempt) is not int or attempt < receipt["ci_run_attempt"]:
        raise ValueError("Release validation cannot precede the original distribution build")
    checked_attempt(repository, receipt, latest, attempt)
    if (
        latest["conclusion"] != "success"
        or latest["workflow_id"] != original["workflow_id"]
        or (attempt == receipt["ci_run_attempt"] and original["conclusion"] != "success")
    ):
        raise ValueError("Release requires the latest successful full CI attempt for the original distribution source")
    checked_job(repository, receipt, receipt["ci_run_attempt"], "Package / Build checked distributions")
    checked_job(repository, receipt, attempt, "CI / required")


def recover_draft(
    repository: str, target: str, version: str, present: dict[str, bytes]
) -> tuple[dict, dict[str, bytes]]:
    """Recover an interrupted draft only from an exact successful CI artifact at its full SHA."""
    if not isinstance(target, str) or not re.fullmatch(r"[0-9a-f]{40}", target):
        raise ValueError("An interrupted draft requires its exact original source SHA")
    runs = github(
        repository,
        f"actions/workflows/ci.yml/runs?branch=main&event=push&status=success&head_sha={target}&per_page=10",
    )["workflow_runs"]
    for run in runs:
        with tempfile.TemporaryDirectory() as temporary:
            try:
                subprocess.run(
                    [
                        "gh",
                        "run",
                        "download",
                        str(run["id"]),
                        "--repo",
                        repository,
                        "--name",
                        ARTIFACT,
                        "--dir",
                        temporary,
                    ],
                    check=True,
                )
                directory = Path(temporary)
                receipt = verify(directory, source_sha=target, version=version)
                if receipt["ci_run_id"] != run["id"] or receipt["ci_run_attempt"] > run["run_attempt"]:
                    raise ValueError("Draft recovery artifact identifies a different CI run or a later attempt")
                checked_run(repository, receipt)
                files = {path.name: path.read_bytes() for path in directory.iterdir()}
                compare_files(
                    {name: digest(data) for name, data in files.items()},
                    {name: digest(data) for name, data in present.items()},
                    "GitHub draft",
                )
                return receipt, files
            except (ValueError, OSError, subprocess.CalledProcessError):
                continue
    raise ValueError("Interrupted draft has no matching retained successful original CI artifact; repair it explicitly")


def prepare(directory: Path, output: Path, retained: Path, *, repository: str, source_sha: str, version: str) -> dict:
    """Preflight both hosts, retain original bytes, then upload only absent GitHub assets."""
    current = verify(directory, source_sha=source_sha, version=version)
    tag = f"v{version}"
    release = release_state(repository, tag)
    commit = tag_commit(repository, tag)
    present = {}
    recovered = {}
    receipt = current
    receipt_bytes = (directory / "build.json").read_bytes()
    if release:
        items = release["assets"]
        if len({item["name"] for item in items}) != len(items):
            raise ValueError("GitHub release contains duplicate asset names")
        present = {item["name"]: asset(repository, item) for item in items}
        if "build.json" in present:
            receipt_bytes = present["build.json"]
            receipt = manifest(json.loads(receipt_bytes))
            if receipt["version"] != version:
                raise ValueError("Existing release receipt has a different version")
        elif not release["draft"]:
            raise ValueError(
                "Existing release lacks its original CI build.json; repair it explicitly before automation"
            )
        elif release["target_commitish"] != source_sha or any(
            name not in current["files"] or digest(content) != current["files"][name]
            for name, content in present.items()
        ):
            receipt, recovered = recover_draft(repository, release["target_commitish"], version, present)
            receipt_bytes = recovered["build.json"]
        if commit is None and (not release["draft"] or release["target_commitish"] != receipt["source_sha"]):
            raise ValueError("Existing release has no tag at its original CI source")
    if commit is not None and commit != receipt["source_sha"]:
        raise ValueError(
            f"{tag}: existing tag points to {commit}, expected {receipt['source_sha']}; no files were written"
        )
    expected = {**receipt["files"], "build.json": digest(receipt_bytes)}
    missing_github = compare_files(expected, {name: digest(data) for name, data in present.items()}, "GitHub")
    missing_pypi = compare_files(receipt["files"], pypi(version), "PyPI")
    if output.exists() or retained.exists():
        raise ValueError("Release staging directories must not already exist")
    with tempfile.TemporaryDirectory() as temporary:
        original = Path(temporary) / "package"
        original.mkdir()
        (original / "build.json").write_bytes(receipt_bytes)
        for name in receipt["files"]:
            if name in present:
                (original / name).write_bytes(present[name])
            elif name in recovered:
                (original / name).write_bytes(recovered[name])
            elif current == receipt:
                shutil.copyfile(directory / name, original / name)
        if missing_github or missing_pypi or (release is not None and release["draft"]):
            checked_run(repository, receipt)
        if any(not (original / name).exists() for name in receipt["files"]):
            with tempfile.TemporaryDirectory() as download:
                subprocess.run(
                    [
                        "gh",
                        "run",
                        "download",
                        str(receipt["ci_run_id"]),
                        "--repo",
                        repository,
                        "--name",
                        ARTIFACT,
                        "--dir",
                        download,
                    ],
                    check=True,
                )
                restored = Path(download)
                if verify(restored) != receipt:
                    raise ValueError("Retained original CI artifact does not match the release receipt")
                for name in receipt["files"]:
                    shutil.copyfile(restored / name, original / name)
        verify(original, source_sha=receipt["source_sha"], version=version)
        if missing_github:
            if release is None:
                command = [
                    "gh",
                    "release",
                    "create",
                    tag,
                    "--repo",
                    repository,
                    "--target",
                    receipt["source_sha"],
                    "--title",
                    f"relflow {version}",
                    "--generate-notes",
                    "--draft",
                ]
                if re.search(r"(?:a|b|rc)[0-9]|\.dev[0-9]", version):
                    command.append("--prerelease")
                subprocess.run(command, check=True)
            subprocess.run(
                [
                    "gh",
                    "release",
                    "upload",
                    tag,
                    "--repo",
                    repository,
                    *[str(original / name) for name in missing_github],
                ],
                check=True,
            )
        if release is None or release["draft"]:
            subprocess.run(["gh", "release", "edit", tag, "--repo", repository, "--draft=false"], check=True)
        if tag_commit(repository, tag) != receipt["source_sha"]:
            raise ValueError("Published release tag differs from the original CI source")
        published = release_state(repository, tag)
        if published is None or published["draft"]:
            raise ValueError("GitHub release was not published")
        final = {item["name"]: digest(asset(repository, item)) for item in published["assets"]}
        if compare_files(expected, final, "GitHub"):
            raise ValueError("Published GitHub release is incomplete")
        shutil.copytree(original, retained)
        output.mkdir()
        for name in missing_pypi:
            shutil.copyfile(original / name, output / name)
    return {"should_publish": str(bool(missing_pypi)).lower(), "source_sha": receipt["source_sha"], "version": version}


def verify_pypi(directory: Path, *, attempts: int = 6) -> None:
    receipt = verify(directory)
    for attempt in range(attempts):
        missing = compare_files(receipt["files"], pypi(receipt["version"]), "PyPI")
        if not missing:
            return
        if attempt + 1 < attempts:
            time.sleep(10)
    raise ValueError(f"PyPI is missing distributions from the original CI build: {missing}")


def outputs(values: dict) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with Path(path).open("a") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")
    print(json.dumps(values, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "verify", "prepare", "verify-pypi"):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
        if name != "verify-pypi":
            command.add_argument("--source-sha", required=True)
        if name in ("verify", "prepare"):
            command.add_argument("--version", required=name == "prepare")
        if name == "inventory":
            command.add_argument("--run-id", type=int, default=os.environ.get("GITHUB_RUN_ID"), required=False)
            command.add_argument("--attempt", type=int, default=os.environ.get("GITHUB_RUN_ATTEMPT"), required=False)
        if name == "prepare":
            command.add_argument("--repository", required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--retained", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        receipt = inventory(args.directory, args.source_sha, args.run_id, args.attempt)
        wheel = next(name for name in receipt["files"] if name.endswith(".whl"))
        outputs(
            {
                "artifact_name": ARTIFACT,
                "source_sha": receipt["source_sha"],
                "version": receipt["version"],
                "wheel_name": wheel,
                "wheel_sha256": receipt["files"][wheel],
            }
        )
    elif args.command == "verify":
        outputs(verify(args.directory, source_sha=args.source_sha, version=args.version))
    elif args.command == "prepare":
        outputs(
            prepare(
                args.directory,
                args.output,
                args.retained,
                repository=args.repository,
                source_sha=args.source_sha,
                version=args.version,
            )
        )
    else:
        verify_pypi(args.directory)


if __name__ == "__main__":
    main()
