"""Publish immutable evidence assets once per verified package version."""

from __future__ import annotations

import argparse
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
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

import yaml
from evidence_ci import task_key
from record import select, validate
from releases import VERSION, comparison, filenames

REPOSITORY = "relflow/relflow"
SHA = re.compile(r"[0-9a-f]{40}")
LIMIT = 512 * 1024 * 1024
HELPER = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/evidence.yml").read_text())["jobs"][
    "run"
]["uses"].rsplit("@", 1)[1]
EXECUTOR = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/evidence-hosted.yml").read_text())[
    "env"
]["EVIDENCE_EXECUTOR"]
if not isinstance(EXECUTOR, str) or not SHA.fullmatch(EXECUTOR):
    raise ValueError("The hosted Evidence workflow must declare one immutable EVIDENCE_EXECUTOR SHA")


def api(path: str, *, missing: bool = False, paginate: bool = False) -> dict | list:
    """Read GitHub JSON, distinguishing an absent object from a failed request."""
    command = ["gh", "api", path]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        if missing and "HTTP 404" in result.stderr:
            return {}
        raise RuntimeError(f"GitHub metadata request failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def check_run(run: dict, main: str, run_id: int) -> None:
    if (
        run.get("id") != run_id
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] < 1
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("event") not in {"push", "workflow_dispatch"}
        or run.get("head_branch") != "main"
        or run.get("head_sha") != main
        or run.get("head_repository", {}).get("full_name") != REPOSITORY
        or run.get("path") != ".github/workflows/ci.yml"
    ):
        raise ValueError("Publication requires successful internal CI for the exact current main commit")


def check_report(manifest: dict, summary: dict, run: dict, main: str, *, baseline: str | None = None) -> None:
    if (
        not summary["passed"]
        or manifest["run_id"] != f"gh-{run['id']}-{run['run_attempt']}"
        or manifest["candidate"]["revision"] != main
        or manifest["baseline"]["revision"] != (baseline or main)
        or manifest["evidence_sha"] != main
        or manifest["executor_sha"] != EXECUTOR
    ):
        raise ValueError("Report identity, complete passing result, or tested revisions differ from trusted main CI")


def helper_sha(run: dict) -> str:
    """Authenticate the exact reusable hosted workflow from GitHub's run metadata."""
    prefix = REPOSITORY + "/.github/workflows/evidence-hosted.yml@"
    references = []
    for row in run.get("referenced_workflows", []):
        if not row.get("path", "").startswith(prefix):
            continue
        revision = row.get("sha", "")
        if not SHA.fullmatch(revision) or row["path"] != prefix + revision:
            raise ValueError("CI must identify one immutable trusted Evidence helper")
        references.append(revision)
    if len(references) != 1:
        raise ValueError("CI must identify one immutable trusted Evidence helper")
    return references[0]


def trusted(run: dict, manifest: dict, request: dict, root: Path) -> None:
    """Bind publication to this main CI attempt and its successful hosted collector."""
    if (
        manifest.get("backend") != "github-hosted-docker"
        or helper_sha(run) != HELPER
        or manifest.get("executor_sha") != EXECUTOR
        or manifest.get("run_id") != f"gh-{run['id']}-{run['run_attempt']}"
    ):
        raise ValueError("Publication requires the current immutable hosted Evidence helper and executor")
    expected = {
        "candidate": manifest["candidate"]["revision"],
        "baseline": manifest["baseline"]["revision"],
        "suite": manifest["evidence_sha"],
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "event": run["event"],
    }
    if request != expected or request["candidate"] != run["head_sha"]:
        raise ValueError("Evidence input request differs from its exact main CI attempt")
    selection = manifest["comparison"]
    selected = comparison(
        root,
        request["candidate"],
        selection["event_base"],
        forced=run["event"] == "push" and selection.get("reason") == "rewritten-push",
    )
    if selection != selected or selection["baseline"] != request["baseline"]:
        raise ValueError("Evidence comparison differs from its supported baseline policy")
    path = f"repos/{REPOSITORY}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100"
    listing = api(path)
    jobs = list(listing["jobs"])
    for page in range(2, (listing.get("total_count", len(jobs)) + 99) // 100 + 1):
        jobs.extend(api(path + f"&page={page}")["jobs"])
    collectors = [job for job in jobs if job["name"].endswith("/ Collect full paired suite")]
    if (
        len(collectors) != 1
        or collectors[0].get("status") != "completed"
        or collectors[0].get("conclusion") != "success"
    ):
        raise ValueError("The current attempt has no complete successful Evidence collector")


def unpack(payload: bytes, destination: Path) -> None:
    """Extract one bounded GitHub artifact without links or path traversal."""
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        total, names = 0, set()
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if (
                not path.parts
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or entry.filename in names
                or path.as_posix() != entry.filename.rstrip("/")
            ):
                raise ValueError("Evidence archive contains unsafe or duplicate paths")
            names.add(entry.filename)
            if stat.S_IFMT(entry.external_attr >> 16) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError("Evidence archive contains special files")
            total += entry.file_size
            if total > LIMIT:
                raise ValueError("Evidence archive exceeds its size bound")
            target = destination / entry.filename
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(entry))


def artifact(run: dict, name: str, destination: Path) -> dict:
    """Download and authenticate one artifact from the exact CI attempt."""
    name = f"{name}-attempt-{run['run_attempt']}"
    path = f"repos/{REPOSITORY}/actions/runs/{run['id']}/artifacts?per_page=100"
    listing = api(path)
    artifacts = list(listing["artifacts"])
    for page in range(2, (listing.get("total_count", len(artifacts)) + 99) // 100 + 1):
        artifacts.extend(api(path + f"&page={page}")["artifacts"])
    matches = [row for row in artifacts if row["name"] == name and not row["expired"]]
    if len(matches) != 1:
        raise ValueError("Publication requires one unexpired immutable artifact per name")
    row = matches[0]
    expected = row.get("digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected) or row.get("size_in_bytes", 0) > LIMIT:
        raise ValueError("Publication requires an API-authenticated artifact digest")
    payload = subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{row['id']}/zip"])
    if len(payload) > LIMIT or "sha256:" + hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Downloaded artifact differs from its GitHub digest")
    unpack(payload, destination)
    return {key: row[key] for key in ("id", "name", "digest")}


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


def integrity(reports: Path) -> None:
    """Require the collector's entire checksum inventory and every task log."""
    files = {}
    for path in reports.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Evidence report contains a linked or special file: {path}")
        if path.is_file() and path != reports / "checksums.json":
            files[path.relative_to(reports).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = json.loads((reports / "checksums.json").read_text())
    if files != recorded:
        raise ValueError("Evidence report checksum inventory differs from the complete retained payload")
    manifest = json.loads((reports / "manifest.json").read_text())
    required = {"manifest.json", "summary.json", "report.html"}
    required.update(f"logs/{task_key(task)}.txt" for task in manifest["tasks"])
    if not required <= files.keys():
        raise ValueError("Evidence report is missing its summary, HTML report, or a task log")


def retain(tag: str, tested: str, output: Path, names: list[str]) -> None:
    # The published-by-tag endpoint can omit interrupted drafts and their tag.
    pages = api(f"repos/{REPOSITORY}/releases?per_page=100", paginate=True)
    matches = [release for page in pages for release in page if release["tag_name"] == tag]
    if len(matches) > 1:
        raise ValueError("Multiple report releases use the requested immutable tag")
    release = matches[0] if matches else {}
    missing = names
    reference = api(f"repos/{REPOSITORY}/git/ref/tags/{tag}", missing=not release or release.get("draft", False))
    if reference and (reference["object"].get("type") != "commit" or reference["object"].get("sha") != tested):
        raise ValueError("Existing report tag differs from the tested commit")
    if release:
        if release.get("target_commitish") != tested:
            raise ValueError("Existing report release targets a different tested commit")
        present = {asset["name"] for asset in release["assets"]}
        if present - set(names):
            raise ValueError("Existing report release has unexpected assets")
        if present:
            with tempfile.TemporaryDirectory() as temporary:
                subprocess.run(["gh", "release", "download", tag, "--repo", REPOSITORY, "--dir", temporary], check=True)
                for name in present:
                    if (Path(temporary) / name).read_bytes() != (output / name).read_bytes():
                        raise ValueError(f"Immutable report asset differs: {name}")
        missing = sorted(set(names) - present)
        if missing and not release.get("draft"):
            raise ValueError("Published report release is missing expected assets")
    else:
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                REPOSITORY,
                "--draft",
                "--latest=false",
                "--target",
                tested,
                "--title",
                f"relflow {tag.removeprefix('evidence-v')} evidence",
                "--notes",
                "Complete paired reports, tested inputs, and immutable provenance.",
            ],
            check=True,
        )
    if missing:
        subprocess.run(
            ["gh", "release", "upload", tag, "--repo", REPOSITORY, *[str(output / name) for name in missing]],
            check=True,
        )
    if not release or release.get("draft"):
        subprocess.run(
            ["gh", "release", "edit", tag, "--repo", REPOSITORY, "--draft=false", "--latest=false"], check=True
        )

    published = api(f"repos/{REPOSITORY}/releases/tags/{tag}")
    reference = api(f"repos/{REPOSITORY}/git/ref/tags/{tag}")["object"]
    if published.get("draft") or reference.get("type") != "commit" or reference.get("sha") != tested:
        raise ValueError("Published report tag does not identify the tested commit")
    if {asset["name"] for asset in published["assets"]} != set(names):
        raise ValueError("Published report release has an incomplete or unexpected asset inventory")
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run(["gh", "release", "download", tag, "--repo", REPOSITORY, "--dir", temporary], check=True)
        for name in names:
            if (Path(temporary) / name).read_bytes() != (output / name).read_bytes():
                raise ValueError(f"Published report asset differs: {name}")


def publish(root: Path, output: Path, run_id: int, *, version: str | None = None) -> None:
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    version = version or declared
    if run_id < 1 or version != declared or not VERSION.fullmatch(version):
        raise ValueError("Publication requires a positive CI run ID and the checkout package version")
    tag = "evidence-v" + version
    files = filenames(version)
    released = api(f"repos/{REPOSITORY}/releases/tags/v{version}", missing=True)
    if released.get("tag_name") == "v" + version and released.get("draft") is False:
        print(f"relflow {version} is already published; its evidence selection remains frozen")
        return
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Evidence publication output must be outside the checkout")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root).strip():
        raise ValueError("Publication requires a clean tracked checkout")
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    main = api(f"repos/{REPOSITORY}/git/ref/heads/main")["object"]["sha"]
    run = api(f"repos/{REPOSITORY}/actions/runs/{run_id}")
    check_run(run, main, run_id)
    workflow = api(f"repos/{REPOSITORY}/actions/workflows/ci.yml")
    if run.get("workflow_id") != workflow["id"]:
        raise ValueError("Evidence run belongs to a different CI workflow")
    if checkout != main:
        raise ValueError("Publication checkout must be the exact tested main commit")
    output.mkdir(parents=True, exist_ok=False)
    reports, inputs = output / "retained", output / "retained-inputs"
    artifact(run, "relflow-evidence", reports)
    retained_manifest = json.loads((reports / "manifest.json").read_text())
    artifact(run, "evidence-inputs", inputs)
    request = json.loads((inputs / "request.json").read_text())
    baseline = request.get("baseline", "")
    expected_request = {
        "candidate": main,
        "baseline": baseline,
        "suite": main,
        "run_id": run_id,
        "run_attempt": run["run_attempt"],
        "event": run["event"],
    }
    if not SHA.fullmatch(baseline) or request != expected_request:
        raise ValueError("Authenticated input request does not bind the exact current main CI attempt")
    check_report(retained_manifest, json.loads((reports / "summary.json").read_text()), run, main, baseline=baseline)
    trusted(run, retained_manifest, request, root)
    integrity(reports)
    suite_revision = retained_manifest["evidence_sha"]
    with tempfile.TemporaryDirectory() as temporary:
        tested = Path(temporary) / "evidence"
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", suite_revision, "--", "evidence"], cwd=root, text=True
        ).splitlines()
        for name in names:
            mode = subprocess.check_output(["git", "ls-tree", suite_revision, "--", name], cwd=root, text=True).split()[
                0
            ]
            if mode not in {"100644", "100755"}:
                raise ValueError(f"Tested source must be a regular tracked file: {name}")
            destination = Path(temporary) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(subprocess.check_output(["git", "show", f"{suite_revision}:{name}"], cwd=root))
            destination.chmod(0o755 if mode == "100755" else 0o644)
        manifest, records, summary = validate(reports, inputs, root=tested)
        if json.loads((reports / "summary.json").read_text()) != summary:
            raise ValueError("Retained report summary differs from the validated task records")
        check_report(manifest, summary, run, main, baseline=baseline)
        receipt = {
            key: run[key]
            for key in ("run_attempt", "head_sha", "head_branch", "event", "html_url", "workflow_id", "path")
        }
        receipt["run_id"] = run["id"]
        github_run = output / files["provenance"]
        github_run.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        archive(reports, output / files["reports"])
        archive(inputs, output / files["inputs"])
        assets = [files["reports"], files["inputs"], files["provenance"]]
        measurements = output / files["measurements"]
        measurements.write_text(
            f"# relflow {version} evidence\n\n<!-- evidence-results:start -->\n<!-- evidence-results:end -->\n"
        )
        select(
            reports,
            root=tested,
            github_run=github_run,
            release=tag,
            bundle=output / files["reports"],
            readme_target=measurements,
            output=output / files["publication"],
            version=version,
            revision=main,
        )
        if manifest.get("observations"):
            from visuals import install, verify

            figures = output / "plots"
            artifact(run, "evidence-plots", figures)
            bundle = figures / "proof-visualizations.zip"
            verify(bundle, manifest, records)
            published_bundle = output / files["plots"]
            shutil.copyfile(bundle, published_bundle)
            document = json.loads((output / files["publication"]).read_text())
            document["report"]["plots"] = {
                "url": f"https://github.com/{REPOSITORY}/releases/download/{tag}/{published_bundle.name}",
                "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "size": bundle.stat().st_size,
            }
            (output / files["publication"]).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
            install(bundle, root / "docs/assets/evidence-plots")
            assets.append(published_bundle.name)
        assets += [files["publication"], files["measurements"]]
        (output / files["checksums"]).write_text(
            "".join(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n" for name in assets)
        )
        assets.append(files["checksums"])
    environment = dict(os.environ, RELFLOW_EVIDENCE_PUBLICATION=str(output / files["publication"]))
    subprocess.run(["make", "check", "check-docs"], cwd=root, env=environment, check=True)
    if subprocess.check_output(["git", "diff", "--name-only"], cwd=root).strip():
        raise ValueError("Evidence publication must not change tracked experiment, README, or documentation sources")
    (output / "docs-base.txt").write_text(checkout + "\n")
    if api(f"repos/{REPOSITORY}/git/ref/heads/main")["object"]["sha"] != main:
        raise ValueError("Main advanced during validation; rerun publication against current main")
    retain(tag, main, output, assets)
    print(f"Retained {tag} for verified relflow {version}; publication asset: {output / files['publication']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--version")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    publish(
        Path(__file__).resolve().parents[1],
        options.output,
        options.run_id,
        version=options.version,
    )


if __name__ == "__main__":
    main()
