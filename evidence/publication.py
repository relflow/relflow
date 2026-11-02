"""Retain trusted main evidence and propose its validated publication selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from evidence_ci import task_key
from experiments.source import fingerprint, frontmatter, metadata, registry
from record import readme, select, validate
from reuse import SHA, archive, verify_receipt
from reuse import artifact as verified_artifact

REPOSITORY = "relflow/relflow"
EXECUTOR = "92b3c887eabc598ee2fc6c9dba4e05b9166cb8ba"
RESULTS = {"evidence-result", "evidence-report"}
MARKERS = ("<!-- evidence-results:start -->", "<!-- evidence-results:end -->")


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


def outside_results(text: str) -> tuple[str, str]:
    if any(text.count(marker) != 1 for marker in MARKERS):
        raise ValueError("README requires one ordered evidence-results marker pair")
    before, remaining = text.split(MARKERS[0])
    _, after = remaining.split(MARKERS[1])
    return before, after


def selection_changes(before: dict[str, str], after: dict[str, str]) -> bool:
    """Accept only selected result headers and the explicitly bounded README table."""
    if not before or before.keys() != after.keys():
        return False
    try:
        for path, old in before.items():
            new = after[path]
            if path == "README.md":
                if outside_results(old) != outside_results(new):
                    return False
            elif re.fullmatch(r"evidence/experiments/[^/]+/[^/]+\.py", path):
                left, right = metadata(old), metadata(new)
                if not left.get("evidence-id") or left.get("evidence-id") != right.get("evidence-id"):
                    return False
                left = {key: value for key, value in left.items() if key not in RESULTS}
                right = {key: value for key, value in right.items() if key not in RESULTS}
                if left != right or fingerprint(old) != fingerprint(new):
                    return False
                if frontmatter(old, left) != frontmatter(new, right):
                    return False
            else:
                return False
    except (ValueError, SyntaxError, KeyError):
        return False
    return True


def selection_only(root: Path, before: str, after: str) -> bool:
    names = (
        subprocess.check_output(["git", "diff", "--name-only", "-z", before, after], cwd=root)
        .decode()
        .strip("\0")
        .split("\0")
    )
    if names == [""]:
        return False
    versions = []
    for revision in (before, after):
        values = {}
        for name in names:
            result = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=root, capture_output=True, text=True)
            if result.returncode:
                return False
            values[name] = result.stdout
        versions.append(values)
    return selection_changes(*versions)


def compatible(root: Path, tested: str, checkout: str) -> None:
    """A bootstrap checkout may change publication tooling, but not tested runtime inputs."""
    changed = (
        subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                "-z",
                tested,
                checkout,
                "--",
                "src",
                "pyproject.toml",
                "uv.lock",
                "evidence/experiments",
            ],
            cwd=root,
        )
        .decode()
        .strip("\0")
        .split("\0")
    )
    for name in filter(None, changed):
        if not re.fullmatch(r"evidence/experiments/[^/]+/[^/]+\.py", name):
            raise ValueError(f"Bootstrap checkout changes tested runtime input: {name}")
        old = subprocess.check_output(["git", "show", f"{tested}:{name}"], cwd=root, text=True)
        new = (root / name).read_text()
        if not selection_changes({name: old}, {name: new}):
            raise ValueError(f"Bootstrap checkout changes tested experiment source: {name}")


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
                f"Evidence {tag}",
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


def propose(root: Path, output: Path, run: dict, tag: str, main: str, changed: list[str]) -> None:
    """Create one bot branch per immutable attempt, without replacing human changes."""
    branch = f"automation/evidence-{run['id']}-{run['run_attempt']}"
    subprocess.run(["git", "add", "--", *changed], cwd=root, check=True)
    tree = subprocess.check_output(["git", "write-tree"], cwd=root, text=True).strip()
    reference = api(f"repos/{REPOSITORY}/git/ref/heads/{branch}", missing=True)
    if reference:
        commit = api(f"repos/{REPOSITORY}/git/commits/{reference['object']['sha']}")
        if commit["tree"]["sha"] != tree or [parent["sha"] for parent in commit["parents"]] != [main]:
            raise ValueError("Existing evidence bot branch differs; refusing to replace its changes")
    if api(f"repos/{REPOSITORY}/git/ref/heads/main")["object"]["sha"] != main:
        raise ValueError("Main advanced before opening the evidence publication PR")
    if not reference:
        subprocess.run(["git", "checkout", "-b", branch], cwd=root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=github-actions[bot]",
                "-c",
                "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                "commit",
                "-m",
                f"docs: publish evidence from CI {run['id']} attempt {run['run_attempt']}",
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(["gh", "auth", "setup-git"], cwd=root, check=True)
        subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], cwd=root, check=True)
    existing = subprocess.check_output(
        ["gh", "pr", "list", "--repo", REPOSITORY, "--head", branch, "--state", "all", "--json", "number"], text=True
    )
    if not json.loads(existing):
        body = output / "pr-body.md"
        body.write_text(
            f"Select the complete paired evidence verified by CI {run['html_url']}.\n\nValidated main: `{main}`.\nReport: https://github.com/{REPOSITORY}/releases/tag/{tag}\n\nMeasurements retain their original execution revisions and run identity; a reused report includes its verified request/source receipt. Individual unmet learning checks remain visible. Only result/report frontmatter and the bounded README metrics table change. Normal required CI and review apply; GitHub may require approval to start the generated PR workflows.\n"
        )
        subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                REPOSITORY,
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                f"docs: publish evidence from CI {run['id']}",
                "--body-file",
                str(body),
            ],
            check=True,
        )


def publish(root: Path, output: Path, run_id: int, *, automatic: bool = False, tag: str | None = None) -> None:
    if run_id < 1 or (tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag)):
        raise ValueError("Use a positive CI run ID and a simple immutable report tag")
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
    if automatic and checkout != main:
        raise ValueError("Automatic publication must execute the current trusted main checkout")
    compatible(root, main, checkout)
    tag = tag or f"evidence-ci-{run_id}-attempt-{run['run_attempt']}"
    output.mkdir(parents=True, exist_ok=False)
    reports, inputs = output / "retained", output / "retained-inputs"
    verified_artifact(run_id, "relflow-evidence", reports)
    retained_manifest = json.loads((reports / "manifest.json").read_text())
    reused = retained_manifest["run_id"] != f"gh-{run_id}-{run['run_attempt']}"
    current_inputs = output / "current-inputs" if reused else inputs
    verified_artifact(run_id, "evidence-inputs", current_inputs)
    request = json.loads((current_inputs / "request.json").read_text())
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
    if automatic and run["event"] == "push" and selection_only(root, baseline, main):
        shutil.rmtree(output)
        print("Skipping recursive publication of selection-only main changes; required CI has still run.")
        return
    names = [("evidence-reused-inputs", inputs)] if reused else []
    if reused:
        names.append(("evidence-reuse", output / "reuse-context"))
    for name, target in names:
        verified_artifact(run_id, name, target)
    integrity(reports)
    execution_run = run
    if reused:
        reuse_receipt = json.loads((output / "reuse-context/reuse.json").read_text())
        execution_run = verify_receipt(
            root, reports, inputs, reuse_receipt, run, baseline, current_inputs=current_inputs, executor=EXECUTOR
        )
        (output / "evidence-reuse.json").write_text(json.dumps(reuse_receipt, sort_keys=True, indent=2) + "\n")
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
        manifest, _, summary = validate(reports, inputs, root=tested)
        if json.loads((reports / "summary.json").read_text()) != summary:
            raise ValueError("Retained report summary differs from the validated task records")
        if not reused:
            check_report(manifest, summary, run, main, baseline=baseline)
        receipt = {
            key: execution_run[key]
            for key in ("run_attempt", "head_sha", "head_branch", "event", "html_url", "workflow_id", "path")
        }
        receipt["run_id"] = execution_run["id"]
        github_run = output / "github-run.json"
        github_run.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        archive(reports, output / "evidence-reports.tar.gz")
        archive(inputs, output / "evidence-inputs.tar.gz")
        assets = ["evidence-reports.tar.gz", "evidence-inputs.tar.gz", "github-run.json"]
        if reused:
            assets.append("evidence-reuse.json")
        (output / "SHA256SUMS").write_text(
            "".join(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n" for name in assets)
        )
        assets.append("SHA256SUMS")
        select(reports, root=tested, github_run=github_run, release=tag, bundle=output / "evidence-reports.tar.gz")
        before, after = {}, {}
        for entry in registry(tested).values():
            name = "evidence/" + entry["script"]
            path = root / name
            before[name] = path.read_text()
            header = metadata(before[name])
            header.update({key: entry["header"][key] for key in RESULTS})
            after[name] = frontmatter(before[name], header)
        before["README.md"] = (root / "README.md").read_text()
        pending = Path(temporary) / "README.md"
        pending.write_text(before["README.md"])
        readme(tested, pending)
        after["README.md"] = pending.read_text()
        if not selection_changes(before, after):
            raise ValueError("Publication changed more than result headers and the bounded README table")
        for name, content in after.items():
            (root / name).write_text(content)
    subprocess.run(["make", "check", "check-docs"], cwd=root, check=True)
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=root, text=True).splitlines()
    actual = {name: (root / name).read_text() for name in changed}
    original = {name: subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root, text=True) for name in changed}
    if not selection_changes(original, actual):
        raise ValueError("Validated publication contains unexpected source changes")
    patch = subprocess.check_output(["git", "diff", "--binary", "--", *changed], cwd=root)
    (output / "evidence-update.patch").write_bytes(patch)
    (output / "docs-base.txt").write_text(checkout + "\n")
    if api(f"repos/{REPOSITORY}/git/ref/heads/main")["object"]["sha"] != main:
        raise ValueError("Main advanced during validation; rerun publication against current main")
    retain(tag, main, output, assets)
    if automatic:
        propose(root, output, run, tag, main, changed)
    print(f"Retained {tag} for validated main {main}; publication patch: {output / 'evidence-update.patch'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--automatic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    publish(
        Path(__file__).resolve().parents[1],
        options.output,
        options.run_id,
        automatic=options.automatic,
        tag=options.tag,
    )


if __name__ == "__main__":
    main()
