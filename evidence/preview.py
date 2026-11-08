"""Retain CI plots as immutable release assets and refresh their documentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from evidence_ci import evaluate
from publication import REPOSITORY, api, integrity, retain
from reuse import SHA, archive, artifact, ensure, trusted
from reuse import api as run_api
from visuals import LIMIT, verify


def collect(root: Path, output: Path, run: dict, *, helper: str, executor: str) -> tuple[dict, list, dict]:
    """Authenticate one original attempt, including complete learning regressions."""
    if (
        run.get("id", 0) < 1
        or run.get("workflow_id") != api(f"repos/{REPOSITORY}/actions/workflows/ci.yml")["id"]
        or run.get("event") not in {"pull_request", "push", "workflow_dispatch", "merge_group"}
        or run.get("head_repository", {}).get("full_name") != REPOSITORY
        or not SHA.fullmatch(run.get("head_sha", ""))
    ):
        raise ValueError("Plot publication requires the original internal CI workflow")
    suffix = f"-attempt-{run['run_attempt']}"
    artifact(run["id"], "relflow-evidence" + suffix, output / "reports")
    artifact(run["id"], "evidence-inputs" + suffix, output / "inputs")
    artifact(run["id"], "evidence-plots" + suffix, output / "plots")
    reports = output / "reports"
    integrity(reports)
    manifest = json.loads((reports / "manifest.json").read_text())
    request = json.loads((output / "inputs/request.json").read_text())
    for revision in {run["head_sha"], manifest["candidate"]["revision"], manifest["baseline"]["revision"]}:
        ensure(root, revision)
    trusted(run, manifest, helper=helper, executor=executor, root=root, request=request, review=True)
    path = f"repos/{REPOSITORY}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100"
    listing = run_api(path)
    jobs = list(listing["jobs"])
    for page in range(2, (listing.get("total_count", len(jobs)) + 99) // 100 + 1):
        jobs.extend(run_api(path + f"&page={page}")["jobs"])
    rendered = [job for job in jobs if job["name"].endswith("/ Render paired proof plots")]
    if len(rendered) != 1 or rendered[0].get("conclusion") != "success" or rendered[0].get("status") != "completed":
        raise ValueError("Plot publication requires the original successful isolated plotting job")
    records = [json.loads(path.read_text()) for path in sorted((reports / "results").rglob("*.json"))]
    summary = evaluate(manifest, records, root=reports / "source", reports=reports)
    if not summary["complete"] or summary["errors"] or summary != json.loads((reports / "summary.json").read_text()):
        raise ValueError(
            "Plot publication requires complete paired measurements without execution or provenance errors"
        )
    for role in ("baseline", "candidate"):
        wheels = list((output / "inputs" / role).glob("*.whl"))
        if len(wheels) != 1 or hashlib.sha256(wheels[0].read_bytes()).hexdigest() != manifest[role]["wheel_sha256"]:
            raise ValueError(f"Plot publication has a different {role} wheel")
    bundle = output / "plots/proof-visualizations.zip"
    receipt = json.loads((output / "plots/plot-receipt.json").read_text())
    if (
        receipt.get("run_id") != manifest["run_id"]
        or receipt.get("report_manifest_sha256") != hashlib.sha256((reports / "manifest.json").read_bytes()).hexdigest()
        or receipt.get("source_sha256") != manifest["suite_sha256"]
        or receipt.get("bytes") != bundle.stat().st_size
        or receipt.get("sha256") != hashlib.sha256(bundle.read_bytes()).hexdigest()
        or bundle.stat().st_size > LIMIT
    ):
        raise ValueError("Plot receipt differs from its original paired report or static bundle")
    verify(bundle, manifest, records)
    return manifest, records, summary


def refresh(run: dict) -> str | None:
    """Refresh only the documentation version still pointing at this exact head."""
    version = None
    if run["event"] == "pull_request":
        pulls = api(f"repos/{REPOSITORY}/commits/{run['head_sha']}/pulls")
        matches = [
            pull
            for pull in pulls
            if pull["state"] == "open"
            and pull["head"]["sha"] == run["head_sha"]
            and pull["head"]["repo"]["full_name"] == REPOSITORY
        ]
        if len(matches) == 1:
            version = str(matches[0]["number"])
    elif run.get("head_branch") == "main":
        if api(f"repos/{REPOSITORY}/git/ref/heads/main")["object"]["sha"] == run["head_sha"]:
            version = "latest"
    if version is None:
        return None
    token = os.environ.get("READTHEDOCS_TOKEN")
    if not token:
        raise ValueError(
            "Plot assets are published. Set the READTHEDOCS_TOKEN Actions secret to rebuild the preview automatically."
        )
    request = Request(
        f"https://app.readthedocs.org/api/v3/projects/relflow/versions/{version}/builds/",
        data=b"{}",
        headers={"Authorization": "Token " + token, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        if response.status != 202:
            raise ValueError("Read the Docs did not accept the plot preview rebuild")
    return version


def publish(root: Path, output: Path, run_id: int, *, helper: str, executor: str) -> None:
    if not SHA.fullmatch(helper) or not SHA.fullmatch(executor):
        raise ValueError("Plot publication requires frozen helper and executor revisions")
    run = api(f"repos/{REPOSITORY}/actions/runs/{run_id}")
    output.mkdir(parents=True, exist_ok=False)
    manifest, records, summary = collect(root, output, run, helper=helper, executor=executor)
    tag = f"evidence-ci-{run_id}-attempt-{run['run_attempt']}"
    shutil.copyfile(output / "plots/proof-visualizations.zip", output / "proof-visualizations.zip")
    shutil.copyfile(output / "plots/plot-receipt.json", output / "plot-receipt.json")
    archive(output / "reports", output / "evidence-reports.tar.gz")
    archive(output / "inputs", output / "evidence-inputs.tar.gz")
    receipt = {
        key: run[key] for key in ("run_attempt", "head_sha", "head_branch", "event", "html_url", "workflow_id", "path")
    }
    receipt["run_id"] = run_id
    (output / "github-run.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    bundle = output / "proof-visualizations.zip"
    selection = {
        "schema": 1,
        "version": "ci-preview",
        "revision": run["head_sha"],
        "report": {
            "run_id": manifest["run_id"],
            "passed": summary["passed"],
            "github_run": receipt,
            "candidate": manifest["candidate"],
            "baseline": manifest["baseline"],
            "evidence_revision": manifest["evidence_sha"],
            "executor_revision": manifest["executor_sha"],
            "release": f"https://github.com/{REPOSITORY}/releases/tag/{tag}",
            "plots": {
                "url": f"https://github.com/{REPOSITORY}/releases/download/{tag}/proof-visualizations.zip",
                "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "size": bundle.stat().st_size,
            },
        },
        "results": {r["task"]["id"]: r["result"] for r in records if r["task"]["role"] == "candidate"},
    }
    (output / "publication.json").write_text(json.dumps(selection, sort_keys=True, indent=2, allow_nan=False) + "\n")
    names = [
        "evidence-reports.tar.gz",
        "evidence-inputs.tar.gz",
        "github-run.json",
        "publication.json",
        "proof-visualizations.zip",
        "plot-receipt.json",
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n" for name in names)
    )
    retain(tag, run["head_sha"], output, names + ["SHA256SUMS"])
    print(f"Published {tag}")
    version = refresh(run)
    print(
        f"Requested Read the Docs build: {version}"
        if version
        else "The source has advanced; retained its assets without rebuilding newer docs"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--executor", required=True)
    options = parser.parse_args()
    publish(
        Path(__file__).resolve().parents[1],
        options.output,
        options.run_id,
        helper=options.helper,
        executor=options.executor,
    )
