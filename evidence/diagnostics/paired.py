"""Bind plotted observations to the exact canonical paired CI records."""

from __future__ import annotations

import json
from pathlib import Path

from evidence.evidence_ci import digest, task_key
from evidence.experiments.source import registry
from evidence.observations import read, unpack


def prepare(reports: Path, observations: Path, destination: Path) -> dict:
    manifest = json.loads((reports / "manifest.json").read_text())
    catalog = registry(reports / "source", measurements=False)
    tasks = manifest["tasks"]
    policy = manifest.get("policy", "paired-passing-checks-v1")
    if policy not in {"paired-passing-checks-v1", "introduced-passing-checks-v1"}:
        raise ValueError("Unknown Evidence plot comparison policy")
    candidate_only = set(manifest.get("introduced", []))
    if not candidate_only <= catalog.keys() or (candidate_only and policy != "introduced-passing-checks-v1"):
        raise ValueError("Plot scope differs from the registered catalog")
    expected = {(key, "candidate") for key in catalog}
    expected |= {(key, "baseline") for key in catalog if key not in candidate_only}
    if (
        manifest.get("mode") != "full"
        or manifest.get("full_catalog") is not True
        or {(task["id"], task["role"]) for task in tasks} != expected
        or len(tasks) != len(expected)
        or not manifest.get("observations")
    ):
        raise ValueError("Plot generation requires the complete full-budget paired CI catalog and observations")
    archives = {}
    for path in observations.rglob("bundle.zip"):
        value = read(path)
        task = value["manifest"]["task"]
        key = task_key(task)
        if key in archives:
            raise ValueError(f"Paired observations duplicate task {key}")
        archives[key] = path
    if set(archives) != {task_key(task) for task in tasks}:
        raise ValueError("Paired observations do not cover the exact measured task union")
    for task in tasks:
        key = task_key(task)
        record = json.loads((reports / "results" / (key + ".json")).read_text())
        if record["task"] != task or record["result"] is None or not record.get("observations"):
            raise ValueError(f"{key}: canonical measured record or observation receipt is missing")
        path = archives[key]
        receipt = record["observations"]
        captured = read(path, receipt)
        header = captured["manifest"]
        if (
            captured["record"] != record["result"]
            or header["task"] != task
            or header["source_sha256"] != task["script_sha256"]
            or header["wheel_sha256"] != manifest[task["role"]]["wheel_sha256"]
            or header["capture_sha256"] != manifest["observations"]["capture_sha256"]
            or header["runtime"]["run_id"] != manifest["run_id"]
        ):
            raise ValueError(
                f"{key}: observations differ from the paired source, wheel, recorder, or exact measurement"
            )
        if task["role"] == "candidate":
            unpack(path, destination / task["id"], receipt)
    return {
        "run_id": manifest["run_id"],
        "evidence_revision": manifest["evidence_sha"],
        "candidate_revision": manifest["candidate"]["revision"],
        "baseline_revision": manifest["baseline"]["revision"],
        "executor_sha": manifest["executor_sha"],
        "report_manifest_sha256": digest(reports / "manifest.json"),
        "capture_sha256": manifest["observations"]["capture_sha256"],
        **({"candidate_only": sorted(candidate_only)} if candidate_only else {}),
    }
