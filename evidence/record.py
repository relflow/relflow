"""Validate retained reports and select their candidate measurements for publication."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from evidence_ci import ROOT, digest, evaluate
from experiments.source import registry


def validate(reports: Path, inputs: Path | None = None, *, root: Path | None = None) -> tuple[dict, list[dict], dict]:
    manifest = json.loads((reports / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted((reports / "results").rglob("*.json"))]
    summary = evaluate(manifest, records, root=root if root is not None else ROOT, reports=reports)
    if not summary["complete"] or not summary["passed"] or summary["errors"] or summary["regressions"]:
        raise ValueError(
            "Publication requires a complete full-catalog run without execution or provenance errors or regressions"
        )
    if root is not None:
        entries = registry(root)
        for task in manifest["tasks"]:
            entry = entries[task["id"]]
            if task.get("script") != entry["script"]:
                raise ValueError(f"{task['id']}: publication source path differs from the tested catalog")
            if digest(root / entry["script"]) != task.get("script_sha256"):
                raise ValueError(f"{task['id']}: publication source fingerprint differs from the tested script")
    try:
        json.dumps(records, allow_nan=False)
    except ValueError as error:
        raise ValueError("Publication requires finite JSON measurements") from error
    if inputs:
        for role in ("baseline", "candidate"):
            wheels = list((inputs / role).glob("*.whl"))
            if len(wheels) != 1 or digest(wheels[0]) != manifest[role]["wheel_sha256"]:
                raise ValueError(f"Retained {role} wheel differs from the tested artifact")
        if digest(inputs / "requirements.txt") != manifest["requirements_sha256"]:
            raise ValueError("Retained runtime requirements differ from the tested lock")
    return manifest, records, summary


def metric(entry: dict, *path: str) -> float:
    """Read one finite numerical measurement from the selected full run."""
    identifier = entry["header"]["evidence-id"]
    run = entry["run"]
    if not run or run.get("mode") != "full" or run.get("outcome") == "error":
        raise ValueError(f"{identifier}: README measurements require a selected full run")
    measured = run.get("metrics", {})
    try:
        for part in path:
            measured = measured[part]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{identifier}: selected run is missing README metric {'.'.join(path)}") from error
    if isinstance(measured, bool) or not isinstance(measured, (int, float)) or not math.isfinite(measured):
        raise ValueError(f"{identifier}: README metric {'.'.join(path)} must be a finite number, got {measured!r}")
    return measured


def readme_content(entries: dict, source: str) -> str:
    """Render the three measured comparisons inside unique, standalone markers."""
    start = "<!-- evidence-results:start -->"
    end = "<!-- evidence-results:end -->"
    if (
        source.count(start) != 1
        or source.count(end) != 1
        or re.search(rf"(?m)^{re.escape(start)}\r?$", source) is None
        or re.search(rf"(?m)^{re.escape(end)}\r?$", source) is None
        or source.index(end) < source.index(start)
    ):
        raise ValueError("README requires one standalone evidence-results:start marker followed by one end marker")
    missing = {"P021", "P025", "P038"} - entries.keys()
    if missing:
        raise ValueError(f"README measurements require catalog experiments: {', '.join(sorted(missing))}")
    identity, recall, states = (entries[identifier] for identifier in ("P021", "P025", "P038"))
    hash_auc = metric(identity, "hash_auc")
    category_auc = metric(identity, "category_oov_auc")
    intact = metric(recall, "median", "intact_nrmse")
    broken = metric(recall, "median", "broken_nrmse")
    state_accuracy = metric(states, "state_accuracy")
    prefilled_accuracy = metric(states, "prefilled_accuracy")
    if any(not 0 <= value <= 1 for value in (hash_auc, category_auc, state_accuracy, prefilled_accuracy)):
        raise ValueError("README AUC and accuracy measurements must be between zero and one")
    if intact < 0 or broken < 0:
        raise ValueError("README normalized RMSE measurements must be nonnegative")
    urls = {
        identifier: "https://www.relflow.ai/en/latest/"
        + entry["page"].removeprefix("docs/").removesuffix(".py")
        + ".html"
        for identifier, entry in (("P021", identity), ("P025", recall), ("P038", states))
    }
    table = (
        "| Experiment | relflow result | Control |\n"
        "| --- | --- | --- |\n"
        f"| [Match IDs absent from training]({urls['P021']}) | **{hash_auc:.4f} AUC** with `Hash` | "
        f"{category_auc:.4f} with unknown `Category` labels |\n"
        f"| [Recall a value by an unfamiliar ID]({urls['P025']}) | **{intact:.3f} normalized RMSE**, "
        f"median of three seeds | {broken:.3f} after corrupting the lookup IDs |\n"
        f"| [Distinguish nulls from zero]({urls['P038']}) | **{state_accuracy * 100:.4g}% accuracy** "
        f"with value states preserved | {prefilled_accuracy * 100:.4g}% after filling nulls with zero |"
    )
    return source[: source.index(start) + len(start)] + "\n\n" + table + "\n\n" + source[source.index(end) :]


def readme(root: Path, target: Path) -> None:
    """Render an explicitly requested comparison table from the selected results."""
    content = readme_content(registry(root), target.read_bytes().decode("utf-8"))
    target.write_bytes(content.encode("utf-8"))


def select(
    reports: Path,
    *,
    root: Path = ROOT,
    github_run: Path | None = None,
    release: str | None = None,
    bundle: Path | None = None,
    readme_target: Path | None = None,
    output: Path | None = None,
    version: str = "unversioned",
    revision: str | None = None,
) -> dict:
    manifest, records, summary = validate(reports, root=root)
    entries = registry(root)
    selection = {
        "run_id": manifest["run_id"],
        "candidate": manifest["candidate"],
        "baseline": manifest["baseline"],
        "evidence_revision": manifest["evidence_sha"],
        "executor_revision": manifest["executor_sha"],
        "manifest_sha256": digest(reports / "manifest.json"),
        "passed": summary["passed"],
    }
    if github_run:
        selection["github_run"] = json.loads(github_run.read_text())
    if release:
        selection["release"] = "https://github.com/relflow/relflow/releases/tag/" + release
    if bundle:
        selection["reports_sha256"] = digest(bundle)
    results = {}
    for record in records:
        if record["task"]["role"] == "candidate":
            entry = entries[record["task"]["id"]]
            results[record["task"]["id"]] = record["result"]
            entries[record["task"]["id"]] = dict(entry, report=selection, run=record["result"])
    document = {
        "schema": 1,
        "version": version,
        "revision": revision or manifest["candidate"]["revision"],
        "report": selection,
        "results": results,
    }
    updates = {}
    if output is not None:
        updates[output] = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if readme_target is not None:
        updates[readme_target] = readme_content(entries, readme_target.read_bytes().decode("utf-8"))
    for script, content in updates.items():
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=script.parent, delete=False
        ) as handle:
            handle.write(content)
            pending = Path(handle.name)
        try:
            if script.exists():
                shutil.copymode(script, pending)
            pending.replace(script)
        finally:
            pending.unlink(missing_ok=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "select"))
    parser.add_argument("reports", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--github-run", type=Path)
    parser.add_argument("--release")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", default="unversioned")
    options = parser.parse_args()
    if options.command == "validate":
        _, _, summary = validate(options.reports, options.inputs)
        print(json.dumps(summary, indent=2))
    else:
        if options.output is None:
            parser.error("select requires --output for the publication JSON asset")
        select(
            options.reports,
            github_run=options.github_run,
            release=options.release,
            bundle=options.bundle,
            readme_target=options.readme,
            output=options.output,
            version=options.version,
        )


if __name__ == "__main__":
    main()
