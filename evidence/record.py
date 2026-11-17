"""Validate retained reports and select their candidate measurements for publication."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from string import Formatter

from evidence_ci import ROOT, digest, evaluate
from experiments.source import registry


def validate(reports: Path, inputs: Path | None = None, *, root: Path | None = None) -> tuple[dict, list[dict], dict]:
    manifest = json.loads((reports / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted((reports / "results").rglob("*.json"))]
    summary = evaluate(manifest, records, root=root if root is not None else ROOT, reports=reports)
    if not summary["complete"] or not summary["passed"] or summary["errors"] or summary["regressions"]:
        raise ValueError(
            "Publication requires a complete full-catalog run without execution or provenance errors or regressions"
            + (": " + "; ".join(summary["errors"][:3]) if summary["errors"] else "")
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


def metric(entry: dict, *path: str | int) -> float:
    """Read one finite numerical measurement from the selected full run."""
    identifier = entry["header"]["evidence-id"]
    run = entry["run"]
    if not run or run.get("mode") != "full" or run.get("outcome") == "error":
        raise ValueError(f"{identifier}: README measurements require a selected full run")
    measured = run.get("metrics", {})
    try:
        for part in path:
            measured = measured[part]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"{identifier}: selected run is missing README metric {'.'.join(map(str, path))}") from error
    if isinstance(measured, bool) or not isinstance(measured, (int, float)) or not math.isfinite(measured):
        raise ValueError(
            f"{identifier}: README metric {'.'.join(map(str, path))} must be a finite number, got {measured!r}"
        )
    return measured


def highlights(entries: dict) -> list[tuple[str, dict, dict]]:
    """Select and order publication highlights declared by the experiments themselves."""
    selected = []
    for identifier, entry in entries.items():
        plan = entry["header"].get("evidence-highlight")
        if plan is None:
            continue
        if (
            not isinstance(plan, dict)
            or type(plan.get("order")) is not int
            or plan["order"] < 0
            or not isinstance(plan.get("title"), str)
            or not plan["title"].strip()
        ):
            raise ValueError(f"{identifier}: evidence-highlight requires a nonnegative order and title")
        selected.append((identifier, entry, plan))
    if len({plan["order"] for _, _, plan in selected}) != len(selected):
        raise ValueError("README highlights require experiment-owned declarations with distinct orders")
    return sorted(selected, key=lambda row: row[2]["order"])


def highlight(entry: dict, plan: dict) -> str:
    """Format one bounded recorded value without evaluating expressions from source metadata."""
    path, text = plan.get("metric"), plan.get("text")
    if (
        not isinstance(path, list)
        or not path
        or any(not ((isinstance(part, str) and part) or (type(part) is int and part >= 0)) for part in path)
        or not isinstance(text, str)
    ):
        raise ValueError("README highlight requires a metric path and text with one value field")
    fields = [(field, spec, conversion) for _, field, spec, conversion in Formatter().parse(text) if field is not None]
    if len(fields) != 1 or fields[0][0] != "value" or fields[0][2] is not None or "{" in fields[0][1]:
        raise ValueError("README highlight text must contain exactly one simple {value} field")
    measured = metric(entry, *path)
    for name in ("minimum", "maximum", "scale"):
        value = plan.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise ValueError(f"README highlight {name} must be finite")
    if measured < plan.get("minimum", -math.inf) or measured > plan.get("maximum", math.inf):
        raise ValueError("README highlight metric is outside its declared bounds")
    return text.format(value=measured * plan.get("scale", 1))


def readme_content(entries: dict, source: str) -> str:
    """Render experiment-owned highlights inside unique, standalone markers."""
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
    rows = []
    for identifier, entry, plan in highlights(entries):
        url = f"https://www.relflow.ai/latest/evidence/{identifier}/"
        result, control = (highlight(entry, plan[name]) for name in ("result", "control"))
        rows.append(f"| [{plan['title']}]({url}) | {result} | {control} |")
    table = (
        "| Experiment | relflow result | Control |\n| --- | --- | --- |\n" + "\n".join(rows)
        if rows
        else "No publication highlights are declared for this catalog."
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
