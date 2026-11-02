"""Validate retained reports and select their candidate measurements for publication."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from evidence_ci import ROOT, digest, evaluate
from experiments.source import frontmatter, registry


def validate(reports: Path, inputs: Path | None = None) -> tuple[dict, list[dict], dict]:
    manifest = json.loads((reports / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted((reports / "results").rglob("*.json"))]
    summary = evaluate(manifest, records)
    if not summary["complete"] or summary["errors"]:
        raise ValueError("Publication requires a complete full-catalog run without execution or provenance errors")
    if inputs:
        for role in ("baseline", "candidate"):
            wheels = list((inputs / role).glob("*.whl"))
            if len(wheels) != 1 or digest(wheels[0]) != manifest[role]["wheel_sha256"]:
                raise ValueError(f"Retained {role} wheel differs from the tested artifact")
        if digest(inputs / "requirements.txt") != manifest["requirements_sha256"]:
            raise ValueError("Retained runtime requirements differ from the tested lock")
    return manifest, records, summary


def select(
    reports: Path,
    *,
    root: Path = ROOT,
    github_run: Path | None = None,
    release: str | None = None,
    bundle: Path | None = None,
) -> None:
    manifest, records, summary = validate(reports)
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
    updates = {}
    for record in records:
        if record["task"]["role"] == "candidate":
            entry = entries[record["task"]["id"]]
            script = root / entry["script"]
            header = dict(entry["header"], **{"evidence-result": record["result"], "evidence-report": selection})
            updates[script] = frontmatter(script.read_text(), header)
    for script, content in updates.items():
        with tempfile.NamedTemporaryFile(mode="w", dir=script.parent, delete=False) as handle:
            handle.write(content)
            pending = Path(handle.name)
        try:
            shutil.copymode(script, pending)
            pending.replace(script)
        finally:
            pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "select"))
    parser.add_argument("reports", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--github-run", type=Path)
    parser.add_argument("--release")
    parser.add_argument("--bundle", type=Path)
    options = parser.parse_args()
    if options.command == "validate":
        _, _, summary = validate(options.reports, options.inputs)
        print(json.dumps(summary, indent=2))
    else:
        select(
            options.reports,
            github_run=options.github_run,
            release=options.release,
            bundle=options.bundle,
        )


if __name__ == "__main__":
    main()
