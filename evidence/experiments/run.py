"""Run standalone experiments by their stable IDs, serially by default."""

# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6"]
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from string import Template

from source import check_experiment, describe, environment, registry

ROOT = Path(__file__).resolve().parents[1]


def command(script: Path, wheel: Path | None = None) -> list[str]:
    """Let uv resolve the script's dependencies, replacing only the SDK when supplied."""
    declared = environment(script.read_text())
    if declared is None:
        raise ValueError(f"Experiment {script} needs a PEP 723 script block declaring its dependencies")
    arguments = ["uv", "run", "--no-project", "--no-config"]
    if wheel is None:
        return [*arguments, "--script", str(script)]
    # Supplying all requirements in one resolution avoids a second overlay whose
    # SDK dependencies could conflict with the script's original environment.
    arguments.append("--isolated")
    if "requires-python" in declared:
        arguments.extend(["--python", declared["requires-python"]])
    replaced = False
    for dependency in declared["dependencies"]:
        match = re.fullmatch(r"relflow(\[[^\]]+\])?(?:\s*[<>=!~][^;]+)?", dependency)
        if match:
            dependency = f"relflow{match[1] or ''} @ {wheel.as_uri()}"
            replaced = True
        arguments.extend(["--with", dependency])
    if not replaced:
        raise ValueError(f"Experiment {script} must declare relflow or relflow[extra] to select a supplied wheel")
    return [*arguments, "python", str(script)]


def create(
    root: Path,
    target: str,
    *,
    title: str | None = None,
    group: str | None = None,
    identifier: str | None = None,
    seed: int = 0,
) -> Path:
    """Create a complete authoring skeleton without importing or running a model."""
    match = re.fullmatch(r"([a-z][a-z0-9_]*)/([a-z][a-z0-9_]*)(?:\.py)?", target)
    if match is None:
        raise ValueError(
            "Use category/name with lowercase letters, digits, and underscores, such as aggregation/median"
        )
    category, name = match.groups()
    if name in {"reporting", "source", "run", "render"}:
        raise ValueError(f"Experiment name {name!r} conflicts with shared tooling; choose another name")
    directory = root / "experiments"
    family = directory / category
    path = family / f"{name}.py"
    for parent in (root, directory, family):
        if parent.is_symlink():
            raise ValueError(f"Experiment directory must not be a symlink: {parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Experiment already exists; choose another name: {path}")

    entries = registry(root, measurements=False)
    identifier = (
        identifier if identifier is not None else f"P{max((int(key[1:]) for key in entries), default=0) + 1:03d}"
    )
    declared = check_experiment(identifier, seed)
    if identifier in entries:
        raise ValueError(f"Experiment ID {identifier} is already registered; choose a new ID")
    title = title if title is not None else name.replace("_", " ").capitalize()
    groups = {
        entry["header"]["evidence-group"]
        for entry in entries.values()
        if Path(entry["script"]).parent.name == category and "evidence-group" in entry["header"]
    }
    if group is None:
        group = next(iter(groups)) if len(groups) == 1 else category.replace("_", " ").title()
    if not title.strip() or not group.strip():
        raise ValueError("Experiment title and group must contain text")
    template = Template((Path(__file__).resolve().parents[1] / "templates/experiment.py.txt").read_text())
    source = template.substitute(
        identifier=json.dumps(declared.id),
        seed=str(declared.seed),
        title=json.dumps(title, ensure_ascii=False),
        group=json.dumps(group, ensure_ascii=False),
        category=json.dumps(category.replace("_", " ").title(), ensure_ascii=False),
    )
    family.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(source)
    return path


def main() -> None:
    root = ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", help="Evidence IDs, such as P014; omit to run all experiments.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="List stable IDs and their scripts without training.")
    mode.add_argument("--new", metavar="CATEGORY/NAME", help="Create a new experiment with the next stable ID.")
    parser.add_argument("--title", help="New experiment's question; defaults to its filename.")
    parser.add_argument("--group", help="New experiment's documentation group; inferred from its category.")
    parser.add_argument("--id", dest="identifier", help="Explicit new P identifier instead of the next available ID.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, help="Directory for independent JSON records.")
    parser.add_argument("--wheel", type=Path, help="Exact relflow wheel to use instead of the published package.")
    options = parser.parse_args()
    if options.new is not None:
        if (
            options.experiments
            or options.steps is not None
            or options.accelerator != "cpu"
            or options.threads != 1
            or options.output_dir is not None
            or options.wheel is not None
        ):
            parser.error("--new creates a source file; select experiments and execution options in a separate command")
        try:
            path = create(
                root,
                options.new,
                title=options.title,
                group=options.group,
                identifier=options.identifier,
                seed=options.seed if options.seed is not None else 0,
            )
        except (ValueError, FileExistsError) as error:
            parser.error(str(error))
        identifier = describe(path.read_text())["evidence-id"]
        print(f"Created {identifier}: {path.relative_to(root.parent)}")
        print("Fill in the question, independent data splits, model, matched control, metrics, and checks.")
        return
    if any(value is not None for value in (options.title, options.group, options.identifier)):
        parser.error("--title, --group, and --id apply only to --new")
    entries = registry(root, measurements=False)
    selected = options.experiments or list(entries)
    unknown = sorted(set(selected) - entries.keys())
    if unknown:
        parser.error(f"Unknown evidence IDs: {', '.join(unknown)}. Use --list to see every experiment.")
    if options.list:
        for identifier in selected:
            print(f"{identifier}  {entries[identifier]['script']}")
        return

    wheel = options.wheel.expanduser().resolve() if options.wheel else None
    if wheel is not None and (not wheel.is_file() or not re.fullmatch(r"relflow-.+\.whl", wheel.name)):
        parser.error(f"--wheel must name an existing relflow wheel; got {wheel}")
    try:
        commands = {identifier: command(root / entries[identifier]["script"], wheel) for identifier in selected}
    except ValueError as error:
        parser.error(str(error))
    arguments = []
    for name in ("seed", "steps", "accelerator", "threads"):
        value = getattr(options, name)
        if value is not None:
            arguments.extend([f"--{name}", str(value)])
    errors = []
    process_environment = os.environ.copy()
    # Inherited import paths can silently select an editable checkout over a wheel.
    process_environment.pop("PYTHONPATH", None)
    process_environment.pop("PYTHONHOME", None)
    process_environment.pop("VIRTUAL_ENV", None)
    if wheel is not None:
        process_environment["EVIDENCE_WHEEL_SHA256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    for identifier in selected:
        print(f"\nRunning {identifier}: {entries[identifier]['script']}", flush=True)
        output = (
            ["--output", str(options.output_dir / f"{identifier}-{time.time_ns()}.json")] if options.output_dir else []
        )
        result = subprocess.run(
            [*commands[identifier], *arguments, *output],
            check=False,
            env=process_environment,
        )
        if result.returncode:
            errors.append(identifier)
    if errors:
        print(f"Execution errors: {', '.join(errors)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
