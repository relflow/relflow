"""CLI output, diagnostics, and Rich report rendering."""

from __future__ import annotations

import json as jsonlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, redirect_stdout
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.filesize import decimal
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from relflow.logging import configure
from relflow.structs.tree import Address

__all__ = ["report", "render"]


@contextmanager
def report(command: str, *, json: bool = False, debug: bool = False) -> Iterator[dict[str, Any]]:
    """Keep backend output on stderr and emit one terminal result per command."""
    result: dict[str, Any] = {"schema_version": 1, "command": command, "status": "running"}
    console = Console(markup=False, highlight=False)
    errors = Console(stderr=True, markup=False, highlight=False)
    configure(level="debug" if debug else "warning", output=sys.stderr)
    activity = (
        errors.status(Text(f"Running {command}…"))
        if errors.is_terminal and not json and command != "serve"
        else nullcontext()
    )
    try:
        with activity, redirect_stdout(sys.stderr):
            yield result
    except (Exception, KeyboardInterrupt) as error:
        # Job has already recorded failures that occurred after output reservation.
        recorded = result["status"] == "failed"
        result.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        if json:
            print(jsonlib.dumps(result, allow_nan=False))
        errors.print(Text.assemble((f"relflow {command}: ", "bold red"), str(error) or "interrupted"))
        if debug:
            errors.print_exception(show_locals=False)
        if isinstance(error, KeyboardInterrupt):
            raise typer.Exit(130) from error
        invalid = isinstance(
            error, (ValueError, TypeError, FileNotFoundError, FileExistsError, ImportError, yaml.YAMLError)
        )
        raise typer.Exit(2 if invalid and not recorded else 1) from error
    if json:
        print(jsonlib.dumps(result, allow_nan=False))
    else:
        render(result, console)


def model_tree(definition: dict[str, Any], parents: tuple[str, ...], report: dict[str, Any]) -> Tree:
    """Follow schema children, preserving names that contain address punctuation."""
    address = str(Address(*parents))
    label = Text(address, style="cyan")
    label.append("  " + definition["type"], style="dim")
    if address in report["objectives"]:
        label.append("  target", style="yellow")
    if address in report["representations"]:
        label.append("  embedding", style="green")
    if definition.get("query") is not None:
        label.append("  query=" + definition["query"])
    tree = Tree(label, guide_style="dim")
    if definition["type"] == "branch":
        length = definition["length"]
        label.append(f"  length={length if length is not None else 'unbounded'}", style="dim")
        for child in definition["fields"]:
            tree.add(model_tree(child, (*parents, child["name"]), report))
    return tree


def render(report: dict[str, Any], console: Console) -> None:
    """Present operational outcomes; the versioned JSON report stays unchanged."""
    command = report["command"]
    if command == "inspect":
        console.print(Text.assemble(("Model ", "bold"), report["model"]))
        details = Table.grid(padding=(0, 2))
        details.add_row("relflow version", report["model_version"])
        if report["preset"] is not None:
            details.add_row("Preset", report["preset"]["name"])
        details.add_row("Stored tensors", f"{report['stored_tensors']:,}")
        details.add_row("Tensor storage", decimal(report["stored_tensor_bytes"]))
        console.print(details)
        console.print(model_tree(report["schema"]["fields"], (), report))
        return
    if command == "check":
        console.print(Text.assemble(("Preflight ", "bold"), report["model"]))
        phases = Table("Phase", "Split", "Result", "Rows", "Detail", expand=True)
        for phase in report["phases"]:
            status = phase["status"]
            detail = "\n".join(str(phase[key]) for key in ("checks", "reason", "limitations") if key in phase)
            rows = str(phase["source_rows"]) if "source_rows" in phase else "—"
            if "processed_rows" in phase:
                rows += f" → {phase['processed_rows']}"
            phases.add_row(
                Text(phase["phase"]),
                Text(phase.get("split", "—")),
                Text(status, style={"passed": "green", "failed": "red", "skipped": "yellow"}[status]),
                Text(rows),
                Text(detail),
            )
        console.print(phases)
        return
    title, fields = {
        "train": (
            "Training complete",
            (
                ("Model", "model"),
                ("Output", "output"),
                ("Last checkpoint", "last_checkpoint"),
                ("Best checkpoint", "best_checkpoint"),
                ("Optimizer steps", "global_step"),
                ("Epoch", "epoch"),
                ("Accelerator", "accelerator"),
                ("Precision", "precision"),
            ),
        ),
        "predict": (
            "Prediction complete",
            (
                ("Output", "output"),
                ("Processed rows", "processed_rows"),
                ("Written rows", "written_rows"),
                ("Format", "format"),
                ("Accelerator", "accelerator"),
                ("Precision", "precision"),
            ),
        ),
        "serve": ("Server stopped", ()),
    }[command]
    console.print(Text(title, style="bold green"))
    details = Table.grid(padding=(0, 2))
    for label, key in fields:
        value = report.get(key)
        details.add_row(label, Text(str(value) if value is not None else "—"))
    console.print(details)
    if report.get("metrics"):
        metrics = Table("Metric", "Value")
        for name, value in report["metrics"].items():
            metrics.add_row(Text(name), "—" if value is None else f"{value:.6g}")
        console.print(metrics)
