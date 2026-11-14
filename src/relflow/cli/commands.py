"""Typed command declarations, generated help, and native shell completion."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from relflow._version import __version__
from relflow.cli.config import Format, Level, Stage
from relflow.cli.workflows import run

__all__ = ["app"]

app = typer.Typer(
    name="relflow",
    help="Inspect, check, train, predict, and serve relflow models.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)

Model = Annotated[Path, typer.Argument(help="Model architecture YAML or learned checkpoint.")]
Config = Annotated[Path | None, typer.Option(help="Version 1 run YAML; data paths are relative to this file.")]
Imports = Annotated[
    list[str] | None, typer.Option("--import", help="Import an installed extension module; repeat for more.")
]
Json = Annotated[bool, typer.Option("--json", help="Write one machine-readable report to stdout.")]
Debug = Annotated[bool, typer.Option(help="Include failure tracebacks on stderr.")]
Output = Annotated[Path, typer.Option(help="Fresh local output directory; existing paths are rejected.")]
Input = Annotated[str | None, typer.Option(help="Input file, directory, glob, or filesystem URI.")]
Batch = Annotated[int | None, typer.Option("--batch-size", min=1, help="Observations per model batch.")]
Accelerator = Annotated[str | None, typer.Option(help="Execution accelerator, such as auto, cpu, cuda, or mps.")]
Precision = Annotated[str | None, typer.Option(help="Lightning execution precision, such as 32-true.")]
Retain = Annotated[list[str] | None, typer.Option(help="Retain a processed input column; repeat for more.")]


@app.callback(invoke_without_command=True)
def root(
    context: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", is_eager=True, help="Show the installed relflow version.")
    ] = False,
) -> None:
    if context.resilient_parsing:
        return
    if version:
        typer.echo(f"relflow {__version__}")
        raise typer.Exit()
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())
        raise typer.Exit(2)


def dispatch(context: typer.Context) -> None:
    """Only explicit CLI parameters override run files, including false flags."""
    context.ensure_object(dict)["executed"] = True
    arguments = {}
    for name, value in context.params.items():
        source = context.get_parameter_source(name)
        if source is not None and source.name == "COMMANDLINE":
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Path):
                value = str(value)
            arguments[name] = value
    arguments["command"] = context.info_name
    raise typer.Exit(run(arguments))


@app.command()
def inspect(
    context: typer.Context,
    model: Model,
    config: Config = None,
    imports: Imports = None,
    json: Json = False,
    debug: Debug = False,
) -> None:
    """Inspect a model's schema and CPU checkpoint metadata without building its graph."""
    dispatch(context)


@app.command()
def check(
    context: typer.Context,
    model: Model,
    config: Config = None,
    imports: Imports = None,
    input: Input = None,
    stage: Annotated[Stage, typer.Option(help="Data split contract used with --input.")] = "predict",
    execute: Annotated[bool, typer.Option(help="Opt into sampled preprocessing and tensorization.")] = False,
    sample: Annotated[int | None, typer.Option(min=1, help="Source-row cap; requires --execute (default 128).")] = None,
    json: Json = False,
    debug: Debug = False,
) -> None:
    """Check declarations and sources, optionally executing sampled tensorization."""
    dispatch(context)


@app.command()
def train(
    context: typer.Context,
    output: Output,
    model: Annotated[Path | None, typer.Argument(help="Architecture YAML or a learned artifact.")] = None,
    config: Config = None,
    imports: Imports = None,
    resume: Annotated[Path | None, typer.Option(help="Resume the full last checkpoint of an existing CLI run.")] = None,
    train: Annotated[str | None, typer.Option(help="Training files, overriding data.train.")] = None,
    validate: Annotated[str | None, typer.Option(help="Validation files, overriding data.validate.")] = None,
    batch_size: Batch = None,
    accelerator: Accelerator = None,
    precision: Precision = None,
    learning_rate: Annotated[float | None, typer.Option(help="Positive AdamW learning rate.")] = None,
    weight_decay: Annotated[float | None, typer.Option(min=0, help="AdamW weight decay.")] = None,
    max_epochs: Annotated[int | None, typer.Option(min=1, help="Maximum training epochs.")] = None,
    max_steps: Annotated[int | None, typer.Option(min=1, help="Maximum optimizer steps.")] = None,
    early_stopping: Annotated[bool | None, typer.Option("--early-stopping/--no-early-stopping")] = None,
    rollback: Annotated[bool | None, typer.Option("--rollback/--no-rollback")] = None,
    monitor: Annotated[str | None, typer.Option(help="Metric used for stopping and model selection.")] = None,
    patience: Annotated[int | None, typer.Option(min=0, help="Checks without improvement before stopping.")] = None,
    seed: Annotated[int | None, typer.Option(min=0, max=4294967295, help="Model and data seed.")] = None,
    json: Json = False,
    debug: Debug = False,
) -> None:
    """Train from a template, adapt an artifact, or resume a full checkpoint."""
    dispatch(context)


@app.command()
def predict(
    context: typer.Context,
    model: Model,
    output: Output,
    config: Config = None,
    imports: Imports = None,
    input: Input = None,
    batch_size: Batch = None,
    accelerator: Accelerator = None,
    precision: Precision = None,
    retain: Retain = None,
    format: Annotated[Format | None, typer.Option(help="Stream Parquet or Arrow IPC files.")] = None,
    json: Json = False,
    debug: Debug = False,
) -> None:
    """Stream file-backed predictions without collecting the full result in memory."""
    dispatch(context)


@app.command()
def serve(
    context: typer.Context,
    model: Model,
    config: Config = None,
    imports: Imports = None,
    host: Annotated[str | None, typer.Option(help="HTTP bind address; defaults to loopback.")] = None,
    port: Annotated[int | None, typer.Option(min=1, max=65535, help="HTTP listen port.")] = None,
    accelerator: Accelerator = None,
    retain: Retain = None,
    max_batch_size: Annotated[int | None, typer.Option(min=1, help="Maximum serving batch size.")] = None,
    batch_timeout: Annotated[float | None, typer.Option(min=0, help="Batch collection timeout in seconds.")] = None,
    max_queue_size: Annotated[int | None, typer.Option(min=1, help="Maximum queued requests.")] = None,
    log_level: Annotated[Level | None, typer.Option(help="HTTP runtime log level.")] = None,
    json: Json = False,
    debug: Debug = False,
) -> None:
    """Start the optional HTTP runtime for a saved model."""
    dispatch(context)
