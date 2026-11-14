"""Shell workflows over relflow's model and file APIs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import lightning.pytorch as lit
import typer

from relflow._version import __version__
from relflow.cli.config import Format, Level, Stage, location, processor, read
from relflow.cli.jobs import Job, digest, snapshot
from relflow.cli.terminal import report
from relflow.data.datasets.arrow import ArrowDataModule
from relflow.data.processors import Postprocessor, Preprocessor
from relflow.data.sources import source
from relflow.inference.callback import Writer
from relflow.inspection import Artifact
from relflow.inspection import check as check_model
from relflow.inspection import inspect as inspect_model
from relflow.structs.enums import Strata

__all__ = ["app"]

app = typer.Typer(
    name="relflow",
    help="Inspect, check, predict, and serve relflow models.",
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
    version: Annotated[
        bool, typer.Option("--version", is_eager=True, help="Show the installed relflow version.")
    ] = False,
) -> None:
    if version:
        typer.echo(f"relflow {__version__}")
        raise typer.Exit()


@app.command()
def inspect(
    model: Model,
    config: Config = None,
    imports: Imports = None,
    json: Json = False,
    debug: Debug = False,
) -> None:
    """Inspect a model's schema and CPU checkpoint metadata without building its graph."""
    with report("inspect", json=json, debug=debug) as result:
        run = read(config)
        run.register(imports)
        result.update(inspect_model(model))


@app.command()
def check(
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
    with report("check", json=json, debug=debug) as result:
        if sample is not None and not execute:
            raise ValueError("--sample requires --execute; sampled checks can initialize encoders and stores")
        run = read(config)
        if input is not None:
            run.data = {stage: location(input, Path.cwd())}
        run.register(imports)
        result.update(
            check_model(
                model,
                data={str(split): value for split, value in run.data.items()},
                execute=execute,
                sample=128 if sample is None else sample,
                preprocessor=processor(run.preprocessor, Preprocessor),
                seed=run.seed,
            )
        )


@app.command()
def predict(
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
    with report("predict", json=json, debug=debug) as result:
        run = read(config)
        run.predict = run.predict.override(
            batch_size=batch_size,
            accelerator=accelerator,
            precision=precision,
            retain=retain,
            format=format,
        )
        if input is not None:
            run.data["predict"] = location(input, Path.cwd())
        if "predict" not in run.data:
            raise ValueError("predict requires data.predict in --config or an explicit --input")
        run.register(imports)
        selected = source(run.data["predict"])
        files = snapshot({Strata.predict: selected})
        artifact = Artifact.read(model)
        if artifact.state is None:
            raise ValueError(f"predict requires a learned checkpoint, got template {model}")
        options = run.predict
        data = ArrowDataModule(
            predict=selected,
            **run.loader.model_dump(),
            seed=run.seed,
            preprocessor=processor(run.preprocessor, Preprocessor),
            retain=tuple(options.retain),
        )
        postprocess = processor(run.postprocessor, Postprocessor)
        configuration = {
            "model": str(artifact.path),
            "model_sha256": digest(artifact.path),
            "run": run.model_dump(mode="json"),
        }
        with Job(output.resolve(), configuration, result) as job:
            job.write("sources.json", files)
            result.update(
                seed=run.seed, model_source=str(artifact.path), model_source_sha256=configuration["model_sha256"]
            )
            lit.seed_everything(run.seed, workers=True)
            instance = artifact.build()
            writer = Writer(
                job.path, format=options.format, postprocessor=postprocess, row_group_size=options.row_group_size
            )
            execution = options.model_dump(include={"accelerator", "precision"}, exclude_none=True)
            instance.predict(
                datamodule=data,
                batch_size=options.batch_size,
                **execution,
                devices=1,
                enable_progress_bar=False,
                callbacks=[writer],
                return_predictions=False,
            )
            if instance.trainer.interrupted:
                raise KeyboardInterrupt
            writer.finish(instance, data.schemas[Strata.predict].processed, retain=tuple(options.retain))
            job.complete(
                processed_rows=writer.source_rows,
                written_rows=writer.written_rows,
                format=options.format,
                shards=[{"path": str(path), "size": path.stat().st_size} for path in sorted(job.path.glob("rank-*.*"))],
                accelerator=type(instance.trainer.accelerator).__name__,
                precision=instance.trainer.precision,
                devices=1,
            )


@app.command()
def serve(
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
    with report("serve", json=json, debug=debug) as result:
        run = read(config)
        run.serve = run.serve.override(
            host=host,
            port=port,
            accelerator=accelerator,
            retain=retain,
            max_batch_size=max_batch_size,
            batch_timeout=batch_timeout,
            max_queue_size=max_queue_size,
            log_level=log_level,
        )
        if model.suffix.lower() in (".yaml", ".yml"):
            raise ValueError(f"serve requires a learned checkpoint, got template {model}")
        run.register(imports)
        from relflow.inference import Deployment

        options = run.serve.model_dump(exclude_none=True)
        options.setdefault("host", os.environ.get("RELFLOW_HOST", "127.0.0.1"))
        if run.serve.retain is not None:
            options["retain"] = tuple(run.serve.retain)
        deployment = Deployment(checkpoint=model.resolve(), workers=1, **options)
        deployment.preprocess(processor(run.preprocessor, Preprocessor))
        deployment.postprocess(processor(run.postprocessor, Postprocessor))
        deployment.serve()
        result["status"] = "success"
