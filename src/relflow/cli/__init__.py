"""Shell entry points for relflow model and file workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import lightning.pytorch as lit
import pyarrow as pa
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from relflow._version import __version__
from relflow.architecture.output import write
from relflow.architecture.template import Loader
from relflow.cli.config import Fit, Predict, Run, Serve, location, processor, read
from relflow.cli.jobs import ExecutionError, Job, digest, manifest
from relflow.data.datasets.arrow import ArrowDataModule
from relflow.data.processors import Postprocessor, Preprocessor
from relflow.data.sources import source
from relflow.inference.callback import Writer
from relflow.inspection import Artifact, check, inspect
from relflow.logging import configure
from relflow.structs.enums import Strata

__all__ = ["main"]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="relflow", description="Inspect, check, train, predict, and serve relflow models."
    )
    root.add_argument("--version", action="version", version=f"relflow {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("inspect", "check", "train", "predict", "serve"):
        command = commands.add_parser(name, argument_default=argparse.SUPPRESS)
        command.add_argument("model", nargs="?" if name == "train" else None)
        command.add_argument("--config", help="Version 1 run YAML; local data paths are relative to this file.")
        command.add_argument("--import", dest="imports", action="append", help="Import an installed extension module.")
        command.add_argument("--json", action="store_true", help="Write one machine-readable report to stdout.")
        command.add_argument("--debug", action="store_true", help="Include failure tracebacks on stderr.")
        if name in ("check", "predict"):
            command.add_argument("--input", help="Input file, directory, glob, or filesystem URI.")
        if name == "check":
            command.add_argument("--stage", choices=("train", "validate", "test", "predict"), default="predict")
            command.add_argument(
                "--execute", action="store_true", help="Opt into sampled preprocessing and tensorization."
            )
            command.add_argument(
                "--sample", type=int, help="Positive source-row cap; requires --execute (default 128)."
            )
        if name == "train":
            command.add_argument("--resume", help="Resume the full last checkpoint of an existing CLI run.")
            command.add_argument("--train", help="Training files, overriding data.train.")
            command.add_argument("--validate", help="Validation files, overriding data.validate.")
            for option in ("max-epochs", "max-steps", "patience"):
                command.add_argument(f"--{option}", type=int)
            for option in ("learning-rate", "weight-decay"):
                command.add_argument(f"--{option}", type=float)
            for option in ("early-stopping", "rollback"):
                command.add_argument(f"--{option}", action=argparse.BooleanOptionalAction)
            command.add_argument("--monitor")
            command.add_argument("--seed", type=int)
        if name in ("train", "predict"):
            command.add_argument(
                "--output", required=True, help="Fresh local output directory; existing paths are rejected."
            )
            command.add_argument("--batch-size", type=int)
            command.add_argument("--precision")
        if name in ("train", "predict", "serve"):
            command.add_argument("--accelerator")
        if name == "predict":
            command.add_argument("--format", choices=("parquet", "ipc"))
        if name in ("predict", "serve"):
            command.add_argument("--retain", action="append", help="Retain a processed input column; repeat for more.")
        if name == "serve":
            command.add_argument("--host")
            command.add_argument("--port", type=int)
            command.add_argument("--max-batch-size", type=int)
            command.add_argument("--batch-timeout", type=float)
            command.add_argument("--max-queue-size", type=int)
            command.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug", "trace"))
    return root


def settings(arguments: dict[str, Any]) -> tuple[Run, str, str | None]:
    """Resolve file paths and explicit flags before importing application code."""
    checkpoint = None
    if arguments.get("resume"):
        allowed = {"command", "resume", "output", "max_epochs", "json", "debug"}
        forbidden = {key for key, value in arguments.items() if key not in allowed and value is not None}
        if forbidden:
            raise ValueError(f"resume cannot replace its model, data, or runtime settings: {sorted(forbidden)}")
        parent = Path(arguments["resume"]).resolve()
        resolved = yaml.load((parent / "resolved.yaml").read_text(encoding="utf-8"), Loader=Loader)
        previous = json.loads((parent / "report.json").read_text(encoding="utf-8"))
        checkpoint = previous.get("last_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError(f"run {parent}: no full last checkpoint is available to resume")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "optimizer_states" not in state or "loops" not in state:
            raise ValueError(f"checkpoint {checkpoint}: resume requires optimizer and loop state")
        run = Run.model_validate(resolved["run"])
        model = checkpoint
    else:
        run = read(arguments.get("config"))
        model = arguments.get("model")
        if model is None:
            raise ValueError("train requires a model template or artifact, or --resume RUN_DIRECTORY")
    payload = run.model_dump(mode="python")
    command = arguments["command"]
    section = {"train": "fit", "predict": "predict", "serve": "serve"}.get(command)
    if section:
        kind = {"fit": Fit, "predict": Predict, "serve": Serve}[section]
        for key in kind.model_fields:
            if key in arguments:
                payload[section][key] = arguments[key]
    for split in ("train", "validate"):
        if split in arguments:
            payload["data"][split] = location(arguments[split], Path.cwd())
    if "input" in arguments:
        split = arguments.get("stage", "predict")
        if command == "check":
            payload["data"] = {}
        payload["data"][split] = location(arguments["input"], Path.cwd())
    if "seed" in arguments:
        payload["seed"] = arguments["seed"]
    payload["imports"].extend(arguments.get("imports", []))
    run = Run.model_validate(payload)
    return run, str(Path(model).resolve()), checkpoint


def train(model, data: ArrowDataModule, run: Run, job: Job, checkpoint: str | None) -> dict[str, Any]:
    options = run.fit.model_dump(exclude_none=True)
    callbacks = []
    if not options["rollback"]:
        callbacks.append(ModelCheckpoint(dirpath=job.path / "checkpoints", save_last=True, save_top_k=0))
    job.report.update(last_checkpoint=str(job.path / "checkpoints" / "last.ckpt"))
    job.write("report.json", job.report)
    model.fit(
        data,
        **options,
        devices=1,
        enable_progress_bar=False,
        checkpoint_dir=job.path / "checkpoints",
        ckpt_path=checkpoint,
        callbacks=callbacks,
        default_root_dir=str(job.path),
    )
    if model.trainer.interrupted:
        raise KeyboardInterrupt
    model.save(job.path / "model.rf")
    trainer = model.trainer
    last = next(
        (callback.last_model_path for callback in trainer.checkpoint_callbacks if callback.last_model_path), None
    )
    best = next(
        (callback.best_model_path for callback in trainer.checkpoint_callbacks if callback.best_model_path), None
    )
    metrics = {}
    for name, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            number = value.item()
            metrics[name] = number if math.isfinite(number) else None
    job.write("sources.json", manifest(data.sources))
    return job.complete(
        model=str(job.path / "model.rf"),
        model_sha256=digest(job.path / "model.rf"),
        last_checkpoint=last,
        best_checkpoint=best,
        parent_checkpoint=checkpoint,
        global_step=trainer.global_step,
        epoch=trainer.current_epoch,
        metrics=metrics,
        accelerator=type(trainer.accelerator).__name__,
        precision=trainer.precision,
        devices=1,
    )


def predict(model, data: ArrowDataModule, run: Run, job: Job, postprocessor) -> dict[str, Any]:
    options = run.predict
    writer = Writer(job.path, format=options.format, postprocessor=postprocessor, row_group_size=options.row_group_size)
    execution: dict[str, Any] = {"accelerator": options.accelerator, "devices": 1, "enable_progress_bar": False}
    if options.precision is not None:
        execution["precision"] = options.precision
    model.predict(
        datamodule=data, batch_size=options.batch_size, callbacks=[writer], return_predictions=False, **execution
    )
    if model.trainer.interrupted:
        raise KeyboardInterrupt
    shards = sorted(job.path.glob("rank-*.*"))
    if not shards:
        schema = data.schemas[Strata.predict].processed
        if schema is None:
            raise RuntimeError("prediction produced no processed schema; no complete job was published")
        empty = write(model, [], source=pa.Table.from_batches([], schema=schema), retain=tuple(options.retain))
        writer.write_on_batch_end(model.trainer, model, empty, None, None, 0, 0)
        writer.close()
        shards = sorted(job.path.glob("rank-*.*"))
    job.write("sources.json", manifest(data.sources))
    return job.complete(
        processed_rows=writer.source_rows,
        written_rows=writer.written_rows,
        format=options.format,
        shards=[{"path": str(path), "size": path.stat().st_size} for path in shards],
        accelerator=type(model.trainer.accelerator).__name__,
        precision=model.trainer.precision,
        devices=1,
    )


def execute(arguments: dict[str, Any], run: Run, pathname: str, checkpoint: str | None) -> dict[str, Any]:
    command = arguments["command"]
    for module in run.imports:
        importlib.import_module(module)
    pre = processor(run.preprocessor, Preprocessor)
    post = processor(run.postprocessor, Postprocessor)
    if command == "train" or (
        command == "check" and "config" in arguments and "input" not in arguments and "train" in run.data
    ):
        run.check_training()
    if command == "inspect":
        return inspect(pathname)
    if command == "check":
        if "sample" in arguments and not arguments.get("execute"):
            raise ValueError("--sample requires --execute; sampled checks can initialize encoders and stores")
        return check(
            pathname,
            data={str(key): value for key, value in run.data.items()},
            execute=arguments.get("execute", False),
            sample=arguments.get("sample", 128),
            preprocessor=pre,
            seed=run.seed,
        )
    artifact = Artifact.read(pathname)
    if command != "train" and artifact.state is None:
        raise ValueError(f"{command} requires a learned checkpoint, got template {pathname}")
    if command == "serve":
        try:
            from relflow.inference.deployment import Deployment
        except ModuleNotFoundError as error:
            if error.name not in ("fastapi", "orjson", "pydantic_settings", "uvicorn"):
                raise
            raise ImportError(
                "serving requires relflow[serving]; install with: uv pip install 'relflow[serving]'"
            ) from error
        options = run.serve.model_dump(exclude_none=True)
        options.setdefault("host", os.environ.get("RELFLOW_HOST", "127.0.0.1"))
        if "retain" in options:
            options["retain"] = tuple(options["retain"])
        deployment = Deployment(checkpoint=pathname, workers=1, **options)
        deployment.preprocess(pre).postprocess(post).serve()
        return {"schema_version": 1, "command": "serve", "status": "success"}
    selected = {
        split: source(value)
        for split, value in run.data.items()
        if split in (("train", "validate") if command == "train" else ("predict",))
    }
    required = "train" if command == "train" else "predict"
    if required not in selected:
        raise ValueError(
            f"{command} requires data.{required} in --config or an explicit --{required if required == 'train' else 'input'}"
        )
    sources = manifest(selected)
    output = Path(arguments["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"output directory {output} already exists; choose a fresh directory")
    data = ArrowDataModule(
        train=selected.get("train"),
        validate=selected.get("validate"),
        predict=selected.get("predict"),
        **run.loader.model_dump(exclude_none=True),
        seed=run.seed,
        preprocessor=pre,
        retain=tuple(run.predict.retain) if command == "predict" else (),
    )
    configuration = {"model": pathname, "model_sha256": digest(pathname), "run": run.model_dump(mode="json")}
    with Job(output, command, configuration) as job:
        job.report.update(seed=run.seed, model_source=pathname, model_source_sha256=configuration["model_sha256"])
        job.write("sources.json", sources)
        lit.seed_everything(run.seed, workers=True)
        model = artifact.build()
        if command == "train":
            return train(model, data, run, job, checkpoint)
        return predict(model, data, run, job, post)


def main(argv: list[str] | None = None) -> int:
    """Run one command, keeping backend output off machine-readable stdout."""
    arguments = vars(parser().parse_args(argv))
    configure(level="debug" if arguments.get("debug") else "warning", output=sys.stderr)
    try:
        with redirect_stdout(sys.stderr):
            run, pathname, checkpoint = settings(arguments)
            report = execute(arguments, run, pathname, checkpoint)
    except KeyboardInterrupt:
        print("relflow: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        code = (
            2
            if isinstance(
                error, (ValueError, TypeError, FileNotFoundError, FileExistsError, ImportError, yaml.YAMLError)
            )
            else 1
        )
        report = {
            "schema_version": 1,
            "command": arguments["command"],
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        if isinstance(error, ExecutionError):
            report = error.report
        if arguments.get("json"):
            print(json.dumps(report, allow_nan=False))
        print(f"relflow {arguments['command']}: {error}", file=sys.stderr)
        if arguments.get("debug"):
            traceback.print_exc(file=sys.stderr)
        return code
    if arguments.get("json"):
        print(json.dumps(report, allow_nan=False))
    elif arguments["command"] == "inspect":
        print(f"Model: {report['model']} (relflow {report['model_version']})")
        for node in report["nodes"]:
            print(f"  {node['address']}: {node['type']}")
        print("Objectives: " + ", ".join(report["objectives"]))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    return 0
