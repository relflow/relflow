"""Exchange a tiny trained checkpoint between package, Python, and OS versions."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch

import relflow as rf


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in ("relflow", "torch", "lightning", "pyarrow", "numpy", "pydantic")},
    }


def compare(expected: Any, actual: Any, path: str = "/") -> None:
    """Compare exact structure and discrete values, allowing CPU floating-point variation."""
    assert type(expected) is type(actual), f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), f"{path}: record fields differ"
        for key in expected:
            compare(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), f"{path}: collection lengths differ"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, f"{path}/{index}")
    elif isinstance(expected, float):
        assert math.isclose(expected, actual, rel_tol=1e-4, abs_tol=1e-5), f"{path}: {expected} != {actual}"
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def snapshot(model: rf.Model) -> dict[str, Any]:
    """Capture extension-owned learned state through the public SDK."""
    return {
        "version": model.version,
        "batch_size": model.batch_size,
        "preset": model.preset.model_dump(mode="json") if model.preset is not None else None,
        "vocabulary": {
            address: {
                "labels": list(rf.Category.vocabulary(model, address)),
                "counts": rf.Category.counts(model, address),
            }
            for address in ("/merchant", "/label", "/items/sku")
        },
        "normalization": {
            address: rf.Number.normalization(model, address)
            for address in ("/amount", "/risk", "/items/total", "/items/adjustments/estimate")
        },
    }


def fit(model: rf.Model, inputs: pa.Table, output: Path, *, checkpoint: Path | None = None) -> None:
    model.fit(
        rf.ArrowDataModule(model=model, train=inputs, shuffle=False),
        batch_size=4,
        learning_rate=1e-3,
        fused=False,
        early_stopping=False,
        rollback=False,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=2 if checkpoint else 1,
        max_steps=3 if checkpoint else 2,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=output,
        ckpt_path=checkpoint,
    )
    assert model.trainer.global_step == (3 if checkpoint else 2)
    state = model.trainer.optimizers[0].state
    assert state, "training did not populate AdamW state"
    assert max(int(values["step"]) for values in state.values()) == model.trainer.global_step, (
        "AdamW update count does not match resumed training progress"
    )


def create(output: Path) -> None:
    output.mkdir(parents=True)
    model = rf.Model.xs(
        d_model=8,
        embed=True,
        amount=rf.Number,
        merchant=rf.Category(p_unavailable=0),
        risk=rf.Number(mask=True),
        label=rf.Category(mask=True, p_unavailable=0, topk=[2]),
        items=rf.Branch(
            length=4,
            embed=True,
            sku=rf.Category(p_unavailable=0),
            total=rf.Number(mask=True),
            adjustments=rf.Branch(length=2, amount=rf.Number, estimate=rf.Number(mask=True)),
        ),
    )
    collections = [
        None,
        [],
        [None],
        [{"sku": "book", "total": 3.0, "adjustments": None}],
        [{"sku": "pen", "total": 2.0, "adjustments": []}],
        [{"sku": "book", "total": 4.0, "adjustments": [None]}],
        [{"sku": "pen", "total": 1.0, "adjustments": [{"amount": 0.5, "estimate": 0.4}]}],
        [None, {"sku": "book", "total": 5.0, "adjustments": [{"amount": 1.5, "estimate": 1.4}, None]}],
    ]
    inputs = pa.Table.from_pylist(
        [
            {
                "amount": float(index + 1),
                "merchant": "shop" if index % 2 else "cafe",
                "risk": index / 10,
                "label": "yes" if index % 2 else "no",
                "items": items,
            }
            for index, items in enumerate(collections)
        ]
    )
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    fit(model, inputs, output)
    assert any(
        name in initial and parameter.shape == initial[name].shape and not torch.equal(parameter, initial[name])
        for name, parameter in model.named_parameters()
    ), "training did not update model parameters"
    model.trainer.save_checkpoint(output / "model.ckpt")
    predictions = model.predict(inputs, retain="*")
    # Null collections, empty collections, and null records remain distinct through both repeated axes.
    rows = predictions["predictions"].to_pylist()
    for source, prediction in zip(collections, rows, strict=True):
        written = prediction["items"]
        assert (source is None) == (written is None)
        if source is not None:
            assert len(source) == len(written)
            for item, result in zip(source, written, strict=True):
                assert (item is None) == (result is None)
                if item is not None:
                    adjustments = item["adjustments"]
                    actual = result["adjustments"]
                    assert (adjustments is None) == (actual is None)
                    if adjustments is not None:
                        assert [value is None for value in adjustments] == [value is None for value in actual]
    for name, table in (("inputs", inputs), ("predictions", predictions)):
        with pa.ipc.new_file(output / f"{name}.arrow", table.schema) as writer:
            writer.write_table(table)
    (output / "schema.json").write_text(model.schema.model_dump_json(indent=2), encoding="utf-8")
    metadata = {"environment": environment(), "model": snapshot(model), "steps": model.trainer.global_step}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"created": str(output), **metadata}, indent=2))


def check(source: Path, output: Path) -> None:
    output.mkdir(parents=True)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    inputs = pa.ipc.open_file(source / "inputs.arrow").read_all()
    expected = pa.ipc.open_file(source / "predictions.arrow").read_all()
    model = rf.Model.load(source / "model.ckpt")
    for loaded in (model, rf.Model.load(model.save(output / "resaved.ckpt"))):
        assert schema == loaded.schema.model_dump(mode="json"), "checkpoint schema changed"
        compare(metadata["model"], snapshot(loaded), "/model")
        actual = loaded.predict(inputs, retain="*")
        assert expected.schema.equals(actual.schema, check_metadata=True), "prediction Arrow schemas differ"
        compare(expected.to_pylist(), actual.to_pylist(), "/rows")
    state = torch.load(source / "model.ckpt", weights_only=False, map_location="cpu")
    assert state["global_step"] == metadata["steps"] == 2
    assert len(state["optimizer_states"]) == 1 and state["optimizer_states"][0]["state"]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    fit(model, inputs, output, checkpoint=source / "model.ckpt")
    assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters()), (
        "resumed training did not update parameters"
    )
    assert model.version == metadata["model"]["version"], "resume changed checkpoint provenance"
    report = {"producer": metadata["environment"], "consumer": environment(), "steps_after_resume": 3}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create").add_argument("--output", type=Path, required=True)
    consumer = commands.add_parser("check")
    consumer.add_argument("--input", type=Path, required=True)
    consumer.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    random.seed(1729)
    np.random.seed(1729)
    if arguments.command == "create":
        create(arguments.output)
    else:
        check(arguments.input, arguments.output)
