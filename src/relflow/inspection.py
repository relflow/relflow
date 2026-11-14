"""Inspect declarations and check data through the existing schema contracts."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import torch

from relflow._version import __version__
from relflow.architecture import template
from relflow.architecture.checkpoint import CheckpointState
from relflow.data.processors import Preprocessor, PreprocessorInput
from relflow.data.ragged import coalesce
from relflow.data.sources import Source, source
from relflow.presets.base import Preset, resolve
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["check", "inspect"]


@dataclass
class Artifact:
    """A validated declaration or CPU checkpoint, before neural graph creation."""

    path: Path
    schema: Schema
    preset: Preset | None
    batch_size: int
    version: str
    state: dict[str, Any] | None = None

    @classmethod
    def read(cls, pathname: str | Path) -> Artifact:
        path = Path(pathname).resolve()
        if path.suffix.lower() in (".yaml", ".yml"):
            options, profile = template.read(path, {})
            batch_size = options.pop("batch_size", 1)
            if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
                raise ValueError(f"model {path}: batch_size must be a positive integer, got {batch_size!r}")
            schema = Schema.from_tree(**options) if profile is None else resolve(profile, **options)
            state = None
            version = __version__
        else:
            state = CheckpointState.read(path)
            schema = Schema.model_validate(state["schema"])
            profile = None if state.get("preset") is None else Preset.model_validate(state["preset"])
            batch_size, version = state["batch_size"], state["version"]
        return cls(path, schema, profile, batch_size, version, state)

    @property
    def resumable(self) -> bool:
        return self.state is not None and "optimizer_states" in self.state and "loops" in self.state

    def build(self) -> Model:
        from relflow.architecture.root import Model

        if self.state is not None:
            return CheckpointState.build(Model, self.state)
        model = Model(schema=self.schema, batch_size=self.batch_size)
        model.preset = self.preset
        return model

    def report(self) -> dict[str, Any]:
        nodes = {**self.schema.branches, **self.schema.requests}
        tensors = (
            []
            if self.state is None
            else [value for value in self.state["state_dict"].values() if isinstance(value, torch.Tensor)]
        )
        return {
            "schema_version": 1,
            "status": "success",
            "model": str(self.path),
            "model_version": self.version,
            "relflow_version": __version__,
            "preset": None if self.preset is None else self.preset.model_dump(mode="json"),
            "batch_size": self.batch_size,
            "schema": self.schema.model_dump(mode="json"),
            "nodes": [
                {"address": str(address), **node.model_dump(mode="json", exclude={"fields"})}
                for address, node in sorted(nodes.items(), key=lambda item: str(item[0]))
            ],
            "objectives": list(map(str, self.schema.objectives)),
            "predictions": list(map(str, self.schema.decodes)),
            "representations": list(map(str, self.schema.embed)),
            "stored_tensors": len(tensors),
            "stored_tensor_bytes": sum(value.numel() * value.element_size() for value in tensors),
            "resumable": self.resumable,
        }


def inspect(pathname: str | Path) -> dict[str, Any]:
    """Describe a YAML template or checkpoint on CPU without building its graph."""
    report = Artifact.read(pathname).report()
    report["command"] = "inspect"
    return report


def check(
    pathname: str | Path,
    *,
    data: Mapping[str, str | Sequence[str] | Source] | None = None,
    execute: bool = False,
    sample: int = 128,
    preprocessor: PreprocessorInput = (),
    seed: int = 0,
) -> dict[str, Any]:
    """Check declarations, sources, and optionally sampled tensorization.

    Execution is opt-in and may initialize optional encoders or disk stores.
    A sample bounds source rows, not nested values or processor expansion.
    The supplied checkpoint is never modified.
    """
    if not isinstance(sample, int) or isinstance(sample, bool) or sample < 1:
        raise ValueError(f"check sample must be a positive integer, got {sample!r}")
    artifact = Artifact.read(pathname)
    report = artifact.report()
    report.update(command="check", phases=[{"phase": "declaration", "status": "passed"}], sources={})
    processors = Preprocessor.normalize(preprocessor)
    model = artifact.build() if execute else None
    for split, location in (data or {}).items():
        strata = Strata.normalize(split)
        selected = location if isinstance(location, Source) else source(location)
        selected.manifest = selected.discover()
        manifest = selected.manifest
        report["sources"][split] = {
            "schema": str(manifest.schema),
            "files": [{"path": item.path, "size": item.size, "modified": item.modified} for item in manifest.files],
        }
        phase: dict[str, Any] = {"phase": "data", "split": split, "status": "passed", "execution": execute}
        if processors and (not execute or any(processor.scope == "dataset" for processor in processors)):
            phase.update(status="skipped", reason="preprocessing requires execution on its declared scope")
        elif not execute:
            # A typed null observation exercises binding without reading source values.
            probe = pa.Table.from_arrays(
                [pa.nulls(1, type=field.type) for field in manifest.schema], schema=manifest.schema
            )
            coalesce(probe, schema=artifact.schema, strata=strata)
            phase.update(
                source_rows=0,
                checks="binding against a typed null observation; source values not tensorized",
                limitations="populated collection paths and datatype semantics require --execute",
            )
        else:
            from relflow.data.datasets.arrow import ArrowDataModule

            batches = []
            rows = 0
            iterator = iter(selected())
            try:
                for batch in iterator:
                    batch = batch.slice(0, sample - rows)
                    batches.append(batch)
                    rows += batch.num_rows
                    if rows >= sample:
                        break
            finally:
                if isinstance(iterator, Generator):
                    iterator.close()
            table = pa.Table.from_batches(batches, schema=manifest.schema)
            assert model is not None
            module = ArrowDataModule(
                model=model,
                preprocessor=processors,
                seed=seed,
                train=table if strata == Strata.train else None,
                validate=table if strata == Strata.validate else None,
                test=table if strata == Strata.test else None,
                predict=table if strata == Strata.predict else None,
            )
            module.setup("fit" if strata in (Strata.train, Strata.validate) else split)
            loader = getattr(module, f"{strata.value if strata != Strata.validate else 'val'}_dataloader")()
            processed = 0
            for encoded in loader:
                processed += encoded.source.num_rows
            phase.update(source_rows=rows, processed_rows=processed, checks="sampled preprocessing and tensorization")
        report["phases"].append(phase)
    return report
