"""Record plain-branch outputs for the golden backwards-compatibility test.

The files record what relflow computed before the branch encoder was split into
stages. Rerun this script only when a change is meant to alter plain-branch
behavior: ``uv run python -m tests.architecture.golden.regenerate``.

The fixtures record the platform and torch version that produced them. Other
platforms (CI on Linux and Windows) or torch versions compare with tolerances
of 1e-5 and 1e-6, so bit-identity is proven only on the recording platform.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pyarrow as pa
import torch
from lightning.pytorch import seed_everything

import relflow as rf
from tests.architecture.test_golden import FIXTURES, batches, platform_key


def atomic(path: Path, write: Callable[[Path], object]) -> None:
    """Write through a temporary file so a crash never leaves a half-written fixture.

    Args:
        path: Final location.
        write: Callable that writes the given temporary path.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    """Record initial parameters, a perturbed checkpoint, outputs and the platform."""
    torch.set_num_threads(1)
    seed_everything(0, workers=True)
    model = rf.Model.from_yaml(FIXTURES / "model.yaml")
    atomic(FIXTURES / "initial.pt", lambda path: torch.save(model.state_dict(), path))

    tables = batches()
    model.encode(pa.concat_tables([tables["mixed"], tables["full"]]), strata="train")
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                parameter.add_(0.05 * noise)
    atomic(FIXTURES / "model.pt", model.save)

    for name, table in tables.items():
        frame: pl.DataFrame = model.predict(table)
        atomic(FIXTURES / f"{name}.arrow", frame.write_ipc)
    atomic(FIXTURES / "meta.json", lambda path: path.write_text(json.dumps(platform_key())))


if __name__ == "__main__":
    main()
