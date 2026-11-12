"""Exercise an installed wheel with absent, incomplete, and complete plotting stacks."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import sysconfig
import tempfile
from pathlib import Path


def smoke(plotting: str) -> None:
    """Check optional imports before a training/checkpoint/prediction round trip."""
    packages = {}
    for name in ("relflow", "lightning", "torchmetrics", "lightning-utilities", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    spec = importlib.util.find_spec("matplotlib")
    print(
        json.dumps(
            {
                "python": sys.version,
                "packages": packages,
                "matplotlib_origin": spec.origin if spec else None,
                "matplotlib_paths": list(spec.submodule_search_locations or ()) if spec else [],
            }
        ),
        flush=True,
    )

    if plotting == "namespace":
        assert packages["matplotlib"] is None, "Run the namespace reproduction without Matplotlib installed"
        assert spec is not None and spec.origin is None, "Expected an incomplete namespace package"
        from lightning_utilities.core.imports import RequirementCache

        assert bool(RequirementCache("matplotlib")), "Upstream no longer misidentifies this namespace package"
        try:
            import relflow  # noqa: F401
        except ModuleNotFoundError as error:
            assert error.name == "matplotlib.axes", str(error)
        else:
            raise AssertionError("The upstream namespace-package failure no longer reproduces; review the remedy")
        print("Reproduced the upstream matplotlib.axes failure", flush=True)
        return

    if plotting == "absent":
        assert packages["matplotlib"] is None and spec is None, "Expected no Matplotlib installation or namespace"
    else:
        assert packages["matplotlib"] is not None, "Expected a complete Matplotlib installation"

    import pyarrow as pa
    import torch

    import relflow as rf

    assert Path(rf.__file__).resolve().is_relative_to(Path(sysconfig.get_paths()["purelib"]).resolve())
    print(json.dumps({"relflow_file": rf.__file__}), flush=True)
    torch.set_num_threads(1)
    model = rf.Model(preset="small", d_model=8, value=rf.Number, target=rf.Boolean(mask=True))
    rows = pa.table({"value": [1.0, 2.0], "target": [False, True]})
    data = rf.ArrowDataModule(train=rows, validate=rows, num_workers=0, persistent_workers=False, pin_memory=False)
    model.fit(
        data,
        batch_size=2,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        rollback=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    expected = model.predict(rows)
    with tempfile.TemporaryDirectory() as directory:
        restored = rf.Model.load(model.save(Path(directory) / "model.ckpt"))
        assert restored.predict(rows).equals(expected)
    if plotting == "installed":
        import matplotlib.pyplot as plt
        from torchmetrics.classification import BinaryAccuracy

        metric = BinaryAccuracy()
        metric.update(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
        figure, _ = metric.plot()
        plt.close(figure)
    else:
        assert "matplotlib" not in sys.modules, "Core modeling unexpectedly imported Matplotlib"
    print(f"Import, training, checkpoint, and prediction passed ({plotting=})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plotting", choices=("absent", "namespace", "installed"), required=True)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.plotting == "namespace":
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "matplotlib").mkdir()
            sys.path.insert(0, directory)
            smoke(args.plotting)
    else:
        smoke(args.plotting)
