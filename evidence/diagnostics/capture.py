"""Retain proof readouts without changing their source, budgets, or predictions.

The paired Evidence worker invokes this recorder around the canonical reporter.
CPython return events expose completed proof calculations; capture performs no
additional model calls, random draws, training, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import sys
from datetime import date, datetime
from pathlib import Path
from types import CodeType

FUNCTIONS = {
    "normalized_rmse",
    "rmse",
    "error",
    "errors",
    "raw_features",
    "raw",
    "least_squares",
    "operating",
    "run",
    "trial",
    "fit",
    "score",
    "scores",
    "evaluate",
    "measure",
    "predict",
    "prediction",
    "probabilities",
    "probability",
    "embedding",
    "features",
    "classify",
    "probe",
    "accuracy",
    "rollout",
}
CONTEXT = {
    "route",
    "target",
    "representation",
    "objective",
    "task",
    "label",
    "operation",
    "condition_name",
    "exposures",
    "phase",
    "prefix",
    "dateparts",
    "seed",
    "arm",
    "name",
    "kind",
    "condition",
    "split",
    "stage",
    "identity",
    "signal",
    "shuffled",
    "hops",
    "capacity",
    "keep",
    "domain",
    "novel",
    "length",
    "pattern",
    "key",
    "count",
    "budget",
    "n",
    "level",
    "variant",
    "rate",
    "size",
    "weight",
    "offset",
    "no_regime",
    "jitter",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Capture:
    """Own immutable snapshots and the return-event lifetime for one proof."""

    def __init__(self, source: Path, output: Path):
        import numpy as np
        import pyarrow as pa

        self.np, self.pa = np, pa
        self.source, self.output = source, output
        self.frames: list[dict] = []
        self.files: dict[str, dict] = {}
        self.omitted: list[dict] = []
        self.tool = 3

    def blob(self, table, *, kind: str, shape: tuple | None = None) -> dict:
        import pyarrow.ipc as ipc

        sink = self.pa.BufferOutputStream()
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        content = sink.getvalue()
        fingerprint = hashlib.sha256(content).hexdigest()
        name = f"observations/{fingerprint}.arrow"
        path = self.output / name
        if name not in self.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
            self.files[name] = {"sha256": fingerprint, "bytes": len(content), "rows": len(table)}
        return {"artifact": name, "kind": kind, "shape": list(shape) if shape is not None else None}

    def value(self, value, *, depth: int = 0):
        np, pa = self.np, self.pa
        if depth > 8:
            return {"unavailable": "Nested diagnostic exceeds eight levels"}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"nonfinite": str(value)}
        if isinstance(value, (date, datetime)):
            return {"datetime": value.isoformat()}
        if isinstance(value, np.generic):
            return self.value(value.item(), depth=depth)
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "bifU" or value.size > 4_000_000:
                return {"unavailable": "Non-numeric or oversized array", "shape": list(value.shape)}
            array = np.ascontiguousarray(value)
            return self.blob(pa.table({"value": array.reshape(-1)}), kind="array", shape=array.shape)
        if isinstance(value, pa.Table):
            return self.blob(value, kind="table")
        if isinstance(value, (pa.Array, pa.ChunkedArray)):
            return self.blob(pa.table({"value": value}), kind="array", shape=(len(value),))
        # Torch is already loaded by the proof. Avoid importing it into static tooling.
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(value, torch.Tensor):
            if value.numel() <= 4_000_000:
                return self.value(value.detach().cpu().numpy(), depth=depth)
            return {"unavailable": "Oversized tensor", "shape": list(value.shape)}
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            if len(value) > 65_536:
                return {"unavailable": "Sequence exceeds 65536 records", "length": len(value)}
            if all(isinstance(item, dict) for item in value) and any(value):
                try:
                    return self.blob(pa.Table.from_pylist(value), kind="table")
                except (pa.ArrowException, TypeError, ValueError):
                    pass
            if len(value) > 32:
                try:
                    array = np.asarray(value)
                    if array.dtype.kind in "bifU":
                        return self.value(array, depth=depth)
                except (TypeError, ValueError, RuntimeError):
                    pass
            return [self.value(item, depth=depth + 1) for item in value]
        if isinstance(value, dict):
            if len(value) > 4096:
                return {"unavailable": "Mapping exceeds 4096 entries", "length": len(value)}
            return {
                str(key): self.value(item, depth=depth + 1)
                for key, item in value.items()
                if isinstance(key, (str, int, float, bool))
            }
        if isinstance(value, set) and all(isinstance(item, (str, int)) for item in value):
            return sorted(value, key=str)
        return {"unavailable": type(value).__name__}

    def returned(self, code: CodeType, offset: int, result: object) -> None:
        frame = sys._getframe(1)
        if frame.f_code is not code:
            raise RuntimeError("Proof capture did not receive its originating return frame")
        # Keep nested call identity explicit instead of guessing an arm from its score.
        context = []
        caller = frame.f_back
        while caller is not None:
            if Path(caller.f_code.co_filename).resolve() == self.source:
                context.append(
                    {
                        "function": caller.f_code.co_name,
                        "line": caller.f_lineno,
                        "values": {
                            key: self.value(value)
                            for key, value in caller.f_locals.items()
                            if key in CONTEXT and isinstance(value, (str, bool, int, float, tuple))
                        },
                    }
                )
            caller = caller.f_back
        values = {}
        for key, value in frame.f_locals.items():
            if key.startswith("__") or inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
                continue
            converted = self.value(value)
            if isinstance(converted, dict) and set(converted) == {"unavailable"}:
                continue
            values[key] = converted
        self.frames.append(
            {
                "ordinal": len(self.frames),
                "function": code.co_name,
                "line": code.co_firstlineno,
                "context": context,
                "values": values,
                "result": self.value(result),
            }
        )

    def start(self, module) -> None:
        monitoring = sys.monitoring
        monitoring.use_tool_id(self.tool, "relflow-proof-diagnostics")
        monitoring.register_callback(self.tool, monitoring.events.PY_RETURN, self.returned)
        for value in vars(module).values():
            if inspect.isfunction(value) and value.__module__ == module.__name__ and value.__name__ in FUNCTIONS:
                monitoring.set_local_events(self.tool, value.__code__, monitoring.events.PY_RETURN)

    def close(self) -> None:
        monitoring = sys.monitoring
        monitoring.set_events(self.tool, 0)
        monitoring.register_callback(self.tool, monitoring.events.PY_RETURN, None)
        monitoring.free_tool_id(self.tool)


def load(source: Path):
    """Use a real import name so declared spawn-based seed panels can import it."""
    source = source.resolve()
    sys.path.insert(0, str(source.parent))
    module = importlib.import_module(source.stem)
    if Path(module.__file__).resolve() != source:
        raise ValueError(f"Proof {source}: import name collides with {module.__file__}")
    return module


def run(source: Path, output: Path, arguments: list[str]) -> dict:
    """Observe the canonical reporter, preserving its exact result and CLI policy."""
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source.parents[2] / "experiments"))
    import reporting
    from source import describe

    header = describe(source.read_text())
    module = load(source)
    record_path = Path(os.environ.get("EVIDENCE_JSON_OUTPUT", str(output / "record.json")))
    os.environ["EVIDENCE_JSON_OUTPUT"] = str(record_path)
    sys.argv = [str(source), *arguments]
    capture = Capture(source, output)
    capture.start(module)
    try:
        reporting.report(module.run)
    except SystemExit as error:
        if error.code not in (None, 0, 1):
            raise
    finally:
        capture.close()
    record = json.loads(record_path.read_text())
    if record_path != output / "record.json":
        (output / "record.json").write_bytes(record_path.read_bytes())
    write(output / "frames.json", capture.frames)
    (output / "proof.py").write_bytes(source.read_bytes())
    for name in ("record.json", "frames.json", "proof.py"):
        capture.files[name] = {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
    run_id = os.environ.get("EVIDENCE_RUN_ID", "")
    parts = run_id.split("-")
    task = json.loads(os.environ.get("EVIDENCE_TASK", "{}"))
    write(
        output / "manifest.json",
        {
            "schema": 1,
            "id": module.run.experiment.id,
            "title": header["title"],
            "source_sha256": digest(source),
            "code_sha256": record["provenance"]["code_sha256"],
            "wheel_sha256": record["provenance"]["wheel_sha256"],
            "capture_sha256": digest(Path(__file__)),
            "capture_method": "CPython 3.12 PY_RETURN snapshots of unchanged proof functions",
            "frames": len(capture.frames),
            "artifacts": capture.files,
            "task": task,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "run_id": run_id,
                "role": task.get("role"),
                "github_run_id": parts[1] if len(parts) == 3 and parts[0] == "gh" else None,
                "github_run_attempt": parts[2] if len(parts) == 3 and parts[0] == "gh" else None,
            },
        },
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    arguments = options.arguments[1:] if options.arguments[:1] == ["--"] else options.arguments
    record = run(options.source, options.output, arguments)
    from observations import package

    package(options.output, options.archive)
    raise SystemExit(1 if record["outcome"] == "error" else 0)


if __name__ == "__main__":
    main()
