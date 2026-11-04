"""Validate a proposed YAML schema and bounded, preprocessed observations.

Schema/Arrow checks do not construct neural modules. --encode additionally builds
a disposable CPU model and tensorizes both training and validation observations;
extension resources such as Text encoders may then be needed. Neither mode trains.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pyarrow as pa
from observations import MAX_BYTES, read


def validate(
    path: Path,
    *,
    data: Path | None = None,
    rows: int = 256,
    max_bytes: int = MAX_BYTES,
    max_slots: int = 1_000_000,
    allow_unbounded: bool = False,
    encode: bool = False,
    extensions: tuple[str, ...] = (),
) -> dict:
    """Use the SDK's template, preset, geometry, and extension-owned contracts."""
    import relflow as rf
    from relflow.architecture import template
    from relflow.data.ragged import coalesce
    from relflow.presets.base import resolve
    from relflow.tensorfields.base import TENSORFIELDS

    if rows < 1 or max_bytes < 1 or max_slots < 1:
        raise ValueError("rows, max_bytes, and max_slots must be positive integers")
    if encode and data is None:
        raise ValueError("--encode requires --data containing already-preprocessed observations")
    for module in extensions:
        importlib.import_module(module)
    options, preset = template.read(path, {})
    batch_size = options.pop("batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"model batch_size must be a positive integer, got {batch_size!r}")
    schema = rf.Schema.from_tree(**options) if preset is None else resolve(preset, **options)

    branches = []
    bounds = {"/": 1}
    for address, branch in schema.branches.items():
        if str(address) == "/":
            continue
        parent = str(branch.parent.address)
        bound = None if branch.length is None or bounds[parent] is None else bounds[parent] * branch.length
        bounds[str(address)] = bound
        branches.append(
            {"address": str(address), "cap": branch.length, "overflow": branch.overflow, "slots_per_row_bound": bound}
        )
    unbounded = [branch["address"] for branch in branches if branch["slots_per_row_bound"] is None]
    if unbounded and not allow_unbounded:
        raise ValueError(
            f"unbounded branch geometry at {unbounded}; set finite lengths or explicitly --allow-unbounded"
        )
    leaf_slots = sum(bounds[str(request.parent.address)] or 0 for request in schema.active_requests.values())
    bound = None if unbounded else sum(bounds.values()) + leaf_slots
    if bound is not None and bound > max_slots:
        raise ValueError(
            f"declared schema uses up to {bound} structural/leaf slots per row, above max_slots={max_slots}"
        )
    result = {
        "ok": True,
        "relflow_version": rf.__version__,
        "schema_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "preset": None if preset is None else preset.name,
        "objectives": [str(address) for address in schema.objectives],
        "representations": [str(address) for address in schema.embed],
        "branches": branches,
        "declared_slots_per_row_bound": bound,
        "checks": {"schema": "passed", "arrow": "not_run", "encoding": "not_run", "training": "not_run"},
        "limits": ["declared geometry is not a VRAM estimate", "sample checks do not establish population validity"],
    }
    if not schema.objectives and not schema.embed:
        result["limits"].append("schema has no reconstruction objective or requested representation output")
    if data is None:
        return result
    table, source = read(data, rows=rows, max_bytes=max_bytes)
    if not table.num_rows:
        raise ValueError(
            "validation observations have no rows; provide labeled/context examples to check compatibility"
        )
    if bound is not None and bound * table.num_rows > max_slots:
        raise ValueError(
            f"declared batch geometry is {bound * table.num_rows} slots, above max_slots={max_slots}; lower --rows"
        )
    result["source"] = source
    result["fields"] = {}
    for strata in (rf.Strata.train, rf.Strata.validate):
        projected = coalesce(table, schema, strata)
        checked = []
        for address, request in schema.active_requests.items():
            projection = projected.fields[address]
            values = projection.pristine.values
            if not projection.vacant:
                TENSORFIELDS[request.type].prepare(values, address=address)
            checked.append(
                {
                    "address": str(address),
                    "type": request.type,
                    "source_missing": projection.vacant,
                    "values": len(values),
                }
            )
        result["fields"][strata.value] = checked
    missing = sorted(
        {field["address"] for fields in result["fields"].values() for field in fields if field["source_missing"]}
    )
    result["checks"]["arrow"] = "partial" if missing else "passed"
    if missing:
        result["limits"].append(f"source values were unavailable for {missing}; compatibility remains unchecked there")
    if encode:
        import torch

        model = rf.Model(schema, batch_size=batch_size)
        with torch.no_grad():
            for strata in (rf.Strata.train, rf.Strata.validate):
                model.encode(table, strata=strata)
        result["checks"]["encoding"] = "passed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--max-slots", type=int, default=1_000_000)
    parser.add_argument("--allow-unbounded", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--extension", action="append", default=[], help="Import a user-selected extension module")
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = validate(
                args.model,
                data=args.data,
                rows=args.rows,
                max_bytes=args.max_bytes,
                max_slots=args.max_slots,
                allow_unbounded=args.allow_unbounded,
                encode=args.encode,
                extensions=tuple(args.extension),
            )
    except (OSError, ImportError, TypeError, ValueError, RuntimeError, pa.ArrowException) as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
