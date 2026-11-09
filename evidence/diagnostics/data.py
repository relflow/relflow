"""Read checksum-bound observations for explicit plot generation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path


def check_artifact(directory: Path, identifier: str, item: tuple) -> None:
    name, asset = item
    path = directory / name
    if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
        raise ValueError(f"{identifier}: diagnostic artifact must be a regular relative file: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
        raise ValueError(f"{identifier}: diagnostic artifact checksum differs: {name}")


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    artifacts = list(manifest["artifacts"].items())
    # Representation proofs can retain over 100,000 small, deduplicated buffers.
    # Bound outstanding reads and futures while checking every original checksum.
    with ThreadPoolExecutor(max_workers=16) as workers:
        for start in range(0, len(artifacts), 256):
            list(workers.map(partial(check_artifact, directory, manifest["id"]), artifacts[start : start + 256]))
    if hashlib.sha256((directory / "proof.py").read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError(f"{manifest['id']}: plotted proof differs from its captured source")
    return manifest


def leaves(value, path: list[str], parents: tuple = ()) -> list[tuple[tuple, object]]:
    """Retain parent ordinals when a target occurs inside repeated records."""
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in leaves(child, path, (*parents, index))]
    if not path:
        return [(parents, value)]
    if not isinstance(value, dict) or path[0] not in value:
        return []
    return leaves(value[path[0]], path[1:], parents)


WORDS = {
    "intact": "Intact evidence",
    "in_range": "Familiar lengths",
    "unseen": "Unseen condition",
    "test": "Held-out records",
    "rows": "Held-out records",
    "observations": "Held-out records",
    "records": "Held-out records",
    "train": "Training diagnostic",
    "training": "Training diagnostic",
    "validate_rows": "Validation records",
    "validation": "Validation records",
    "original": "Original records",
    "short": "Short collections",
    "duplicated": "Duplicated collections",
    "duplicate": "Duplicated collections",
    "duplicate_items": "Duplicated collections",
    "permuted": "Reordered complete records",
    "permutation": "Reordered complete records",
    "permute": "Reordered complete records",
    "permute_items": "Reordered complete records",
    "broken": "Broken dependency",
    "corrupted": "Corrupted evidence",
    "shuffled": "Shuffled control",
    "shuffle_y": "Pair alignment broken",
    "rotate_group_labels": "Group labels rotated",
    "swap_weights_within_groups": "Weights mispaired",
    "permute_weights": "Weights mispaired",
    "scale_weights": "All weights scaled",
    "scaled": "All weights scaled",
    "translated": "Translated values",
    "probes": "Fixed diagnostic probes",
    "paired": "Matched probes",
    "nulls": "Null inputs",
    "novel": "Unseen labels",
    "renamed": "Unseen labels renamed",
    "known": "Known labels",
    "complete": "Complete evidence",
    "selected": "Selected visibility",
    "visible": "Visible evidence",
    "tail": "Rare valid records",
    "familiar": "Familiar function family",
    "withheld": "Withheld population",
    "seen": "Previously observed population",
    "old": "Previously learned classes",
    "new": "New classes",
    "reconstruction": "Reconstruction training",
    "untrained": "Untrained model",
    "permuted_views": "Permuted-view training",
    "aligned": "Aligned training",
    "query_only": "Query-only control",
    "context": "Context examples",
    "raw": "Prepared raw features",
    "stable": "Stable identity regimes",
    "no_regime": "No stable identity regimes",
    "no_replay": "No replay",
    "small_replay": "Small replay",
    "joint": "Joint-data reference",
    "continued_A": "Continue only cohort A",
    "compressed": "Compressed route",
    "preserved": "Preserved route",
    "hash": "Hash identities",
    "category": "Category identities",
    "number": "Number",
    "quantile": "Quantile",
    "informative": "Informative training",
    "shuffled_training": "Shuffled training",
    "zero": "Centered at zero",
    "wide": "Large source offset",
    "centered": "Explicitly centered",
    "float32": "Float32 ingress",
    "visiblevisible": "Both views visible",
    "hiddenvisible": "First hidden; second visible",
    "visiblehidden": "First visible; second hidden",
    "hiddenhidden": "Both views hidden",
}


def human(value: object) -> str:
    text = str(value)
    return WORDS.get(text, text.replace("_", " ").capitalize())


class Bundle:
    """Own the verified record and lazily decoded observation buffers."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.manifest = verify(directory)
        self.record = json.loads((directory / "record.json").read_text())
        self.frames = json.loads((directory / "frames.json").read_text())
        self.source = (directory / "proof.py").read_text()
        self.syntax = ast.parse(self.source)
        self.cache = {}
        self.labels = {}
        self.exclude_context = set()

    def value(self, value):
        if isinstance(value, dict) and "artifact" in value:
            import numpy as np
            import pyarrow as pa
            import pyarrow.ipc as ipc

            name = value["artifact"]
            key = (name, tuple(value.get("shape") or ()), value["kind"])
            if key not in self.cache:
                with ipc.open_file(self.directory / name) as reader:
                    table = reader.read_all()
                if value["kind"] == "array":
                    column = table["value"]
                    array = column.to_numpy(zero_copy_only=False)
                    if (
                        pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
                    ) and not column.null_count:
                        array = np.asarray(array, dtype=str)
                    self.cache[key] = array.reshape(value["shape"])
                else:
                    self.cache[key] = table.to_pylist()
            return self.cache[key]
        if isinstance(value, dict):
            if "nonfinite" in value:
                return float(value["nonfinite"])
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        return value

    def values(self, frame: dict, *names: str) -> list:
        return [self.value(frame["values"].get(name)) for name in names]

    def context(self, frame: dict) -> dict:
        result = {}
        for caller in reversed(frame.get("context", [])):
            result.update(caller["values"])
        return self.value(result)

    def call(self, frame: dict) -> str:
        for caller in frame.get("context", []):
            line = caller.get("line")
            if line is None:
                continue
            calls = [
                node
                for node in ast.walk(self.syntax)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == frame["function"]
                and node.lineno <= line <= node.end_lineno
            ]
            if calls:
                return ast.unparse(min(calls, key=lambda node: node.end_lineno - node.lineno))
        return frame["function"]

    def label(self, frame: dict) -> str:
        context = self.context(frame)
        call = self.call(frame)
        terms = []
        for key in (
            "arm",
            "route",
            "identity",
            "representation",
            "objective",
            "task",
            "kind",
            "name",
            "condition",
            "condition_name",
        ):
            if key in self.exclude_context:
                continue
            value = context.get(key)
            if isinstance(value, (str, int, float)) and human(value) not in terms:
                terms.append(human(value))
        for key in ("hops", "capacity", "keep", "cohort", "phase", "cut", "length", "level"):
            value = context.get(key)
            if value is None:
                continue
            if isinstance(value, (str, int, float)):
                terms.append(f"{key.capitalize()} {value}")
        argument = None
        try:
            node = ast.parse(call, mode="eval").body
            if isinstance(node, ast.Call) and len(node.args) > 1:
                argument = node.args[1]
                expression = ast.unparse(argument)
                if expression in self.labels:
                    terms.append(self.labels[expression])
                elif isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name):
                    terms.append(human(argument.func.id))
                elif isinstance(argument, ast.Name):
                    terms.append(human(argument.id))
                elif isinstance(argument, ast.Subscript):
                    terms.append(ast.unparse(argument).replace("test[", "Cohort ").replace("]", "").replace("'", ""))
                elif isinstance(argument, ast.ListComp):
                    hidden = [
                        key.value
                        for item in ast.walk(argument)
                        if isinstance(item, ast.Dict)
                        for key, value in zip(item.keys, item.values, strict=True)
                        if isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and isinstance(value, ast.Constant)
                        and value.value is None
                    ]
                    if hidden:
                        terms.append("Hidden " + ", ".join(hidden))
        except SyntaxError:
            pass
        for key, positive, negative in [
            ("shuffled", "Shuffled training", "Informative training"),
            ("novel", "Unseen identities", "Known identities"),
            ("no_regime", "No stable regimes", "Stable regimes"),
        ]:
            if isinstance(context.get(key), bool):
                terms.insert(0, positive if context[key] else negative)
        if "trials" in self.record.get("metrics", {}) and "seed" in context:
            terms.insert(0, f"Seed {context['seed']}")
        terms = list(dict.fromkeys(terms))
        return " · ".join(terms[:4]) or "Held-out records"

    def pairs(self, target: str) -> list[dict]:
        """Bind predictions to explicitly named answers and preserve coordinates."""
        import numpy as np

        pairs, seen = [], set()
        frames = [
            frame
            for frame in self.frames
            if frame["function"] in {"predict", "prediction", "probabilities", "probability", "normalized_rmse"}
        ]
        for frame in frames:
            call = self.call(frame)
            if re.search(r"poison|tamper", call, re.I):
                continue
            values = frame["values"]
            raw = None
            actual, predicted = self.values(frame, "actual", "predicted")
            result = self.value(frame["result"])
            requested = self.value(values.get("target"))
            field = requested if isinstance(requested, str) else target
            for name in ("observations", "rows", "records", "test"):
                candidate = self.value(values.get(name))
                if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
                    raw = candidate
                    break
            outputs = result if isinstance(result, dict) and "unavailable" not in result else {field: result}
            if frame["function"] == "normalized_rmse":
                outputs = {field: predicted}
            for output_field, output in outputs.items():
                if output is None or isinstance(output, dict):
                    continue
                # A rich categorical readout carries a value and confidence.
                confidence = None
                if isinstance(output, list) and output and isinstance(output[0], dict):
                    if "value" not in output[0]:
                        continue
                    confidence = [item.get("probability") for item in output]
                    output = [item["value"] for item in output]
                try:
                    prediction = np.asarray(output)
                except (TypeError, ValueError):
                    continue
                if prediction.dtype.kind not in "bifuUOS" or prediction.size < 2 or prediction.dtype.kind == "O":
                    continue
                coordinates = []
                truth = actual
                if raw is not None:
                    selected = leaves(raw, str(output_field).split("/"))
                    if selected and len(selected) == prediction.size:
                        coordinates, truth = zip(*selected, strict=True)
                if truth is None or isinstance(truth, dict):
                    continue
                truth = np.asarray(truth)
                if truth.size != prediction.size or truth.size < 2:
                    continue
                truth, prediction = truth.reshape(-1), prediction.reshape(-1)
                if truth.dtype.kind == "O":
                    continue
                label = self.label(frame)
                signature = (
                    label,
                    str(output_field),
                    hashlib.sha256(truth.tobytes() + prediction.tobytes()).hexdigest(),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                pairs.append(
                    {
                        "label": label,
                        "field": str(output_field),
                        "actual": truth,
                        "predicted": prediction,
                        "confidence": confidence,
                        "frame": frame["ordinal"],
                        "call": call,
                        "context": self.context(frame),
                        "rows": raw,
                        "coordinates": list(coordinates),
                        "population": "training"
                        if re.search(r"\btrain(?:ing)?\b", call)
                        else "validation"
                        if re.search(r"validat", call)
                        else "test",
                    }
                )
        if not pairs:
            for frame in reversed(self.frames):
                if frame["function"] not in {"score", "scores", "measure", "run"}:
                    continue
                actual = self.value(frame["values"].get("actual"))
                predicted = self.value(frame["values"].get("predicted"))
                if actual is None or predicted is None or isinstance(actual, dict) or isinstance(predicted, dict):
                    continue
                a, p = np.asarray(actual), np.asarray(predicted)
                if a.size == p.size and a.size > 1 and a.dtype.kind != "O" and p.dtype.kind != "O":
                    pairs.append(
                        {
                            "label": self.label(frame),
                            "field": target,
                            "actual": a.reshape(-1),
                            "predicted": p.reshape(-1),
                            "confidence": None,
                            "frame": frame["ordinal"],
                            "call": self.call(frame),
                            "context": self.context(frame),
                            "rows": None,
                            "coordinates": [],
                            "population": "test",
                        }
                    )
                    break
        return pairs
