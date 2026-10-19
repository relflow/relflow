"""Cross-check episode sources, docs mappings, provenance and actual API traces."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, default=HERE.parents[1] / "explainer-videos")
    parser.add_argument("--docs", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.framework.resolve()))
    from videos.ml.relflow.manifest import EPISODES

    assert len({e["slug"] for e in EPISODES}) == len(EPISODES) == 25
    assert [e["number"] for e in EPISODES] == list(range(1, 26))
    for episode in EPISODES:
        module = importlib.import_module("videos.ml.relflow." + episode["module"])
        assert hasattr(module, episode["cls"])
        end = importlib.import_module("videos.ml.relflow.endcards")
        assert hasattr(end, episode["end"][1])
        if args.docs:
            for doc in episode["docs"]:
                assert (args.docs / "docs" / doc).is_file(), f"Missing doc: {doc}"
        tree = ast.parse((HERE / (episode["module"] + ".py")).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            call = node.items[0].context_expr
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Attribute)
                or call.func.attr != "voiceover"
            ):
                continue
            text = ast.unparse(call.args[0])
            defined = set(re.findall(r'<bookmark\s+mark=[\\]*[\'"]([^\'"]+)', text))
            # Validate literal waits in this block; loop-computed waits are covered by rendering.
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "wait_until"
                ):
                    if child.args and isinstance(child.args[0], ast.Constant):
                        assert child.args[0].value in defined, (episode["cls"], child.lineno, defined)
    provenance = json.loads((HERE / "data/provenance.json").read_text())
    for name, expected in provenance["archived_files"].items():
        actual = hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest()
        assert actual == expected, f"Historical measurements changed: {name}"
    census = json.loads((HERE / "data/schema_current.json").read_text())
    assert census["provenance"]["version"] == "0.3.2"
    assert census["customer"]["total_params"] == sum(
        sum(node["params"].values()) for node in census["customer"]["nodes"]
    )
    current = json.loads((HERE / "data/current.json").read_text())
    assert current["provenance"]["version"] == "0.3.2"
    assert current["parameters_unchanged"]
    assert [b["shape"][-1] for b in current["geometry"]] == [2, 5, 1]
    assert current["metadata"]["relflow.prediction_format"] == "2"
    assert current["output"][2]["predictions"]["trxns"][1] is None
    assert all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(current["query"]["retained_values"], [56.2, 84.5]))
    assert current["fit"]["train_batches"] == current["fit"]["validation_batches"] == 1
    assert all(4.8 <= v <= 5.2 for v in current["jitter"]["samples"])
    learning = json.loads((HERE / "data/learning_current.json").read_text())
    assert learning["provenance"]["version"] == "0.3.2"
    snapshots = learning["snapshots"]
    assert [s["step"] for s in snapshots] == [0, 1, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    assert len(learning["loss"]) == 256
    assert len(learning["labels"]) == learning["test_rows"] == 512
    for snapshot in snapshots:
        assert len(snapshot["xy"]) == 512
        assert all(0 <= p <= 1 for p in snapshot["prob"])
        for probe in snapshot["attention"]:
            assert len(probe) == 4
            assert all(len(h) == 18 and math.isclose(sum(h), 1, abs_tol=1e-5) for h in probe)
    first, kept, moved = [p["line_items"] for p in learning["probes"]]
    assert first[2]["quantity"] == 4 and kept[2]["quantity"] == 1
    assert all(a == b for i, (a, b) in enumerate(zip(first, kept)) if i != 2)
    assert {k: v for k, v in first[2].items() if k != "quantity"} == {
        k: v for k, v in kept[2].items() if k != "quantity"
    }
    assert moved == [first[2], *first[:2], *first[3:]]
    assert snapshots[-1]["prob"][0] > 0.95 and snapshots[-1]["prob"][1] < 0.05
    assert abs(snapshots[-1]["prob"][2] - snapshots[-1]["prob"][0]) < 0.01
    update = learning["update"]
    assert all(v > 0 and math.isfinite(v) for v in update["gradient_norms"].values())
    assert update["after"] != update["before"]
    assert [max(range(18), key=lambda i: h[i]) for h in snapshots[0]["attention"][0][:2]] == [15, 15]
    assert [max(range(18), key=lambda i: h[i]) for h in snapshots[-1]["attention"][0][:2]] == [2, 2]
    print(
        f"Validated {len(EPISODES)} scene imports/end cards, docs mappings, bookmarks, {len(provenance['archived_files'])} archived files and current traces."
    )


if __name__ == "__main__":
    main()
