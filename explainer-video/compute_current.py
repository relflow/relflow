"""Record current API traces, separate from the immutable 0.2.3 learning runs.

Run with the Relflow environment: .venv/bin/python explainer-video/compute_current.py
No model downloads. All random demonstrations are seeded and labeled as untrained.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

import relflow as rf

HERE = Path(__file__).resolve().parent


def params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def customer_model():
    import relflow as rf

    return rf.Model.md(
        batch_size=128,
        embed=True,
        tenure_days=rf.Number,
        orders=rf.Branch(
            length=64,
            overflow="tail",
            age_days=rf.Number,
            line_items=rf.Branch(length=16, sku=rf.Category, quantity=rf.Number),
        ),
        churned=rf.Boolean(mask=True),
        next_month_spend=rf.Number(mask=True),
    )


def schema(provenance):
    import relflow as rf
    from relflow.structs.tree import Leaf

    model = customer_model()
    s = model.schema
    nodes = []
    for node in (s.fields, *s.fields.descendants):
        addr = str(node.address) or "/"
        kind = "root" if addr == "/" else ("leaf" if isinstance(node, Leaf) else "branch")
        entry = {"address": addr, "kind": kind, "type": getattr(node, "type", None)}
        if kind == "leaf":
            entry["shape"] = list(s.shapes[node.address])
            entry["target"] = node.address in s.objectives
            entry["heritage"] = [str(a) or "/" for a in node.heritage]
            mod = model.nodes[node.address]
            entry["params"] = {"embedder": params(mod.embedder)}
            if hasattr(mod, "decoder"):
                entry["params"]["decoder"] = params(mod.decoder)
                entry["decoder_context"] = [str(a) for a in mod.decoder.context_addresses]
                entry["decoder_queries"] = int(mod.decoder.n_context)
        else:
            enc = model.nodes[node.address].encoder
            entry["length"] = int(node.length)
            entry["outputs"] = int(s.branch_outputs[node.address])
            entry["input_width"] = int(enc.input_width)
            entry["n_layers"] = len(enc.encoder)
            entry["params"] = {
                "coordinate": params(enc.coordinate_encoder) if enc.coordinate_encoder is not None else 0,
                "sequence": params(enc.encoder),
                "reduction": params(enc.pool) if enc.pool is not None else 0,
            }
        nodes.append(entry)
    total = params(model)
    assert total == sum(sum(n["params"].values()) for n in nodes)

    presets = []
    for name in ("XS", "SM", "MD", "LG", "XL"):
        p = getattr(rf.presets, name)
        presets.append(
            {
                "name": p.name,
                "d_model": p.d_model,
                "root_layers": p.root.n_layers,
                "branch_layers": p.branch.n_layers,
                "heads": p.root.n_heads,
                "root_outputs": p.root.reduction.n_outputs,
                "branch_outputs": p.branch.reduction.n_outputs,
            }
        )

    # The README's order model, at its own settings.
    order = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        batch_size=128,
        line_items=rf.Branch(length=32, sku=rf.Category, quantity=rf.Number, price=rf.Number),
        returned=rf.Boolean(mask=True),
    )
    payload = {
        "customer": {
            "nodes": nodes,
            "total_params": total,
            "d_model": s.d_model,
            "depthwise": [[str(a) or "/" for a in d] for d in s.depthwise],
        },
        "order": {"total_params": params(order), "nodes": {str(a) or "/": params(m) for a, m in order.nodes.items()}},
        "presets": presets,
    }
    payload["provenance"] = provenance
    (HERE / "data/schema_current.json").write_text(json.dumps(payload, indent=2) + "\n")


def record():
    if importlib.metadata.version("relflow") != "0.3.2":
        raise SystemExit(
            "Current traces are pinned to relflow==0.3.2; update provenance deliberately for a new target."
        )
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(4)
    torch.manual_seed(20261019)
    np.random.seed(20261019)
    # Holes, null lists and empty lists must survive independently of leaf masks.
    records = [
        {"payment_id": "missing", "trxns": None},
        {"payment_id": "empty", "trxns": []},
        {"payment_id": "mixed", "trxns": [{"amount": 32.0}, None, {"amount": 84.5}]},
    ]
    source = pa.Table.from_pylist(
        records,
        schema=pa.schema(
            [
                ("payment_id", pa.string()),
                ("trxns", pa.list_(pa.struct([("amount", pa.float64())]))),
            ]
        ),
    )
    model = rf.Model.xs(
        d_model=16,
        n_layers=1,
        n_heads=2,
        embed=True,
        trxns=rf.Branch(amount=rf.Number, disputed=rf.Boolean(mask=True, embed=True)),
    )
    parameters = {name: id(p) for name, p in model.named_parameters()}
    geometry = []
    for lengths in ([2, 1, 0], [5, 2, 1], [1, 1, 0]):
        batch = pa.Table.from_pylist([{"trxns": [{"amount": float(i)} for i in range(n)]} for n in lengths])
        tensors = model.encode(batch, strata="predict")
        geometry.append({"lengths": lengths, "shape": list(tensors["/trxns/amount"].state.shape)})
    assert parameters == {name: id(p) for name, p in model.named_parameters()}
    output = model.predict(source, batch_size=2, retain=("payment_id",))
    rows = output.to_pylist()
    assert rows[0]["predictions"]["trxns"] is None
    assert rows[1]["predictions"]["trxns"] == []
    assert rows[2]["predictions"]["trxns"][1] is None
    assert len(rows[2]["predictions"]["trxns"]) == 3
    assert pa.types.is_list(output.schema.field("predictions").type.field("trxns").type)
    assert all(x["inferred"] for x in [rows[2]["predictions"]["trxns"][i]["disputed"] for i in (0, 2)])
    summaries = []
    for r in rows:
        summaries.append(
            {
                **r,
                "representations": [
                    {
                        **v,
                        "embedding": v["embedding"][:6],
                        "width": len(v["embedding"]),
                        "norm": float(np.linalg.norm(v["embedding"])),
                    }
                    for v in r["representations"]
                ],
            }
        )
    # Query navigation is a structural trace, not a learned result.
    query_source = pa.Table.from_pylist(
        [{"payload": {"transactions": [{"gross_amount": 32.0}, {"gross_amount": 56.2}, {"gross_amount": 84.5}]}}]
    )
    queried = rf.Model.xs(
        d_model=16,
        query="payload",
        trxns=rf.Branch(
            query="transactions[-2:]", length=2, amount=rf.Number(query="gross_amount"), disputed=rf.Boolean(mask=True)
        ),
    )
    q = queried.encode(query_source, strata="predict")
    # Number raw content is preserved at tensorization, before embedder normalization.
    query_values = q["/trxns/amount"].content.reshape(-1).tolist()
    assert np.allclose(query_values, [56.2, 84.5])
    jitter = rf.Jitter(add=0.1, multiply=0.02, normalize=False)
    base = torch.full((512,), 5.0)
    noisy = jitter.apply(base.clone(), torch.ones(512, dtype=torch.bool))
    assert float(noisy.min()) >= 4.8 and float(noisy.max()) <= 5.2
    # Real current fit API smoke run, deliberately no claim about predictive quality.
    train = pa.table(
        {
            "amount": [0.0, 1.0, 2.0, 3.0] * 8,
            "merchant": ["A", "B", "A", "B"] * 8,
            "disputed": [False, False, True, True] * 8,
        }
    )
    tiny = rf.Model.xs(
        d_model=16, n_layers=1, n_heads=2, amount=rf.Number, merchant=rf.Category, disputed=rf.Boolean(mask=True)
    )
    data = rf.ArrowDataModule(
        model=tiny, train=train, validate=train, num_workers=0, persistent_workers=False, pin_memory=False
    )
    tiny.fit(
        data,
        batch_size=8,
        learning_rate=1e-3,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        precision="32-true",
        devices=1,
        early_stopping=False,
        rollback=False,
        enable_progress_bar=False,
    )
    fitted = tiny.predict(train.slice(0, 2).drop(["disputed"]))
    assert fitted.num_rows == 2 and tiny.batch_size == 8
    payload = {
        "provenance": {
            "version": importlib.metadata.version("relflow"),
            "runtime_commit": subprocess.check_output(
                ["git", "rev-parse", "origin/main"], cwd=HERE.parent, text=True
            ).strip(),
            "docs_commit": "b46f1b0d6e5fc511993bc56fcdba77e56441d926",
            "seed": 20261019,
            "machine": platform.machine(),
            "torch": torch.__version__,
            "pyarrow": pa.__version__,
        },
        "geometry": geometry,
        "parameters_unchanged": True,
        "source": records,
        "output": summaries,
        "output_schema": str(output.schema),
        "metadata": {k.decode(): v.decode() for k, v in output.schema.metadata.items()},
        "query": {"retained_values": query_values, "source_values": [32.0, 56.2, 84.5]},
        "jitter": {"base": 5.0, "add": 0.1, "multiply": 0.02, "samples": noisy.tolist()},
        "fit": {
            "train_batches": 1,
            "validation_batches": 1,
            "batch_size": tiny.batch_size,
            "prediction_rows": fitted.num_rows,
            "purpose": "API smoke check, no learning-quality claim",
        },
    }
    schema(payload["provenance"])
    (HERE / "data/current.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("Recorded current geometry, output, queries, jitter and fit traces.")


if __name__ == "__main__":
    record()
