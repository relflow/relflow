"""Real footage for the relflow video.

Every grid, curve and number the scenes show comes from running relflow itself
(release 0.2.3) on this machine's CPU: real schemas and parameter counts, real
Arrow-to-tensor layouts, the real Number and Hash embedders, real vocabulary
growth with the optimizer state checked before and after, the real packing code
timed against dense attention, and small real training runs.

    python -m videos.relflow.compute              # everything (about 5 min on 4 cores)
    python -m videos.relflow.compute counting     # one experiment

Results are written to ``videos/relflow/data/<experiment>.json`` and committed,
so rendering the video does not need relflow or torch.  relflow 0.2.3 needs
Polars < 2 (it calls ``Series.hash(seed_1=...)``) and was locked to PyArrow 24:

    uv pip install relflow==0.2.3 polars==1.44.1 pyarrow==24.0.0
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RELFLOW_REPO = Path(os.environ.get("RELFLOW_REPO", "/home/user/relflow/relflow"))


def quiet():
    import torch

    torch.set_num_threads(4)
    for name in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:  # relflow logs through loguru; keep the console readable
        from loguru import logger

        logger.remove()
    except Exception:
        pass


def save(name: str, payload: dict) -> None:
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=float))
    print(f"  wrote {path.relative_to(HERE.parents[1])} ({path.stat().st_size / 1e3:.1f} kB)")


def trainer(**kw):
    import lightning.pytorch as lit

    kw.setdefault("accelerator", "cpu")
    kw.setdefault("devices", 1)
    return lit.Trainer(enable_progress_bar=False, logger=False, enable_checkpointing=False,
                       enable_model_summary=False, num_sanity_val_steps=0, **kw)


def params(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# ---------------------------------------------------------------------------
# 1. The schema is the architecture
# ---------------------------------------------------------------------------
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


def schema():
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
        presets.append({
            "name": p.name, "d_model": p.d_model, "root_layers": p.root.n_layers, "branch_layers": p.branch.n_layers,
            "heads": p.root.n_heads, "root_outputs": p.root.reduction.n_outputs,
            "branch_outputs": p.branch.reduction.n_outputs,
        })

    # The README's order model, at its own settings.
    order = rf.Model(
        d_model=64, n_layers=2, n_heads=4, batch_size=128,
        line_items=rf.Branch(length=32, sku=rf.Category, quantity=rf.Number, price=rf.Number),
        returned=rf.Boolean(mask=True),
    )
    save("schema", {
        "customer": {"nodes": nodes, "total_params": total, "d_model": s.d_model,
                     "depthwise": [[str(a) or "/" for a in d] for d in s.depthwise]},
        "order": {"total_params": params(order),
                  "nodes": {str(a) or "/": params(m) for a, m in order.nodes.items()}},
        "presets": presets,
    })


# ---------------------------------------------------------------------------
# 2. Records -> rectangles
# ---------------------------------------------------------------------------
ORDERS = [
    {"line_items": [{"sku": "A12", "quantity": 2, "price": 19.99}, {"sku": "B07", "quantity": 1, "price": 45.50}],
     "returned": False},
    {"line_items": [{"sku": "C33", "quantity": None, "price": 5.00}], "returned": True},
    {"line_items": [], "returned": False},
    {"line_items": [{"sku": "A12", "quantity": 1, "price": 19.99}, {"sku": "A12", "quantity": 3, "price": 19.99},
                    {"sku": "D01", "quantity": 1, "price": 2.50}, {"sku": "E90", "quantity": 1, "price": 9.00},
                    {"sku": "F11", "quantity": 1, "price": 1.00}],
     "returned": True},
]


def encode():
    import pyarrow as pa
    import relflow as rf
    from relflow.data import ragged
    from relflow.structs.enums import Strata

    out = {"records": ORDERS}
    table = pa.Table.from_pylist(ORDERS)
    out["offsets"] = table["line_items"].chunks[0].offsets.to_pylist()
    out["skus_flat"] = [it["sku"] for r in ORDERS for it in r["line_items"]]
    for overflow in ("head", "tail"):
        model = rf.Model(
            d_model=32, n_layers=1, n_heads=4, batch_size=4,
            line_items=rf.Branch(length=4, overflow=overflow, sku=rf.Category, quantity=rf.Number, price=rf.Number),
            returned=rf.Boolean(mask=True),
        )
        lay = ragged.root(table, model.schema.fields)
        branch = model.schema.branches["/line_items"]
        sub = ragged.descend(lay, branch, branch.address)
        enc = model.encode(table, strata="train")
        fields = {}
        for addr in ("/line_items/sku", "/line_items/quantity", "/line_items/price", "/returned"):
            f = enc[addr]
            fields[addr] = {"state": f.state.reshape(4, -1).tolist(), "present": f.present.reshape(4, -1).tolist(),
                            "content": f.content.reshape(4, -1).tolist()}
        out[overflow] = {"placement": sub.placement.tolist(), "shape": list(sub.shape),
                         "retained_skus": [r["sku"] for r in sub.records.to_pylist()], "fields": fields}
    # the head layout keeps items 0-3 of the last order, the tail layout items 1-4
    assert out["head"]["retained_skus"][-4:] == ["A12", "A12", "D01", "E90"]
    assert out["tail"]["retained_skus"][-4:] == ["A12", "D01", "E90", "F11"]
    assert out["head"]["placement"] == [0, 1, 4, 12, 13, 14, 15]

    # Deterministic sampled masking on a bigger layout: 12 orders x 16 slots.
    rng = np.random.default_rng(7)
    lengths = rng.integers(2, 17, 12)
    rows = [{"line_items": [{"sku": "x", "quantity": 1, "price": 1.0}] * int(n), "returned": False} for n in lengths]
    big = pa.Table.from_pylist(rows)
    model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=12,
                     line_items=rf.Branch(length=16, sku=rf.Category, quantity=rf.Number, price=rf.Number),
                     returned=rf.Boolean(mask=True))
    lay = ragged.descend(ragged.root(big, model.schema.fields), model.schema.branches["/line_items"], "/line_items")
    mask = rf.Mask(rate=0.15, reconstruct=True)
    picks = {}
    for strata, epoch in (("train", 0), ("train", 0), ("train", 1), ("train", 2), ("validate", 5)):
        sel = ragged.select(lay, mask, address="/line_items/quantity", seed=0, epoch=epoch, strata=Strata(strata))
        picks.setdefault(f"{strata}{epoch}", []).append(sel.reshape(12, 16).astype(int).tolist())
    assert picks["train0"][0] == picks["train0"][1]  # same seed, stage and epoch: same selection
    present = np.zeros((12, 16), dtype=int)
    for i, n in enumerate(lengths):
        present[i, :n] = 1
    out["masking"] = {"lengths": lengths.tolist(), "present": present.tolist(),
                      "epochs": {k: v[0] for k, v in picks.items()},
                      "rate": 0.15}
    save("encode", out)


# ---------------------------------------------------------------------------
# 3. Numbers
# ---------------------------------------------------------------------------
def number():
    import relflow as rf
    import torch
    from relflow.tensorfields.extensions.number import GlobalOnlineNormalizer

    model = rf.Model(d_model=64, n_layers=1, n_heads=4, batch_size=8, price=rf.Number, y=rf.Number(mask=True))
    emb = model.nodes["/price"].embedder
    weights = emb.weights.reshape(-1).tolist()
    assert len(weights) == 13 and abs(weights[0] - math.pi / 256) < 1e-9 and abs(weights[-1] - 16 * math.pi) < 1e-5

    # Streaming moments: merge batch summaries, compare with the exact statistics of all values seen.
    rng = np.random.default_rng(3)
    norm = GlobalOnlineNormalizer()
    seen, stream = [], []
    for b in range(6):
        batch = np.round(rng.lognormal(3.0, 0.6, 128 if b else 64) * (1.0 if b < 3 else 1.4), 2).astype(np.float32)
        norm.update(torch.from_numpy(batch))
        seen.append(batch)
        allv = np.concatenate(seen).astype(np.float64)
        stream.append({"batch": [float(x) for x in batch[:40]], "n": int(batch.size),
                       "batch_mean": float(batch.mean()), "batch_var": float(batch.var()),
                       "mean": float(norm.mean.item()), "var": float(norm.var.item()), "count": int(norm.count.item()),
                       "exact_mean": float(allv.mean()), "exact_var": float(allv.var())})
    for s in stream:
        assert abs(s["mean"] - s["exact_mean"]) < 1e-3 and abs(s["var"] / s["exact_var"] - 1) < 1e-4
    save("number", {"weights": weights, "max_fourier_input": float(emb.max_fourier_input.item()),
                    "n_bands": 8, "offset": 4, "stream": stream})


# ---------------------------------------------------------------------------
# 4. Vocabulary growth (and what happens to Adam's moments)
# ---------------------------------------------------------------------------
def vocab():
    import pyarrow as pa
    import relflow as rf
    import relflow.architecture.root as runtime  # Model.on_before_batch_transfer calls root.bind
    import torch

    rng = np.random.default_rng(11)
    effects = {f"m{i:02d}": float(rng.normal(0, 1)) for i in range(40)}
    names = list(effects)
    rows = []
    # new merchants keep arriving: batch b may use the first 1 + 3b merchants
    for b in range(14):
        k = min(len(names), 1 + 3 * b)
        for _ in range(32):
            items = [names[int(rng.integers(0, k))] for _ in range(4)]
            rows.append({"orders": [{"merchant": m} for m in items],
                         "spend": float(sum(effects[m] for m in items) + rng.normal(0, 0.1))})
    table = pa.Table.from_pylist(rows)
    torch.manual_seed(0)
    model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=32,
                     orders=rf.Branch(length=4, merchant=rf.Category), spend=rf.Number(mask=True))
    model.optimizer = rf.adamw(learning_rate=3e-3)
    address = "/orders/merchant"

    def table_rows():
        emb = model.nodes[address].embedder
        return emb.embeddings["content"].weight

    events = []
    original = runtime.bind

    def spy(module, batch, strata):
        weight = table_rows()
        opt = module.trainer.optimizers[0] if module._trainer is not None else None
        state = opt.state.get(weight, {}) if opt is not None else {}
        before = {"rows": int(weight.shape[0]), "labels": len(rf.Category.vocabulary(model, rf.Address("orders", "merchant"))),
                  "w": weight.detach().clone(),
                  "m": state["exp_avg"].detach().clone() if "exp_avg" in state else None,
                  "v": state["exp_avg_sq"].detach().clone() if "exp_avg_sq" in state else None}
        result = original(module, batch, strata)
        weight = table_rows()
        state = opt.state.get(weight, {}) if opt is not None else {}
        n = before["rows"]
        ev = {"batch": len(events), "rows_before": n, "rows_after": int(weight.shape[0]),
              "labels_before": before["labels"],
              "labels_after": len(rf.Category.vocabulary(model, rf.Address("orders", "merchant")))}
        ev["weights_kept"] = bool(torch.equal(weight.detach()[:n], before["w"]))
        if before["m"] is not None and "exp_avg" in state:
            m, v = state["exp_avg"], state["exp_avg_sq"]
            ev["moments_kept"] = bool(torch.equal(m[:n], before["m"]) and torch.equal(v[:n], before["v"]))
            ev["new_moments_zero"] = bool((m[n:] == 0).all() and (v[n:] == 0).all())
            ev["moment_norm_old_rows"] = float(before["m"].norm(dim=1).mean())
        events.append(ev)
        return result

    runtime.bind = spy
    try:
        data = rf.ArrowDataModule(model=model, train=table, shuffle=False, num_workers=0, persistent_workers=False,
                                  pin_memory=False)
        trainer(max_epochs=1, limit_val_batches=0).fit(model=model, datamodule=data)
    finally:
        runtime.bind = original
    growth = [e for e in events if e["rows_after"] > e["rows_before"]]
    if os.environ.get("RELFLOW_DEBUG"):
        for e in events:
            print("   ", e)
    assert growth and all(e["weights_kept"] for e in events)
    assert all(e.get("moments_kept", True) and e.get("new_moments_zero", True) for e in growth)
    assert [e["rows_after"] for e in growth] == sorted({e["rows_after"] for e in growth})
    labels = list(rf.Category.vocabulary(model, rf.Address("orders", "merchant")))
    counts = rf.Category.counts(model, rf.Address("orders", "merchant"))
    counts = [int(counts[label]) for label in labels] if hasattr(counts, "keys") else [int(c) for c in counts]
    save("vocab", {"events": events, "labels": labels, "counts": counts})


# ---------------------------------------------------------------------------
# 5. Hash: equality without identity
# ---------------------------------------------------------------------------
def hashing():
    import pyarrow as pa
    import relflow as rf
    import torch

    ids = ["acct-17", "acct-92", "acct-17", "acct-5", "acct-92", "acct-33"]
    table = pa.Table.from_pylist([{
        "account_id": "acct-17",
        "transfers": [{"counterparty_id": c, "amount": 1.0} for c in ids],
        "fraud": False,
    }])
    model = rf.Model(d_model=64, n_layers=1, n_heads=4, batch_size=1,
                     account_id=rf.Hash(n_hashes=2),
                     transfers=rf.Branch(length=6, reduction=None, counterparty_id=rf.Hash(n_hashes=2), amount=rf.Number),
                     fraud=rf.Boolean(mask=True))
    model.eval()
    batches = []
    for strata in ("train", "train", "train", "predict"):
        enc = model.encode(table, strata=strata)
        with torch.no_grad():
            a = model.nodes["/account_id"].embedder.embed(enc["/account_id"]).payload.reshape(-1, 64)
            c = model.nodes["/transfers/counterparty_id"].embedder.embed(enc["/transfers/counterparty_id"]).payload
            c = c.reshape(-1, 64)
        vecs = torch.cat([a, c[: len(ids)]])
        unit = torch.nn.functional.normalize(vecs, dim=1)
        sim = (unit @ unit.T).numpy()
        lanes = np.concatenate([enc["/account_id"].content.reshape(-1, 2).numpy(),
                                enc["/transfers/counterparty_id"].content.reshape(-1, 2).numpy()[: len(ids)]])
        batches.append({"strata": strata, "lanes": (lanes / 2.0**63).tolist(), "vectors": vecs.numpy().tolist(),
                        "similarity": sim.tolist()})
        labels = ["acct-17", *ids]
        for i, x in enumerate(labels):
            for j, y in enumerate(labels):
                if x == y:
                    assert sim[i, j] > 1 - 1e-5
    assert batches[0]["lanes"] != batches[1]["lanes"]  # a new salt every training batch
    save("hash", {"ids": ["acct-17", *ids], "batches": batches})


# ---------------------------------------------------------------------------
# 6. Packing: do not pay for padding
# ---------------------------------------------------------------------------
def packing():
    import relflow as rf
    import torch
    from relflow.architecture.packed import pack, unpack

    # (a) the grouping, on a small batch
    rng = np.random.default_rng(5)
    lengths = np.minimum(rng.geometric(0.18, 16), 12)
    present = torch.zeros(16, 12, dtype=torch.bool)
    for i, n in enumerate(lengths):
        present[i, :n] = True
    x = torch.randn(16, 12, 8)
    values, packing = pack(x, present)
    groups = [{"length": g.length, "rows": g.rows,
               "row_ids": [int(r) for r in sorted({int(t) // 1 for t in []})]} for g in packing.groups]
    order = {}
    for i, n in enumerate(lengths.tolist()):
        order.setdefault(n, []).append(i)
    small = {"lengths": lengths.tolist(), "capacity": 12, "live": int(present.sum()), "dense": 16 * 12,
             "groups": [{"length": g["length"], "rows": order[g["length"]]} for g in groups],
             "pairs_dense": int(16 * 12 * 12), "pairs_packed": int((lengths**2).sum())}

    # (b) timing an encoder stack, dense and packed, on this CPU
    torch.manual_seed(0)
    model = rf.Model(d_model=128, n_layers=1, n_heads=8, batch_size=256,
                     events=rf.Branch(length=128, n_layers=2, n_heads=8, amount=rf.Number), y=rf.Number(mask=True))
    enc = model.nodes["/events"].encoder.eval()
    timing = []
    for p in (0.25, 0.08, 0.03):
        lens = np.minimum(np.random.default_rng(1).geometric(p, 256), 128)
        pres = torch.zeros(256, 128, dtype=torch.bool)
        for i, n in enumerate(lens):
            pres[i, :n] = True
        inp = torch.randn(256, 128, 128).masked_fill(~pres.unsqueeze(-1), 0.0)

        def dense():
            return enc.compute(inp, pres)

        def packed():
            vals, pk = pack(inp, pres)
            vals = enc.compute(vals, pres, packing=pk)
            return unpack(vals, pk, inp, pres, enc.encoder)

        with torch.inference_mode():
            a, b = dense(), packed()
            diff = float((a - b).abs().max())
            ts = {}
            for name, fn in (("dense", dense), ("packed", packed)):
                fn()
                runs = []
                for _ in range(7):
                    t = time.perf_counter()
                    fn()
                    runs.append(time.perf_counter() - t)
                ts[name] = float(np.median(runs))
        timing.append({"mean_length": float(lens.mean()), "live_fraction": float(pres.float().mean()),
                       "dense_ms": 1e3 * ts["dense"], "packed_ms": 1e3 * ts["packed"], "max_abs_diff": diff,
                       "pair_fraction": float((lens.astype(float) ** 2).sum() / (256 * 128 * 128))})
        assert diff < 1e-4
        print(f"    mean length {lens.mean():5.1f}: dense {1e3 * ts['dense']:.0f} ms, packed {1e3 * ts['packed']:.0f} ms")
    save("packing", {"small": small, "timing": timing, "rows": 256, "capacity": 128, "d_model": 128, "layers": 2,
                     "threads": torch.get_num_threads()})


# ---------------------------------------------------------------------------
# 7. Counting: softmax attention is an average; the mass lane is not
# ---------------------------------------------------------------------------
def counting():
    import pyarrow as pa
    import relflow as rf
    import torch

    def bags(n, seed, lo, hi):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            k = int(rng.integers(lo, hi + 1))
            a = float(rng.uniform(0.5, 1.5))
            out.append({"items": [{"amount": a} for _ in range(k)], "total": k * a, "count": k, "amount": a})
        return out

    train, val = bags(4096, 1, 1, 6), bags(512, 2, 1, 6)
    test_seen, test_unseen = bags(512, 3, 1, 6), bags(512, 4, 7, 10)
    probes = [{"items": [{"amount": 1.25}] * k, "total": 1.25 * k, "count": k, "amount": 1.25} for k in range(1, 11)]

    def strip(rows):
        return pa.Table.from_pylist([{"items": r["items"], "total": r["total"]} for r in rows])

    arms = {}
    for arm in ("mean", "attention_no_mass", "attention"):
        torch.manual_seed(0)
        reduction = rf.Mean() if arm == "mean" else rf.Attention()
        model = rf.Model(d_model=64, n_layers=1, n_heads=4, batch_size=128,
                         items=rf.Branch(length=10, attention=None, reduction=reduction, amount=rf.Number),
                         total=rf.Number(mask=True))
        if arm == "attention_no_mass":  # ablation: remove the additive mass and count lanes from every reducer
            for module in model.modules():
                if type(module).__name__ == "LearnedQueryCrossAttention" and module.mass_projection is not None:
                    module.mass_projection = None
                    module.mass_direction = None
        model.optimizer = rf.adamw(learning_rate=2e-3)
        losses = []

        class Track(__import__("lightning.pytorch", fromlist=["Callback"]).Callback):
            def on_train_batch_end(self, tr, pl, outputs, batch, idx):
                if outputs is not None:
                    losses.append(float(outputs["loss"]))

        data = rf.ArrowDataModule(model=model, train=strip(train), validate=strip(val), num_workers=0,
                                  persistent_workers=False, pin_memory=False)
        t = time.time()
        trainer(max_epochs=40, callbacks=[Track()]).fit(model=model, datamodule=data)
        model.eval()

        def predict(rows):
            out = model.predict(pa.Table.from_pylist([{"items": r["items"]} for r in rows])).to_pylist()
            return [float(o["predictions"]["/total"]["content"]) for o in out]

        res = {"steps": len(losses), "seconds": time.time() - t,
               "loss": [float(np.mean(losses[i:i + 32])) for i in range(0, len(losses), 32)]}
        for name, rows in (("seen", test_seen), ("unseen", test_unseen), ("probes", probes)):
            pred = predict(rows)
            truth = [r["total"] for r in rows]
            res[name] = {"truth": truth, "pred": pred, "count": [r["count"] for r in rows],
                         "rmse": float(np.sqrt(np.mean((np.array(pred) - np.array(truth)) ** 2))),
                         "baseline_rmse": float(np.sqrt(np.mean((np.mean([r["total"] for r in train]) - np.array(truth)) ** 2)))}
        arms[arm] = res
        print(f"    {arm:20s} seen rmse {res['seen']['rmse']:.3f}  unseen {res['unseen']['rmse']:.3f}  "
              f"probes 1x/6x: {res['probes']['pred'][0]:.3f} {res['probes']['pred'][5]:.3f}  ({res['seconds']:.0f} s)")
    save("counting", {"arms": arms})


# ---------------------------------------------------------------------------
# 8. A real run: which item made the order come back?
# ---------------------------------------------------------------------------
SKUS = [f"S{i:02d}" for i in range(20)]
FRAGILE = ("S03", "S11")


def order(rng):
    n = int(rng.integers(1, 9))
    items = []
    for _ in range(n):
        items.append({"sku": SKUS[int(rng.integers(len(SKUS)))], "quantity": int(rng.integers(1, 6)),
                      "price": float(np.round(rng.lognormal(3, 0.6), 2))})
    returned = any(it["sku"] in FRAGILE and it["quantity"] >= 3 for it in items)
    return {"line_items": items, "returned": bool(returned)}


def misalign(rows, seed):
    """Shuffle quantities among the items of each order: same SKUs, same quantities, wrong pairs."""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        q = [it["quantity"] for it in r["line_items"]]
        perm = rng.permutation(len(q))
        items = [dict(it, quantity=q[p]) for it, p in zip(r["line_items"], perm)]
        out.append({"line_items": items, "returned": r["returned"]})
    return out


def auc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    order_ = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order_] = np.arange(1, len(p) + 1)
    pos = y.sum()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def returns():
    import lightning.pytorch as lit
    import pyarrow as pa
    import relflow as rf
    import torch
    from relflow.tensorfields.base import TensorInput

    rng = np.random.default_rng(0)
    train = [order(rng) for _ in range(4096)]
    val = [order(rng) for _ in range(512)]
    test = [order(rng) for _ in range(2048)]
    torch.manual_seed(0)
    model = rf.Model(d_model=64, n_layers=1, n_heads=4, batch_size=128, embed=True,
                     line_items=rf.Branch(length=8, sku=rf.Category, quantity=rf.Number, price=rf.Number),
                     returned=rf.Boolean(mask=True))
    model.optimizer = rf.adamw(learning_rate=2e-3)
    losses, epochs = [], []

    class Track(lit.Callback):
        def on_train_batch_end(self, tr, pl, outputs, batch, idx):
            if outputs is not None:
                losses.append(float(outputs["loss"]))

        def on_validation_epoch_end(self, tr, pl):
            m = tr.callback_metrics
            epochs.append({"step": len(losses), "auc": float(m.get(".returned/validate.auc.content", float("nan"))),
                           "loss": float(m.get("loss/validate", float("nan")))})

    data = rf.ArrowDataModule(model=model, train=pa.Table.from_pylist(train), validate=pa.Table.from_pylist(val),
                              num_workers=0, persistent_workers=False, pin_memory=False)
    t = time.time()
    trainer(max_epochs=4, callbacks=[Track()]).fit(model=model, datamodule=data)
    seconds = time.time() - t
    model.eval()

    def predict(rows):
        out = model.predict(pa.Table.from_pylist([{"line_items": r["line_items"]} for r in rows])).to_pylist()
        prob = [float(o["predictions"]["/returned"]["content"]["probability"]) for o in out]
        emb = [o["predictions"]["/"]["embedding"] for o in out]
        return np.array(prob), np.array(emb, dtype=np.float32)

    y = [r["returned"] for r in test]
    p_intact, emb = predict(test)
    p_broken, _ = predict(misalign(test, 1))
    res = {"intact": {"auc": auc(y, p_intact), "accuracy": float(np.mean((p_intact > 0.5) == np.array(y)))},
           "misaligned": {"auc": auc(y, p_broken), "accuracy": float(np.mean((p_broken > 0.5) == np.array(y)))}}
    print(f"    intact AUC {res['intact']['auc']:.4f}   misaligned AUC {res['misaligned']['auc']:.4f}  ({seconds:.0f} s)")

    # PCA of the exported root embeddings (one 64-d vector per order)
    centered = emb.reshape(len(test), -1) - emb.reshape(len(test), -1).mean(0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    pca = {"xy": xy[:600].tolist(), "label": y[:600], "explained": (sv[:2] ** 2 / (sv**2).sum()).tolist()}

    # Where does the branch's learned summary query look? Recompute the softmax weights of the
    # reduction's cross-attention from its own projections (SDPA does not return them).
    pool = model.nodes["/line_items"].encoder.pool
    attn = pool.blocks[0].attention
    captured = {}

    def hook(mod, args, kwargs):
        captured["q"], captured["k"] = args[0].detach(), args[1].detach()
        captured["pad"] = kwargs.get("key_padding_mask")

    handle = attn.register_forward_pre_hook(hook, with_kwargs=True)
    picks = []
    for r in test:
        n = len(r["line_items"])
        hits = [i for i, it in enumerate(r["line_items"]) if it["sku"] in FRAGILE and it["quantity"] >= 3]
        decoys = [i for i, it in enumerate(r["line_items"]) if it["sku"] in FRAGILE and it["quantity"] < 3]
        if n >= 5 and ((len(hits) == 1 and len(picks) < 2) or (not hits and decoys and len(picks) >= 2 and len(picks) < 4)):
            picks.append(r)
        if len(picks) == 4:
            break
    weights = []
    for r in picks:
        captured.clear()
        model.predict(pa.Table.from_pylist([{"line_items": r["line_items"]}]))
        with torch.inference_mode():
            q = attn.rotate(attn.splitheads(attn.q_proj(captured["q"]), attn.nhead))
            k = attn.rotate(attn.splitheads(attn.k_proj(captured["k"]), attn.n_kv_heads))
            s = (q @ k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
            pad = captured["pad"]
            if pad is not None:
                s = s.masked_fill(pad[:, None, None, :], float("-inf"))
            w = torch.softmax(s, dim=-1)[0, :, 0, :]  # heads x tokens
        weights.append(w.tolist())
    handle.remove()
    prob_picks, _ = predict(picks)
    example = model.predict(pa.Table.from_pylist([{"line_items": picks[0]["line_items"]}])).to_pylist()[0]["predictions"]
    emb0 = np.array(example["/"]["embedding"], dtype=np.float64)
    example = {"/returned": example["/returned"],
               "/": {"embedding_head": emb0[:3].tolist(), "embedding_norm": float(np.linalg.norm(emb0)),
                     "width": len(emb0)}}

    # The number line as the trained price embedder sees it.
    emb_mod = model.nodes["/line_items/price"].embedder
    prices = np.geomspace(2, 300, 160).astype(np.float32)
    with torch.inference_mode():
        out = emb_mod(TensorInput(state=torch.zeros(len(prices), dtype=torch.long),
                                  content=torch.from_numpy(prices), batch_size=[len(prices)])).payload.numpy()
    c = out - out.mean(0)
    _, s2, v2 = np.linalg.svd(c, full_matrices=False)
    line = (c @ v2[:3].T)
    stats = rf.Number.normalization(model, rf.Address("line_items", "price"))

    save("returns", {
        "skus": SKUS, "fragile": list(FRAGILE), "rule": "returned iff a fragile SKU has quantity >= 3",
        "train_rows": len(train), "test_rows": len(test), "positive_rate": float(np.mean(y)), "seconds": seconds,
        "loss": losses, "epochs": epochs, "results": res, "pca": pca,
        "example_prediction": example,
        "attention": {"orders": picks, "weights": weights, "prob": prob_picks.tolist(),
                      "layout": "tokens 0-7 sku slots, 8-15 quantity slots, 16-23 price slots"},
        "number_line": {"prices": prices.tolist(), "xyz": line.tolist(),
                        "explained": (s2[:3] ** 2 / (s2**2).sum()).tolist(),
                        "mean": float(stats["mean"]), "std": float(stats["std"])},
    })


# ---------------------------------------------------------------------------
# 9. Rotary positions in the concatenated branch sequence
# ---------------------------------------------------------------------------
def rope():
    import torch
    from relflow.architecture.rotary import RotaryEmbedding

    head_dim = 16
    rot = RotaryEmbedding(d_model=head_dim)
    torch.manual_seed(0)
    q, k = torch.randn(head_dim), torch.randn(head_dim)
    n = 24
    pos = torch.arange(n, dtype=torch.float32)
    rq = rot(q.expand(n, head_dim).unsqueeze(0), positions=pos)[0]
    rk = rot(k.expand(n, head_dim).unsqueeze(0), positions=pos)[0]
    scores = (rq @ rk.T).numpy()
    # depends only on the offset m - n
    for d in range(-n + 1, n):
        diag = np.diagonal(scores, offset=d)
        assert np.ptp(diag) < 1e-4
    save("rope", {"inv_freq": rot.inv_freq.tolist(), "scores": scores.tolist(), "head_dim": head_dim})


# ---------------------------------------------------------------------------
# 10. The proof catalog
# ---------------------------------------------------------------------------
def proofs():
    import yaml

    raw = yaml.safe_load((RELFLOW_REPO / "proofs" / "results.yaml").read_text())["proofs"]
    catalog = []
    for pid, p in raw.items():
        runs = [r for r in p.get("runs", []) if r.get("mode") == "full"]
        catalog.append({"id": pid, "family": p["script"].split("/")[1], "name": Path(p["script"]).stem,
                        "status": p.get("historical", {}).get("status"),
                        "latest_full": runs[-1]["outcome"] if runs else None})
    pick = {pid: next(r for r in reversed(raw[pid]["runs"]) if r["mode"] == "full")["metrics"]
            for pid in ("P001", "P002", "P021", "P028", "P038", "P040", "P041", "P043", "P044", "P051", "P052",
                        "P053", "P054", "P062", "P066", "P068", "P070", "P076", "P077")}
    save("proofs", {"catalog": catalog, "metrics": pick, "source": "relflow/proofs/results.yaml @ 5077716"})


# ---------------------------------------------------------------------------
# 11. The happy path: a schema, a data module, a fit
# ---------------------------------------------------------------------------
def happy():
    import tempfile

    import lightning.pytorch as lit
    import pyarrow as pa
    import relflow as rf
    import torch

    rng = np.random.default_rng(0)
    train = pa.Table.from_pylist([order(rng) for _ in range(4096)])
    validate = pa.Table.from_pylist([order(rng) for _ in range(512)])
    test_rows = [order(rng) for _ in range(2048)]
    test = pa.Table.from_pylist(test_rows)
    torch.manual_seed(0)
    schema = dict(line_items=rf.Branch(length=8, sku=rf.Category, quantity=rf.Number, price=rf.Number),
                  returned=rf.Boolean(mask=True))
    # The three lines, as the video shows them (Lightning picks the device; on this M1 that is set to the CPU,
    # which is faster here than the GPU: see the benchmarks).
    model = rf.Model.sm(**schema, batch_size=128, optimizer=rf.adamw(learning_rate=2e-3))
    data = rf.ArrowDataModule(model=model, train=train, validate=validate, test=test)
    epochs, batch_losses = [], []

    class Track(lit.Callback):
        def on_train_batch_end(self, tr, pl, outputs, batch, idx):
            if outputs is not None:
                batch_losses.append(float(outputs["loss"]))

        def on_validation_epoch_end(self, tr, pl):
            m = tr.callback_metrics
            epochs.append({"epoch": len(epochs), "steps": len(batch_losses),
                           "train_loss": float(np.mean(batch_losses[-32:])) if batch_losses else None,
                           "validate_loss": float(m["loss/validate"]),
                           "auc": float(m[".returned/validate.auc.content"])})

    t = time.time()
    fit = trainer(max_epochs=6, callbacks=[Track()])
    fit.fit(model, datamodule=data)
    seconds = time.time() - t
    tested = trainer().test(model, datamodule=data, verbose=False)[0]
    defaults = rf.Model.sm(**schema)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "orders.rf"
        model.save(path)
        size = path.stat().st_size
        restored = rf.Model.load(path)
    request = pa.Table.from_pylist([{"order_id": f"order-{i}", "line_items": r["line_items"]}
                                    for i, r in enumerate(test_rows[:4])])
    out = restored.predict(request, retain=("order_id",)).to_pylist()
    rows = [{"order_id": o["inputs"]["order_id"], "returned": test_rows[i]["returned"],
             "prediction": o["predictions"]["/returned"]} for i, o in enumerate(out)]
    print(f"    {seconds:.0f} s, validation AUC {epochs[-1]['auc']:.3f}, test AUC {tested['.returned/test.auc.content']:.3f}")
    save("happy", {
        "params": params(model), "batch_size": model.batch_size, "preset": "sm", "seconds": seconds,
        "train_rows": train.num_rows, "validate_rows": validate.num_rows, "test_rows": test.num_rows,
        "steps": len(batch_losses), "epochs": epochs, "loss": batch_losses,
        "test": {k: float(v) for k, v in tested.items() if k.startswith(".returned/test.") or k == "loss/test"},
        "artifact_bytes": size, "predictions": rows,
        "defaults": {"batch_size": defaults.batch_size, "optimizer_set": defaults.optimizer is not None},
    })


# ---------------------------------------------------------------------------
# 12. More tensorfields: Quantile and Enum; DateParts, Vector and Text
# ---------------------------------------------------------------------------
def quantile_enum():
    import pyarrow as pa
    import relflow as rf
    import torch

    # Quantile vs Number on heavy-tailed amounts with eight huge outliers.
    rng = np.random.default_rng(0)
    amounts = rng.lognormal(3.5, 1.0, 4000)
    amounts[rng.choice(4000, 8, replace=False)] *= 400
    rows = [{"amount": float(a), "rank": float(a), "large": bool(a > 50)} for a in amounts]
    torch.manual_seed(0)
    model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=256, optimizer=rf.adamw(learning_rate=1e-3),
                     amount=rf.Number, rank=rf.Quantile, large=rf.Boolean(mask=True))
    table = pa.Table.from_pylist(rows)
    trainer(max_epochs=1).fit(model, datamodule=rf.ArrowDataModule(model=model, train=table, validate=table.slice(0, 256)))
    q = rf.Quantile.normalization(model, rf.Address("rank"))
    n = rf.Number.normalization(model, rf.Address("amount"))
    probe = np.geomspace(0.5, 60000, 160)
    enc = model.encode(pa.Table.from_pylist([{"amount": float(p), "rank": float(p)} for p in probe]), strata="predict")
    ranks = enc["/rank"].content.reshape(-1).numpy().astype(float)
    standardized = (probe - n["mean"]) / n["std"]
    assert np.all(np.diff(ranks) >= -1e-6)
    marks = [5, 20, 33, 100, 1000, 50000]
    enc = model.encode(pa.Table.from_pylist([{"amount": float(p), "rank": float(p)} for p in marks]), strata="predict")
    mark_ranks = enc["/rank"].content.reshape(-1).tolist()
    emb = model.nodes["/rank"].embedder
    quantile = {
        "values_sample": np.sort(amounts)[::8].tolist(), "count": int(q["count"]), "compression": int(q["compression"]),
        "minimum": q["minimum"], "median": q["median"], "maximum": q["maximum"],
        "number_mean": float(n["mean"]), "number_std": float(n["std"]),
        "share_within_tenth_sd": float(np.mean(np.abs((amounts - n["mean"]) / n["std"]) < 0.1)),
        "probe": probe.tolist(), "ranks": ranks.tolist(), "standardized": standardized.tolist(),
        "marks": marks, "mark_ranks": mark_ranks, "mark_standardized": ((np.array(marks) - n["mean"]) / n["std"]).tolist(),
        "frequencies": emb.frequencies.tolist(), "outliers": int(8),
    }
    print(f"    Number: mean {n['mean']:.1f} std {n['std']:.1f};  Quantile median {q['median']:.1f}")

    # Enum vs Category: three declared tiers, only two of which appear while training.
    rows = [{"tier": str(rng.choice(["bronze", "silver"])), "y": bool(rng.random() < 0.5)} for _ in range(512)]
    torch.manual_seed(0)
    model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=128, optimizer=rf.adamw(learning_rate=1e-3),
                     tier=rf.Enum(values=("bronze", "silver", "gold")), tier_learned=rf.Category(query="tier"),
                     y=rf.Boolean(mask=True))
    table = pa.Table.from_pylist(rows)
    trainer(max_epochs=1).fit(model, datamodule=rf.ArrowDataModule(model=model, train=table, validate=table.slice(0, 128)))
    enum_rows = int(model.nodes["/tier"].embedder.embeddings["content"].num_embeddings)
    cat_labels = list(rf.Category.vocabulary(model, rf.Address("tier_learned")))
    error = None
    try:
        model.predict(pa.Table.from_pylist([{"tier": "platinum"}]))
    except ValueError as e:
        error = str(e)
    assert error and "undeclared value 'platinum'" in error
    enum = {"values": list(rf.Enum.vocabulary(model, rf.Address("tier"))),
            "counts": rf.Enum.counts(model, rf.Address("tier")), "rows": enum_rows,
            "category_labels": cat_labels, "error": error}
    assert enum["counts"]["gold"] == 0 and enum_rows == 3
    save("quantile_enum", {"quantile": quantile, "enum": enum})


def time_vector_text():
    import pyarrow as pa
    import relflow as rf
    import torch

    stamps = ["2026-05-28T23:30:00", "2026-05-29T00:30:00", "2026-05-29T01:30:00", "2026-05-29T12:00:00"]
    day = [f"2026-05-29T{h:02d}:15:00" for h in range(24)]
    week = [f"2026-05-{d:02d}T09:00:00" for d in range(25, 32)]
    model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=64,
                     t=rf.DateParts(dateparts=["hour_of_day", "day_of_week"], pattern="%Y-%m-%dT%H:%M:%S"),
                     y=rf.Boolean(mask=True))

    def coords(values):
        enc = model.encode(pa.Table.from_pylist([{"t": v} for v in values]), strata="predict")
        return {str(k): v.reshape(len(values), 2).tolist() for k, v in enc["/t"].content.items()}

    dateparts = {"stamps": stamps, "examples": coords(stamps), "day": day, "day_coords": coords(day)["hour_of_day"],
                 "week": week, "week_coords": coords(week)["day_of_week"]}
    a, b = np.array(dateparts["examples"]["hour_of_day"][0]), np.array(dateparts["examples"]["hour_of_day"][2])
    dateparts["distance_2330_0130"] = float(np.linalg.norm(a - b))
    dateparts["distance_0130_1200"] = float(np.linalg.norm(b - np.array(dateparts["examples"]["hour_of_day"][3])))

    model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=8, embedding=rf.Vector(n_dim=4),
                     y=rf.Boolean(mask=True))
    enc = model.encode(pa.Table.from_pylist([{"embedding": [0.12, -0.03, 0.44, 0.08]}, {"embedding": None}]),
                       strata="predict")
    error = None
    try:
        model.encode(pa.Table.from_pylist([{"embedding": [0.12, -0.03, 0.44]}]), strata="predict")
    except ValueError as e:
        error = str(e)
    linears = [m for m in model.nodes["/embedding"].embedder.modules() if isinstance(m, torch.nn.Linear)]
    vector = {"shape": list(enc["/embedding"].content.shape), "states": enc["/embedding"].state.reshape(-1).tolist(),
              "error": error, "projection": [linears[0].in_features, linears[0].out_features]}

    messages = ["The international transfer has not arrived.", "My wire payment still hasn't shown up.",
                "How do I change my password?", "The payment is awaiting review."]
    model = rf.Model(d_model=64, n_layers=1, n_heads=4, batch_size=8, body=rf.Text(max_length=64, encoder_pooling="mean"),
                     y=rf.Boolean(mask=True))
    enc = model.encode(pa.Table.from_pylist([{"body": m} for m in messages]), strata="predict")
    emb = model.nodes["/body"].embedder
    with torch.inference_mode():
        frozen = emb.encode(content=enc["/body"].content, state=enc["/body"].state).reshape(len(messages), -1)
    unit = torch.nn.functional.normalize(frozen, dim=1)
    sim = (unit @ unit.T).numpy()
    ids = enc["/body"].content["input_ids"].reshape(len(messages), -1)
    mask = enc["/body"].content["attention_mask"].reshape(len(messages), -1)
    text = {"messages": messages, "model": emb.request.model, "hidden": int(frozen.shape[1]),
            "tokens": mask.sum(1).tolist(), "max_length": int(ids.shape[1]), "similarity": sim.tolist(),
            "projection": [int(emb.hidden_size), 64],
            "frozen_params": int(sum(p.numel() for p in emb.__dict__["_cached_model"].module_for(torch.device("cpu")).parameters())),
            "trainable_params": params(emb)}
    print(f"    hour circle 23:30-01:30 {dateparts['distance_2330_0130']:.3f}; text sim(0,1) {sim[0, 1]:.3f} "
          f"sim(0,2) {sim[0, 2]:.3f}")
    save("time_vector_text", {"dateparts": dateparts, "vector": vector, "text": text})


# ---------------------------------------------------------------------------
# 13. Masks: what one batch looks like under each policy
# ---------------------------------------------------------------------------
def masks():
    import pyarrow as pa
    import relflow as rf

    rng = np.random.default_rng(5)
    rows = [order(rng) for _ in range(8)]
    table = pa.Table.from_pylist(rows)
    policies = {
        "visible": dict(quantity=rf.Number),
        "dropout": dict(quantity=rf.Number(mask=0.25)),
        "skip": dict(quantity=rf.Number(mask=rf.Mask(rate=0.25, skip=True))),
        "reconstruct": dict(quantity=rf.Number(mask=rf.Mask(rate=0.25, reconstruct=True))),
        "target": dict(quantity=rf.Number(mask=True)),
    }
    out = {"lengths": [len(r["line_items"]) for r in rows]}
    for name, fields in policies.items():
        model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=8,
                         line_items=rf.Branch(length=8, sku=rf.Category, price=rf.Number, **fields),
                         returned=rf.Boolean(mask=True))
        res = {}
        for strata in ("train", "validate", "predict"):
            enc = model.encode(table, strata=strata)
            f = enc["/line_items/quantity"]
            res[strata] = {"state": f.state.reshape(8, 8).tolist(), "present": f.present.reshape(8, 8).int().tolist(),
                           "inferred": f.inferred.reshape(8, 8).int().tolist()}
        out[name] = res
    # A branch-level policy: one decision hides a whole line item (all of its fields).
    model = rf.Model(d_model=32, n_layers=1, n_heads=4, batch_size=8,
                     line_items=rf.Branch(length=8, mask=rf.Mask(rate=0.25, skip=True), sku=rf.Category,
                                          quantity=rf.Number, price=rf.Number),
                     returned=rf.Boolean(mask=True))
    enc = model.encode(table, strata="train")
    out["branch"] = {a: enc[a].present.reshape(8, 8).int().tolist()
                     for a in ("/line_items/sku", "/line_items/quantity", "/line_items/price")}
    assert out["branch"]["/line_items/sku"] == out["branch"]["/line_items/price"]
    assert out["dropout"]["predict"]["state"] == out["visible"]["predict"]["state"]
    save("masks", out)


# ---------------------------------------------------------------------------
# 14. Pretraining without labels, then fine-tuning with a few
# ---------------------------------------------------------------------------
GROUPS, PER_GROUP = 8, 25
CATALOG = [f"{'ABCDEFGH'[g]}{i:02d}" for g in range(GROUPS) for i in range(PER_GROUP)]
GROUP_PRICE = [25, 25, 60, 8, 12, 90, 40, 150]  # group A (the fragile one) shares its price level with group B


def basket(rng, labeled=True):
    """An order whose items mostly come from one product group; returned iff a group-A item has quantity >= 3."""
    theme, n = int(rng.integers(GROUPS)), int(rng.integers(2, 9))
    items = []
    for _ in range(n):
        g = theme if rng.random() < 0.8 else int(rng.integers(GROUPS))
        items.append({"sku": CATALOG[g * PER_GROUP + int(rng.integers(PER_GROUP))], "quantity": int(rng.integers(1, 6)),
                      "price": float(np.round(GROUP_PRICE[g] * rng.lognormal(0, 0.25), 2))})
    row = {"line_items": items}
    if labeled:
        row["returned"] = any(it["sku"][0] == "A" and it["quantity"] >= 3 for it in items)
    return row


def pretraining():
    import tempfile

    import lightning.pytorch as lit
    import pyarrow as pa
    import relflow as rf
    import torch

    rng = np.random.default_rng(1)
    unlabeled = [basket(rng, False) for _ in range(20000)]
    unlabeled_val = [basket(rng, False) for _ in range(1000)]
    test = [basket(rng) for _ in range(2000)]
    y = [r["returned"] for r in test]
    request = pa.Table.from_pylist([{"line_items": r["line_items"]} for r in test])

    def branch(sku=False, price=False):
        return rf.Branch(length=8, sku=rf.Category(mask=sku), quantity=rf.Number, price=rf.Number(mask=price))

    sampled = rf.Mask(rate=0.15, reconstruct=True)
    torch.manual_seed(0)
    pre = rf.Model(d_model=64, n_layers=2, n_heads=4, batch_size=128, optimizer=rf.adamw(learning_rate=2e-3),
                   line_items=branch(sampled, sampled))
    curve = []

    class Track(lit.Callback):
        def on_validation_epoch_end(self, tr, pl):
            m = tr.callback_metrics
            curve.append({k: float(v) for k, v in m.items() if "validate" in k})

    t = time.time()
    trainer(max_epochs=6, callbacks=[Track()]).fit(pre, datamodule=rf.ArrowDataModule(
        model=pre, train=pa.Table.from_pylist(unlabeled), validate=pa.Table.from_pylist(unlabeled_val)))
    pre_seconds = time.time() - t

    # What the SKU table learned without a single label: PCA of the embedding rows, colored by product group.
    emb = pre.nodes["/line_items/sku"].embedder
    vocab = list(rf.Category.vocabulary(pre, rf.Address("line_items", "sku")))
    weight = emb.embeddings["content"].weight.detach().numpy()
    assert weight.shape[0] >= len(vocab)
    w = weight[: len(vocab)].astype(np.float64)
    w = w - w.mean(0)
    _, sv, vt = np.linalg.svd(w, full_matrices=False)
    xy = w @ vt[:2].T
    unit = w / np.linalg.norm(w, axis=1, keepdims=True)
    groups = np.array([s[0] for s in vocab])
    same = np.mean([np.mean(unit[groups == g] @ unit[groups == g].T) for g in "ABCDEFGH"])
    cross = float(np.mean(unit @ unit.T))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pretrained.rf"
        pre.save(path)

        def finetune(labeled, seed, pretrained):
            torch.manual_seed(seed)
            if pretrained:
                m = rf.Model.load(path)
                inputs = rf.where("address").is_in([rf.Address("line_items", "sku"), rf.Address("line_items", "price")])
                m.update(inputs, mask=False)
                m.extend(rf.where("address") == rf.Address(), returned=rf.Boolean(mask=True))
            else:
                m = rf.Model(d_model=64, n_layers=2, n_heads=4, batch_size=128, line_items=branch(),
                             returned=rf.Boolean(mask=True))
            m.batch_size = 32
            m.optimizer = rf.adamw(learning_rate=1e-3)
            table = pa.Table.from_pylist(labeled)
            steps = 300
            epochs = int(np.ceil(steps / np.ceil(len(labeled) / 32)))
            trainer(max_steps=steps, max_epochs=epochs, limit_val_batches=0).fit(
                m, datamodule=rf.ArrowDataModule(model=m, train=table, validate=table))
            m.eval()
            out = m.predict(request).to_pylist()
            p = [o["predictions"]["/returned"]["content"]["probability"] for o in out]
            seen = {it["sku"] for r in labeled for it in r["line_items"]}
            return auc(y, p), len(seen)

        budgets = [32, 64, 128, 256, 512, 1024]
        results = {"scratch": {}, "pretrained": {}}
        coverage = {}
        for n in budgets:
            for arm in ("scratch", "pretrained"):
                runs = []
                for seed in range(3):
                    g = np.random.default_rng(100 + seed)
                    score, seen = finetune([basket(g) for _ in range(n)], seed, arm == "pretrained")
                    runs.append(score)
                    coverage.setdefault(n, []).append(seen)
                results[arm][n] = runs
            print(f"    {n:5d} labels: scratch {np.mean(results['scratch'][n]):.3f}  "
                  f"pretrained {np.mean(results['pretrained'][n]):.3f}")

    save("pretraining", {
        "catalog": len(CATALOG), "groups": GROUPS, "unlabeled_rows": len(unlabeled), "test_rows": len(test),
        "positive_rate": float(np.mean(y)), "pretrain_seconds": pre_seconds, "pretrain_curve": curve,
        "embedding": {"labels": vocab, "rows": np.round(weight[: len(vocab)], 5).tolist(), "xy": xy.tolist(),
                      "explained": (sv[:2] ** 2 / (sv**2).sum()).tolist(),
                      "same_group_cosine": float(same), "all_pairs_cosine": cross},
        "budgets": budgets, "results": {a: {str(k): v for k, v in r.items()} for a, r in results.items()},
        "skus_seen": {str(k): float(np.mean(v)) for k, v in coverage.items()},
        "steps": 300, "seeds": 3,
    })


# ---------------------------------------------------------------------------
# 15. Preprocessors and postprocessors around a real model
# ---------------------------------------------------------------------------
ACCOUNTS = [f"account-{i:03d}" for i in range(400)]
WATCHLIST = ACCOUNTS[:8]


def transfer(rng, i):
    payer, payee = ACCOUNTS[int(rng.integers(400))], ACCOUNTS[int(rng.integers(400))]
    if rng.random() < 0.08:
        payee = WATCHLIST[int(rng.integers(len(WATCHLIST)))]
    if rng.random() < 0.08:
        payer = WATCHLIST[int(rng.integers(len(WATCHLIST)))]
    amount_minor = int(rng.lognormal(10, 1.2))
    return {"transfer_id": f"transfer-{i}", "accounts": {"payer": payer, "payee": payee}, "amount_minor": amount_minor,
            "flagged": payee in WATCHLIST or amount_minor > 250_000}


def processors():
    import polars as pl
    import pyarrow as pa
    import relflow as rf
    import torch

    def reshape(frame: pl.DataFrame, *, scale: float = 100.0) -> pl.DataFrame:
        accounts = pl.col("accounts")
        return frame.with_columns(
            amount=pl.col("amount_minor") / scale,
            parties=pl.concat_list(
                pl.struct(account=accounts.struct.field("payer"), role=pl.lit("payer")),
                pl.struct(account=accounts.struct.field("payee"), role=pl.lit("payee")),
            ),
        ).drop("accounts", "amount_minor")

    def review(frame: pl.DataFrame, *, threshold: float) -> pl.DataFrame:
        flagged = pl.col("predictions").struct.field(rf.Address("flagged"))
        probability = flagged.struct.field("content").struct.field("probability")
        return frame.select(transfer_id=pl.col("inputs").struct.field("transfer_id"),
                            flag_probability=probability, review=probability >= threshold)

    prepare, decision = rf.preprocess(reshape), rf.postprocess(review)
    rng = np.random.default_rng(0)
    train = [transfer(rng, i) for i in range(6000)]
    val = [transfer(rng, 10000 + i) for i in range(500)]
    test = [transfer(rng, 20000 + i) for i in range(2000)]
    torch.manual_seed(0)
    model = rf.Model.sm(batch_size=128, optimizer=rf.adamw(learning_rate=2e-3),
                        parties=rf.Branch(length=2, overflow="error", account=rf.Category, role=rf.Category),
                        amount=rf.Number, flagged=rf.Boolean(mask=True))
    data = rf.ArrowDataModule(model=model, train=pa.Table.from_pylist(train), validate=pa.Table.from_pylist(val),
                              preprocessor=prepare.partial(scale=100.0))
    t = time.time()
    trainer(max_epochs=15).fit(model, datamodule=data)
    seconds = time.time() - t

    def strip(rows):
        return pa.Table.from_pylist([{k: v for k, v in r.items() if k != "flagged"} for r in rows])

    out = model.predict(strip(test), preprocess=prepare.partial(scale=100.0), retain=("transfer_id",),
                        postprocess=decision.partial(threshold=0.5))
    p = np.array(out["flag_probability"].to_pylist())
    y = np.array([r["flagged"] for r in test])
    watched_payee = np.array([r["accounts"]["payee"] in WATCHLIST for r in test])
    large = np.array([r["amount_minor"] > 250_000 for r in test])
    watched_payer = np.array([r["accounts"]["payer"] in WATCHLIST and r["accounts"]["payee"] not in WATCHLIST
                              for r in test]) & ~large
    stats = {"accuracy": float(np.mean((p >= 0.5) == y)), "auc": auc(y, p),
             "watched_payee_recall": float(np.mean(p[watched_payee] >= 0.5)),
             "large_recall": float(np.mean(p[large & ~watched_payee] >= 0.5)),
             "watched_payer_flag_rate": float(np.mean(p[watched_payer] >= 0.5)),
             "ordinary_flag_rate": float(np.mean(p[~y & ~watched_payer] >= 0.5))}

    # Four requests worth showing: an ordinary transfer, a watched payee, a watched payer, a large amount.
    def pick(cond):
        return next(r for r in test if cond(r))

    shown = [pick(lambda r: not r["flagged"] and r["accounts"]["payer"] not in WATCHLIST and r["amount_minor"] < 20000),
             pick(lambda r: r["accounts"]["payee"] in WATCHLIST and r["amount_minor"] < 50000),
             pick(lambda r: r["accounts"]["payer"] in WATCHLIST and not r["flagged"]),
             pick(lambda r: r["amount_minor"] > 250_000 and r["accounts"]["payee"] not in WATCHLIST)]
    processed = reshape(pl.from_arrow(strip(shown[1:2])), scale=100.0).to_dicts()[0]
    result = model.predict(strip(shown), preprocess=prepare.partial(scale=100.0), retain=("transfer_id",),
                           postprocess=decision.partial(threshold=0.5)).to_pylist()
    raw = model.predict(strip(shown[1:2]), preprocess=prepare.partial(scale=100.0), retain=("transfer_id",)).to_pylist()[0]
    print(f"    {seconds:.0f} s; accuracy {stats['accuracy']:.4f}; watched payer flagged {stats['watched_payer_flag_rate']:.3f}")
    save("processors", {"seconds": seconds, "train_rows": len(train), "test_rows": len(test),
                        "watchlist_size": len(WATCHLIST), "stats": stats,
                        "source": {k: v for k, v in shown[1].items() if k != "flagged"}, "processed": processed,
                        "requests": [{k: v for k, v in r.items() if k != "flagged"} for r in shown],
                        "truth": [r["flagged"] for r in shown], "result": result,
                        "envelope": {"inputs": raw["inputs"], "flagged": raw["predictions"]["/flagged"]}})


# ---------------------------------------------------------------------------
# 16. Benchmarks: time and memory for each preset, and for longer contexts
# ---------------------------------------------------------------------------
BENCH_BATCH, BENCH_STEPS, BENCH_WARMUP = 64, 12, 3


def bench_customer(rng, length):
    orders = []
    for _ in range(min(length, int(rng.poisson(10)))):
        items = [{"sku": f"P{int(rng.integers(500)):03d}", "quantity": int(rng.integers(1, 5))}
                 for _ in range(int(rng.integers(1, 7)))]
        orders.append({"age_days": int(rng.integers(1, 365)), "line_items": items})
    return {"tenure_days": int(rng.integers(30, 2000)), "orders": orders, "churned": bool(rng.random() < 0.2),
            "next_month_spend": float(np.round(rng.lognormal(4, 1), 2))}


def bench_one(preset: str, length: int, mode: str, device: str) -> dict:
    """One configuration, measured in its own process (peak memory is per process)."""
    import resource

    import lightning.pytorch as lit
    import pyarrow as pa
    import relflow as rf
    import torch

    def rss_mib():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    table = pa.Table.from_pylist([bench_customer(rng, length) for _ in range(BENCH_BATCH * (BENCH_STEPS + 1))])
    baseline = rss_mib()
    model = getattr(rf.Model, preset)(
        batch_size=BENCH_BATCH, optimizer=rf.adamw(learning_rate=1e-3, fused=None),
        tenure_days=rf.Number,
        orders=rf.Branch(length=length, overflow="tail", age_days=rf.Number,
                         line_items=rf.Branch(length=16, sku=rf.Category, quantity=rf.Number)),
        churned=rf.Boolean(mask=True), next_month_spend=rf.Number(mask=True))
    out = {"preset": preset, "length": length, "mode": mode, "device": device, "batch": BENCH_BATCH,
           "params": params(model), "baseline_mib": baseline}
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    if mode == "train":
        times = []

        class Clock(lit.Callback):
            def on_train_batch_start(self, *a):
                sync()
                self.t = time.perf_counter()

            def on_train_batch_end(self, *a):
                sync()
                times.append(time.perf_counter() - self.t)

        trainer(max_steps=BENCH_STEPS, limit_val_batches=0, callbacks=[Clock()], accelerator=device).fit(
            model, datamodule=rf.ArrowDataModule(model=model, train=table, validate=table.slice(0, BENCH_BATCH)))
        step = float(np.median(times[BENCH_WARMUP:]))
    else:
        model.eval()
        if device == "mps":
            model.to("mps")
        request = table.drop_columns(["churned", "next_month_spend"])
        model.predict(request.slice(0, BENCH_BATCH))
        times = []
        for i in range(1, BENCH_STEPS):
            t = time.perf_counter()
            model.predict(request.slice(i * BENCH_BATCH, BENCH_BATCH))
            sync()
            times.append(time.perf_counter() - t)
        step = float(np.median(times))
    out["ms_per_batch"] = 1000 * step
    out["rows_per_s"] = BENCH_BATCH / step
    out["peak_mib"] = rss_mib()
    if device == "mps":
        out["mps_driver_mib"] = torch.mps.driver_allocated_memory() / 2**20
    return out


def benchmarks():
    import platform
    import subprocess

    import torch

    def run(*cfg):
        cmd = [sys.executable, "-m", "videos.relflow.compute", "_bench", *map(str, cfg)]
        proc = subprocess.run(cmd, cwd=HERE.parents[1], capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            raise RuntimeError(f"benchmark {cfg} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        r = json.loads(line[7:])
        print(f"    {cfg}: {r['ms_per_batch']:.0f} ms/batch, {r['rows_per_s']:.0f} rows/s, peak {r['peak_mib']:.0f} MiB")
        return r

    presets = ["xs", "sm", "md", "lg", "xl"]
    runs = [run(p, 64, mode, "cpu") for p in presets for mode in ("train", "predict")]
    if torch.backends.mps.is_available():
        runs += [run(p, 64, mode, "mps") for p in ("sm", "md", "xl") for mode in ("train", "predict")]
    runs += [run("sm", n, "train", "cpu") for n in (16, 32, 128, 256)]
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if sys.platform == "darwin" else platform.processor()
    memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30 \
        if sys.platform == "darwin" else None
    save("benchmarks", {"machine": {"chip": chip, "memory_gib": memory, "threads": 4, "torch": torch.__version__,
                                    "python": platform.python_version()},
                        "batch": BENCH_BATCH, "steps": BENCH_STEPS, "warmup": BENCH_WARMUP, "runs": runs})


EXPERIMENTS = {"schema": schema, "encode": encode, "number": number, "vocab": vocab, "hash": hashing,
               "packing": packing, "counting": counting, "returns": returns, "rope": rope, "proofs": proofs,
               "happy": happy, "quantile_enum": quantile_enum, "time_vector_text": time_vector_text, "masks": masks,
               "pretraining": pretraining, "processors": processors, "benchmarks": benchmarks}


def main():
    quiet()
    if sys.argv[1:2] == ["_bench"]:  # one benchmark configuration, in a fresh process (see benchmarks())
        preset, length, mode, device = sys.argv[2:6]
        print("RESULT " + json.dumps(bench_one(preset, int(length), mode, device)), flush=True)
        return
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        t = time.time()
        print(f"[{name}]")
        EXPERIMENTS[name]()
        print(f"  done in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
