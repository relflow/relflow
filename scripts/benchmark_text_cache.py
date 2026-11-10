"""Manual production-path cache benchmark on a saved nested text-only workload.

Requires relflow[text], psutil, and CUDA. Supply observations.arrow, fields.json
(Text/Branch request definitions), and observe.json (sampling receipt). Run the
same script against baseline and candidate PYTHONPATHs. This does not fit a
benchmark model: it measures the real loader, frozen text lookup, and a tiny
backward head. No runtime functions are patched. Each output records scope,
input hashes, warm-up, latency, process-tree resources, and cache counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import threading
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "2")

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

import relflow as rf
from relflow.structs.enums import Strata


class Monitor:
    """Process-tree PSS counts shared pages proportionally; RSS is also retained."""

    def __init__(self):
        self.stop = threading.Event()
        self.peak = np.zeros(3)
        self.cpu = {}
        self.sample()
        self.initial = self.cpu.copy()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def sample(self):
        totals = np.zeros(3)
        root = psutil.Process()
        for process in [root, *root.children(recursive=True)]:
            try:
                memory = process.memory_full_info()
                totals += [memory.pss, memory.uss, memory.rss]
                cpu = process.cpu_times()
                self.cpu[process.pid] = cpu.user + cpu.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak = np.maximum(self.peak, totals)

    def run(self):
        while not self.stop.wait(0.25):
            self.sample()

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.sample()
        return dict(
            zip(("peak_tree_pss_mib", "peak_tree_uss_mib", "peak_tree_rss_mib"), (self.peak / 2**20).tolist()),
            cpu_seconds=sum(value - self.initial.get(pid, 0) for pid, value in self.cpu.items()),
        )


def definitions(fields, args):
    output = {}
    for name, field in fields.items():
        options = {key: value for key, value in field.items() if key != "type"}
        if field["type"] == "branch":
            options["fields"] = definitions(options["fields"], args)
            output[name] = rf.Branch(**options)
        elif field["type"] == "text":
            options.update(store=str(args.store), cache=args.mode != "legacy-store")
            if args.mode == "automatic":
                options.update(memory=args.memory, resident=args.resident)
            constructor = rf.Text
            if args.native_backend != "python":
                from relflow_cache.text import Request

                constructor = Request
            output[name] = constructor(**options)
        else:
            raise ValueError("this benchmark expects only Text and Branch definitions")
    return output


def stats(embedders):
    stores = {}
    for embedder in embedders.values():
        store = embedder.store
        if hasattr(store, "stats"):
            stores[store.client.endpoint.identity] = store.stats()
    return list(stores.values())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    os.environ["NATIVE_TEXT_MODE"] = args.native_backend
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    table = ipc.open_file(pa.memory_map(str(args.input / "observations.arrow"))).read_all()
    fields = json.loads((args.input / "fields.json").read_text())
    receipt = json.loads((args.input / "observe.json").read_text())
    model = rf.Model(
        d_model=16, n_layers=1, n_heads=4, batch_size=args.batch_size, fields=definitions(fields, args)
    ).cuda()
    embedders = {
        address: model.nodes[address].embedder
        for address, request in model.schema.requests.items()
        if request.type in ("text", "text_native")
    }
    width = next(iter(embedders.values())).hidden_size
    data = rf.ArrowDataModule(
        model=model,
        train=table,
        num_workers=args.workers,
        persistent_workers=bool(args.workers),
        pin_memory=False,
        shuffle=False,
        retain=(),
    )
    loader = data.train_dataloader()
    head = torch.nn.Linear(width, 1, device="cuda")
    frozen = next(iter(embedders.values())).__dict__["_cached_model"].module_for(torch.device("cuda"))
    encoder_calls = [0]

    def count(module, inputs):
        encoder_calls[0] += 1

    hook = frozen.register_forward_pre_hook(count)
    first_outputs = {}
    parity = 0.0
    content_bytes = None

    def consume(batch, check=False):
        nonlocal parity, content_bytes
        tensors = batch.tensors.to("cuda")
        size = tensors.batch_size[0]
        total = torch.zeros((size, width), device="cuda")
        if content_bytes is None:
            content_bytes = sum(
                t.numel() * t.element_size() for a in embedders for t in batch.tensors[a].content.values()
            )
        for address, embedder in embedders.items():
            field = tensors[address]
            if field.state.numel() == 0:
                continue
            output = embedder.encode(content=field.content, state=field.state)
            total += output.reshape(size, -1, width).sum(dim=1) if output.numel() else 0
            if check:
                actual = output.detach().cpu()
                if address in first_outputs:
                    torch.testing.assert_close(actual, first_outputs[address], rtol=1e-5, atol=1e-6)
                    if actual.numel():
                        parity = max(parity, float((actual - first_outputs[address]).abs().max()))
                else:
                    first_outputs[address] = actual
        head(total / len(embedders)).square().mean().backward()
        head.zero_grad(set_to_none=True)
        return size

    torch.cuda.synchronize()
    start = time.perf_counter()
    for ordinal, batch in enumerate(loader):
        consume(batch, check=ordinal == 0)
    if args.mode != "automatic":
        for store in {id(embedder.store): embedder.store for embedder in embedders.values()}.values():
            store.flush()
        # A fresh legacy store uploads its resident index only after flushing.
        for ordinal, batch in enumerate(loader):
            consume(batch, check=ordinal == 0)
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - start
    before = stats(embedders)
    before_calls = encoder_calls[0]
    monitor = Monitor()
    torch.cuda.reset_peak_memory_stats()
    epochs, waits, steps = [], [], []
    start = time.perf_counter()
    while len(epochs) < args.epochs or sum(epochs) < args.seconds:
        data.epochs[Strata.train] = len(epochs) + 1
        epoch_start = time.perf_counter()
        iterator = iter(loader)
        count_rows = 0
        ordinal = 0
        while True:
            then = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            waits.append(time.perf_counter() - then)
            then = time.perf_counter()
            count_rows += consume(batch, check=ordinal == 0)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - then)
            ordinal += 1
        assert count_rows == len(table)
        epochs.append(time.perf_counter() - epoch_start)
    elapsed = time.perf_counter() - start
    resources = monitor.finish()
    timed_calls = encoder_calls[0] - before_calls
    hook.remove()
    assert timed_calls == 0, f"warm loop performed {timed_calls} frozen encoder calls"
    if args.save_vectors:
        np.savez(args.output.with_suffix(".vectors.npz"), **{str(k): v.numpy() for k, v in first_outputs.items()})
    ready = torch.zeros((args.batch_size, width), device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        head(ready).square().mean().backward()
        head.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    result = dict(
        scope="Saved user-badge text-only nested snapshots; actual loader, Text runtime, tiny backward head. No full model training or quality measurement.",
        task=receipt["task"],
        mode=args.mode,
        native_backend=args.native_backend,
        workers=args.workers,
        memory_mib=args.memory,
        resident_mib=args.resident,
        seed=args.seed,
        batch_size=args.batch_size,
        observations_sha256=digest(args.input / "observations.arrow"),
        fields_sha256=digest(args.input / "fields.json"),
        rows=len(table),
        text_values=receipt["text_values"],
        text_fields=len(embedders),
        warm_epoch_seconds=warm_seconds,
        warm_encoder_calls=before_calls,
        timed_encoder_calls=timed_calls,
        epochs=len(epochs),
        epoch_seconds=epochs,
        rows_per_second=len(table) * len(epochs) / sum(epochs),
        text_values_per_second=receipt["text_values"] * len(epochs) / sum(epochs),
        loader_wait_p50_ms=float(np.median(waits) * 1000),
        loader_wait_p95_ms=float(np.percentile(waits, 95) * 1000),
        step_p50_ms=float(np.median(steps) * 1000),
        step_p95_ms=float(np.percentile(steps, 95) * 1000),
        first_batch_text_bytes=content_bytes,
        warm_repeat_max_error=parity,
        average_cpu_cores=resources["cpu_seconds"] / elapsed,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        head_ceiling_rows_per_second=200 * args.batch_size / (time.perf_counter() - start),
        cache_before=before,
        cache_after=stats(embedders),
        host=platform.node(),
        python=platform.python_version(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        **resources,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--mode", choices=("automatic", "legacy-store", "legacy-local"), default="automatic")
    parser.add_argument("--native-backend", choices=("python", "owner", "full"), default="python")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--memory", type=int, default=64)
    parser.add_argument("--resident", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=7302)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-vectors", action="store_true")
    run(parser.parse_args())
