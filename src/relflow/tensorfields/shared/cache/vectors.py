"""Resolve pristine float32 vectors through the device table and shared artifact store."""

from __future__ import annotations

import numpy as np
import torch

from relflow.tensorfields.shared.cache import MAX_FRAME, MAX_ROWS, Cache, Endpoint, Owner, Payloads
from relflow.tensorfields.shared.cache.device import Device


def vectors(payloads: Payloads, width: int) -> np.ndarray:
    if np.any(np.diff(payloads.offsets) != width * 4):
        raise ValueError("cached vector width does not match encoder namespace")
    return np.frombuffer(payloads.data, dtype="<f4").reshape(-1, width).copy()


class VectorCache:
    vector_namespace: str
    input_bytes: int

    def __init__(self, path: str | None, memory: int | None, capacity: int | None, endpoint: Endpoint | None = None):
        self.path, self.memory, self.capacity = path, memory, capacity
        self.endpoint = endpoint
        self.owner = (
            Owner.open(path, (64 if memory is None else memory) * 2**20, capacity)
            if endpoint is None
            else Owner.hold(endpoint)
        )
        self.connection: Cache | None = None
        self.views: dict[str, Device] = {}

    def __reduce__(self) -> tuple[type, tuple]:
        return type(self), (self.path, self.memory, self.capacity, self.endpoint)

    @property
    def client(self) -> Cache:
        if self.connection is None:
            if self.endpoint is None:
                assert self.owner is not None
                self.endpoint = self.owner.start()
            self.connection = Cache(endpoint=self.endpoint)
        return self.connection

    def share(self) -> None:
        self.client

    @torch.compiler.disable
    def resolve(
        self, identifiers: torch.Tensor, *, variant: str, width: int, resident: int | None, pool
    ) -> torch.Tensor:
        client = self.client
        variant = f"{self.vector_namespace}:{variant}"
        name = f"{identifiers.device}|{variant}"
        view = self.views.get(name)
        if view is None:
            view = Device.open(client.endpoint.identity, variant, width, identifiers.device, resident, self.capacity)
            if view is not None:
                self.views[name] = view
        with client.resolve_lock:
            if view is None:
                return self.fetch(identifiers, variant, width, pool)
            with view.lock:
                found, cached = view.lookup(identifiers)
                if found.all():
                    # A fresh tensor is safe for later training after an
                    # inference_mode warm-up and cannot alias the cache table.
                    return cached.clone()
                output = torch.empty((len(identifiers), width), device=identifiers.device, dtype=torch.float32)
                output[found] = cached[found]
                missing = identifiers[~found]
                host = missing.detach().cpu().numpy()
                unique, inverse, counts = np.unique(host, return_inverse=True, return_counts=True)
                values = self.fetch(torch.from_numpy(unique).to(identifiers.device), variant, width, pool)
                output[~found] = values.index_select(0, torch.from_numpy(inverse).to(identifiers.device))
                view.admit(unique, values, counts)
                return output

    def fetch(self, identifiers: torch.Tensor, variant: str, width: int, pool) -> torch.Tensor:
        host = identifiers.detach().cpu().numpy()
        unique, inverse = np.unique(host, return_inverse=True)
        output = np.empty((len(unique), width), dtype=np.float32)
        chunk = min(MAX_ROWS, (MAX_FRAME - 65536) // (max(width * 4, self.input_bytes) + 8))
        if chunk <= 0:
            raise ValueError("encoder vector or prepared input exceeds the cache frame budget")
        try:
            for start in range(0, len(unique), chunk):
                pending = [np.arange(start, min(start + chunk, len(unique)))]
                while pending:
                    positions = pending.pop()
                    try:
                        fetched = self.client.fetch(unique[positions], variant)
                    except (RuntimeError, ValueError) as error:
                        # A variable-width prepared input can fill a frame before the
                        # vector budget does. Both owners release claims on this error.
                        if len(positions) == 1 or "artifact lookup exceeds 32 MiB" not in str(error):
                            raise
                        middle = len(positions) // 2
                        pending.extend((positions[middle:], positions[:middle]))
                        continue
                    status = fetched.status
                    output[positions[status == 1]] = vectors(fetched.hits, width)
                    owned = positions[status == 0]
                    if len(owned):
                        fresh = pool(fetched.inputs, identifiers.device)
                        values = np.ascontiguousarray(fresh.detach().cpu().numpy(), dtype="<f4")
                        committed = self.client.publish(unique[owned], variant, Payloads.fixed(values, width * 4))
                        output[owned] = vectors(committed, width)
                    waiting = positions[status == 2]
                    if len(waiting):
                        self.client.wait()
                        pending.append(waiting)
        except BaseException:
            self.client.release()
            raise
        return torch.from_numpy(output[inverse]).to(identifiers.device)

    def stats(self) -> dict:
        return self.client.stats()
