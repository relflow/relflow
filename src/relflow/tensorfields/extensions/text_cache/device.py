"""Bounded, frequency-aware device tables addressed by persistent text IDs."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import ClassVar
from weakref import WeakValueDictionary

import numpy as np
import torch


def budget(device: torch.device, resident: int | None) -> int:
    if resident is not None:
        return resident * 2**20
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return min(free // 8, 512 * 2**20)
    return 0


class Device:
    """Only admitted IDs and vectors are indexed; corpus size never sizes this table."""

    instances: ClassVar[WeakValueDictionary[tuple, Device]] = WeakValueDictionary()
    registry_lock: ClassVar[threading.RLock] = threading.RLock()

    @torch.inference_mode(False)
    def __init__(self, key: tuple, width: int, device: torch.device, limit: int, maximum: int | None):
        self.key, self.width, self.device, self.limit, self.maximum = key, width, device, limit, maximum
        self.lock = threading.RLock()
        self.ids = torch.empty(0, dtype=torch.int64, device=device)
        self.vectors = torch.empty((0, width), dtype=torch.float32, device=device)
        self.frequency = torch.empty(0, dtype=torch.int64, device=device)
        self.recency = torch.empty(0, dtype=torch.int64, device=device)
        self.clock = self.accesses = 0
        self.capacity = 0
        self.ghost: OrderedDict[int, int] = OrderedDict()

    @classmethod
    def open(
        cls, identity: str, variant: str, width: int, device: torch.device, resident: int | None, maximum: int | None
    ) -> Device | None:
        limit = budget(device, resident)
        if limit == 0:
            return None
        key = (identity, variant, width, str(device))
        with cls.registry_lock:
            view = cls.instances.get(key)
            if view is None:
                view = cls.instances[key] = cls(key, width, device, limit, maximum)
            else:
                view.limit = min(view.limit, limit)
                if maximum is not None:
                    view.maximum = maximum if view.maximum is None else min(view.maximum, maximum)
            peers = [v for v in cls.instances.values() if v.device == device]
            total = max(budget(device, None), max(v.limit for v in peers))
            quota = total // len(peers)
            for peer in peers:
                # Reserve room for a replacement table and index workspace.
                capacity = min(peer.limit, quota) // (3 * (peer.width * 4 + 32))
                if peer.maximum is not None:
                    capacity = min(capacity, peer.maximum)
                peer.resize(capacity)
            return view

    @torch.inference_mode(False)
    def resize(self, capacity: int) -> None:
        with self.lock:
            self.capacity = capacity
            if len(self.ids) > capacity:
                selected = torch.argsort(self.frequency, descending=True, stable=True)[:capacity]
                selected = selected[torch.argsort(self.ids[selected])]
                self.ids = self.ids[selected]
                self.vectors = self.vectors[selected]
                self.frequency = self.frequency[selected]
                self.recency = self.recency[selected]

    @torch.inference_mode(False)
    def lookup(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not len(self.ids):
            return torch.zeros(len(ids), dtype=torch.bool, device=ids.device), torch.zeros(
                (len(ids), self.width), device=ids.device
            )
        positions = torch.searchsorted(self.ids, ids.contiguous()).clamp_max(len(self.ids) - 1)
        found = self.ids[positions] == ids
        selected = positions[found]
        self.clock += 1
        self.frequency.index_add_(0, selected, torch.ones_like(selected))
        self.recency[selected] = self.clock
        self.accesses += len(ids)
        if self.accesses >= max(self.capacity * 10, 1024):
            self.frequency.div_(2, rounding_mode="floor")
            self.accesses = 0
        return found, self.vectors.index_select(0, positions)

    @torch.inference_mode(False)
    def admit(self, ids: np.ndarray, vectors: torch.Tensor, counts: np.ndarray) -> None:
        if self.capacity == 0 or not len(ids):
            return
        scores = np.empty(len(ids), dtype=np.int64)
        for index, (identifier, count) in enumerate(zip(ids.tolist(), counts.tolist(), strict=True)):
            scores[index] = min(int(self.ghost.get(identifier, 0)) + count, 2**31 - 1)
            self.ghost[identifier] = int(scores[index])
            self.ghost.move_to_end(identifier)
            if len(self.ghost) > 65536:
                self.ghost.popitem(last=False)
        scores = torch.from_numpy(scores).to(self.device)
        candidates = torch.from_numpy(ids).to(self.device)
        if len(self.ids) >= self.capacity:
            keep = scores > self.frequency.min()
            if not keep.any():
                return
            candidates, scores, vectors = candidates[keep], scores[keep], vectors[keep]
        keys = torch.cat((self.ids, candidates))
        values = torch.cat((self.vectors, vectors.detach().to(dtype=torch.float32)))
        frequency = torch.cat((self.frequency, scores))
        recency = torch.cat((self.recency, torch.full_like(candidates, self.clock + 1)))
        order = torch.argsort(recency, descending=True, stable=True)
        order = order[torch.argsort(frequency[order], descending=True, stable=True)[: self.capacity]]
        order = order[torch.argsort(keys[order])]
        self.ids, self.vectors, self.frequency, self.recency = (
            keys[order],
            values[order],
            frequency[order],
            recency[order],
        )
